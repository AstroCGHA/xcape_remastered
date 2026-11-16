"""CAPE / CIN and parcel LCL in numba.

A line-by-line port of the Fortran kernel (George H. Bryan's ``getcape``,
see ``fortran/xcaper_kernels.f90`` for the licence notice) so that xcaper
also works where the Fortran extensions could not be built, and so that
numpy inputs can use every core through ``numba.prange``.

The arithmetic is done in double precision (the Fortran kernel is single
precision), so results agree with ``method='fortran'`` to a small fraction
of a J/kg rather than bit for bit.

Two flavours of every grid loop are compiled:

* ``*_parallel`` - ``prange`` over grid points, for numpy inputs;
* ``*_serial``   - plain loop that releases the GIL, for use inside dask
  tasks (dask threads then provide the parallelism; nesting a numba
  parallel region inside dask threads would oversubscribe the machine and
  is not safe with numba's default ``workqueue`` threading layer).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ._parallel import run_loop

# ---------------------------------------------------------------------------
# constants (identical to getcape)
# ---------------------------------------------------------------------------
_G = 9.81
_P00 = 100000.0
_CP = 1005.7
_RD = 287.04
_RV = 461.5
_XLV = 2501000.0
_XLS = 2836017.0
_T0 = 273.15
_CPV = 1875.0
_CPL = 4190.0
_CPI = 2118.636
_LV1 = _XLV + (_CPL - _CPV) * _T0
_LV2 = _CPL - _CPV
_LS1 = _XLS + (_CPI - _CPV) * _T0
_LS2 = _CPI - _CPV

_RP00 = 1.0 / _P00
_EPS = _RD / _RV
_REPS = _RV / _RD
_RDDCP = _RD / _CP
_CPDG = _CP / _G

_CONVERGE = 0.0002
_MISSING = -999999.0
_MISSING_INT = -999999


# ---------------------------------------------------------------------------
# thermodynamic helpers
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _qvs_liquid(p, t):
    """Saturation mixing ratio (kg/kg) over liquid; p in Pa, t in K."""
    es = 611.2 * math.exp(17.67 * (t - 273.15) / (t - 29.65))
    return _EPS * es / (p - es)


@njit(cache=True, nogil=True)
def _qvs_ice(p, t):
    """Saturation mixing ratio (kg/kg) over ice; p in Pa, t in K."""
    es = 611.2 * math.exp(21.8745584 * (t - 273.15) / (t - 7.66))
    return _EPS * es / (p - es)


@njit(cache=True, nogil=True)
def _theta_e(p, t, td, q):
    """Equivalent potential temperature (K), Bolton (1980)."""
    if (td - t) >= -0.1:
        tlcl = t
    else:
        tlcl = 56.0 + 1.0 / (1.0 / (td - 56.0) + 0.00125 * math.log(t / td))
    return (t * ((100000.0 / p) ** (0.2854 * (1.0 - 0.28 * q)))
            * math.exp(((3376.0 / tlcl) - 2.54) * q * (1.0 + 0.81 * q)))


# ---------------------------------------------------------------------------
# profile set-up and source-parcel selection (shared by CAPE and LCL)
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _setup_profile(p_in, t_in, td_in, ps, ts, tds, w):
    """Fill the work array ``w`` (8, nk) with the sounding in MKS units.

    Rows of ``w``: 0 p (Pa), 1 t (K), 2 td (K), 3 exner, 4 q (kg/kg),
    5 z (m above the surface level), 6 theta (K), 7 theta_v (K).
    The surface values become level 0.
    """
    nk = p_in.shape[0] + 1
    w[0, 0] = 100.0 * ps
    w[1, 0] = 273.15 + ts
    w[2, 0] = 273.15 + tds
    for k in range(1, nk):
        w[0, k] = 100.0 * p_in[k - 1]
        w[1, k] = 273.15 + t_in[k - 1]
        w[2, k] = 273.15 + td_in[k - 1]
    for k in range(nk):
        w[3, k] = (w[0, k] * _RP00) ** _RDDCP
        w[4, k] = _qvs_liquid(w[0, k], w[2, k])
        w[6, k] = w[1, k] / w[3, k]
        w[7, k] = w[6, k] * (1.0 + _REPS * w[4, k]) / (1.0 + w[4, k])
    # height from the hydrostatic equation
    w[5, 0] = 0.0
    for k in range(1, nk):
        dz = -_CPDG * 0.5 * (w[7, k] + w[7, k - 1]) * (w[3, k] - w[3, k - 1])
        w[5, k] = w[5, k - 1] + dz
    return nk


@njit(cache=True, nogil=True)
def _select_parcel(source, ml_depth, nk, w):
    """Pick the source parcel.

    Returns ``(kmax, mulvl, th, qv)`` with ``kmax`` the 0-based start level
    of the ascent, ``mulvl`` the 1-based most-unstable level (or -999999)
    and, for the mixed-layer parcel, its mean theta and mixing ratio.
    """
    p, t, td, q, z, th = w[0], w[1], w[2], w[4], w[5], w[6]
    kmax = 0
    mulvl = _MISSING_INT
    avgth = 0.0
    avgqv = 0.0

    if source == 2:
        # most unstable parcel: maximum theta-e below 500 hPa
        if p[0] >= 50000.0:
            maxthe = 0.0
            for k in range(nk):
                if p[k] >= 50000.0:
                    the = _theta_e(p[k], t[k], td[k], q[k])
                    if the > maxthe:
                        mulvl = k + 1
                        maxthe = the
                        kmax = k
    elif source == 3:
        if nk < 2 or (z[1] - z[0]) > ml_depth:
            # second level is above the mixed layer: use the lowest level
            avgth = th[0]
            avgqv = q[0]
            kmax = 0
        elif z[nk - 1] < ml_depth:
            # top-most level is within the mixed layer: use it
            avgth = th[nk - 1]
            avgqv = q[nk - 1]
            kmax = nk - 1
        else:
            k = 1
            while k < nk and z[k] <= ml_depth:
                avgth += 0.5 * (z[k] - z[k - 1]) * (th[k] + th[k - 1])
                avgqv += 0.5 * (z[k] - z[k - 1]) * (q[k] + q[k - 1])
                k += 1
            if k < nk:
                frac = (ml_depth - z[k - 1]) / (z[k] - z[k - 1])
                th2 = th[k - 1] + (th[k] - th[k - 1]) * frac
                qv2 = q[k - 1] + (q[k] - q[k - 1]) * frac
                avgth += 0.5 * (ml_depth - z[k - 1]) * (th2 + th[k - 1])
                avgqv += 0.5 * (ml_depth - z[k - 1]) * (qv2 + q[k - 1])
            avgth /= ml_depth
            avgqv /= ml_depth
            kmax = 0
    return kmax, mulvl, avgth, avgqv


# ---------------------------------------------------------------------------
# single-column CAPE / CIN
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def getcape(p_in, t_in, td_in, ps, ts, tds, pinc, source, ml_depth, adiabat, w):
    """CAPE and CIN of one sounding.

    Parameters
    ----------
    p_in, t_in, td_in : 1-D arrays
        Pressure (hPa), temperature (C) and dewpoint (C), bottom to top,
        *without* the surface level.
    ps, ts, tds : float
        Surface pressure (hPa), temperature (C) and dewpoint (C).
    pinc : float
        Pressure increment (Pa) of the ascent.
    source : int
        1 surface, 2 most unstable, 3 mixed layer.
    ml_depth : float
        Mixed-layer depth (m).
    adiabat : int
        1 pseudo-liquid, 2 reversible-liquid, 3 pseudo-ice, 4 reversible-ice.
    w : 2-D float64 work array, at least ``(8, len(p_in) + 1)``.

    Returns
    -------
    cape, cin, mulvl, z_out
        ``z_out`` is the height (m above the surface) of the last level
        reached by the ascent, as in the Fortran kernel.
    """
    cape = 0.0
    cin = 0.0
    z_out = _MISSING
    if source < 1 or source > 3 or adiabat < 1 or adiabat > 4:
        return cape, cin, _MISSING_INT, z_out

    nk = _setup_profile(p_in, t_in, td_in, ps, ts, tds, w)
    p, t, pi, q, z, th, thv = w[0], w[1], w[3], w[4], w[5], w[6], w[7]

    kmax, mulvl, avgth, avgqv = _select_parcel(source, ml_depth, nk, w)

    # ---- parcel properties at the initial location ----
    narea = 0.0
    k = kmax
    if source == 3:
        th2 = avgth
        qv2 = avgqv
        thv2 = th2 * (1.0 + _REPS * qv2) / (1.0 + qv2)
        pi2 = pi[kmax]
        p2 = p[kmax]
        t2 = th2 * pi2
        b2 = _G * (thv2 - thv[kmax]) / thv[kmax]
    else:
        th2 = th[kmax]
        pi2 = pi[kmax]
        p2 = p[kmax]
        t2 = t[kmax]
        thv2 = thv[kmax]
        qv2 = q[kmax]
        b2 = 0.0

    ql2 = 0.0
    qi2 = 0.0
    qt = qv2

    doit = True
    ice = adiabat == 3 or adiabat == 4
    pseudo = adiabat == 1 or adiabat == 3

    # ---- ascent of the parcel ----
    while doit and k < nk - 1:
        k += 1
        b1 = b2

        dp = p[k - 1] - p[k]
        if dp < pinc:
            nloop = 1
        elif dp == dp and dp < 1.0e7:
            nloop = 1 + int(dp / pinc)
            dp = dp / nloop
        else:
            # NaN (or absurd) pressure: nothing sensible can be computed
            return np.nan, np.nan, mulvl, z_out

        for _ in range(nloop):
            p1 = p2
            t1 = t2
            th1 = th2
            qv1 = qv2
            ql1 = ql2
            qi1 = qi2

            p2 = p2 - dp
            pi2 = (p2 * _RP00) ** _RDDCP

            thlast = th1
            i = 0
            not_converged = True
            while not_converged:
                i += 1
                t2 = thlast * pi2
                if ice:
                    fliq = max(min((t2 - 233.15) / (273.15 - 233.15), 1.0), 0.0)
                    fice = 1.0 - fliq
                    qv2 = min(qt, fliq * _qvs_liquid(p2, t2) + fice * _qvs_ice(p2, t2))
                else:
                    fliq = 1.0
                    fice = 0.0
                    qv2 = min(qt, _qvs_liquid(p2, t2))
                qi2 = max(fice * (qt - qv2), 0.0)
                ql2 = max(qt - qv2 - qi2, 0.0)

                tbar = 0.5 * (t1 + t2)
                qvbar = 0.5 * (qv1 + qv2)
                qlbar = 0.5 * (ql1 + ql2)
                qibar = 0.5 * (qi1 + qi2)

                lhv = _LV1 - _LV2 * tbar
                lhs = _LS1 - _LS2 * tbar

                rm = _RD + _RV * qvbar
                cpm = _CP + _CPV * qvbar + _CPL * qlbar + _CPI * qibar
                th2 = th1 * math.exp(lhv * (ql2 - ql1) / (cpm * tbar)
                                     + lhs * (qi2 - qi1) / (cpm * tbar)
                                     + (rm / cpm - _RD / _CP) * math.log(p2 / p1))

                if i > 100:
                    # lack of convergence: give up on this column
                    return 0.0, 0.0, mulvl, z_out
                if abs(th2 - thlast) > _CONVERGE:
                    thlast = thlast + 0.3 * (th2 - thlast)
                else:
                    not_converged = False

            if pseudo:
                qt = qv2
                ql2 = 0.0
                qi2 = 0.0

        thv2 = th2 * (1.0 + _REPS * qv2) / (1.0 + qv2 + ql2 + qi2)
        b2 = _G * (thv2 - thv[k]) / thv[k]
        dz = -_CPDG * 0.5 * (thv[k] + thv[k - 1]) * (pi[k] - pi[k - 1])

        # contributions to CAPE and CIN
        if b2 >= 0.0 and b1 < 0.0:
            # first trip into positive area
            frac = b2 / (b2 - b1)
            parea = 0.5 * b2 * dz * frac
            narea = narea - 0.5 * b1 * dz * (1.0 - frac)
            cin = cin + narea
            narea = 0.0
        elif b2 < 0.0 and b1 > 0.0:
            # first trip into negative area
            frac = b1 / (b1 - b2)
            parea = 0.5 * b1 * dz * frac
            narea = -0.5 * b2 * dz * (1.0 - frac)
        elif b2 < 0.0:
            # still collecting negative buoyancy
            parea = 0.0
            narea = narea - 0.5 * dz * (b1 + b2)
        else:
            # still collecting positive buoyancy
            parea = 0.5 * dz * (b1 + b2)
            narea = 0.0

        cape = cape + max(0.0, parea)

        if p[k] <= 10000.0 and b2 < 0.0:
            # stop if b < 0 and p < 100 mb
            doit = False
        z_out = z[k]

    return cape, cin, mulvl, z_out


# ---------------------------------------------------------------------------
# single-column LCL of the source parcel
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def parcel_lcl(p_in, t_in, td_in, ps, ts, tds, source, ml_depth, w):
    """Lifting condensation level of the surface / most-unstable / mixed-layer parcel.

    The parcel is defined exactly as in :func:`getcape`.  Its LCL
    temperature follows Bolton (1980, eq. 15); the LCL pressure follows
    from dry-adiabatic ascent (Poisson) and the LCL height from the
    conservation of dry static energy, ``z = z0 + cp/g * (T - T_lcl)``.

    Returns
    -------
    z_lcl : float
        Height of the LCL in m above the surface.
    p_lcl : float
        Pressure of the LCL in hPa.
    t_lcl : float
        Temperature of the LCL in K.
    """
    if source < 1 or source > 3:
        return np.nan, np.nan, np.nan
    nk = _setup_profile(p_in, t_in, td_in, ps, ts, tds, w)
    kmax, _, avgth, avgqv = _select_parcel(source, ml_depth, nk, w)

    p0 = w[0, kmax]
    z0 = w[5, kmax]
    if source == 3:
        t0 = avgth * w[3, kmax]
        qv = avgqv
    else:
        t0 = w[1, kmax]
        qv = w[4, kmax]

    if not (qv > 0.0) or not (t0 > 0.0):
        return np.nan, np.nan, np.nan

    # dewpoint of the parcel: invert es(T) = 611.2 exp(17.67 (T-273.15)/(T-29.65))
    e = qv * p0 / (_EPS + qv)
    lne = math.log(e / 611.2)
    td0 = (243.5 * lne) / (17.67 - lne) + 273.15
    if td0 >= t0:
        # saturated (or super-saturated) parcel: LCL at the parcel level
        return z0, p0 * 0.01, t0

    t_lcl = 56.0 + 1.0 / (1.0 / (td0 - 56.0) + math.log(t0 / td0) / 800.0)
    p_lcl = p0 * (t_lcl / t0) ** (1.0 / _RDDCP)
    z_lcl = z0 + _CPDG * (t0 - t_lcl)
    return z_lcl, p_lcl * 0.01, t_lcl


# ---------------------------------------------------------------------------
# grid loops
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _cape_point(i, p2d, t2d, td2d, ps, ts, tds, start, pinc, source, ml_depth,
                adiabat, cape, cin, mulev, zmulev):
    # Points whose surface temperature is not above 0 C (which includes
    # NaN) are skipped, exactly like the Fortran drivers.
    if ts[i] > 0.0:
        nk = t2d.shape[1]
        k0 = min(max(start[i] - 1, 0), nk)
        ip = np.intp(i) if p2d.shape[0] > 1 else np.intp(0)
        w = np.empty((8, nk - k0 + 1))
        c, ci, mu, zo = getcape(p2d[ip, k0:], t2d[i, k0:], td2d[i, k0:],
                                ps[i], ts[i], tds[i],
                                pinc, source, ml_depth, adiabat, w)
        cape[i] = c
        cin[i] = ci
        mulev[i] = mu
        zmulev[i] = zo
    else:
        cape[i] = 0.0
        cin[i] = 0.0
        mulev[i] = 0
        zmulev[i] = 0.0


@njit(cache=True, nogil=True)
def _loopcape_serial(p2d, t2d, td2d, ps, ts, tds, start, pinc, source, ml_depth, adiabat):
    n = t2d.shape[0]
    cape = np.empty(n, dtype=np.float32)
    cin = np.empty(n, dtype=np.float32)
    mulev = np.empty(n, dtype=np.int32)
    zmulev = np.empty(n, dtype=np.float32)
    for i in range(n):
        _cape_point(i, p2d, t2d, td2d, ps, ts, tds, start, pinc, source, ml_depth,
                    adiabat, cape, cin, mulev, zmulev)
    return cape, cin, mulev, zmulev


@njit(cache=True, nogil=True, parallel=True)
def _loopcape_parallel(p2d, t2d, td2d, ps, ts, tds, start, pinc, source, ml_depth, adiabat):
    n = t2d.shape[0]
    cape = np.empty(n, dtype=np.float32)
    cin = np.empty(n, dtype=np.float32)
    mulev = np.empty(n, dtype=np.int32)
    zmulev = np.empty(n, dtype=np.float32)
    for i in prange(n):
        _cape_point(i, p2d, t2d, td2d, ps, ts, tds, start, pinc, source, ml_depth,
                    adiabat, cape, cin, mulev, zmulev)
    return cape, cin, mulev, zmulev


@njit(cache=True, nogil=True)
def _lcl_point(i, p2d, t2d, td2d, ps, ts, tds, start, source, ml_depth, out):
    nk = t2d.shape[1]
    k0 = min(max(start[i] - 1, 0), nk)
    ip = np.intp(i) if p2d.shape[0] > 1 else np.intp(0)
    w = np.empty((8, nk - k0 + 1))
    z_lcl, p_lcl, t_lcl = parcel_lcl(p2d[ip, k0:], t2d[i, k0:], td2d[i, k0:],
                                     ps[i], ts[i], tds[i], source, ml_depth, w)
    out[i, 0] = z_lcl
    out[i, 1] = p_lcl
    out[i, 2] = t_lcl


@njit(cache=True, nogil=True)
def _looplcl_serial(p2d, t2d, td2d, ps, ts, tds, start, source, ml_depth):
    n = t2d.shape[0]
    out = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        _lcl_point(i, p2d, t2d, td2d, ps, ts, tds, start, source, ml_depth, out)
    return out


@njit(cache=True, nogil=True, parallel=True)
def _looplcl_parallel(p2d, t2d, td2d, ps, ts, tds, start, source, ml_depth):
    n = t2d.shape[0]
    out = np.empty((n, 3), dtype=np.float32)
    for i in prange(n):
        _lcl_point(i, p2d, t2d, td2d, ps, ts, tds, start, source, ml_depth, out)
    return out


# ---------------------------------------------------------------------------
# python-level entry points (same calling convention as cape_fortran.cape)
# ---------------------------------------------------------------------------
def _as_columns(a_2d):
    """(nk, ngrid) array (any layout / dtype) -> C-contiguous float64 (ngrid, nk)."""
    return np.ascontiguousarray(np.asarray(a_2d).T, dtype=np.float64)


def _as_vector(a, n=None, dtype=np.float64):
    a = np.asarray(a)
    if n is not None:
        a = np.broadcast_to(a, (n,))
    return np.ascontiguousarray(a, dtype=dtype)


def cape(p_2d, t_2d, td_2d, p_s, t_s, td_s, flag_1d, pres_lev_pos, source,
         ml_depth, adiabat, pinc, type_grid, parallel=True):
    """numba twin of :func:`xcaper.cape_fortran.cape` (same arguments).

    ``parallel=False`` runs a serial, GIL-free loop (for dask tasks).
    """
    nlev, ngrid = t_2d.shape
    t = _as_columns(t_2d)
    td = _as_columns(td_2d)
    p = _as_columns(p_2d)  # (ngrid, nk) or (1, nk)
    if type_grid == 2:
        start = _as_vector(pres_lev_pos, ngrid, np.int64)
    else:
        start = np.ones(ngrid, dtype=np.int64)
    return run_loop(_loopcape_parallel, _loopcape_serial, parallel,
                    p, t, td, _as_vector(p_s), _as_vector(t_s), _as_vector(td_s), start,
                    float(pinc), int(source), float(ml_depth), int(adiabat))


def lcl(p_2d, t_2d, td_2d, p_s, t_s, td_s, pres_lev_pos, source, ml_depth,
        type_grid, parallel=True):
    """LCL of the source parcel for a collapsed grid.

    Arguments as :func:`cape`.  Returns ``(z_lcl, p_lcl, t_lcl)``: height
    above the surface (m), pressure (hPa) and temperature (K), each of
    shape ``(ngrid,)``.
    """
    nlev, ngrid = t_2d.shape
    t = _as_columns(t_2d)
    td = _as_columns(td_2d)
    p = _as_columns(p_2d)
    if type_grid == 2:
        start = _as_vector(pres_lev_pos, ngrid, np.int64)
    else:
        start = np.ones(ngrid, dtype=np.int64)
    out = run_loop(_looplcl_parallel, _looplcl_serial, parallel,
                   p, t, td, _as_vector(p_s), _as_vector(t_s), _as_vector(td_s), start,
                   int(source), float(ml_depth))
    return out[:, 0], out[:, 1], out[:, 2]
