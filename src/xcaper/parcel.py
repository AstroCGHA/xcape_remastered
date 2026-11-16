"""Parcel diagnostics: LFC, EL, effective inflow layer and DCAPE (numba).

One column kernel computes, for every grid point,

* the level of free convection (LFC) and equilibrium level (EL) of the
  surface, mixed-layer and most-unstable parcels;
* the effective inflow layer (Thompson et al. 2007): the lowest contiguous
  layer whose parcels have CAPE >= 100 J/kg and CIN <= 250 J/kg;
* downdraft CAPE (DCAPE), following MetPy's ``downdraft_cape``.

The parcels are lifted exactly as in :func:`xcaper.core.calc_cape` (George
Bryan's ``getcape``; the ascent is the numba port in
:mod:`xcaper.cape_numba`), so LFC, EL and the effective layer are consistent
with the CAPE and CIN of xcaper.

Heights are metres above the surface, from the same hydrostatic
integration (virtual temperature) that getcape uses; pressures are hPa.

numpy inputs are processed with a ``numba.prange`` loop over all cores;
dask inputs lazily, chunk by chunk.
"""
from __future__ import annotations

import math

import dask.array as da
import numpy as np
from numba import njit, prange

from ._parallel import run_loop
from .cape_numba import (_CP, _CPDG, _CPI, _CPL, _CPV, _CONVERGE, _EPS, _G, _LS1, _LS2, _LV1,
                         _LV2, _RD, _RDDCP, _REPS, _RP00, _RV, _qvs_ice, _qvs_liquid,
                         _select_parcel, _setup_profile, _theta_e)

__all__ = ["calc_parcel_diagnostics", "PARCEL_NAMES"]

_XLV = 2501000.0          # latent heat of vaporisation used by the moist lapse rate
_LN_STEP = 0.005          # max ln(p) step of the moist-adiabat integration (~5 hPa)

# order of the kernel outputs
PARCEL_NAMES = (
    "SFC_LFC", "SFC_LFC_P", "SFC_EL", "SFC_EL_P",
    "ML_LFC", "ML_LFC_P", "ML_EL", "ML_EL_P",
    "MU_LFC", "MU_LFC_P", "MU_EL", "MU_EL_P",
    "EIL_BASE", "EIL_BASE_P", "EIL_TOP", "EIL_TOP_P",
    "DCAPE", "DCAPE_SRC_P",
)
_NOUT = len(PARCEL_NAMES) + 1     # + internal validity flag


# ---------------------------------------------------------------------------
# parcel ascent with LFC / EL tracking
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _lcl_pressure(p0, t0, qv):
    """LCL pressure (Pa) of a parcel at p0 (Pa), t0 (K), mixing ratio qv (Bolton 1980)."""
    if not (qv > 0.0):
        return np.nan
    e = qv * p0 / (_EPS + qv)
    lne = math.log(e / 611.2)
    td0 = (243.5 * lne) / (17.67 - lne) + 273.15
    if td0 >= t0:
        return p0
    t_lcl = 56.0 + 1.0 / (1.0 / (td0 - 56.0) + math.log(t0 / td0) / 800.0)
    return p0 * (t_lcl / t0) ** (1.0 / _RDDCP)


@njit(cache=True, nogil=True)
def _interp_lnp(p1, p2, f):
    """Pressure a fraction f of the way from p1 to p2, linear in ln(p).

    Exact at the end points: exp(log(p)) is not always p in floating point,
    which used to misplace crossings that sit exactly on a level (e.g. the
    start of a saturated parcel) relative to the LCL.
    """
    if f <= 0.0:
        return p1
    if f >= 1.0:
        return p2
    return math.exp(math.log(p1) + f * (math.log(p2) - math.log(p1)))


@njit(cache=True, nogil=True)
def _ascent(w, nk, kstart, th2, qv2, t2, b2, pinc, adiabat, cape_stop, res):
    """Lift a parcel from level ``kstart`` of the work array ``w``.

    The arithmetic is that of :func:`xcaper.cape_numba.getcape` (same CAPE and
    CIN); in addition the LFC and EL are tracked.  When ``cape_stop`` is
    finite the ascent stops as soon as CAPE reaches it (CIN is then the
    inhibition accumulated below that point).

    ``res`` (6 floats) receives cape, cin, lfc_p, lfc_z, el_p, el_z
    (Pa and m above the surface; NaN when undefined).
    """
    p, pi, z, thv = w[0], w[3], w[5], w[7]
    cape = 0.0
    cin = 0.0
    narea = 0.0
    lfc_p = np.nan
    lfc_z = np.nan
    el_p = np.nan
    el_z = np.nan

    pi2 = pi[kstart]
    p2 = p[kstart]
    p_lcl = _lcl_pressure(p2, t2, qv2)
    ql2 = 0.0
    qi2 = 0.0
    qt = qv2
    ice = adiabat == 3 or adiabat == 4
    pseudo = adiabat == 1 or adiabat == 3

    k = kstart
    doit = True
    stopped = False
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
            res[0] = np.nan
            res[1] = np.nan
            res[2] = np.nan
            res[3] = np.nan
            res[4] = np.nan
            res[5] = np.nan
            return

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
                    # no convergence: same outcome as getcape
                    res[0] = 0.0
                    res[1] = 0.0
                    res[2] = np.nan
                    res[3] = np.nan
                    res[4] = np.nan
                    res[5] = np.nan
                    return
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

        # ---- CAPE / CIN (identical to getcape) ----
        if b2 >= 0.0 and b1 < 0.0:
            frac = b2 / (b2 - b1)
            parea = 0.5 * b2 * dz * frac
            narea = narea - 0.5 * b1 * dz * (1.0 - frac)
            cin = cin + narea
            narea = 0.0
        elif b2 < 0.0 and b1 > 0.0:
            frac = b1 / (b1 - b2)
            parea = 0.5 * b1 * dz * frac
            narea = -0.5 * b2 * dz * (1.0 - frac)
        elif b2 < 0.0:
            parea = 0.0
            narea = narea - 0.5 * dz * (b1 + b2)
        else:
            parea = 0.5 * dz * (b1 + b2)
            narea = 0.0
        cape = cape + max(0.0, parea)

        # ---- LFC: lowest point at or above the LCL where the parcel is
        #      positively buoyant (the LCL itself when buoyant there) ----
        if lfc_p != lfc_p:
            cand_p = np.nan
            cand_z = np.nan
            if p[k - 1] >= p_lcl and p_lcl > p[k]:
                f = math.log(p[k - 1] / p_lcl) / math.log(p[k - 1] / p[k])
                if b1 + f * (b2 - b1) > 0.0:
                    cand_p = p_lcl
                    cand_z = z[k - 1] + f * (z[k] - z[k - 1])
            if b1 <= 0.0 and b2 > 0.0:
                f = -b1 / (b2 - b1)
                px = _interp_lnp(p[k - 1], p[k], f)
                if px <= p_lcl * (1.0 + 1e-12) and not (px <= cand_p):
                    cand_p = px
                    cand_z = z[k - 1] + f * (z[k] - z[k - 1])
            if cand_p == cand_p:
                lfc_p = cand_p
                lfc_z = cand_z

        # ---- EL: top of the highest positively buoyant layer above the LFC ----
        if lfc_p == lfc_p and b1 > 0.0 and b2 <= 0.0:
            f = b1 / (b1 - b2)
            px = _interp_lnp(p[k - 1], p[k], f)
            if px < lfc_p:
                el_p = px
                el_z = z[k - 1] + f * (z[k] - z[k - 1])

        if cape >= cape_stop:
            stopped = True
            break
        if p[k] <= 10000.0 and b2 < 0.0:
            doit = False

    # still buoyant at the top of the data: the EL is not resolved, use the top
    if not stopped and lfc_p == lfc_p and b2 > 0.0 and k > kstart:
        el_p = p[k]
        el_z = z[k]

    res[0] = cape
    res[1] = cin
    res[2] = lfc_p
    res[3] = lfc_z
    res[4] = el_p
    res[5] = el_z


@njit(cache=True, nogil=True)
def _lift_source(w, nk, source, ml_depth, pinc, adiabat, cape_stop, res):
    """Lift the surface (1), most-unstable (2) or mixed-layer (3) parcel."""
    kmax, _, avgth, avgqv = _select_parcel(source, ml_depth, nk, w)
    if source == 3:
        th2 = avgth
        qv2 = avgqv
        thv2 = th2 * (1.0 + _REPS * qv2) / (1.0 + qv2)
        t2 = th2 * w[3, kmax]
        b2 = _G * (thv2 - w[7, kmax]) / w[7, kmax]
    else:
        th2 = w[6, kmax]
        qv2 = w[4, kmax]
        t2 = w[1, kmax]
        b2 = 0.0
    _ascent(w, nk, kmax, th2, qv2, t2, b2, pinc, adiabat, cape_stop, res)


@njit(cache=True, nogil=True)
def _passes(w, nk, k, pinc, adiabat, cape_min, cin_max, res):
    """Does the parcel of level k meet the effective-layer criteria?"""
    _ascent(w, nk, k, w[6, k], w[4, k], w[1, k], 0.0, pinc, adiabat, cape_min, res)
    return res[0] >= cape_min and res[1] <= cin_max


# ---------------------------------------------------------------------------
# moist adiabat (pseudoadiabatic, liquid) for DCAPE
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _moist_dtdlnp(p, t):
    """dT/dln(p) along the pseudoadiabat (same form as MetPy's moist_lapse)."""
    rs = _qvs_liquid(p, t)
    return (_RD * t + _XLV * rs) / (_CP + _XLV * _XLV * rs * _EPS / (_RD * t * t))


@njit(cache=True, nogil=True)
def _moist_follow(t, p_from, p_to):
    """Temperature (K) at p_to of a saturated parcel at (p_from, t), RK4 in ln(p)."""
    x0 = math.log(p_from)
    x1 = math.log(p_to)
    n = max(1, int(math.ceil(abs(x1 - x0) / _LN_STEP)))
    h = (x1 - x0) / n
    x = x0
    for _ in range(n):
        k1 = _moist_dtdlnp(math.exp(x), t)
        k2 = _moist_dtdlnp(math.exp(x + 0.5 * h), t + 0.5 * h * k1)
        k3 = _moist_dtdlnp(math.exp(x + 0.5 * h), t + 0.5 * h * k2)
        k4 = _moist_dtdlnp(math.exp(x + h), t + h * k3)
        t = t + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        x = x + h
    return t


@njit(cache=True, nogil=True)
def _wet_bulb(p, t, td):
    """Wet-bulb temperature (K): LCL (Bolton 1980), then down the moist adiabat."""
    if td >= t:
        return t
    t_lcl = 56.0 + 1.0 / (1.0 / (td - 56.0) + math.log(t / td) / 800.0)
    p_lcl = p * (t_lcl / t) ** (1.0 / _RDDCP)
    return _moist_follow(t_lcl, p_lcl, p)


@njit(cache=True, nogil=True)
def _dcape(w, nk, p_bot, p_top, res):
    """DCAPE (J/kg) and its source pressure (Pa); MetPy ``downdraft_cape`` recipe.

    Source: the level of minimum equivalent potential temperature between
    ``p_bot`` (or the surface if it is higher up) and ``p_top``.  The parcel
    starts at its wet-bulb temperature and descends along the pseudoadiabat
    to the surface; DCAPE = Rd * integral of (Tv_env - Tv_parcel) d ln(p).
    """
    p, t, td, q = w[0], w[1], w[2], w[4]
    res[0] = np.nan
    res[1] = np.nan
    kmin = -1
    the_min = 1.0e30
    for k in range(nk):
        if p[k] <= p_bot and p[k] >= p_top:
            the = _theta_e(p[k], t[k], td[k], q[k])
            if the < the_min:
                the_min = the
                kmin = k
    if kmin < 0:
        return
    tp = _wet_bulb(p[kmin], t[kmin], td[kmin])
    total = 0.0
    rs = _qvs_liquid(p[kmin], tp)
    diff_prev = (t[kmin] * (1.0 + _REPS * q[kmin]) / (1.0 + q[kmin])
                 - tp * (1.0 + _REPS * rs) / (1.0 + rs))
    for k in range(kmin - 1, -1, -1):
        tp = _moist_follow(tp, p[k + 1], p[k])
        rs = _qvs_liquid(p[k], tp)
        diff = (t[k] * (1.0 + _REPS * q[k]) / (1.0 + q[k])
                - tp * (1.0 + _REPS * rs) / (1.0 + rs))
        total += 0.5 * (diff + diff_prev) * math.log(p[k] / p[k + 1])
        diff_prev = diff
    res[0] = _RD * total
    res[1] = p[kmin]


# ---------------------------------------------------------------------------
# refined profile for the effective-layer search
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _refined_profile(w, nk, dp_ref, p_ref_top):
    """Work array of the sounding with extra levels every ``dp_ref`` Pa.

    Temperature and dewpoint are interpolated linearly in ln(p) between the
    input levels; extra levels are only added below ``p_ref_top`` (Pa).
    Returns the new work array and its number of levels.
    """
    nmax = nk + int((w[0, 0] - p_ref_top) / dp_ref) + nk + 2
    pr = np.empty(nmax)
    tr = np.empty(nmax)
    tdr = np.empty(nmax)
    n = 0
    for k in range(1, nk):
        p1 = w[0, k - 1]
        p2 = w[0, k]
        if p1 > p_ref_top and p1 - p2 > dp_ref:
            nsub = int(math.ceil((p1 - p2) / dp_ref))
            for m in range(1, nsub):
                pm = p1 - m * (p1 - p2) / nsub
                f = math.log(p1 / pm) / math.log(p1 / p2)
                pr[n] = pm
                tr[n] = w[1, k - 1] + f * (w[1, k] - w[1, k - 1])
                tdr[n] = w[2, k - 1] + f * (w[2, k] - w[2, k - 1])
                n += 1
        pr[n] = p2
        tr[n] = w[1, k]
        tdr[n] = w[2, k]
        n += 1
    w2 = np.empty((8, n + 1))
    nk2 = _setup_profile(pr[:n] * 0.01, tr[:n] - 273.15, tdr[:n] - 273.15,
                         w[0, 0] * 0.01, w[1, 0] - 273.15, w[2, 0] - 273.15, w2)
    return w2, nk2


# ---------------------------------------------------------------------------
# column driver and grid loops
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _parcel_point(i, p2d, t2d, td2d, ps, ts, tds, start, pinc, ml_depth, adiabat,
                  cape_min, cin_max, p_search_top, eil_dp, dc_bot, dc_top, out):
    for j in range(out.shape[1]):
        out[i, j] = np.nan
    out[i, _NOUT - 1] = 0.0
    if not (np.isfinite(ps[i]) and np.isfinite(ts[i]) and np.isfinite(tds[i])):
        return
    nkin = t2d.shape[1]
    k0 = min(max(start[i] - 1, 0), nkin)
    ip = np.intp(i) if p2d.shape[0] > 1 else np.intp(0)
    w = np.empty((8, nkin - k0 + 1))
    nk = _setup_profile(p2d[ip, k0:], t2d[i, k0:], td2d[i, k0:], ps[i], ts[i], tds[i], w)
    out[i, _NOUT - 1] = 1.0
    res = np.empty(6)

    # ---- LFC / EL of the three parcels ----
    mucape = np.nan
    mucin = np.nan
    for s in range(3):
        source = (1, 3, 2)[s]
        _lift_source(w, nk, source, ml_depth, pinc, adiabat, np.inf, res)
        out[i, 4 * s + 0] = res[3]
        out[i, 4 * s + 1] = res[2] * 0.01
        out[i, 4 * s + 2] = res[5]
        out[i, 4 * s + 3] = res[4] * 0.01
        if source == 2:
            mucape = res[0]
            mucin = res[1]

    # ---- effective inflow layer (Thompson et al. 2007) ----
    if mucape >= cape_min and mucin <= cin_max:
        if eil_dp > 0.0:
            we, nke = _refined_profile(w, nk, eil_dp, p_search_top)
        else:
            we = w
            nke = nk
        base = -1
        for k in range(nke):
            if we[0, k] < p_search_top:
                break
            if _passes(we, nke, k, pinc, adiabat, cape_min, cin_max, res):
                base = k
                break
        if base >= 0:
            top = base
            for k in range(base + 1, nke):
                if _passes(we, nke, k, pinc, adiabat, cape_min, cin_max, res):
                    top = k
                else:
                    break
            out[i, 12] = we[5, base]
            out[i, 13] = we[0, base] * 0.01
            out[i, 14] = we[5, top]
            out[i, 15] = we[0, top] * 0.01

    # ---- DCAPE ----
    _dcape(w, nk, dc_bot, dc_top, res)
    out[i, 16] = res[0]
    out[i, 17] = res[1] * 0.01


@njit(cache=True, nogil=True)
def _loop_parcel_serial(p2d, t2d, td2d, ps, ts, tds, start, pinc, ml_depth, adiabat,
                        cape_min, cin_max, p_search_top, eil_dp, dc_bot, dc_top):
    out = np.empty((t2d.shape[0], _NOUT), dtype=np.float64)
    for i in range(t2d.shape[0]):
        _parcel_point(i, p2d, t2d, td2d, ps, ts, tds, start, pinc, ml_depth, adiabat,
                      cape_min, cin_max, p_search_top, eil_dp, dc_bot, dc_top, out)
    return out


@njit(cache=True, nogil=True, parallel=True)
def _loop_parcel_parallel(p2d, t2d, td2d, ps, ts, tds, start, pinc, ml_depth, adiabat,
                          cape_min, cin_max, p_search_top, eil_dp, dc_bot, dc_top):
    out = np.empty((t2d.shape[0], _NOUT), dtype=np.float64)
    for i in prange(t2d.shape[0]):
        _parcel_point(i, p2d, t2d, td2d, ps, ts, tds, start, pinc, ml_depth, adiabat,
                      cape_min, cin_max, p_search_top, eil_dp, dc_bot, dc_top, out)
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def _parcel_numpy(p, t, td, ps, ts, tds, *, vertical_lev, ml_depth, pinc, adiabat,
                  cape_min, cin_max, eil_search_top, eil_dp, dcape_layer, _parallel=True):
    from .core import _ADIABAT_OPTIONS, _collapse_thermo
    from .cape_numba import _as_columns, _as_vector

    p, t, td, ps, ts, tds = (np.asarray(a) for a in (p, t, td, ps, ts, tds))
    shape = t.shape[:-1] if t.ndim > 1 else ()
    p_2d, (t_2d, td_2d), (p_s, t_s, td_s), _, pres_lev_pos = _collapse_thermo(
        p, t, td, ps, ts, tds, vertical_lev)
    ngrid = t_2d.shape[1]
    if vertical_lev == "pressure":
        start = _as_vector(pres_lev_pos, ngrid, np.int64)
    else:
        start = np.ones(ngrid, dtype=np.int64)
    out = run_loop(_loop_parcel_parallel, _loop_parcel_serial, _parallel,
                   _as_columns(p_2d), _as_columns(t_2d), _as_columns(td_2d),
                   _as_vector(p_s), _as_vector(t_s), _as_vector(td_s), start,
                   float(pinc), float(ml_depth), int(_ADIABAT_OPTIONS[adiabat]),
                   float(cape_min), float(cin_max), float(eil_search_top) * 100.0,
                   float(eil_dp or 0.0) * 100.0,
                   float(dcape_layer[0]) * 100.0, float(dcape_layer[1]) * 100.0)
    return out.astype(np.float32).reshape(shape + (_NOUT,))


def calc_parcel_diagnostics(p, t, td, ps, ts, tds, *, vertical_lev="sigma", ml_depth=500.0,
                            pinc=500.0, adiabat="pseudo-liquid", cape_min=100.0,
                            cin_max=250.0, eil_search_top=500.0, eil_dp=10.0,
                            dcape_layer=(700.0, 500.0), return_valid=False):
    """
    LFC, EL, effective inflow layer and DCAPE for every column.

    Parameters
    ----------
    p, t, td, ps, ts, tds : array-like
        As in :func:`xcaper.core.calc_cape`: pressure (hPa; 1-D for pressure
        levels), temperature and dewpoint (Celsius) with the vertical axis
        last (bottom to top), surface pressure (hPa), 2 m temperature and
        dewpoint (Celsius).
    vertical_lev : {'sigma', 'pressure'}
    ml_depth : float
        Mixed-layer depth (m) of the mixed-layer parcel.
    pinc : float
        Pressure increment (Pa) of the parcel ascent.
    adiabat : str
        Moist adiabat of the ascent, see :func:`xcaper.core.calc_cape`.
    cape_min, cin_max : float
        Effective-layer criteria (J/kg); CIN is a positive number.
    eil_search_top : float
        The base of the effective layer is searched below this pressure (hPa).
    eil_dp : float or None
        Spacing (hPa) of the refined profile used for the effective-layer
        search (default 10).  Temperature and dewpoint are interpolated
        linearly in ln(p) wherever the input levels below
        ``eil_search_top`` are further apart.  With None (or 0) parcels start
        from the input levels only; with coarse data (ERA5: 25 hPa near the
        ground) a layer whose top lies between the surface and the next
        level then has zero depth, hence zero ESRH, ESCP and ESTP.
    dcape_layer : (float, float)
        Bottom and top (hPa) of the layer in which the DCAPE source parcel
        (minimum equivalent potential temperature) is sought.  Where the
        surface is above the bottom, the search starts at the surface.

    Returns
    -------
    dict of str -> array, each with the shape of ``ps`` (float32)

        ``SFC_LFC``, ``ML_LFC``, ``MU_LFC``
            LFC height (m AGL): lowest level at or above the LCL where the
            parcel is positively buoyant (virtual temperature).
        ``SFC_EL``, ``ML_EL``, ``MU_EL``
            EL height (m AGL): top of the highest positively buoyant layer.
            When the parcel is still buoyant at the top of the data, the top
            of the data is used.
        ``*_LFC_P``, ``*_EL_P``
            The same levels as pressures (hPa).  All four are NaN where the
            parcel has no LFC.
        ``EIL_BASE``, ``EIL_TOP``, ``EIL_BASE_P``, ``EIL_TOP_P``
            Effective inflow layer bounds (m AGL, hPa); NaN where there is
            no effective layer.  0 m means the layer is surface based.
        ``DCAPE``
            Downdraft CAPE (J/kg), positive when the downdraft is negatively
            buoyant; NaN where the source layer is below ground.
        ``DCAPE_SRC_P``
            Pressure (hPa) of the DCAPE source parcel.
        ``_VALID`` (only with ``return_valid=True``)
            1 where the surface data are finite, else 0.

    Notes
    -----
    Effective inflow layer (Thompson et al. 2007): when the most-unstable
    parcel meets the criteria, the base is the first level from the surface
    upward whose parcel has CAPE >= ``cape_min`` and CIN <= ``cin_max``; the
    top is the last level of the uninterrupted run of such levels above it.
    Each candidate ascent stops once CAPE reaches ``cape_min``, so CIN is the
    inhibition below that point.  The layer bounds are levels of the input
    (the surface or a pressure / model level).

    DCAPE follows MetPy's ``downdraft_cape``: minimum theta-e between 700 and
    500 hPa, wet-bulb temperature as starting point, pseudoadiabatic
    descent to the surface, net integral of the virtual-temperature
    difference.  xcaper's constants and Bolton (1980) saturation vapour
    pressure are used, so values differ from MetPy by a few percent.
    """
    from .core import _ADIABAT_OPTIONS, _VERTICAL_LEV_OPTIONS, _check_option

    _check_option("vertical_lev", vertical_lev, _VERTICAL_LEV_OPTIONS)
    _check_option("adiabat", adiabat, _ADIABAT_OPTIONS)
    if dcape_layer[0] <= dcape_layer[1]:
        raise ValueError("dcape_layer must be (bottom, top) with bottom > top, in hPa")
    if eil_dp is not None and not eil_dp >= 0:
        raise ValueError("eil_dp must be a positive spacing in hPa, or None / 0 for input levels")
    kwargs = dict(vertical_lev=vertical_lev, ml_depth=ml_depth, pinc=pinc, adiabat=adiabat,
                  cape_min=cape_min, cin_max=cin_max, eil_search_top=eil_search_top,
                  eil_dp=eil_dp, dcape_layer=tuple(dcape_layer))
    args = (p, t, td, ps, ts, tds)
    if any(isinstance(a, da.Array) for a in args):
        out = da.apply_gufunc(_parcel_numpy, "(i),(i),(i),(),(),()->(n)", *args,
                              output_dtypes="f4", output_sizes={"n": _NOUT},
                              vectorize=False, allow_rechunk=True, _parallel=False, **kwargs)
    else:
        out = _parcel_numpy(*args, **kwargs)
    result = {name: out[..., j] for j, name in enumerate(PARCEL_NAMES)}
    if return_valid:
        result["_VALID"] = out[..., _NOUT - 1]
    return result
