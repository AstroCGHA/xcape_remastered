"""numba ports of the legacy xcaper height / Bunkers / SRH Fortran kernels.

These reproduce ``fortran/xcaper_kernels.f90`` (``stdheight``,
``bunkers_calc``, ``calc_srh``) so that :func:`xcaper.core.calc_srh` works
without the compiled extensions and can use every core on numpy inputs.
Double precision is used throughout.

For the newer, more flexible wind diagnostics (configurable storm motion,
several SRH layers, streamwise vorticity, ...) see :mod:`xcaper.kinematics`.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ._parallel import run_loop

_MISSING = -999999.0

# xcape 0.1 wrote the constants of the hypsometric routine as single
# precision literals; their float32 values are kept on purpose so that the
# heights (and therefore SRH) match the Fortran kernel to round-off.
_R = float(np.float32(287.04))
_GNEG = float(np.float32(-9.80665))
_EPS = float(np.float32(0.6219800858985514))
_T0 = float(np.float32(273.15))
_E0 = float(np.float32(6.112))
_C1 = float(np.float32(53.49))
_C3 = float(np.float32(5.09))


# ---------------------------------------------------------------------------
# hypsometric height
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _virtual_temperature(t_c, td_c, p_hpa):
    tin = t_c + _T0
    tdin = td_c + _T0
    e = _E0 * math.exp(_C1 - (6808.0 / tdin) - _C3 * math.log(tdin))
    w = _EPS * (e / (p_hpa - e))
    return tin * ((w + _EPS) / (_EPS * (1.0 + w)))


@njit(cache=True, nogil=True)
def _stdheight_point(i, p2d, t2d, td2d, ps, ts, tds, hin, start, h):
    nk = t2d.shape[1]
    k0 = min(max(start[i] - 1, 0), nk)
    ip = np.intp(i) if p2d.shape[0] > 1 else np.intp(0)
    for k in range(k0):
        h[i, k] = _MISSING
    tv_prev = _virtual_temperature(ts[i], tds[i], ps[i])
    p_prev = ps[i]
    h_prev = hin[i]
    for k in range(k0, nk):
        tv = _virtual_temperature(t2d[i, k], td2d[i, k], p2d[ip, k])
        h_prev = h_prev + ((_R * ((tv + tv_prev) / 2.0) / _GNEG)) * math.log(p2d[ip, k] / p_prev)
        h[i, k] = h_prev
        tv_prev = tv
        p_prev = p2d[ip, k]


@njit(cache=True, nogil=True)
def _loop_stdheight_serial(p2d, t2d, td2d, ps, ts, tds, hin, start):
    h = np.empty(t2d.shape, dtype=np.float64)
    for i in range(t2d.shape[0]):
        _stdheight_point(i, p2d, t2d, td2d, ps, ts, tds, hin, start, h)
    return h


@njit(cache=True, nogil=True, parallel=True)
def _loop_stdheight_parallel(p2d, t2d, td2d, ps, ts, tds, hin, start):
    h = np.empty(t2d.shape, dtype=np.float64)
    for i in prange(t2d.shape[0]):
        _stdheight_point(i, p2d, t2d, td2d, ps, ts, tds, hin, start, h)
    return h


# ---------------------------------------------------------------------------
# Bunkers storm motion + SRH (legacy sampling: 13 points every 500 m)
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _interp_to_height(v, z, n, height):
    """Linear interpolation of ``v(z)`` (first ``n`` entries) to ``height``.

    Returns -999999 when ``height`` is not bracketed (legacy behaviour).
    """
    ip = 0
    im = 1
    if z[0] > z[n - 1]:
        ip = 1
        im = 0
    for kp in range(n - 1, 0, -1):
        if z[kp - im] <= height and z[kp - ip] > height:
            w2 = (height - z[kp - im]) / (z[kp - ip] - z[kp - im])
            return (1.0 - w2) * v[kp - im] + w2 * v[kp - ip]
    return _MISSING


@njit(cache=True, nogil=True)
def _bunkers_srh_point(i, u2d, v2d, aglh2d, us, vs, aglhs, start, top, out):
    nk = u2d.shape[1]
    k0 = min(max(start[i] - 1, 0), nk)
    n = nk - k0 + 1

    u = np.empty(n)
    v = np.empty(n)
    z = np.empty(n)
    u[0] = us[i]
    v[0] = vs[i]
    z[0] = aglhs[i]
    for k in range(k0, nk):
        u[k - k0 + 1] = u2d[i, k]
        v[k - k0 + 1] = v2d[i, k]
        z[k - k0 + 1] = aglh2d[i, k]

    # ---- Bunkers et al. (2000) ----
    us6 = np.empty(13)
    vs6 = np.empty(13)
    us6[0] = u[0]
    vs6[0] = v[0]
    for j in range(1, 13):
        us6[j] = _interp_to_height(u, z, n, 500.0 * j)
        vs6[j] = _interp_to_height(v, z, n, 500.0 * j)
    mean_u = 0.0
    mean_v = 0.0
    for j in range(13):
        mean_u += us6[j]
        mean_v += vs6[j]
    mean_u /= 13.0
    mean_v /= 13.0
    ushr = (us6[11] + us6[12]) / 2.0 - (us6[0] + us6[1]) / 2.0
    vshr = (vs6[11] + vs6[12]) / 2.0 - (vs6[0] + vs6[1]) / 2.0
    shr = math.sqrt(ushr * ushr + vshr * vshr)
    if shr > 0.0:
        du = 7.5 * vshr / shr
        dv = 7.5 * ushr / shr
    else:
        du = np.nan
        dv = np.nan
    rm_u = mean_u + du
    rm_v = mean_v - dv
    lm_u = mean_u - du
    lm_v = mean_v + dv

    # ---- storm relative helicity between the lowest level and `top` ----
    ktop = 0
    for k in range(1, n):
        if z[k] > top:
            ktop = k
            x1 = z[k]
            x3 = z[k - 1]
            if x3 == x1:
                x1 = x1 - 0.01
            f = (top - x1) / (x3 - x1)
            u[k] = u[k] + (u[k - 1] - u[k]) * f
            v[k] = v[k] + (v[k - 1] - v[k]) * f
            break
    srh_rm = 0.0
    srh_lm = 0.0
    for k in range(1, ktop + 1):
        dvk = v[k] - v[k - 1]
        duk = u[k] - u[k - 1]
        srh_rm += (u[k] - rm_u) * dvk - (v[k] - rm_v) * duk
        srh_lm += (u[k] - lm_u) * dvk - (v[k] - lm_v) * duk

    out[i, 0] = -srh_rm
    out[i, 1] = -srh_lm
    out[i, 2] = rm_u
    out[i, 3] = rm_v
    out[i, 4] = lm_u
    out[i, 5] = lm_v
    out[i, 6] = mean_u
    out[i, 7] = mean_v


@njit(cache=True, nogil=True)
def _loop_bunkers_srh_serial(u2d, v2d, aglh2d, us, vs, aglhs, start, top):
    out = np.empty((u2d.shape[0], 8), dtype=np.float64)
    for i in range(u2d.shape[0]):
        _bunkers_srh_point(i, u2d, v2d, aglh2d, us, vs, aglhs, start, top, out)
    return out


@njit(cache=True, nogil=True, parallel=True)
def _loop_bunkers_srh_parallel(u2d, v2d, aglh2d, us, vs, aglhs, start, top):
    out = np.empty((u2d.shape[0], 8), dtype=np.float64)
    for i in prange(u2d.shape[0]):
        _bunkers_srh_point(i, u2d, v2d, aglh2d, us, vs, aglhs, start, top, out)
    return out


# ---------------------------------------------------------------------------
# python-level entry points (same conventions as stdheight.py / srh.py)
# ---------------------------------------------------------------------------
def _as_columns(a_2d):
    return np.ascontiguousarray(np.asarray(a_2d).T, dtype=np.float64)


def _as_vector(a, n=None, dtype=np.float64):
    a = np.asarray(a)
    if n is not None:
        a = np.broadcast_to(a, (n,))
    return np.ascontiguousarray(a, dtype=dtype)


def _start(pres_lev_pos, type_grid, ngrid):
    if type_grid == 2:
        return _as_vector(pres_lev_pos, ngrid, np.int64)
    return np.ones(ngrid, dtype=np.int64)


def stdheight_numba(p_2d, t_2d, td_2d, p_s, t_s, td_s, pres_lev_pos, aglh_in,
                    type_grid, parallel=True):
    """numba twin of the Fortran ``loop_stdheight_*`` drivers.

    Returns ``(H2D, H_s)`` with ``H2D`` of shape ``(nk, ngrid)``.
    """
    nlev, ngrid = t_2d.shape
    hin = _as_vector(aglh_in, ngrid)
    h = run_loop(_loop_stdheight_parallel, _loop_stdheight_serial, parallel,
                 _as_columns(p_2d), _as_columns(t_2d), _as_columns(td_2d),
                 _as_vector(p_s), _as_vector(t_s), _as_vector(td_s), hin,
                 _start(pres_lev_pos, type_grid, ngrid))
    return h.T, hin.copy()


def srh_numba(u_2d, v_2d, aglh_2d, u_s, v_s, aglh_s, pres_lev_pos, depth,
              type_grid, parallel=True):
    """numba twin of the Fortran Bunkers + SREH drivers.

    Returns ``(srh_rm, srh_lm, rm, lm, mean_6km)``; the last three have
    shape ``(2, ngrid)`` (u, v).
    """
    nlev, ngrid = u_2d.shape
    out = run_loop(_loop_bunkers_srh_parallel, _loop_bunkers_srh_serial, parallel,
                   _as_columns(u_2d), _as_columns(v_2d), _as_columns(aglh_2d),
                   _as_vector(u_s), _as_vector(v_s), _as_vector(aglh_s),
                   _start(pres_lev_pos, type_grid, ngrid), float(depth))
    return (out[:, 0], out[:, 1], out[:, 2:4].T.copy(), out[:, 4:6].T.copy(),
            out[:, 6:8].T.copy())
