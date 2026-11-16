"""Thermodynamic helpers: dewpoint, LCL height and lapse rates."""
from __future__ import annotations

import math

import dask.array as da
import numpy as np
from numba import njit, prange

from ._parallel import run_loop

from .core import calc_lcl  # noqa: F401  (re-exported)

__all__ = ["dewpoint_from_specific_humidity", "calc_lcl", "calc_lapse_rates"]


def dewpoint_from_specific_humidity(q, p):
    """Dewpoint temperature (K) from specific humidity and pressure.

    Uses the Bolton (1980) saturation vapour pressure, the same formula
    as the parcel code of xcaper.

    Parameters
    ----------
    q : array-like
        Specific humidity (kg/kg).  Values below 1e-10 are raised to 1e-10.
    p : array-like
        Pressure (hPa), broadcastable against ``q``.

    Returns
    -------
    array-like
        Dewpoint temperature in Kelvin (numpy, dask or xarray, like the input).
    """
    q = q.clip(1.0e-10, None) if hasattr(q, "clip") else np.clip(q, 1.0e-10, None)
    r = q / (1.0 - q)                  # mixing ratio
    e = r / (0.62197 + r) * p          # vapour pressure, hPa
    lne = np.log(e / 6.112)
    return 243.5 * lne / (17.67 - lne) + 273.15


# ---------------------------------------------------------------------------
# lapse rates
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _lapse_point(i, p2d, t2d, z2d, ps, ts, h_bot, h_top, p_bot, p_top, out):
    out[i, 0] = np.nan
    out[i, 1] = np.nan
    nlev = t2d.shape[1]
    ip = np.intp(i) if p2d.shape[0] > 1 else np.intp(0)

    # column above ground, surface first; sorted by height
    zc = np.empty(nlev + 1)
    tc = np.empty(nlev + 1)
    pc = np.empty(nlev + 1)
    zc[0] = 0.0
    tc[0] = ts[i]
    pc[0] = ps[i]
    n = 1
    for k in range(nlev):
        zk = z2d[i, k]
        tk = t2d[i, k]
        pk = p2d[ip, k]
        if np.isfinite(zk) and np.isfinite(tk) and zk > 0.0 and pk < ps[i]:
            zc[n] = zk
            tc[n] = tk
            pc[n] = pk
            n += 1
    if n < 3 or not np.isfinite(ts[i]):
        return
    for a in range(2, n):
        za, ta, pa = zc[a], tc[a], pc[a]
        b = a - 1
        while b >= 1 and zc[b] > za:
            zc[b + 1], tc[b + 1], pc[b + 1] = zc[b], tc[b], pc[b]
            b -= 1
        zc[b + 1], tc[b + 1], pc[b + 1] = za, ta, pa

    # ---- height layer: temperature interpolated linearly in height ----
    if zc[n - 1] >= h_top and h_top > h_bot:
        tb = np.nan
        tt = np.nan
        for k in range(n - 1):
            if zc[k] <= h_bot <= zc[k + 1] and zc[k + 1] > zc[k] and not np.isfinite(tb):
                tb = tc[k] + (tc[k + 1] - tc[k]) * (h_bot - zc[k]) / (zc[k + 1] - zc[k])
            if zc[k] <= h_top <= zc[k + 1] and zc[k + 1] > zc[k] and not np.isfinite(tt):
                tt = tc[k] + (tc[k + 1] - tc[k]) * (h_top - zc[k]) / (zc[k + 1] - zc[k])
        out[i, 0] = -(tt - tb) / (h_top - h_bot) * 1000.0

    # ---- pressure layer: T and z interpolated linearly in ln(p) ----
    if pc[0] >= p_bot and pc[n - 1] <= p_top:
        tb = np.nan
        zb = np.nan
        tt = np.nan
        zt = np.nan
        for k in range(n - 1):
            if pc[k] >= p_bot >= pc[k + 1] and pc[k] > pc[k + 1] and not np.isfinite(tb):
                f = math.log(pc[k] / p_bot) / math.log(pc[k] / pc[k + 1])
                tb = tc[k] + (tc[k + 1] - tc[k]) * f
                zb = zc[k] + (zc[k + 1] - zc[k]) * f
            if pc[k] >= p_top >= pc[k + 1] and pc[k] > pc[k + 1] and not np.isfinite(tt):
                f = math.log(pc[k] / p_top) / math.log(pc[k] / pc[k + 1])
                tt = tc[k] + (tc[k + 1] - tc[k]) * f
                zt = zc[k] + (zc[k + 1] - zc[k]) * f
        if zt > zb:
            out[i, 1] = -(tt - tb) / (zt - zb) * 1000.0


@njit(cache=True, nogil=True)
def _loop_lapse_serial(p2d, t2d, z2d, ps, ts, h_bot, h_top, p_bot, p_top):
    out = np.empty((t2d.shape[0], 2), dtype=np.float32)
    for i in range(t2d.shape[0]):
        _lapse_point(i, p2d, t2d, z2d, ps, ts, h_bot, h_top, p_bot, p_top, out)
    return out


@njit(cache=True, nogil=True, parallel=True)
def _loop_lapse_parallel(p2d, t2d, z2d, ps, ts, h_bot, h_top, p_bot, p_top):
    out = np.empty((t2d.shape[0], 2), dtype=np.float32)
    for i in prange(t2d.shape[0]):
        _lapse_point(i, p2d, t2d, z2d, ps, ts, h_bot, h_top, p_bot, p_top, out)
    return out


def _virtual_celsius(t, td, p):
    """Virtual temperature (C) from temperature and dewpoint (C) and pressure (hPa).

    Same formula as SHARPpy's ``virtemp`` (Bolton vapour pressure, mixing
    ratio from the dewpoint); NaN dewpoints leave the temperature unchanged.
    """
    e = 6.112 * np.exp(17.67 * td / (td + 243.5))
    w = 0.62197 * e / (p - e)
    tv = (t + 273.15) * (1.0 + w / 0.62197) / (1.0 + w) - 273.15
    return np.where(np.isfinite(tv), tv, t)


def _lapse_numpy(p, t, z_agl, ps, ts, td=None, tds=None, *, height_layer, pressure_layer,
                 _parallel=True):
    p, t, z_agl, ps, ts = (np.asarray(a) for a in (p, t, z_agl, ps, ts))
    if t.shape != z_agl.shape:
        raise ValueError("t and z_agl must have the same shape")
    if td is not None:
        # virtual temperature lapse rate (SPC / SHARPpy convention)
        t = _virtual_celsius(t, np.asarray(td), p)
        ts = _virtual_celsius(ts, np.asarray(tds), ps)
    nlev = t.shape[-1]
    as2d = lambda a: np.ascontiguousarray(a.reshape(-1, nlev), dtype=np.float64)  # noqa: E731
    as1d = lambda a: np.ascontiguousarray(a.reshape(-1), dtype=np.float64)  # noqa: E731
    p2d = as2d(p) if p.ndim > 1 else np.ascontiguousarray(p[np.newaxis, :], dtype=np.float64)
    out = run_loop(_loop_lapse_parallel, _loop_lapse_serial, _parallel,
                   p2d, as2d(t), as2d(z_agl), as1d(ps), as1d(ts),
                   float(height_layer[0]), float(height_layer[1]),
                   float(pressure_layer[0]), float(pressure_layer[1]))
    shape = t.shape[:-1]
    return out[:, 0].reshape(shape), out[:, 1].reshape(shape)


def calc_lapse_rates(p, t, z_agl, ps, ts, *, td=None, tds=None, height_layer=(0.0, 3000.0),
                     pressure_layer=(700.0, 500.0)):
    """Environmental lapse rates of a height layer and a pressure layer.

    With ``td`` and ``tds`` given, the lapse rates are those of the virtual
    temperature, as in SHARPpy, the SPC mesoanalysis and SounderPy; without
    them, of the temperature.

    Parameters
    ----------
    p : array-like
        Pressure (hPa): ``(nlev,)`` for pressure levels or ``(..., nlev)``.
    t : array-like, ``(..., nlev)``
        Temperature (Celsius or Kelvin), vertical axis last.
    z_agl : array-like, ``(..., nlev)``
        Height above ground (m).  Levels below ground are ignored.
    ps : array-like, ``(...)``
        Surface pressure (hPa).
    ts : array-like, ``(...)``
        Surface (2 m) temperature, same unit as ``t``; used at 0 m.
    td, tds : array-like, optional
        Dewpoint on the levels (``(..., nlev)``) and at 2 m (``(...)``), in
        Celsius.  When given, ``t`` and ``ts`` must be Celsius too and the
        virtual-temperature lapse rate is returned.
    height_layer : (bottom, top) in m above ground, default 0-3 km
    pressure_layer : (bottom, top) in hPa, default 700-500 hPa

    Returns
    -------
    lr_height, lr_pressure : array-like
        Lapse rates in K/km, positive when temperature decreases with
        height.  NaN where the layer is not covered by the column (for
        example 700-500 hPa where the surface pressure is below 700 hPa).
    """
    kwargs = dict(height_layer=tuple(height_layer), pressure_layer=tuple(pressure_layer))
    if kwargs["pressure_layer"][0] <= kwargs["pressure_layer"][1]:
        raise ValueError("pressure_layer must be (bottom, top) with bottom > top, in hPa")
    if (td is None) != (tds is None):
        raise ValueError("give both td and tds (virtual lapse rate) or neither")
    args = (p, t, z_agl, ps, ts)
    signature = "(k),(k),(k),(),()->(),()"
    if td is not None:
        args = args + (td, tds)
        signature = "(k),(k),(k),(),(),(k),()->(),()"
    if any(isinstance(a, da.Array) for a in args):
        return da.apply_gufunc(_lapse_numpy, signature, *args,
                               output_dtypes=("f4", "f4"), vectorize=False,
                               allow_rechunk=True, _parallel=False, **kwargs)
    return _lapse_numpy(*args, **kwargs)
