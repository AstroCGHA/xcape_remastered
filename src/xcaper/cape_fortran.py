"""CAPE / CIN through the compiled Fortran kernels (f2py)."""
from __future__ import annotations

import numpy as np

try:
    from .fortran import CAPE_CODE_model_lev, CAPE_CODE_pressure_lev

    HAS_FORTRAN = True
    _IMPORT_ERROR = None
except ImportError as _err:  # pure-python install or docs build
    CAPE_CODE_model_lev = CAPE_CODE_pressure_lev = None
    HAS_FORTRAN = False
    _IMPORT_ERROR = _err


def _require_fortran():
    if not HAS_FORTRAN:
        raise ImportError(
            "The xcaper Fortran extension modules are not available "
            f"({_IMPORT_ERROR}). Reinstall xcaper with a Fortran compiler or use "
            "method='numba'."
        )


def cape(p_2d, t_2d, td_2d, p_s, t_s, td_s, flag_1d, pres_lev_pos, source,
         ml_depth, adiabat, pinc, type_grid):
    """Integrate CAPE / CIN for a collapsed grid with the Fortran kernels.

    Called by :func:`xcaper.core.calc_cape`; see there for the science.

    Parameters
    ----------
    p_2d : array-like
        Pressure in hPa, shape ``(nk, ngrid)`` or ``(nk, 1)`` when
        ``flag_1d == 1``.
    t_2d, td_2d : array-like
        Temperature and dewpoint in Celsius, shape ``(nk, ngrid)``.
    p_s, t_s, td_s : array-like
        Surface pressure (hPa), temperature and dewpoint (Celsius),
        shape ``(ngrid,)``.
    flag_1d : int
        1 if pressure is one vector shared by every grid point, else 0.
    pres_lev_pos : int or array-like of int
        For pressure-level data, the 1-based index of the first level at
        or above the surface of every grid point.
    source : int
        1 surface, 2 most-unstable, 3 mixed-layer parcel.
    ml_depth : float
        Depth (m) of the mixed layer (``source == 3``).
    adiabat : int
        1 pseudo-liquid, 2 reversible-liquid, 3 pseudo-ice, 4 reversible-ice.
    pinc : float
        Pressure increment (Pa) of the parcel ascent.
    type_grid : int
        1 for model (sigma) levels, 2 for pressure levels.

    Returns
    -------
    cape, cin : ndarray of float32, shape ``(ngrid,)``
    mulev : ndarray of int32
    zmulev : ndarray of float32
    """
    _require_fortran()
    nlev, ngrid = t_2d.shape

    if type_grid == 1:
        return CAPE_CODE_model_lev.loopcape_ml(
            p_2d, t_2d, td_2d, p_s, t_s, td_s, pinc, source, ml_depth, adiabat, nlev, ngrid
        )
    if type_grid == 2:
        if flag_1d != 1:
            raise ValueError("pressure-level data need a 1-D pressure vector")
        start = np.ascontiguousarray(np.broadcast_to(pres_lev_pos, (ngrid,)), dtype=np.intc)
        return CAPE_CODE_pressure_lev.loopcape_pl1d(
            t_2d, td_2d, p_2d, p_s, t_s, td_s, pinc, source, ml_depth, adiabat, start, nlev, ngrid
        )
    raise ValueError("type_grid must be 1 (model levels) or 2 (pressure levels)")
