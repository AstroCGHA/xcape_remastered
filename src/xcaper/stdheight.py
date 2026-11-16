"""Height above ground from the hypsometric equation."""
from __future__ import annotations

import numpy as np

try:
    from .fortran import stdheight_2D_model_lev, stdheight_2D_pressure_lev

    HAS_FORTRAN = True
except ImportError:
    stdheight_2D_model_lev = stdheight_2D_pressure_lev = None
    HAS_FORTRAN = False


def stdheight(p_2d, t_2d, td_2d, p_s, t_s, td_s, flag_1d, pres_lev_pos, aglh0,
              type_grid, method="fortran", parallel=True):
    """Height above ground (m) of every level of a collapsed grid.

    Parameters
    ----------
    p_2d : array-like
        Pressure in hPa, ``(nk, ngrid)`` or ``(nk, 1)`` when ``flag_1d == 1``.
    t_2d, td_2d : array-like
        Temperature and dewpoint in Celsius, ``(nk, ngrid)``.
    p_s, t_s, td_s : array-like
        Surface pressure (hPa), temperature and dewpoint (Celsius), ``(ngrid,)``.
    flag_1d : int
        1 if pressure is one vector shared by every grid point, else 0.
    pres_lev_pos : int or array-like of int
        For pressure-level data, the 1-based index of the first level at
        or above the surface.
    aglh0 : float or array-like
        Height above ground (m) of the surface level (e.g. 2 m).
    type_grid : int
        1 for model (sigma) levels, 2 for pressure levels.
    method : {'fortran', 'numba'}
    parallel : bool
        numba only: use all numba threads (set False inside dask tasks).

    Returns
    -------
    H2D : ndarray, ``(nk, ngrid)``
        Height above ground; -999999 for pressure levels below the surface.
    H_s : ndarray, ``(ngrid,)``
        Height above ground of the surface level.
    """
    nlev, ngrid = t_2d.shape

    if np.isscalar(aglh0):
        aglh_in = np.full(ngrid, float(aglh0))
    else:
        aglh_in = np.asarray(aglh0, dtype=np.float64)
        if aglh_in.shape != (ngrid,):
            raise ValueError(f"aglh0 must be a scalar or have shape ({ngrid},)")

    if method == "numba":
        from .srh_numba import stdheight_numba

        return stdheight_numba(p_2d, t_2d, td_2d, p_s, t_s, td_s, pres_lev_pos,
                               aglh_in, type_grid, parallel=parallel)
    if method != "fortran":
        raise ValueError("method must be 'fortran' or 'numba'")
    if not HAS_FORTRAN:
        raise ImportError("xcaper Fortran extensions are not available; use method='numba'")

    if type_grid == 1:
        return stdheight_2D_model_lev.loop_stdheight_ml(
            p_2d, t_2d, td_2d, p_s, t_s, td_s, aglh_in, nlev, ngrid
        )
    if type_grid == 2:
        if flag_1d != 1:
            raise ValueError("pressure-level data need a 1-D pressure vector")
        start = np.ascontiguousarray(np.broadcast_to(pres_lev_pos, (ngrid,)), dtype=np.intc)
        return stdheight_2D_pressure_lev.loop_stdheight_pl1d(
            t_2d, td_2d, p_2d, p_s, t_s, td_s, aglh_in, start, nlev, ngrid
        )
    raise ValueError("type_grid must be 1 (model levels) or 2 (pressure levels)")
