"""Bunkers storm motion and storm relative helicity (legacy xcaper algorithm)."""
from __future__ import annotations

import numpy as np

try:
    from .fortran import (Bunkers_model_lev, Bunkers_pressure_lev,
                          SREH_model_lev, SREH_pressure_lev)

    HAS_FORTRAN = True
except ImportError:
    Bunkers_model_lev = Bunkers_pressure_lev = None
    SREH_model_lev = SREH_pressure_lev = None
    HAS_FORTRAN = False


def srh(u_2d, v_2d, aglh_2d, u_s, v_s, aglh_s, pres_lev_pos, depth, type_grid,
        output, method="fortran", parallel=True):
    """Bunkers storm motion and SRH for a collapsed grid.

    Called by :func:`xcaper.core.calc_srh`; see there for the science.

    Parameters
    ----------
    u_2d, v_2d : array-like
        Wind components in m/s, ``(nk, ngrid)``.
    aglh_2d : array-like
        Height above ground in m, ``(nk, ngrid)``.
    u_s, v_s : array-like
        Wind at the lowest model level or at 10 m, ``(ngrid,)``.
    aglh_s : array-like
        Height above ground (m) of that level, ``(ngrid,)``.
    pres_lev_pos : int or array-like of int
        For pressure-level data, the 1-based index of the first level at
        or above the surface.
    depth : float
        Depth (m) of the SRH integration layer.
    type_grid : int
        1 for model (sigma) levels, 2 for pressure levels.
    output : int
        1 returns only SRH; anything else also returns the storm motions.
    method : {'fortran', 'numba'}
    parallel : bool
        numba only: use all numba threads (set False inside dask tasks).

    Returns
    -------
    srh_rm, srh_lm : ndarray, ``(ngrid,)``
    rm_sup, lm_sup, mean_6km : ndarray, ``(2, ngrid)``
        Only when ``output != 1``.
    """
    nlev, ngrid = u_2d.shape

    if method == "numba":
        from .srh_numba import srh_numba

        srh_rm, srh_lm, rm_sup, lm_sup, mean_6km = srh_numba(
            u_2d, v_2d, aglh_2d, u_s, v_s, aglh_s, pres_lev_pos, depth, type_grid,
            parallel=parallel,
        )
    elif method == "fortran":
        if not HAS_FORTRAN:
            raise ImportError("xcaper Fortran extensions are not available; use method='numba'")
        if type_grid == 1:
            rm_sup, lm_sup, mean_6km = Bunkers_model_lev.bunkers_loop_ml(
                u_2d, v_2d, aglh_2d, u_s, v_s, aglh_s
            )
            srh_rm, srh_lm = SREH_model_lev.loop_sreh_ml(
                u_2d, v_2d, aglh_2d, u_s, v_s, aglh_s,
                rm_sup[0, :], rm_sup[1, :], lm_sup[0, :], lm_sup[1, :], depth,
            )
        elif type_grid == 2:
            start = np.ascontiguousarray(np.broadcast_to(pres_lev_pos, (ngrid,)), dtype=np.intc)
            rm_sup, lm_sup, mean_6km = Bunkers_pressure_lev.bunkers_loop_pl(
                u_2d, v_2d, aglh_2d, u_s, v_s, aglh_s, start
            )
            srh_rm, srh_lm = SREH_pressure_lev.loop_sreh_pl(
                u_2d, v_2d, aglh_2d, u_s, v_s, aglh_s,
                rm_sup[0, :], rm_sup[1, :], lm_sup[0, :], lm_sup[1, :], depth, start,
            )
        else:
            raise ValueError("type_grid must be 1 (model levels) or 2 (pressure levels)")
    else:
        raise ValueError("method must be 'fortran' or 'numba'")

    if output == 1:
        return srh_rm, srh_lm
    return srh_rm, srh_lm, rm_sup, lm_sup, mean_6km
