"""Composite severe-weather indices.

All functions are plain array arithmetic, so they accept (and return) numpy
arrays, dask arrays and xarray objects alike.

Sign conventions used throughout xcaper:

* CAPE and **CIN are positive numbers** (J/kg): a CIN of 80 means 80 J/kg of
  inhibition.  Formulas from the literature written for negative CIN are
  translated accordingly.
* SRH is positive for cyclonic (Northern Hemisphere right-mover) streamwise
  vorticity.  The indices keep the sign of SRH, so they are negative where
  SRH is negative (left movers / Southern Hemisphere right movers).
"""
from __future__ import annotations

__all__ = [
    "supercell_composite",
    "significant_tornado",
    "significant_tornado_cin",
    "energy_helicity_index",
    "wmaxshear",
    "effective_supercell_composite",
    "effective_significant_tornado",
]


def _clip(a, lo=None, hi=None):
    if hasattr(a, "clip"):
        return a.clip(lo, hi)
    import numpy as np

    return np.clip(a, lo, hi)


def supercell_composite(mucape, srh, shear, cin=None):
    """Supercell Composite Parameter (SCP), fixed-layer form.

    ``SCP = (MUCAPE / 1000) * (SRH / 50) * shear_term [* cin_term]``

    * ``shear_term`` = shear / 20 m/s, set to 0 below 10 m/s and capped at 1
      above 20 m/s;
    * ``cin_term`` (only when ``cin`` is given) = 40 / CIN, capped at 1 for
      CIN below 40 J/kg.

    Thompson et al. (2004) define SCP with the effective-layer SRH and the
    effective bulk wind difference.  This fixed-layer form, usual for
    gridded reanalysis work, takes 0-3 km SRH and the 0-6 km bulk wind
    difference instead.

    Parameters
    ----------
    mucape : most-unstable CAPE (J/kg)
    srh : 0-3 km storm relative helicity (m2/s2)
    shear : 0-6 km bulk wind difference (m/s)
    cin : most-unstable CIN (J/kg, positive), optional
    """
    shear_term = _clip(shear / 20.0, None, 1.0) * (shear >= 10.0)
    scp = (mucape / 1000.0) * (srh / 50.0) * shear_term
    if cin is not None:
        scp = scp * _clip(40.0 / _clip(cin, 1.0e-6, None), None, 1.0)
    return scp


def _lcl_term(lcl_height):
    # 1 below 1000 m, 0 above 2000 m, linear in between
    return _clip((2000.0 - lcl_height) / 1000.0, 0.0, 1.0)


def _stp_shear_term(shear):
    # 0 below 12.5 m/s, capped at 1.5 above 30 m/s
    return _clip(shear / 20.0, None, 1.5) * (shear >= 12.5)


def significant_tornado(sbcape, lcl_height, srh, shear):
    """Significant Tornado Parameter (STP), fixed-layer form.

    ``STP = (SBCAPE / 1500) * lcl_term * (SRH / 150) * shear_term``

    * ``lcl_term`` = (2000 - LCL) / 1000, 1 below 1000 m and 0 above 2000 m;
    * ``shear_term`` = shear / 20 m/s, 0 below 12.5 m/s, capped at 1.5.

    Thompson et al. (2003, 2012).

    Parameters
    ----------
    sbcape : surface-based CAPE (J/kg)
    lcl_height : surface-based LCL height (m above ground)
    srh : 0-1 km storm relative helicity (m2/s2)
    shear : 0-6 km bulk wind difference (m/s)
    """
    return (sbcape / 1500.0) * _lcl_term(lcl_height) * (srh / 150.0) * _stp_shear_term(shear)


def significant_tornado_cin(mlcape, lcl_height, srh, shear, mlcin):
    """Significant Tornado Parameter with the CIN term (Thompson et al. 2012).

    ``STP = (MLCAPE / 1500) * lcl_term * (SRH / 150) * shear_term * cin_term``

    with ``cin_term`` = (200 - CIN) / 150, 1 for CIN below 50 J/kg and 0 for
    CIN above 200 J/kg.  The original uses the effective SRH and effective
    bulk wind difference; pass 0-1 km SRH and 0-6 km shear for the usual
    fixed-layer approximation.

    Parameters
    ----------
    mlcape : mixed-layer CAPE (J/kg)
    lcl_height : mixed-layer LCL height (m above ground)
    srh : storm relative helicity (m2/s2)
    shear : bulk wind difference (m/s)
    mlcin : mixed-layer CIN (J/kg, positive)
    """
    cin_term = _clip((200.0 - mlcin) / 150.0, 0.0, 1.0)
    return ((mlcape / 1500.0) * _lcl_term(lcl_height) * (srh / 150.0)
            * _stp_shear_term(shear) * cin_term)


def energy_helicity_index(cape, srh):
    """Energy Helicity Index: ``EHI = CAPE * SRH / 160000`` (Hart and Korotky 1991)."""
    return cape * srh / 160000.0


def wmaxshear(cape, shear):
    """WMAXSHEAR = sqrt(2 CAPE) * shear, in m2/s2 (Taszarek et al. 2020)."""
    return (2.0 * cape) ** 0.5 * shear


def effective_supercell_composite(mucape, esrh, ebwd, mucin):
    """Effective-layer Supercell Composite Parameter (ESCP; SPC form).

    ``ESCP = (MUCAPE / 1000) * (ESRH / 50) * ebwd_term * cin_term``

    * ``ebwd_term`` = EBWD / 20 m/s, 0 below 10 m/s, capped at 1 above 20 m/s;
    * ``cin_term`` = 40 / MUCIN, capped at 1 for MUCIN below 40 J/kg
      (MUCIN positive).

    Thompson et al. (2004, 2007); the CIN term follows the SPC mesoanalysis.

    Parameters
    ----------
    mucape : most-unstable CAPE (J/kg)
    esrh : effective storm relative helicity (m2/s2)
    ebwd : effective bulk wind difference (m/s)
    mucin : most-unstable CIN (J/kg, positive)
    """
    return supercell_composite(mucape, esrh, ebwd, cin=mucin)


def effective_significant_tornado(mlcape, ml_lcl, esrh, ebwd, mlcin, eil_base):
    """Effective-layer Significant Tornado Parameter (ESTP; Thompson et al. 2012).

    ``ESTP = (MLCAPE / 1500) * lcl_term * (ESRH / 150) * ebwd_term * cin_term``

    with the terms of :func:`significant_tornado_cin` (EBWD in place of the
    0-6 km shear).  ESTP is 0 where the effective inflow layer is elevated
    (base above the ground) or absent, as on the SPC mesoanalysis.

    Parameters
    ----------
    mlcape : mixed-layer CAPE (J/kg)
    ml_lcl : mixed-layer LCL height (m above ground)
    esrh : effective storm relative helicity (m2/s2)
    ebwd : effective bulk wind difference (m/s)
    mlcin : mixed-layer CIN (J/kg, positive)
    eil_base : effective inflow base height (m above ground; NaN = no layer)
    """
    surface_based = eil_base == 0.0
    return significant_tornado_cin(mlcape, ml_lcl, esrh, ebwd, mlcin) * surface_based
