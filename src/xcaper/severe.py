"""All severe-weather parameters of xcaper in one call (numpy / dask arrays).

:func:`calc_severe` chains the building blocks of the package:

* CAPE / CIN of the surface, mixed-layer and most-unstable parcels
  (:func:`xcaper.core.calc_cape`);
* LCL heights (:func:`xcaper.core.calc_lcl`) and lapse rates
  (:func:`xcaper.thermo.calc_lapse_rates`);
* storm motion, SRH, bulk shear, streamwise vorticity, streamwiseness and
  critical angle from one common wind profile
  (:func:`xcaper.kinematics.calc_kinematics`);
* the composite indices SCP, STP, EHI and WMAXSHEAR
  (:mod:`xcaper.composites`).

For xarray objects use :func:`xcaper.xarray.severe_parameters`; for ERA5
files or downloads use :mod:`xcaper.era5`.
"""
from __future__ import annotations

import numpy as np

from . import composites
from .core import calc_cape, calc_lcl
from .kinematics import StormMotion, calc_kinematics, layer_tag
from .parcel import calc_parcel_diagnostics
from .thermo import calc_lapse_rates

__all__ = ["calc_severe", "variable_attrs"]

_REQUIRED_SRH = (1000.0, 3000.0)
_REQUIRED_SHEAR = (6000.0,)


def _with_required(layers, required):
    layers = tuple(float(x) for x in layers)
    return tuple(sorted(set(layers) | set(required)))


def calc_severe(p, t, td, u, v, z_agl, ps, ts, tds, us, vs, *,
                vertical_lev="pressure", ml_depth=500.0, pinc=500.0,
                adiabat="pseudo-liquid", method="auto",
                storm_motion="bunkers",
                srh_layers=(500.0, 1000.0, 3000.0),
                shear_layers=(1000.0, 3000.0, 6000.0),
                swv_layers=(1000.0,), dz=100.0, effective=True,
                cape_min=100.0, cin_max=250.0, eil_dp=10.0, dcape_layer=(700.0, 500.0),
                virtual_lapse_rate=True):
    """
    Compute the full set of convective parameters for a gridded field.

    Parameters
    ----------
    p : array-like
        Pressure (hPa): ``(nlev,)`` for ``vertical_lev='pressure'`` or
        ``(..., nlev)`` for ``vertical_lev='sigma'``.  Bottom to top.
    t, td : array-like, ``(..., nlev)``
        Temperature and dewpoint (Celsius).  Vertical axis last, bottom up.
    u, v : array-like, ``(..., nlev)``
        Wind components (m/s).
    z_agl : array-like, ``(..., nlev)``
        Height above ground (m), e.g. from geopotential.  Levels at or
        below the ground are ignored.
    ps, ts, tds : array-like, ``(...)``
        Surface pressure (hPa), 2 m temperature and 2 m dewpoint (Celsius).
    us, vs : array-like, ``(...)``
        10 m wind components (m/s).
    vertical_lev : {'pressure', 'sigma'}
    ml_depth : float
        Depth (m) of the mixed layer of the mixed-layer parcel.
    pinc : float
        Pressure increment (Pa) of the parcel ascent.
    adiabat : str
        Moist adiabat, see :func:`xcaper.core.calc_cape`.
    method : {'auto', 'fortran', 'numba'}
        Backend of the CAPE calculation.
    storm_motion, srh_layers, shear_layers, swv_layers, dz
        See :func:`xcaper.kinematics.calc_kinematics`.  The 0-1 km and 0-3 km
        SRH and the 0-6 km shear are always computed because the composite
        indices need them.  ``storm_motion='bunkers-effective'`` needs
        ``effective=True``.
    effective : bool
        Compute LFC, EL, DCAPE, the effective inflow layer and the
        effective-layer parameters (the most expensive part).
    cape_min, cin_max : float
        Effective inflow layer criteria (J/kg, CIN positive).
    eil_dp : float or None
        Spacing (hPa) of the refined profile of the effective-layer search
        (default 10), see :func:`xcaper.parcel.calc_parcel_diagnostics`.
        None searches the input levels only.
    dcape_layer : (float, float)
        Layer (hPa) searched for the DCAPE source parcel.
    virtual_lapse_rate : bool
        Lapse rates of the virtual temperature (default; SPC / SHARPpy /
        SounderPy convention) rather than of the temperature.

    Returns
    -------
    dict of str -> array
        Every field has the shape of ``ps``; dask inputs give lazy dask
        outputs.  See :func:`variable_attrs` for units and descriptions.

        Thermodynamics
            ``SFC_CAPE, SFC_CIN, ML_CAPE, ML_CIN, MU_CAPE, MU_CIN`` (J/kg, CIN
            positive), ``SFC_LCL, ML_LCL`` (m AGL), ``LR03`` (0-3 km),
            ``LR36`` (3-6 km) and ``LR75`` (700-500 hPa) lapse rates (K/km).
        Kinematics
            everything returned by :func:`~xcaper.kinematics.calc_kinematics`,
            plus ``SRH<layer>`` = max(|RM|, |LM|).
        Parcel levels and DCAPE (``effective=True``)
            ``SFC_/ML_/MU_LFC`` and ``*_EL`` (m AGL) with ``*_LFC_P`` and
            ``*_EL_P`` (hPa), ``DCAPE`` (J/kg) and ``DCAPE_SRC_P`` (hPa).
        Effective inflow layer (``effective=True``)
            ``EIL_BASE``, ``EIL_TOP`` (m AGL), ``EIL_BASE_P``, ``EIL_TOP_P``
            (hPa), ``ESRH_RM``, ``ESRH_LM``, ``ESRH`` = max(|RM|, |LM|),
            ``EBWD``, ``E_SWV``, ``E_SWNE``, and the composites ``ESCP`` and
            ``ESTP``.  ESRH, EBWD, ESCP and ESTP are 0 where there is no
            effective inflow layer.
        Composites
            ``SCP`` (MUCAPE, 0-3 km SRH, 0-6 km shear), ``STP`` (fixed layer:
            SBCAPE, SBLCL, 0-1 km SRH, 0-6 km shear), ``STP_CIN`` (MLCAPE,
            MLLCL, 0-1 km SRH, 0-6 km shear, MLCIN), ``EHI1``, ``EHI3``
            (MLCAPE with 0-1 / 0-3 km SRH) and ``WMAXSHEAR`` (MLCAPE, 0-6 km
            shear).  Right-mover SRH is used.
    """
    srh_layers = _with_required(srh_layers, _REQUIRED_SRH)
    shear_layers = _with_required(shear_layers, _REQUIRED_SHEAR)

    out = {}
    thermo_args = (p, t, td, ps, ts, tds)
    cape_kw = dict(pinc=pinc, adiabat=adiabat, method=method, vertical_lev=vertical_lev)

    # ---- parcels ----
    out["SFC_CAPE"], out["SFC_CIN"] = calc_cape(*thermo_args, source="surface", **cape_kw)
    out["ML_CAPE"], out["ML_CIN"] = calc_cape(*thermo_args, source="mixed-layer",
                                              ml_depth=ml_depth, **cape_kw)
    out["MU_CAPE"], out["MU_CIN"] = calc_cape(*thermo_args, source="most-unstable",
                                              **cape_kw)[:2]
    out["SFC_LCL"] = calc_lcl(*thermo_args, source="surface", vertical_lev=vertical_lev)[0]
    out["ML_LCL"] = calc_lcl(*thermo_args, source="mixed-layer", ml_depth=ml_depth,
                             vertical_lev=vertical_lev)[0]
    lr_kw = dict(td=td, tds=tds) if virtual_lapse_rate else {}
    out["LR03"], out["LR75"] = calc_lapse_rates(p, t, z_agl, ps, ts, **lr_kw)
    out["LR36"] = calc_lapse_rates(p, t, z_agl, ps, ts, height_layer=(3000.0, 6000.0), **lr_kw)[0]

    sm = StormMotion.from_any(storm_motion)
    if sm.effective and not effective:
        raise ValueError("storm_motion='bunkers-effective' needs effective=True")
    eff_kw = {}
    valid = None
    if effective:
        par = calc_parcel_diagnostics(*thermo_args, vertical_lev=vertical_lev,
                                      ml_depth=ml_depth, pinc=pinc, adiabat=adiabat,
                                      cape_min=cape_min, cin_max=cin_max, eil_dp=eil_dp,
                                      dcape_layer=dcape_layer, return_valid=True)
        valid = par.pop("_VALID") > 0
        out.update(par)
        eff_kw = dict(eil_base=par["EIL_BASE"], eil_top=par["EIL_TOP"],
                      el_height=par["MU_EL"])

    # ---- wind profile ----
    kin = calc_kinematics(z_agl, u, v, us, vs, storm_motion=sm,
                          srh_layers=srh_layers, shear_layers=shear_layers,
                          swv_layers=swv_layers, dz=dz, **eff_kw)
    if effective:
        # ESRH / EBWD are 0 where there is no effective layer, but must stay
        # missing where the thermodynamic data themselves are missing
        for name in ("ESRH_RM", "ESRH_LM", "EBWD"):
            kin[name] = np.where(valid, kin[name], np.float32(np.nan))
    out.update(kin)
    for top in srh_layers:
        tag = layer_tag(top)
        out[f"SRH{tag}"] = np.maximum(abs(kin[f"SRH{tag}_RM"]), abs(kin[f"SRH{tag}_LM"]))
    if effective:
        out["ESRH"] = np.maximum(abs(kin["ESRH_RM"]), abs(kin["ESRH_LM"]))

    # ---- composites (right-mover SRH) ----
    srh1, srh3, s06 = kin["SRH1_RM"], kin["SRH3_RM"], kin["S06"]
    out["SCP"] = composites.supercell_composite(out["MU_CAPE"], srh3, s06)
    out["STP"] = composites.significant_tornado(out["SFC_CAPE"], out["SFC_LCL"], srh1, s06)
    out["STP_CIN"] = composites.significant_tornado_cin(out["ML_CAPE"], out["ML_LCL"],
                                                        srh1, s06, out["ML_CIN"])
    out["EHI1"] = composites.energy_helicity_index(out["ML_CAPE"], srh1)
    out["EHI3"] = composites.energy_helicity_index(out["ML_CAPE"], srh3)
    out["WMAXSHEAR"] = composites.wmaxshear(out["ML_CAPE"], s06)
    if effective:
        out["ESCP"] = composites.effective_supercell_composite(
            out["MU_CAPE"], kin["ESRH_RM"], kin["EBWD"], out["MU_CIN"])
        estp = composites.effective_significant_tornado(
            out["ML_CAPE"], out["ML_LCL"], kin["ESRH_RM"], kin["EBWD"], out["ML_CIN"],
            out["EIL_BASE"])
        out["ESTP"] = np.where(np.isfinite(kin["ESRH_RM"]), estp, np.float32(np.nan))
    return out


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------
_STATIC_ATTRS = {
    "SFC_CAPE": ("J kg-1", "Surface-based CAPE"),
    "SFC_CIN": ("J kg-1", "Surface-based CIN (positive = inhibition)"),
    "ML_CAPE": ("J kg-1", "Mixed-layer CAPE"),
    "ML_CIN": ("J kg-1", "Mixed-layer CIN (positive = inhibition)"),
    "MU_CAPE": ("J kg-1", "Most-unstable CAPE"),
    "MU_CIN": ("J kg-1", "Most-unstable CIN (positive = inhibition)"),
    "SFC_LCL": ("m", "LCL height above ground of the surface parcel"),
    "ML_LCL": ("m", "LCL height above ground of the mixed-layer parcel"),
    "LR03": ("K km-1", "0-3 km AGL lapse rate (virtual temperature unless virtual_lapse_rate=False)"),
    "LR36": ("K km-1", "3-6 km AGL lapse rate (virtual temperature unless virtual_lapse_rate=False)"),
    "LR75": ("K km-1", "700-500 hPa lapse rate (virtual temperature unless virtual_lapse_rate=False)"),
    "BUNKERS_RM_U": ("m s-1", "Right-moving storm motion, U component"),
    "BUNKERS_RM_V": ("m s-1", "Right-moving storm motion, V component"),
    "BUNKERS_LM_U": ("m s-1", "Left-moving storm motion, U component"),
    "BUNKERS_LM_V": ("m s-1", "Left-moving storm motion, V component"),
    "CRITICAL_ANGLE": ("degree", "Critical angle: 0-500 m shear vector vs. surface "
                                 "storm-relative inflow (right mover)"),
    "EIL_BASE": ("m", "Effective inflow layer base above ground (CAPE >= 100 J/kg, "
                      "CIN <= 250 J/kg; NaN = no layer)"),
    "EIL_TOP": ("m", "Effective inflow layer top above ground"),
    "EIL_BASE_P": ("hPa", "Effective inflow layer base pressure"),
    "EIL_TOP_P": ("hPa", "Effective inflow layer top pressure"),
    "ESRH_RM": ("m2 s-2", "Effective storm relative helicity, right mover (0 = no layer)"),
    "ESRH_LM": ("m2 s-2", "Effective storm relative helicity, left mover (0 = no layer)"),
    "ESRH": ("m2 s-2", "Effective storm relative helicity, max(|RM|, |LM|)"),
    "EBWD": ("m s-1", "Effective bulk wind difference: effective inflow base to 50% of the "
                      "most-unstable EL depth (0 = no layer)"),
    "E_SWV": ("s-1", "Mean streamwise vorticity magnitude over the effective inflow layer "
                     "(right mover)"),
    "E_SWNE": ("%", "Mean streamwiseness over the effective inflow layer (right mover)"),
    "DCAPE": ("J kg-1", "Downdraft CAPE (MetPy recipe: min theta-e 700-500 hPa, wet-bulb, "
                        "pseudoadiabatic descent)"),
    "DCAPE_SRC_P": ("hPa", "Pressure of the DCAPE source parcel"),
    "BUNKERS_MEAN_U": ("m s-1", "Mean wind of the Bunkers storm motion, U component"),
    "BUNKERS_MEAN_V": ("m s-1", "Mean wind of the Bunkers storm motion, V component"),
    "BUNKERS_EFF_FLAG": ("1", "1 = effective-layer Bunkers (2014) motion, 0 = 0-6 km fallback"),
    "ESCP": ("1", "Effective Supercell Composite Parameter "
                  "(MUCAPE, ESRH RM, EBWD, MUCIN)"),
    "ESTP": ("1", "Effective-layer Significant Tornado Parameter "
                  "(MLCAPE, MLLCL, ESRH RM, EBWD, MLCIN; 0 when the layer is elevated)"),
    "SCP": ("1", "Supercell Composite Parameter, fixed layer "
                 "(MUCAPE, 0-3 km SRH RM, 0-6 km bulk wind difference)"),
    "STP": ("1", "Significant Tornado Parameter, fixed layer "
                 "(SBCAPE, SBLCL, 0-1 km SRH RM, 0-6 km bulk wind difference)"),
    "STP_CIN": ("1", "Significant Tornado Parameter with CIN term "
                     "(MLCAPE, MLLCL, 0-1 km SRH RM, 0-6 km bulk wind difference, MLCIN)"),
    "EHI1": ("1", "0-1 km Energy Helicity Index (MLCAPE, SRH RM)"),
    "EHI3": ("1", "0-3 km Energy Helicity Index (MLCAPE, SRH RM)"),
    "WMAXSHEAR": ("m2 s-2", "WMAXSHEAR = sqrt(2 MLCAPE) * 0-6 km bulk wind difference"),
}


def _layer_text(tag):
    """'1' -> '0-1 km', '06' -> '0-6 km', '500M' -> '0-500 m'."""
    if tag.endswith("M"):
        return f"0-{int(tag[:-1])} m"
    return f"0-{int(tag)} km"


def variable_attrs(name, storm_motion="bunkers"):
    """``{'units': ..., 'long_name': ...}`` of an output variable of xcaper."""
    if name in _STATIC_ATTRS:
        units, long_name = _STATIC_ATTRS[name]
    elif name[:3] in ("SFC", "ML_", "MU_") and name.split("_", 1)[1] in (
            "LFC", "LFC_P", "EL", "EL_P"):
        parcel = {"SFC": "surface", "ML": "mixed-layer", "MU": "most-unstable"}[name.split("_")[0]]
        level = "level of free convection" if "LFC" in name else "equilibrium level"
        if name.endswith("_P"):
            units, long_name = "hPa", f"Pressure of the {level}, {parcel} parcel"
        else:
            units, long_name = "m", f"Height above ground of the {level}, {parcel} parcel"
    elif name.startswith("MEAN") and name[-2:] in ("_U", "_V"):
        sm = StormMotion.from_any(storm_motion)
        units = "m s-1"
        long_name = ("%g-%g m AGL non-pressure-weighted mean wind, %s component"
                     % (sm.mean_layer + (name[-1],)))
    elif name.startswith("SRH"):
        units = "m2 s-2"
        body = name[3:]
        if body.endswith("_RM"):
            long_name = f"{_layer_text(body[:-3])} storm relative helicity, right mover"
        elif body.endswith("_LM"):
            long_name = f"{_layer_text(body[:-3])} storm relative helicity, left mover"
        else:
            long_name = f"{_layer_text(body)} storm relative helicity, max(|RM|, |LM|)"
    elif name.startswith("SWNESS"):
        units = "%"
        long_name = (f"{_layer_text(name[6:])} mean streamwiseness of the horizontal "
                     "vorticity (right mover)")
    elif name.startswith("SWV"):
        units = "s-1"
        long_name = f"{_layer_text(name[3:])} mean streamwise vorticity magnitude (right mover)"
    elif name.startswith("S") and name[1:2].isdigit():
        units = "m s-1"
        long_name = f"{_layer_text(name[1:])} bulk wind difference"
    else:
        units, long_name = "", name
    return {"units": units, "long_name": long_name}
