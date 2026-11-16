"""xarray API of xcaper.

The functions here take :class:`xarray.DataArray` objects in the units of
:mod:`xcaper.core` (hPa, Celsius, m/s, m), move the vertical dimension to
the last axis, call the numpy / dask implementation and wrap the results
with coordinates and CF-style attributes.  Dask-backed inputs stay lazy
and are computed chunk by chunk in parallel.

For ERA5 files (Kelvin, Pa, specific humidity, geopotential) use
:func:`xcaper.era5.compute_severe_parameters`, which converts the units and
then calls :func:`severe_parameters`.
"""
from __future__ import annotations

import xarray as xr

from . import core as _core
from . import severe as _severe
from .kinematics import StormMotion
from .kinematics import calc_kinematics as _calc_kinematics

__all__ = ["calc_cape", "calc_srh", "calc_kinematics", "severe_parameters"]


def _prepare(profiles, surfaces, vertical_dim, pressure=None):
    """Align inputs; return raw arrays with the vertical axis last + a template."""
    first = profiles[0]
    if vertical_dim not in first.dims:
        raise ValueError(f"vertical_dim {vertical_dim!r} is not a dimension of the profile data")
    horiz = [d for d in first.dims if d != vertical_dim]
    order = horiz + [vertical_dim]

    def profile(da_):
        da_ = da_.transpose(*order)
        if da_.chunks is not None:
            da_ = da_.chunk({vertical_dim: -1})
        return da_.data

    def surface(da_):
        missing = [d for d in horiz if d not in da_.dims]
        if missing or len(da_.dims) != len(horiz):
            raise ValueError(
                f"surface field {da_.name!r} has dimensions {da_.dims}, expected {tuple(horiz)}"
            )
        return da_.transpose(*horiz).data

    prof = [profile(a) for a in profiles]
    sfc = [surface(a) for a in surfaces]
    template = surfaces[0].transpose(*horiz)

    p_data = None
    if pressure is not None:
        if pressure.ndim == 1:
            p_data = pressure.values.astype("f8")
        else:
            p_data = profile(pressure)
    return prof, sfc, p_data, template


def _wrap(data, template, name, attrs):
    out = xr.DataArray(data, dims=template.dims, coords=template.coords, name=name)
    out.attrs.update(attrs)
    return out


def _vertical_lev(p, vertical_lev):
    if vertical_lev is None:
        return "pressure" if p.ndim == 1 else "sigma"
    return vertical_lev


def calc_cape(p, t, td, ps, ts, tds, *, vertical_dim, vertical_lev=None, **kwargs):
    """xarray wrapper of :func:`xcaper.core.calc_cape`.

    Parameters
    ----------
    p : DataArray
        Pressure (hPa): 1-D along ``vertical_dim`` (pressure levels) or
        with the dimensions of ``t`` (model levels).  Bottom to top.
    t, td : DataArray
        Temperature and dewpoint (Celsius).
    ps, ts, tds : DataArray
        Surface pressure (hPa), temperature and dewpoint (Celsius).
    vertical_dim : str
        Name of the vertical dimension.
    vertical_lev : {'pressure', 'sigma'}, optional
        Inferred from the dimensionality of ``p`` when omitted.
    **kwargs
        ``source``, ``ml_depth``, ``adiabat``, ``pinc``, ``method``.

    Returns
    -------
    xarray.Dataset
        ``cape`` and ``cin`` (plus ``mulev`` and ``zmulev`` for the
        most-unstable parcel).
    """
    (t_, td_), (ps_, ts_, tds_), p_, template = _prepare([t, td], [ps, ts, tds],
                                                         vertical_dim, pressure=p)
    res = _core.calc_cape(p_, t_, td_, ps_, ts_, tds_,
                          vertical_lev=_vertical_lev(p, vertical_lev), **kwargs)
    source = kwargs.get("source", "surface")
    meta = [("cape", "J kg-1", f"CAPE ({source} parcel)"),
            ("cin", "J kg-1", f"CIN ({source} parcel, positive = inhibition)"),
            ("mulev", "1", "index of the most unstable level (1 = surface)"),
            ("zmulev", "m", "height above the surface of the top of the parcel ascent")]
    return xr.Dataset({name: _wrap(a, template, name, {"units": units, "long_name": long_name})
                       for a, (name, units, long_name) in zip(res, meta)})


def calc_srh(p, t, td, u, v, ps, ts, tds, us, vs, *, vertical_dim, vertical_lev=None,
             **kwargs):
    """xarray wrapper of :func:`xcaper.core.calc_srh` (legacy Bunkers / SRH).

    Returns a Dataset with ``srh_rm`` and ``srh_lm`` and, for
    ``output_var='all'``, the storm motions and the 0-6 km mean wind.
    """
    (t_, td_, u_, v_), (ps_, ts_, tds_, us_, vs_), p_, template = _prepare(
        [t, td, u, v], [ps, ts, tds, us, vs], vertical_dim, pressure=p)
    res = _core.calc_srh(p_, t_, td_, u_, v_, ps_, ts_, tds_, us_, vs_,
                         vertical_lev=_vertical_lev(p, vertical_lev), **kwargs)
    depth = kwargs.get("depth", 3000)
    names = [("srh_rm", "m2 s-2", f"0-{depth:g} m storm relative helicity, right mover"),
             ("srh_lm", "m2 s-2", f"0-{depth:g} m storm relative helicity, left mover"),
             ("bunkers_rm_u", "m s-1", "Bunkers right-mover motion, U"),
             ("bunkers_rm_v", "m s-1", "Bunkers right-mover motion, V"),
             ("bunkers_lm_u", "m s-1", "Bunkers left-mover motion, U"),
             ("bunkers_lm_v", "m s-1", "Bunkers left-mover motion, V"),
             ("mean_6km_u", "m s-1", "0-6 km mean wind, U"),
             ("mean_6km_v", "m s-1", "0-6 km mean wind, V")]
    return xr.Dataset({name: _wrap(a, template, name, {"units": units, "long_name": long_name})
                       for a, (name, units, long_name) in zip(res, names)})


def calc_kinematics(z_agl, u, v, us, vs, *, vertical_dim, **kwargs):
    """xarray wrapper of :func:`xcaper.kinematics.calc_kinematics`."""
    (z_, u_, v_), (us_, vs_), _, template = _prepare([z_agl, u, v], [us, vs], vertical_dim)
    res = _calc_kinematics(z_, u_, v_, us_, vs_, **kwargs)
    sm = kwargs.get("storm_motion", "bunkers")
    return xr.Dataset({name: _wrap(a, template, name, _severe.variable_attrs(name, sm))
                       for name, a in res.items()})


def severe_parameters(p, t, td, u, v, z_agl, ps, ts, tds, us, vs, *, vertical_dim,
                      vertical_lev=None, **kwargs):
    """
    All severe-weather parameters of xcaper as an :class:`xarray.Dataset`.

    Parameters
    ----------
    p : DataArray
        Pressure (hPa), 1-D along ``vertical_dim`` or full field; bottom up.
    t, td : DataArray
        Temperature and dewpoint (Celsius).
    u, v : DataArray
        Wind components (m/s).
    z_agl : DataArray
        Height above ground (m).
    ps, ts, tds, us, vs : DataArray
        Surface pressure (hPa), 2 m temperature and dewpoint (Celsius),
        10 m wind (m/s).
    vertical_dim : str
        Name of the vertical dimension.
    vertical_lev : {'pressure', 'sigma'}, optional
        Inferred from the dimensionality of ``p`` when omitted.
    **kwargs
        Options of :func:`xcaper.severe.calc_severe` (``ml_depth``, ``pinc``,
        ``adiabat``, ``method``, ``storm_motion``, ``srh_layers``,
        ``shear_layers``, ``swv_layers``, ``dz``).

    Returns
    -------
    xarray.Dataset
        One variable per parameter on the horizontal (and time) dimensions
        of the surface fields.  Lazy when the inputs are dask-backed.
    """
    (t_, td_, u_, v_, z_), (ps_, ts_, tds_, us_, vs_), p_, template = _prepare(
        [t, td, u, v, z_agl], [ps, ts, tds, us, vs], vertical_dim, pressure=p)
    res = _severe.calc_severe(p_, t_, td_, u_, v_, z_, ps_, ts_, tds_, us_, vs_,
                              vertical_lev=_vertical_lev(p, vertical_lev), **kwargs)
    sm = StormMotion.from_any(kwargs.get("storm_motion", "bunkers"))
    ds = xr.Dataset({name: _wrap(a, template, name, _severe.variable_attrs(name, sm))
                     for name, a in res.items()})
    ds.attrs.update({
        "storm_motion": sm.describe(),
        "wind_profile": (
            "10-m wind at 0 m AGL + levels above ground, linearly interpolated to a "
            "%g-m AGL grid" % kwargs.get("dz", 100.0)
        ),
        "cape_method": (
            "xcaper calc_cape (G. Bryan getcape); adiabat=%s; pinc=%g Pa; "
            "mixed-layer depth=%g m; CIN reported as a positive number"
            % (kwargs.get("adiabat", "pseudo-liquid"), kwargs.get("pinc", 500.0),
               kwargs.get("ml_depth", 500.0))
        ),
        "srh_method": "discrete SRH on the uniform AGL profile; SRH<layer> = max(|RM|, |LM|)",
        "swv_method": (
            "projection of the horizontal vorticity onto the storm-relative wind of the "
            "right mover (SounderPy style); layer mean of the magnitude"
        ),
        "composites": "SCP, STP, STP_CIN, EHI, WMAXSHEAR, ESCP and ESTP use right-mover SRH",
    })
    if kwargs.get("effective", True):
        ds.attrs.update({
            "effective_inflow_layer": (
                "Thompson et al. (2007): lowest contiguous layer of input levels whose parcels "
                "have CAPE >= %g J/kg and CIN <= %g J/kg (ascent stopped once CAPE reaches the "
                "threshold); searched only where the most-unstable parcel meets the criteria; "
                "search profile: %s"
                % (kwargs.get("cape_min", 100.0), kwargs.get("cin_max", 250.0),
                   "input levels" if not kwargs.get("eil_dp", 10.0) else
                   "refined to %g hPa (T, Td linear in ln p)" % kwargs.get("eil_dp", 10.0))
            ),
            "parcel_levels": (
                "LFC: lowest point at or above the LCL where the parcel is positively buoyant "
                "(virtual temperature); EL: top of the highest positively buoyant layer, or the "
                "top of the data when the parcel is still buoyant there; heights from the "
                "hydrostatic integration of the parcel code"
            ),
            "dcape_method": (
                "MetPy downdraft_cape recipe: minimum theta-e between %g and %g hPa, wet-bulb "
                "start, pseudoadiabatic descent to the surface, Rd * integral (Tv_env - "
                "Tv_parcel) dln(p)" % tuple(kwargs.get("dcape_layer", (700.0, 500.0)))
            ),
        })
    return ds
