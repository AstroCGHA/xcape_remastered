# Copyright (c) 2020 xcaper Developers.
"""
Numpy / dask API of xcaper: Convective Available Potential Energy (CAPE),
Convective Inhibition (CIN) and Storm Relative Helicity (SRH).

Every function accepts numpy arrays (computed eagerly) or dask arrays
(computed lazily, chunk by chunk, in parallel).  The vertical dimension is
always the **last** axis and runs from the bottom of the atmosphere to the
top.

Two interchangeable numerical backends exist:

``method='fortran'``
    the compiled Fortran kernels (single precision, identical to xcape 0.1);
``method='numba'``
    a numba port (double precision) that needs no compiler at install time
    and uses all cores on numpy inputs;
``method='auto'`` (default)
    Fortran when the extension modules are importable, numba otherwise.
"""
from __future__ import annotations

from functools import reduce

import dask.array as da
import numpy as np

from .duck_array_ops import (broadcast_arrays, concatenate, ravel_multi_index,  # noqa: F401
                             reshape)

# The Fortran extensions are optional: xcaper still works (through numba)
# when they could not be built, and the docs can be built without them.
try:
    from .cape_fortran import HAS_FORTRAN
    from .cape_fortran import cape as _cape_fortran
except ImportError:  # pragma: no cover
    HAS_FORTRAN = False
    _cape_fortran = None

from .srh import srh as _srh
from .stdheight import stdheight as _stdheight

__all__ = ["calc_cape", "calc_srh", "calc_lcl", "HAS_FORTRAN"]

_SOURCE_OPTIONS = {"surface": 1, "most-unstable": 2, "mixed-layer": 3}
_ADIABAT_OPTIONS = {"pseudo-liquid": 1, "reversible-liquid": 2,
                    "pseudo-ice": 3, "reversible-ice": 4}
_VERTICAL_LEV_OPTIONS = {"sigma": 1, "pressure": 2}
_OUTPUT_VAR_OPTIONS = {"srh": 1, "all": 2}
_METHODS = ("auto", "fortran", "numba")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _prod(v):
    """Product of the elements of a sequence."""
    return reduce(lambda x, y: x * y, v)


def _resolve_method(method):
    """Translate ``'auto'`` into the backend that is actually available."""
    if method == "auto":
        return "fortran" if HAS_FORTRAN else "numba"
    if method == "fortran" and not HAS_FORTRAN:
        raise ImportError(
            "method='fortran' was requested but the xcaper Fortran extension "
            "modules are not available. Reinstall xcaper with a Fortran "
            "compiler, or use method='numba' (or 'auto')."
        )
    return method


def _check_option(name, value, allowed):
    if value not in allowed:
        raise ValueError(f"`{name}` must be one of: {list(allowed)}; got {value!r}")


def _reshape_inputs(*args):
    """Collapse arrays ``(X1, X2, ..., Z)`` to ``(Z, X1*X2*...)``.

    The kernels work on a 2-D (level, grid point) layout; the transpose is
    a view, so no data is copied for Fortran-ordered consumption.
    """
    a0 = args[0]
    shape = a0.shape
    for a in args:
        if a.shape != shape:
            raise ValueError("Input arrays must have the same shape.")

    args_al2d = [np.atleast_2d(a) for a in args]
    original_shape = args_al2d[0].shape
    new_shape = (_prod(original_shape[:-1]),) + (original_shape[-1],)
    # transpose to move nlevs to first axis
    return [np.reshape(a, new_shape).transpose() for a in args_al2d]


def _reshape_outputs(*args, shape=None):
    """Reshape kernel outputs ``(X1*X2*...,)`` to the input shape minus Z."""
    target_shape = (1,) if len(shape) == 1 else shape[:-1]
    return [np.reshape(np.asarray(a).transpose(), target_shape) for a in args]


def _reshape_outputs_uv_components(*args, shape=None):
    """Reshape two-component outputs ``(2, X1*X2*...)`` to ``(2, X1, X2, ...)``."""
    target_shape = (2,) if len(shape) == 1 else (2,) + shape[:-1]
    return [np.reshape(a, target_shape) for a in args]


def _reshape_surface_inputs(*args):
    """Collapse surface arrays ``(X1, X2, ...)`` to ``(X1*X2*...,)``."""
    a0 = args[0]
    shape = a0.shape
    for a in args:
        if a.shape != shape:
            raise ValueError("Input arrays must have the same shape.")

    args_al1d = [np.atleast_1d(a) for a in args]
    new_shape = (_prod(args_al1d[0].shape),)
    return [np.reshape(a, new_shape) for a in args_al1d]


def _pressure_level_start(p_2d, p_s1d):
    """1-based index of the first pressure level at or above the surface.

    Pressure-level products are defined on every level, including those
    that lie below the ground where surface pressure is low.  Only levels
    with ``p <= ps`` must enter the calculation.  ``p_2d`` has shape
    ``(nlev, 1)`` (bottom to top), ``p_s1d`` shape ``(ngrid,)``.
    """
    diff = p_s1d[np.newaxis, :] - p_2d
    # smallest non-negative difference = closest level above the surface
    start = np.where(diff >= 0, diff, np.inf).argmin(axis=0)
    return (start + 1).astype(np.intc)  # fortran convention


def _collapse_thermo(p, t, td, ps, ts, tds, vertical_lev, extra_3d=(), extra_sfc=()):
    """Shared input handling of calc_cape / calc_srh / calc_lcl.

    Returns ``(p_2d, [3-D fields as (nlev, ngrid)], [surface fields as
    (ngrid,)], flag_1d, pres_lev_pos)``.
    """
    sfc = _reshape_surface_inputs(ps, ts, tds, *extra_sfc)
    p_s1d = sfc[0]
    if p.ndim == 1:
        fields = _reshape_inputs(t, td, *extra_3d)
        p_2d = _reshape_inputs(p)[0]
        flag_1d = 1
        pres_lev_pos = _pressure_level_start(p_2d, p_s1d)
    elif p.shape == t.shape and vertical_lev == "sigma":
        p_2d, *fields = _reshape_inputs(p, t, td, *extra_3d)
        flag_1d = 0
        pres_lev_pos = 1
    elif p.shape == t.shape and vertical_lev == "pressure":
        raise ValueError("P should be 1d")
    else:
        raise ValueError(
            "pressure must either be 1-D (one value per level) or have the same "
            f"shape as temperature; got p.shape={p.shape}, t.shape={t.shape}"
        )
    if fields[0].shape[1] != p_s1d.shape[0]:
        raise ValueError(
            "Surface arrays must have the shape of the 3-D arrays without the "
            "vertical (last) dimension."
        )
    return p_2d, fields, sfc, flag_1d, pres_lev_pos


def _cape_dummy(*args, **kwargs):
    """Stand-in kernel used by the shape tests (returns ones)."""
    p, t, td, ps, ts, tds = args
    # cape is a reduction along the second axis.
    # this tests that reshaping works.
    assert p.ndim == 2
    shape = t.shape
    cape = np.ones((1, shape[1]))
    cin = np.ones((1, shape[1]))
    mulev = np.ones((1, shape[1]))
    zmulev = np.ones((1, shape[1]))
    return cape, cin, mulev, zmulev


def _any_dask_array(*args):
    """True when any of the inputs is a dask array (lazy, parallel path)."""
    return any(isinstance(a, da.Array) for a in args)


# ---------------------------------------------------------------------------
# CAPE / CIN
# ---------------------------------------------------------------------------
def calc_cape(*args, source="surface", ml_depth=500.0, adiabat="pseudo-liquid",
              pinc=500.0, method="auto", vertical_lev="sigma"):
    r"""
    Calculate Convective Available Potential Energy (CAPE) and Convective Inhibition (CIN).

    CAPE and CIN are obtained for every column of an N-dimensional gridded
    field by lifting a parcel and integrating the buoyancy between the
    parcel and its environment (George Bryan's ``getcape``).  The parcel can
    be surface based, a mixed layer of given depth, or the most unstable
    parcel; its ascent can follow a pseudo-adiabat or a reversible adiabat,
    with or without ice.

    Parameters
    ----------
    p : array-like
        Atmospheric pressure at each vertical level in hPa.
        When ``vertical_lev='sigma'``, ``p.shape == t.shape == (..., nlev)``.
        When ``vertical_lev='pressure'``, ``p.shape == (nlev,)``.
    t : array-like
        Atmospheric temperature in Celsius, shape ``(..., nlev)``; the
        vertical axis is last and runs from the bottom up.
    td : array-like
        Atmospheric dew point temperature in Celsius, same shape as ``t``.
    ps : array-like
        Surface pressure in hPa, shape ``t.shape[:-1]``.
    ts : array-like
        Surface (2 m) temperature in Celsius.
    tds : array-like
        Surface (2 m) dew point temperature in Celsius.
    source : {'surface', 'mixed-layer', 'most-unstable'}, optional
        Parcel to lift.

        - 'surface': the surface parcel; sensitive to surface heating and
          to shallow moisture.
        - 'mixed-layer': mean of the lowest ``ml_depth`` metres; a good
          assumption when the boundary layer is well mixed.
        - 'most-unstable': the level below 500 hPa with the highest
          equivalent potential temperature.
    ml_depth : float, optional
        Depth (m) of the mixed layer. Only used when ``source='mixed-layer'``.
    adiabat : {'pseudo-liquid', 'reversible-liquid', 'pseudo-ice', 'reversible-ice'}, optional
        Formulation of the moist adiabat.
    pinc : float, optional
        Pressure increment (Pa) of the integration. Values between 100 and
        1000 are sensible: smaller is more accurate, larger is faster.
    method : {'auto', 'fortran', 'numba'}, optional
        Numerical backend. 'auto' uses the Fortran kernels when available
        and numba otherwise.
    vertical_lev : {'sigma', 'pressure'}, optional
        Vertical grid of the input: model levels ('sigma') or fixed
        pressure levels ('pressure'). On pressure levels, the levels below
        the surface (``p > ps``) are ignored.

    Returns
    -------
    cape : array-like
        Convective available potential energy (J/kg).
    cin : array-like
        Convective inhibition (J/kg), as a positive number.
    MUlev : array-like
        Index of the most unstable level, counted from 1 at the surface
        (only returned for ``source='most-unstable'``).
    zMUlev : array-like
        Height (m above the surface) of the highest level reached by the
        parcel ascent (only returned for ``source='most-unstable'``).

    Notes
    -----
    .. math:: CAPE = g \int_{LFC}^{EL} \frac{T_{v,parcel} - T_{v,env}}{T_{v,env}} \, dz

    .. math:: CIN = -g \int_{SFC}^{LFC} \frac{T_{v,parcel} - T_{v,env}}{T_{v,env}} \, dz

    * :math:`LFC` = level of free convection
    * :math:`EL` = equilibrium level
    * :math:`T_{v,parcel}`, :math:`T_{v,env}` = virtual potential
      temperature of the parcel and of the environment [1]_

    Columns whose surface temperature is not above 0 Celsius (which
    includes missing values) are returned as zero CAPE and CIN; this is the
    behaviour of xcape 0.1 and is kept for backward compatibility.

    With dask inputs the computation is lazy and runs chunk by chunk; the
    vertical dimension is automatically gathered into a single chunk.

    Examples
    --------
    >>> cape, cin = calc_cape(p, t, td, ps, ts, tds, source='mixed-layer',
    ...                       ml_depth=500., adiabat='pseudo-liquid',
    ...                       pinc=500., vertical_lev='pressure')

    References
    ----------
    .. [1] Emanuel, K. A. (1994). Atmospheric convection. Oxford University Press.
    """
    if len(args) < 6:
        raise ValueError("Too few arguments.")
    if len(args) > 6:
        raise ValueError("Too many arguments.")

    _check_option("vertical_lev", vertical_lev, _VERTICAL_LEV_OPTIONS)
    _check_option("source", source, _SOURCE_OPTIONS)
    _check_option("adiabat", adiabat, _ADIABAT_OPTIONS)
    if method != "dummy":
        _check_option("method", method, _METHODS)
        method = _resolve_method(method)

    kwargs = dict(source=source, ml_depth=ml_depth, adiabat=adiabat, pinc=pinc,
                  method=method, vertical_lev=vertical_lev)
    if _any_dask_array(*args):
        return _calc_cape_gufunc(*args, **kwargs)
    return _calc_cape_numpy(*args, **kwargs)


def _calc_cape_gufunc(*args, **kwargs):
    """Lazy, chunk-wise CAPE for dask arrays (``dask.array.apply_gufunc``)."""
    # The vertical dimension "i" is the only core dimension.  A 1-D pressure
    # vector (pressure levels) simply has no loop dimensions, so the same
    # signature serves both vertical grids.
    signature = "(i),(i),(i),(),(),()->(),()"
    output_dtypes = ("f4", "f4")
    if kwargs["source"] == "most-unstable":
        signature += ",(),()"
        output_dtypes = output_dtypes + ("i4", "f4")

    return da.apply_gufunc(_calc_cape_numpy, signature, *args,
                           output_dtypes=output_dtypes,
                           axis=-1,
                           vectorize=False,
                           allow_rechunk=True,
                           _parallel=False,
                           **kwargs)


def _calc_cape_numpy(*args, source="surface", ml_depth=500.0,
                     adiabat="pseudo-liquid", pinc=500.0, method="fortran",
                     vertical_lev="sigma", _parallel=True):
    """Eager CAPE on numpy arrays: collapse the grid, call the kernel, reshape."""
    p, t, td, ps, ts, tds = (np.asarray(a) for a in args)

    original_shape = t.shape  # shape of 3D variable, i.e. t (p could be 1d)

    # after this, all 3-D arrays have shape (nlevs, npoints)
    p_2d, (t_2d, td_2d), (p_s1d, t_s1d, td_s1d), flag_1d, pres_lev_pos = _collapse_thermo(
        p, t, td, ps, ts, tds, vertical_lev)

    kwargs = dict(source=_SOURCE_OPTIONS[source],
                  ml_depth=ml_depth,
                  adiabat=_ADIABAT_OPTIONS[adiabat],
                  pinc=pinc,
                  type_grid=_VERTICAL_LEV_OPTIONS[vertical_lev])

    if method == "auto":
        method = _resolve_method(method)

    if method == "fortran":
        cape_2d, cin_2d, mulev, zmulev = _cape_fortran(p_2d, t_2d, td_2d,
                                                       p_s1d, t_s1d, td_s1d,
                                                       flag_1d, pres_lev_pos,
                                                       **kwargs)
    elif method == "numba":
        from .cape_numba import cape as _cape_numba

        cape_2d, cin_2d, mulev, zmulev = _cape_numba(p_2d, t_2d, td_2d,
                                                     p_s1d, t_s1d, td_s1d,
                                                     flag_1d, pres_lev_pos,
                                                     parallel=_parallel,
                                                     **kwargs)
    elif method == "dummy":
        cape_2d, cin_2d, mulev, zmulev = _cape_dummy(p_2d, t_2d, td_2d,
                                                     p_s1d, t_s1d, td_s1d,
                                                     **kwargs)
    else:
        raise ValueError("invalid method")

    if _SOURCE_OPTIONS[source] == 2:
        cape, cin, mulev, zmulev = _reshape_outputs(cape_2d, cin_2d, mulev, zmulev,
                                                    shape=original_shape)
        return cape, cin, mulev, zmulev
    cape, cin = _reshape_outputs(cape_2d, cin_2d, shape=original_shape)
    return cape, cin


# ---------------------------------------------------------------------------
# LCL of the lifted parcel
# ---------------------------------------------------------------------------
def calc_lcl(*args, source="surface", ml_depth=500.0, vertical_lev="sigma"):
    """
    Lifting condensation level of the parcel that :func:`calc_cape` lifts.

    The surface, mixed-layer or most-unstable parcel is defined exactly as
    in :func:`calc_cape`.  The LCL temperature follows Bolton (1980), the
    LCL height follows from dry-adiabatic ascent.

    Parameters
    ----------
    p, t, td, ps, ts, tds : array-like
        As in :func:`calc_cape` (hPa and Celsius, vertical axis last).
    source : {'surface', 'mixed-layer', 'most-unstable'}, optional
    ml_depth : float, optional
        Depth (m) of the mixed layer for ``source='mixed-layer'``.
    vertical_lev : {'sigma', 'pressure'}, optional

    Returns
    -------
    lcl_height : array-like
        Height of the LCL above the surface (m).
    lcl_pressure : array-like
        Pressure of the LCL (hPa).

    Notes
    -----
    Implemented in numba only (it is cheap); numpy inputs are processed in
    parallel, dask inputs lazily chunk by chunk.
    """
    if len(args) != 6:
        raise ValueError("calc_lcl needs exactly six arrays: p, t, td, ps, ts, tds")
    _check_option("vertical_lev", vertical_lev, _VERTICAL_LEV_OPTIONS)
    _check_option("source", source, _SOURCE_OPTIONS)
    kwargs = dict(source=source, ml_depth=ml_depth, vertical_lev=vertical_lev)
    if _any_dask_array(*args):
        return da.apply_gufunc(_calc_lcl_numpy, "(i),(i),(i),(),(),()->(),()", *args,
                               output_dtypes=("f4", "f4"), axis=-1, vectorize=False,
                               allow_rechunk=True, _parallel=False, **kwargs)
    return _calc_lcl_numpy(*args, **kwargs)


def _calc_lcl_numpy(*args, source="surface", ml_depth=500.0, vertical_lev="sigma",
                    _parallel=True):
    from .cape_numba import lcl as _lcl_numba

    p, t, td, ps, ts, tds = (np.asarray(a) for a in args)
    original_shape = t.shape
    p_2d, (t_2d, td_2d), (p_s1d, t_s1d, td_s1d), _, pres_lev_pos = _collapse_thermo(
        p, t, td, ps, ts, tds, vertical_lev)
    z_lcl, p_lcl, _ = _lcl_numba(p_2d, t_2d, td_2d, p_s1d, t_s1d, td_s1d,
                                 pres_lev_pos, _SOURCE_OPTIONS[source], ml_depth,
                                 _VERTICAL_LEV_OPTIONS[vertical_lev], parallel=_parallel)
    z_lcl, p_lcl = _reshape_outputs(z_lcl, p_lcl, shape=original_shape)
    return z_lcl, p_lcl


# ---------------------------------------------------------------------------
# Storm relative helicity (legacy algorithm)
# ---------------------------------------------------------------------------
def calc_srh(*args, depth=3000, vertical_lev="sigma", output_var="srh", method="auto"):
    r"""
    Calculate storm relative helicity for every column of a gridded field.

    SRH is integrated from the surface over ``depth`` metres, relative to
    the storm motion of Bunkers et al. (2000) [1]_: the un-weighted mean
    0-6 km wind (sampled every 500 m) plus / minus a 7.5 m/s deviation
    perpendicular to the 0-6 km shear for right / left moving supercells.
    Heights above ground are obtained from the hypsometric equation, which
    is why pressure, temperature and dewpoint are needed.

    For other storm-motion layers, several SRH layers at once, bulk shear
    or streamwise vorticity see :func:`xcaper.kinematics.calc_kinematics`.

    Parameters
    ----------
    p : array-like
        Atmospheric pressure in hPa.
        When ``vertical_lev='sigma'``, ``p.shape == t.shape == (..., nlev)``.
        When ``vertical_lev='pressure'``, ``p.shape == (nlev,)``.
    t : array-like
        Atmospheric temperature in Celsius (vertical axis last, bottom up).
    td : array-like
        Atmospheric dew point temperature in Celsius.
    u : array-like
        Zonal wind component in m/s.
    v : array-like
        Meridional wind component in m/s.
    ps : array-like
        Surface pressure in hPa.
    ts : array-like
        Surface temperature in Celsius.
    tds : array-like
        Surface dew point temperature in Celsius.
    us : array-like
        10 m zonal wind in m/s.
    vs : array-like
        10 m meridional wind in m/s.
    depth : float, optional
        Depth (m) of the layer over which SRH is integrated.
    vertical_lev : {'sigma', 'pressure'}, optional
        Vertical grid of the input.
    output_var : {'srh', 'all'}, optional
        'srh' returns only the two helicities; 'all' also returns the
        Bunkers storm motions and the mean 0-6 km wind.
    method : {'auto', 'fortran', 'numba'}, optional
        Numerical backend.

    Returns
    -------
    srh_rm : array-like
        Storm relative helicity for the right-moving storm (m2/s2).
    srh_lm : array-like
        Storm relative helicity for the left-moving storm (m2/s2).
    bunkers_rm_u, bunkers_rm_v : array-like
        Right-mover storm motion (only for ``output_var='all'``).
    bunkers_lm_u, bunkers_lm_v : array-like
        Left-mover storm motion (only for ``output_var='all'``).
    mean_6km_u, mean_6km_v : array-like
        Mean 0-6 km wind, not pressure weighted (only for ``output_var='all'``).

    Notes
    -----
    .. math:: SRH = \int_{0}^{h} (\bar v - \bar c) \cdot \bar\omega_{h} \, dz

    * :math:`\bar v` environmental wind vector
    * :math:`\bar c` storm motion vector
    * :math:`\bar\omega_{h}` horizontal vorticity vector

    For further details see Markowski and Richardson (2010), pages 230-231 [2]_.

    Examples
    --------
    >>> srh_rm, srh_lm = calc_srh(p, t, td, u, v, ps, ts, tds, us, vs,
    ...                           depth=1000, output_var='srh',
    ...                           vertical_lev='pressure')

    >>> (srh_rm, srh_lm, rm_u, rm_v, lm_u, lm_v,
    ...  mean_6km_u, mean_6km_v) = calc_srh(p, t, td, u, v, ps, ts, tds, us, vs,
    ...                                     depth=3000, output_var='all',
    ...                                     vertical_lev='pressure')

    References
    ----------
    .. [1] Bunkers, M. J., B. A. Klimowski, J. W. Zeitler, R. L. Thompson, and
       M. L. Weisman, 2000: Predicting supercell motion using a new hodograph
       technique. Wea. Forecasting, 15, 61-79.
    .. [2] Markowski, P., & Richardson, Y. (2010). Mesoscale meteorology in
       midlatitudes. John Wiley & Sons.
    """
    if len(args) < 10:
        raise ValueError("Too few arguments.")
    if len(args) > 10:
        raise ValueError("Too many arguments.")

    _check_option("vertical_lev", vertical_lev, _VERTICAL_LEV_OPTIONS)
    _check_option("output_var", output_var, _OUTPUT_VAR_OPTIONS)
    _check_option("method", method, _METHODS)
    method = _resolve_method(method)

    kwargs = dict(depth=depth, vertical_lev=vertical_lev, output_var=output_var,
                  method=method)
    if _any_dask_array(*args):
        return _calc_srh_gufunc(*args, **kwargs)
    return _calc_srh_numpy(*args, **kwargs)


def _calc_srh_gufunc(*args, **kwargs):
    """Lazy, chunk-wise SRH for dask arrays (``dask.array.apply_gufunc``)."""
    signature = "(i),(i),(i),(i),(i),(),(),(),(),()->(),()"
    # dtypes actually produced by the kernels: the Fortran Bunkers kernel is
    # single precision, everything else is double precision.
    motion_dtype = "f4" if kwargs["method"] == "fortran" else "f8"
    output_dtypes = ("f8", "f8")
    if kwargs["output_var"] == "all":
        signature += ",(),(),(),(),(),()"
        output_dtypes = output_dtypes + (motion_dtype,) * 6

    return da.apply_gufunc(_calc_srh_numpy, signature, *args,
                           output_dtypes=output_dtypes,
                           axis=-1,
                           vectorize=False,
                           allow_rechunk=True,
                           _parallel=False,
                           **kwargs)


def _calc_srh_numpy(*args, depth=3000, vertical_lev="sigma", output_var="srh",
                    method="fortran", _parallel=True):
    """Eager SRH on numpy arrays: heights -> Bunkers motion -> helicity."""
    p, t, td, u, v, ps, ts, tds, us, vs = (np.asarray(a) for a in args)
    original_shape = t.shape  # shape of 3D variable

    # after this, all 3-D arrays have shape (nlevs, npoints)
    (p_2d, (t_2d, td_2d, u_2d, v_2d), (p_s1d, t_s1d, td_s1d, u_s1d, v_s1d),
     flag_1d, pres_lev_pos) = _collapse_thermo(p, t, td, ps, ts, tds, vertical_lev,
                                               extra_3d=(u, v), extra_sfc=(us, vs))

    if method == "auto":
        method = _resolve_method(method)
    type_grid = _VERTICAL_LEV_OPTIONS[vertical_lev]
    output = _OUTPUT_VAR_OPTIONS[output_var]

    # height above ground from the hypsometric equation
    aglh_2d, aglh_s1d = _stdheight(p_2d, t_2d, td_2d, p_s1d, t_s1d, td_s1d,
                                   flag_1d, pres_lev_pos, aglh0=2.0,
                                   type_grid=type_grid, method=method,
                                   parallel=_parallel)

    result = _srh(u_2d, v_2d, aglh_2d, u_s1d, v_s1d, aglh_s1d, pres_lev_pos, depth,
                  type_grid=type_grid, output=output, method=method,
                  parallel=_parallel)

    if output == 1:
        srh_2d_rm, srh_2d_lm = result
        srh_rm, srh_lm = _reshape_outputs(srh_2d_rm, srh_2d_lm, shape=original_shape)
        return srh_rm, srh_lm

    srh_2d_rm, srh_2d_lm, rm_2d, lm_2d, mean_6km_2d = result
    srh_rm, srh_lm = _reshape_outputs(srh_2d_rm, srh_2d_lm, shape=original_shape)
    rm, lm, mean_6km = _reshape_outputs_uv_components(rm_2d, lm_2d, mean_6km_2d,
                                                      shape=original_shape)
    return srh_rm, srh_lm, rm[0], rm[1], lm[0], lm[1], mean_6km[0], mean_6km[1]
