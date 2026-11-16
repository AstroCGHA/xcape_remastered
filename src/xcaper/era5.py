"""ERA5 workflow: query a bounding box, compute severe-weather parameters.

Typical use::

    from xcaper import era5

    ds = era5.era5_severe_parameters(
        "2026-08-08T14:00",
        north=30, west=100, south=-10, east=145,
        storm_motion="bunkers-shallow",
        output="ERA5_Severe_Params_20260808_14.nc",
    )

which

1. downloads the ERA5 pressure-level and single-level fields for the box
   from the Copernicus Climate Data Store (files are cached and reused),
2. combines them into columns: 2 m temperature / dewpoint and surface
   pressure become the lowest thermodynamic level, the 10 m wind becomes
   the wind at 0 m above ground, heights above ground come from the
   geopotential, and pressure levels below the ground are discarded,
3. computes every parameter of :func:`xcaper.severe.calc_severe`,
4. writes a compressed, CF-style NetCDF file.

Files that are already on disk can be processed without any download with
:func:`open_files` + :func:`compute_severe_parameters`; the same workflow is
available from the shell as ``xcaper-era5`` (see ``xcaper-era5 --help``).

One-time CDS set-up
-------------------
1. ``pip install "cdsapi>=0.7.2"``
2. Register at https://cds.climate.copernicus.eu and accept the licences of
   ``reanalysis-era5-single-levels`` and ``reanalysis-era5-pressure-levels``
   (open the "Download" tab of both datasets once).
3. Create ``~/.cdsapirc``::

       url: https://cds.climate.copernicus.eu/api
       key: <YOUR-PERSONAL-ACCESS-TOKEN>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ._version import __version__
from .thermo import dewpoint_from_specific_humidity
from .xarray import severe_parameters

__all__ = [
    "PRESSURE_LEVELS",
    "time_range",
    "build_requests",
    "download",
    "open_files",
    "subset_bbox",
    "compute_severe_parameters",
    "save",
    "era5_severe_parameters",
    "main",
]

SINGLE_LEVEL_DATASET = "reanalysis-era5-single-levels"
PRESSURE_LEVEL_DATASET = "reanalysis-era5-pressure-levels"

#: the 37 ERA5 pressure levels (hPa)
PRESSURE_LEVELS = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70,
    100, 125, 150, 175, 200, 225, 250, 300, 350, 400,
    450, 500, 550, 600, 650, 700, 750, 775, 800, 825,
    850, 875, 900, 925, 950, 975, 1000,
)

SURFACE_VARIABLES = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "geopotential",
)

PRESSURE_VARIABLES = (
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    "geopotential",
)

#: standard gravity used by ECMWF to convert geopotential to height (m s-2)
STANDARD_GRAVITY = 9.80665

# names under which the fields appear, depending on CDS version / decoder
_NAMES = {
    "time": ("valid_time", "time"),
    "lat": ("latitude", "lat"),
    "lon": ("longitude", "lon"),
    "level": ("pressure_level", "level", "isobaricInhPa", "plev"),
    "u": ("u",), "v": ("v",), "t": ("t",), "q": ("q",), "z": ("z",),
    "u10": ("u10", "10u"), "v10": ("v10", "10v"),
    "t2m": ("t2m", "2t"), "d2m": ("d2m", "2d"),
    "sp": ("sp",), "zsfc": ("z", "geopotential"),
}


# ---------------------------------------------------------------------------
# time handling
# ---------------------------------------------------------------------------
def _normalize_times(times):
    """Sorted, unique, whole-hour UTC timestamps from one time or a sequence."""
    if isinstance(times, (str, _dt.datetime, np.datetime64, pd.Timestamp)):
        times = [times]
    stamps = pd.DatetimeIndex([pd.Timestamp(t) for t in times])
    if stamps.tz is not None:
        stamps = stamps.tz_convert("UTC").tz_localize(None)
    if len(stamps) == 0:
        raise ValueError("no time given")
    if ((stamps.minute != 0) | (stamps.second != 0)).any():
        raise ValueError("ERA5 is hourly: times must be on the hour (UTC)")
    return list(stamps.unique().sort_values())


def time_range(start, end, step_hours=1):
    """Hourly (or ``step_hours``-ly) UTC times from ``start`` to ``end`` inclusive."""
    if step_hours < 1 or int(step_hours) != step_hours:
        raise ValueError("step_hours must be a positive whole number")
    return list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=f"{int(step_hours)}h"))


def _check_area(north, west, south, east):
    north, west, south, east = (float(x) for x in (north, west, south, east))
    if not -90.0 <= south < north <= 90.0:
        raise ValueError(f"need -90 <= south < north <= 90; got south={south}, north={north}")
    if not west < east:
        raise ValueError(f"need west < east; got west={west}, east={east}")
    if east - west > 360.0:
        raise ValueError("the box is wider than 360 degrees of longitude")
    return north, west, south, east


# ---------------------------------------------------------------------------
# CDS requests and download
# ---------------------------------------------------------------------------
def _hours_tag(hours):
    if len(hours) == 24:
        return "all"
    return "h" + "".join(f"{h:02d}" for h in hours)


def build_requests(times, *, north, west, south, east, grid=0.25, pressure_levels=None):
    """CDS API requests (one pair per calendar day) for a box and a set of times.

    Parameters
    ----------
    times : time-like or sequence of time-like
        UTC times on the hour (strings, ``datetime``, ``numpy.datetime64``).
    north, west, south, east : float
        Bounding box in degrees.  Longitudes may be given as -180..180 or
        0..360.
    grid : float or (float, float)
        Grid spacing in degrees (0.25 is the native ERA5 resolution).
    pressure_levels : sequence of int, optional
        Levels in hPa; the 37 ERA5 levels by default.

    Returns
    -------
    list of dict
        One entry per day with keys ``date``, ``hours``, ``pressure``,
        ``surface`` (each ``{'dataset', 'request', 'filename'}``).
    """
    north, west, south, east = _check_area(north, west, south, east)
    stamps = _normalize_times(times)
    grid = [float(grid), float(grid)] if np.isscalar(grid) else [float(g) for g in grid]
    levels = PRESSURE_LEVELS if pressure_levels is None else pressure_levels
    levels = [str(int(lv)) for lv in sorted(int(x) for x in levels)]

    by_day = {}
    for ts in stamps:
        by_day.setdefault(ts.date(), []).append(ts.hour)

    requests = []
    for date, hours in sorted(by_day.items()):
        common = {
            "product_type": ["reanalysis"],
            "year": [f"{date.year:04d}"],
            "month": [f"{date.month:02d}"],
            "day": [f"{date.day:02d}"],
            "time": [f"{h:02d}:00" for h in hours],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "grid": grid,
            "area": [north, west, south, east],
        }
        if len(hours) == 1:
            stem = f"{date:%Y%m%d}{hours[0]:02d}"
        else:
            stem = f"{date:%Y%m%d}_{_hours_tag(hours)}"
        requests.append({
            "date": date,
            "hours": hours,
            "pressure": {
                "dataset": PRESSURE_LEVEL_DATASET,
                "request": dict(common, variable=list(PRESSURE_VARIABLES), pressure_level=levels),
                "filename": f"era5_pl_{stem}.nc",
            },
            "surface": {
                "dataset": SINGLE_LEVEL_DATASET,
                "request": dict(common, variable=list(SURFACE_VARIABLES)),
                "filename": f"era5_sfc_{stem}.nc",
            },
        })
    return requests


def _cached_file_problem(path, request):
    """Why a cached file does not satisfy ``request`` (None when it does)."""
    try:
        with _open_one(path) as ds:
            lat = ds[_find(ds, "lat")].values
            lon = ds[_find(ds, "lon")].values
            times = pd.DatetimeIndex(np.atleast_1d(ds[_find(ds, "time")].values))
    except Exception as err:  # unreadable / truncated file
        return f"cannot be read ({err})"
    north, west, south, east = request["area"]
    tol = max(request["grid"]) * 0.51
    if (abs(lat.max() - north) > tol or abs(lat.min() - south) > tol
            or abs(lon.min() - west) > tol or abs(lon.max() - east) > tol):
        return (f"covers lat {lat.min():g}..{lat.max():g}, lon {lon.min():g}..{lon.max():g} "
                f"instead of the requested lat {south:g}..{north:g}, lon {west:g}..{east:g}")
    date = f"{request['year'][0]}-{request['month'][0]}-{request['day'][0]}"
    wanted = pd.DatetimeIndex([f"{date}T{t}" for t in request["time"]])
    if not wanted.isin(times).all():
        return "does not contain all requested times"
    return None


def download(times, *, north, west, south, east, grid=0.25, pressure_levels=None,
             cache_dir=".", client=None, overwrite=False, verbose=True):
    """Download ERA5 fields for a bounding box from the Climate Data Store.

    Files already present in ``cache_dir`` are reused after checking that
    they cover the requested box and times.  A single hour is stored as
    ``era5_pl_YYYYMMDDHH.nc`` / ``era5_sfc_YYYYMMDDHH.nc``.

    Parameters
    ----------
    times, north, west, south, east, grid, pressure_levels
        See :func:`build_requests`.
    cache_dir : path-like
        Directory of the downloaded files.
    client : object, optional
        Anything with a ``retrieve(dataset, request, target)`` method; a
        ``cdsapi.Client()`` is created when omitted.
    overwrite : bool
        Download again even when a cached file exists.

    Returns
    -------
    pl_paths, sfc_paths : list of pathlib.Path
        One pressure-level and one single-level file per calendar day.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    requests = build_requests(times, north=north, west=west, south=south, east=east,
                              grid=grid, pressure_levels=pressure_levels)
    say = print if verbose else (lambda *a, **k: None)

    pl_paths, sfc_paths = [], []
    for req in requests:
        for kind, paths in (("pressure", pl_paths), ("surface", sfc_paths)):
            item = req[kind]
            target = cache_dir / item["filename"]
            if target.exists() and not overwrite:
                problem = _cached_file_problem(target, item["request"])
                if problem is not None:
                    raise ValueError(
                        f"The cached file {target} {problem}. Use another cache_dir, "
                        "delete the file, or pass overwrite=True."
                    )
                say(f"Reusing cached file -> {target}")
            else:
                if client is None:
                    try:
                        import cdsapi
                    except ImportError as err:
                        raise ImportError(
                            "Downloading ERA5 needs the 'cdsapi' package: "
                            "pip install 'cdsapi>=0.7.2' (and a ~/.cdsapirc file)."
                        ) from err
                    client = cdsapi.Client()
                say(f"Downloading -> {target}")
                partial = target.with_name(target.name + ".part")
                client.retrieve(item["dataset"], item["request"], str(partial))
                os.replace(partial, target)
            paths.append(target)
    return pl_paths, sfc_paths


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def _find(ds, key):
    for name in _NAMES[key]:
        if name in ds.variables:
            return name
    raise KeyError(f"None of {_NAMES[key]} found for '{key}'. Available: {list(ds.variables)}")


def _open_one(path, chunks=None):
    """Open one CDS file (plain NetCDF or a zip of NetCDF files)."""
    path = Path(path)
    if zipfile.is_zipfile(path):
        outdir = path.with_name(path.name + ".unzipped")
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.endswith(".nc")]
            zf.extractall(outdir, members)
        parts = [xr.open_dataset(outdir / n, chunks=chunks) for n in members]
        return xr.merge(parts, compat="override", join="exact")
    return xr.open_dataset(path, chunks=chunks)


def _standardize(ds):
    """Drop ensemble / experiment scalars and make sure a time dimension exists."""
    ds = ds.drop_vars([n for n in ("number", "expver") if n in ds.variables])
    tname = _find(ds, "time")
    if tname not in ds.dims:
        ds = ds.expand_dims(tname)
    return ds


def open_files(pl_paths, sfc_paths, chunks=None):
    """Open ERA5 pressure-level and single-level files.

    Parameters
    ----------
    pl_paths, sfc_paths : path-like or sequence of path-like
        Files as delivered by the CDS (NetCDF or zip).  Several files are
        concatenated along time.
    chunks : dict, optional
        Dask chunks, e.g. ``{'latitude': 40, 'longitude': 40}``.  Leave as
        None to read numpy arrays.

    Returns
    -------
    ds_pl, ds_sfc : xarray.Dataset
    """
    def _open_many(paths):
        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]
        parts = [_standardize(_open_one(p, chunks=chunks)) for p in paths]
        if len(parts) == 1:
            return parts[0]
        tname = _find(parts[0], "time")
        return xr.concat(parts, dim=tname).sortby(tname)

    return _open_many(pl_paths), _open_many(sfc_paths)


def subset_bbox(ds, *, north, west, south, east):
    """Select a latitude / longitude box from an ERA5 dataset.

    Works for ascending or descending coordinates and for longitudes stored
    as 0..360 or -180..180.  Boxes that cross the seam of the longitude
    axis of the file are not supported.
    """
    north, west, south, east = _check_area(north, west, south, east)
    lat, lon = _find(ds, "lat"), _find(ds, "lon")
    lons = ds[lon].values
    if lons.max() > 180.0 and west < 0.0:      # file is 0..360, box is -180..180
        west, east = west % 360.0, east % 360.0
    elif lons.min() < 0.0 and east > 180.0:    # file is -180..180, box is 0..360
        west, east = (west + 180.0) % 360.0 - 180.0, (east + 180.0) % 360.0 - 180.0
    if west > east:
        raise ValueError("the box crosses the longitude seam of the file; split it in two")
    lats = ds[lat].values
    lat_mask = (lats >= south - 1e-9) & (lats <= north + 1e-9)
    lon_mask = (lons >= west - 1e-9) & (lons <= east + 1e-9)
    if not lat_mask.any() or not lon_mask.any():
        raise ValueError("the bounding box does not overlap the data")
    return ds.isel({lat: np.flatnonzero(lat_mask), lon: np.flatnonzero(lon_mask)})


# ---------------------------------------------------------------------------
# computation
# ---------------------------------------------------------------------------
def _auto_chunks(ds):
    return {_find(ds, "time"): 1, _find(ds, "lat"): 40, _find(ds, "lon"): 40}


def compute_severe_parameters(ds_pl, ds_sfc, *, north=None, west=None, south=None,
                              east=None, chunks=None, gravity=STANDARD_GRAVITY,
                              rename_dims=True, **options):
    """
    Severe-weather parameters from ERA5 pressure-level + single-level data.

    Parameters
    ----------
    ds_pl : xarray.Dataset
        Pressure-level fields ``u, v, t, q, z`` (m/s, K, kg/kg, m2/s2).
    ds_sfc : xarray.Dataset
        Single-level fields ``u10, v10, t2m, d2m, sp, z`` (m/s, K, Pa,
        m2/s2), on the same grid and times.
    north, west, south, east : float, optional
        Bounding box to cut out before computing (all four or none).
    chunks : dict or 'auto', optional
        Rechunk with dask before computing; the result is then lazy and is
        evaluated in parallel (on ``.compute()``, ``.load()`` or when
        saved).  'auto' uses one time step and 40 x 40 grid points per
        chunk.  None keeps the data as they are.
    gravity : float
        Gravity (m/s2) used to turn geopotential into height.
    rename_dims : bool
        Name the output dimensions ``time, lat, lon``.
    **options
        Passed to :func:`xcaper.severe.calc_severe`: ``storm_motion``
        ('bunkers', 'bunkers-shallow' or a :class:`~xcaper.kinematics.StormMotion`),
        ``srh_layers``, ``shear_layers``, ``swv_layers``, ``dz``,
        ``ml_depth``, ``pinc``, ``adiabat``, ``method``.

    Returns
    -------
    xarray.Dataset
        See :func:`xcaper.severe.calc_severe` for the variables.
    """
    box = (north, west, south, east)
    if any(x is not None for x in box):
        if any(x is None for x in box):
            raise ValueError("give all of north, west, south, east or none of them")
        ds_pl = subset_bbox(ds_pl, north=north, west=west, south=south, east=east)
        ds_sfc = subset_bbox(ds_sfc, north=north, west=west, south=south, east=east)

    ds_pl, ds_sfc = _standardize(ds_pl), _standardize(ds_sfc)
    lev = _find(ds_pl, "level")
    tname, lat, lon = (_find(ds_sfc, k) for k in ("time", "lat", "lon"))
    for name in (tname, lat, lon):
        if name not in ds_pl.dims:
            raise ValueError(f"dimension {name!r} of the surface data is missing in the "
                             "pressure-level data")
        if not np.array_equal(ds_pl[name].values, ds_sfc[name].values):
            raise ValueError(f"pressure-level and surface data differ along {name!r}; "
                             "they must share grid and times")

    # bottom-to-top levels, in hPa
    ds_pl = ds_pl.sortby(lev, ascending=False)
    p = ds_pl[lev].astype("f8")
    if float(p.max()) > 2000.0:
        p = p / 100.0
    if float(p.min()) > 500.0:
        raise ValueError("the pressure levels must reach at least up to 500 hPa")

    if chunks == "auto":
        chunks = _auto_chunks(ds_sfc)
    if chunks is not None:
        ds_sfc = ds_sfc.chunk({k: v for k, v in chunks.items() if k in ds_sfc.dims})
        ds_pl = ds_pl.chunk({**{k: v for k, v in chunks.items() if k in ds_pl.dims}, lev: -1})

    def f8(da_):
        return da_.astype("f8")

    # ---- combine surface and pressure-level data into xcaper inputs ----
    t = f8(ds_pl[_find(ds_pl, "t")]) - 273.15                       # C
    td = dewpoint_from_specific_humidity(f8(ds_pl[_find(ds_pl, "q")]), p) - 273.15   # C
    u = f8(ds_pl[_find(ds_pl, "u")])                                # m/s
    v = f8(ds_pl[_find(ds_pl, "v")])
    zsfc = f8(ds_sfc[_find(ds_sfc, "zsfc")]) / gravity              # m MSL
    z_agl = f8(ds_pl[_find(ds_pl, "z")]) / gravity - zsfc           # m AGL

    ps = f8(ds_sfc[_find(ds_sfc, "sp")]) * 0.01                     # hPa
    ts = f8(ds_sfc[_find(ds_sfc, "t2m")]) - 273.15                  # C
    tds = f8(ds_sfc[_find(ds_sfc, "d2m")]) - 273.15                 # C
    us = f8(ds_sfc[_find(ds_sfc, "u10")])                           # m/s
    vs = f8(ds_sfc[_find(ds_sfc, "v10")])

    out = severe_parameters(p, t, td, u, v, z_agl, ps, ts, tds, us, vs,
                            vertical_dim=lev, vertical_lev="pressure", **options)

    if rename_dims:
        out = out.rename({tname: "time", lat: "lat", lon: "lon"})
        out["lat"].attrs.update(units="degrees_north", standard_name="latitude", axis="Y")
        out["lon"].attrs.update(units="degrees_east", standard_name="longitude", axis="X")
        out["time"].attrs.update(standard_name="time", axis="T")
    out.attrs.update({
        "title": "ERA5 severe-weather parameters",
        "source": "ERA5 reanalysis (Copernicus Climate Change Service), pressure levels "
                  "+ single levels",
        "Conventions": "CF-1.8",
        "history": f"{_dt.datetime.now(_dt.timezone.utc):%Y-%m-%dT%H:%MZ} "
                   f"created by xcaper {__version__}",
        "height_above_ground": f"(geopotential - surface geopotential) / {gravity:g} m s-2",
    })
    return out


def save(ds, path, complevel=4):
    """Write the parameters to a compressed NetCDF-4 file (float32)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": complevel, "dtype": "float32",
                       "_FillValue": np.float32(np.nan)}
                for name in ds.data_vars}
    tname = "time" if "time" in ds.coords else None
    if tname:
        encoding[tname] = {"units": "hours since 1900-01-01 00:00:00", "calendar": "gregorian",
                           "dtype": "float64"}
    ds.to_netcdf(path, format="NETCDF4", encoding=encoding)
    return path


def era5_severe_parameters(times, *, north, west, south, east, grid=0.25,
                           pressure_levels=None, cache_dir=".", output=None,
                           client=None, overwrite=False, chunks="auto", verbose=True,
                           **options):
    """
    Query ERA5 for a bounding box and compute the severe-weather parameters.

    Parameters
    ----------
    times : time-like or sequence of time-like
        UTC time(s) on the hour, e.g. ``"2026-08-08T14:00"`` or the result of
        :func:`time_range`.
    north, west, south, east : float
        Bounding box in degrees.
    grid : float
        Grid spacing in degrees.
    pressure_levels : sequence of int, optional
        Levels in hPa (all 37 by default).
    cache_dir : path-like
        Where the ERA5 downloads are kept and reused.
    output : path-like, optional
        NetCDF file to write.
    client : object, optional
        CDS client (``cdsapi.Client()`` by default).
    overwrite : bool
        Force a new download.
    chunks : dict, 'auto' or None
        Dask chunking of the computation (see
        :func:`compute_severe_parameters`); None computes eagerly.
    **options
        ``storm_motion``, ``srh_layers``, ``shear_layers``, ``swv_layers``,
        ``dz``, ``ml_depth``, ``pinc``, ``adiabat``, ``method``, ``gravity``.

    Returns
    -------
    xarray.Dataset
        The parameters, loaded in memory, on dimensions ``time, lat, lon``.
    """
    pl_paths, sfc_paths = download(times, north=north, west=west, south=south, east=east,
                                   grid=grid, pressure_levels=pressure_levels,
                                   cache_dir=cache_dir, client=client, overwrite=overwrite,
                                   verbose=verbose)
    ds_pl, ds_sfc = open_files(pl_paths, sfc_paths)
    wanted = _normalize_times(times)
    tname = _find(ds_sfc, "time")
    ds_pl = ds_pl.sel({_find(ds_pl, "time"): wanted})
    ds_sfc = ds_sfc.sel({tname: wanted})
    result = compute_severe_parameters(ds_pl, ds_sfc, chunks=chunks, **options).load()
    if output is not None:
        save(result, output)
        if verbose:
            print(f"Successfully saved: {output}")
    return result


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def _parser():
    p = argparse.ArgumentParser(
        prog="xcaper-era5",
        description="Download ERA5 for a bounding box (or read local ERA5 files) and "
                    "compute severe-weather parameters with xcaper.",
    )
    p.add_argument("--area", nargs=4, type=float, metavar=("NORTH", "WEST", "SOUTH", "EAST"),
                   help="bounding box in degrees (required for downloads)")
    p.add_argument("--time", action="append", default=[], metavar="YYYY-MM-DDTHH",
                   help="UTC time on the hour; may be repeated")
    p.add_argument("--start", metavar="YYYY-MM-DDTHH", help="first time of a range")
    p.add_argument("--end", metavar="YYYY-MM-DDTHH", help="last time of a range")
    p.add_argument("--step", type=int, default=1, help="hours between times of a range")
    p.add_argument("--grid", type=float, default=0.25, help="grid spacing in degrees")
    p.add_argument("--cache-dir", default=".", help="directory of the ERA5 downloads")
    p.add_argument("--pl-file", action="append", default=[],
                   help="local pressure-level file (skips the download); may be repeated")
    p.add_argument("--sfc-file", action="append", default=[],
                   help="local single-level file (skips the download); may be repeated")
    p.add_argument("-o", "--output", help="output NetCDF file")
    p.add_argument("--storm-motion", default="bunkers",
                   choices=["bunkers", "bunkers-shallow", "bunkers-effective"],
                   help="standard 0-6 km Bunkers motion, the shallow 0-3 km variant, or the "
                        "effective-layer method of Bunkers et al. (2014)")
    p.add_argument("--eil-dp", type=float, default=10.0, metavar="HPA",
                   help="pressure spacing of the refined profile on which the effective inflow "
                        "layer is searched (default 10); 0 = input levels only")
    p.add_argument("--temperature-lapse-rates", action="store_true",
                   help="lapse rates of the temperature instead of the virtual temperature")
    p.add_argument("--no-effective", action="store_true",
                   help="skip LFC, EL, DCAPE and the effective-layer parameters (faster)")
    p.add_argument("--ml-depth", type=float, default=500.0, help="mixed-layer depth (m)")
    p.add_argument("--pinc", type=float, default=500.0, help="parcel pressure increment (Pa)")
    p.add_argument("--adiabat", default="pseudo-liquid",
                   choices=["pseudo-liquid", "reversible-liquid", "pseudo-ice", "reversible-ice"])
    p.add_argument("--method", default="auto", choices=["auto", "fortran", "numba"],
                   help="CAPE backend")
    p.add_argument("--no-dask", action="store_true",
                   help="compute eagerly instead of in parallel dask chunks")
    p.add_argument("--overwrite", action="store_true", help="download again")
    p.add_argument("--version", action="version", version=f"xcaper {__version__}")
    return p


def main(argv=None):
    """Entry point of the ``xcaper-era5`` command."""
    args = _parser().parse_args(argv)
    options = dict(storm_motion=args.storm_motion, ml_depth=args.ml_depth, pinc=args.pinc,
                   adiabat=args.adiabat, method=args.method, effective=not args.no_effective,
                   eil_dp=args.eil_dp or None,
                   virtual_lapse_rate=not args.temperature_lapse_rates)
    chunks = None if args.no_dask else "auto"
    area = {}
    if args.area:
        area = dict(zip(("north", "west", "south", "east"), args.area))

    times = list(args.time)
    if args.start or args.end:
        if not (args.start and args.end):
            sys.exit("xcaper-era5: --start and --end must be given together")
        times += time_range(args.start, args.end, args.step)

    if args.pl_file or args.sfc_file:
        if not (args.pl_file and args.sfc_file):
            sys.exit("xcaper-era5: --pl-file and --sfc-file must be given together")
        ds_pl, ds_sfc = open_files(args.pl_file, args.sfc_file)
        if times:
            wanted = _normalize_times(times)
            ds_pl = ds_pl.sel({_find(ds_pl, "time"): wanted})
            ds_sfc = ds_sfc.sel({_find(ds_sfc, "time"): wanted})
        result = compute_severe_parameters(ds_pl, ds_sfc, chunks=chunks, **area, **options).load()
    else:
        if not args.area or not times:
            sys.exit("xcaper-era5: give --area and --time (or --start/--end), "
                     "or local files with --pl-file/--sfc-file")
        result = era5_severe_parameters(times, grid=args.grid, cache_dir=args.cache_dir,
                                        overwrite=args.overwrite, chunks=chunks,
                                        **area, **options)

    stamps = pd.DatetimeIndex(result["time"].values)
    output = args.output
    if output is None:
        tag = f"{stamps[0]:%Y%m%d_%H}"
        if len(stamps) > 1:
            tag += f"-{stamps[-1]:%Y%m%d_%H}"
        output = f"ERA5_Severe_Params_{tag}.nc"
    save(result, output)
    print(f"Successfully saved: {output}  ({len(stamps)} time(s), "
          f"{result.sizes['lat']} x {result.sizes['lon']} grid points, "
          f"{len(result.data_vars)} variables)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
