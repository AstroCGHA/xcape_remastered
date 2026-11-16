"""Shared fixtures: a small synthetic ERA5-like pair of datasets."""
import numpy as np
import pytest
import xarray as xr

G = 9.80665
LEVELS = np.array([1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600,
                   550, 500, 450, 400, 350, 300, 250, 225, 200, 175, 150, 125, 100, 70, 50],
                  dtype="f8")


def make_era5(times=("2026-08-08T14:00",), nlat=7, nlon=9, north=20.0, west=118.0, seed=0):
    """Synthetic fields with the names, units and layout of CDS ERA5 NetCDF files."""
    rng = np.random.default_rng(seed)
    time = np.array(times, dtype="datetime64[ns]")
    lat = north - 0.25 * np.arange(nlat)            # descending, like ERA5
    lon = west + 0.25 * np.arange(nlon)
    nt, nk = len(time), LEVELS.size
    shape2, shape3 = (nt, nlat, nlon), (nt, nk, nlat, nlon)

    # terrain: surface pressure between ~1008 and ~840 hPa
    sp = 100.0 * (1008.0 - 170.0 * rng.random(shape2) ** 3)
    z_of_p = lambda p_hpa: 44330.8 * (1.0 - (p_hpa / 1013.25) ** 0.190263)   # m
    zsfc = z_of_p(sp / 100.0)

    zl = z_of_p(LEVELS)[None, :, None, None] + 20.0 * rng.standard_normal(shape2)[:, None]
    t2m = 300.0 - 0.0065 * zsfc + rng.standard_normal(shape2)
    t = (300.0 - 0.0065 * zl + rng.standard_normal(shape3) * 0.3).clip(200.0, None)
    rh = np.clip(0.85 - 0.6 * (zl / 12000.0), 0.05, 0.95)
    es = 6.112 * np.exp(17.67 * (t - 273.15) / (t - 29.65))               # hPa
    e = rh * es
    q = 0.62197 * e / (LEVELS[None, :, None, None] - 0.378 * e)
    d2m = t2m - 2.0 - 3.0 * rng.random(shape2)
    # veering, sheared wind profile
    ang = np.deg2rad(120.0 + 100.0 * np.clip(zl / 6000.0, 0, 1.5)) + rng.standard_normal(shape2)[:, None] * 0.3
    spd = 4.0 + 18.0 * np.clip(zl / 9000.0, 0, 1.2) + rng.random(shape2)[:, None] * 3
    u, v = -spd * np.sin(ang), -spd * np.cos(ang)
    u10 = 0.6 * u[:, 0] + rng.standard_normal(shape2)
    v10 = 0.6 * v[:, 0] + rng.standard_normal(shape2)

    coords = {"valid_time": time, "latitude": lat, "longitude": lon}
    d3 = ("valid_time", "pressure_level", "latitude", "longitude")
    d2 = ("valid_time", "latitude", "longitude")
    f4 = lambda a: a.astype("f4")
    ds_pl = xr.Dataset(
        {"u": (d3, f4(u)), "v": (d3, f4(v)), "t": (d3, f4(t)), "q": (d3, f4(q)),
         "z": (d3, f4(zl * G))},
        coords={**coords, "pressure_level": LEVELS, "number": 0, "expver": "0001"})
    ds_sfc = xr.Dataset(
        {"u10": (d2, f4(u10)), "v10": (d2, f4(v10)), "t2m": (d2, f4(t2m)),
         "d2m": (d2, f4(d2m)), "sp": (d2, f4(sp)), "z": (d2, f4(zsfc * G))},
        coords={**coords, "number": 0, "expver": "0001"})
    return ds_pl, ds_sfc


@pytest.fixture(scope="session")
def era5_pair():
    return make_era5()


def era5_to_arrays(ds_pl, ds_sfc, gravity=G):
    """The 'combine surface and pressure levels' step, written out by hand."""
    pl = ds_pl.isel(valid_time=0).sortby("pressure_level", ascending=False)
    sf = ds_sfc.isel(valid_time=0)
    p = pl.pressure_level.values.astype("f8")
    lev_last = lambda a: np.moveaxis(a.values.astype("f8"), 0, -1)
    q = lev_last(pl.q)
    r = q / (1.0 - q)
    e = r / (0.62197 + r) * p
    td = 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))
    return dict(
        p=p, t=lev_last(pl.t) - 273.15, td=td, u=lev_last(pl.u), v=lev_last(pl.v),
        z_agl=lev_last(pl.z) / gravity - (sf.z.values.astype("f8") / gravity)[..., None],
        ps=sf.sp.values.astype("f8") / 100.0, ts=sf.t2m.values.astype("f8") - 273.15,
        tds=sf.d2m.values.astype("f8") - 273.15,
        us=sf.u10.values.astype("f8"), vs=sf.v10.values.astype("f8"))
