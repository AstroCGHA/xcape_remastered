"""ERA5 workflow, xarray API and the all-in-one severe-parameter driver."""
import dask.array as dsa
import numpy as np
import pytest
import xarray as xr

import xcaper
from xcaper import era5
from xcaper import xarray as xx
from xcaper.core import calc_cape
from xcaper.severe import calc_severe, variable_attrs

from .conftest import LEVELS, era5_to_arrays, make_era5

EXPECTED = {"SFC_CAPE", "SFC_CIN", "ML_CAPE", "ML_CIN", "MU_CAPE", "MU_CIN", "SFC_LCL", "ML_LCL",
            "LR03", "LR36", "LR75", "MEAN06_U", "MEAN06_V", "BUNKERS_RM_U", "BUNKERS_RM_V",
            "BUNKERS_LM_U", "BUNKERS_LM_V", "CRITICAL_ANGLE", "SRH500M_RM", "SRH500M_LM",
            "SRH1_RM", "SRH1_LM", "SRH3_RM", "SRH3_LM", "SRH500M", "SRH1", "SRH3",
            "S01", "S03", "S06", "SWV01", "SWNESS01", "SCP", "STP", "STP_CIN", "EHI1", "EHI3",
            "WMAXSHEAR"}
EFFECTIVE = {"SFC_LFC", "SFC_LFC_P", "SFC_EL", "SFC_EL_P", "ML_LFC", "ML_LFC_P", "ML_EL", "ML_EL_P",
             "MU_LFC", "MU_LFC_P", "MU_EL", "MU_EL_P", "EIL_BASE", "EIL_BASE_P", "EIL_TOP",
             "EIL_TOP_P", "DCAPE", "DCAPE_SRC_P", "ESRH_RM", "ESRH_LM", "ESRH", "EBWD", "E_SWV",
             "E_SWNE", "ESCP", "ESTP"}
EXPECTED = EXPECTED | EFFECTIVE


def _same(a, b):
    """Same variables, coordinates and values (attributes such as history aside)."""
    assert set(a.data_vars) == set(b.data_vars)
    xr.testing.assert_equal(a, b)


class FakeCDS:
    """Stands in for cdsapi.Client: serves slices of a synthetic ERA5 archive."""

    def __init__(self, times):
        self.pl, self.sfc = make_era5(times=times)
        self.calls = []

    def retrieve(self, dataset, request, target):
        self.calls.append((dataset, request, target))
        ds = self.pl if dataset == era5.PRESSURE_LEVEL_DATASET else self.sfc
        date = f"{request['year'][0]}-{request['month'][0]}-{request['day'][0]}"
        wanted = np.array([f"{date}T{t}" for t in request["time"]], dtype="datetime64[ns]")
        ds = ds.sel(valid_time=wanted)
        north, west, south, east = request["area"]
        ds = ds.sel(latitude=slice(north, south), longitude=slice(west, east))
        ds.to_netcdf(target)


def test_severe_on_arrays(era5_pair):
    a = era5_to_arrays(*era5_pair)
    args = [a[k] for k in ("p", "t", "td", "u", "v", "z_agl", "ps", "ts", "tds", "us", "vs")]
    out = calc_severe(*args)
    assert set(out) == EXPECTED
    for name, arr in out.items():
        assert arr.shape == a["ps"].shape, name
        assert variable_attrs(name)["units"], name
    assert (out["SFC_CAPE"] >= 0).all() and (out["SFC_CIN"] >= 0).all()
    assert (out["MU_CAPE"] >= out["SFC_CAPE"] - 1.0).all()
    assert ((out["SWNESS01"] >= 0) & (out["SWNESS01"] <= 100.0 + 1e-3)).all()
    np.testing.assert_array_equal(out["SRH1"], np.maximum(np.abs(out["SRH1_RM"]), np.abs(out["SRH1_LM"])))
    # the pieces are the documented combinations of each other
    sb, _ = calc_cape(a["p"], a["t"], a["td"], a["ps"], a["ts"], a["tds"], source="surface",
                      vertical_lev="pressure")
    np.testing.assert_array_equal(out["SFC_CAPE"], sb)
    np.testing.assert_allclose(out["EHI3"], out["ML_CAPE"] * out["SRH3_RM"] / 160000.0, rtol=1e-6)


def test_era5_pipeline_matches_manual_combination(era5_pair):
    ds_pl, ds_sfc = era5_pair
    out = era5.compute_severe_parameters(ds_pl, ds_sfc, storm_motion="bunkers-shallow")
    assert set(out.data_vars) == (EXPECTED - {"MEAN06_U", "MEAN06_V"}) | {"MEAN03_U", "MEAN03_V"}
    assert out.sizes == {"time": 1, "lat": 7, "lon": 9}
    assert out["SWV01"].attrs["units"] == "s-1" and "0-3000" in out.attrs["storm_motion"]
    a = era5_to_arrays(ds_pl, ds_sfc)
    args = [a[k] for k in ("p", "t", "td", "u", "v", "z_agl", "ps", "ts", "tds", "us", "vs")]
    ref = calc_severe(*args, storm_motion="bunkers-shallow")
    for name in ref:
        np.testing.assert_allclose(out[name].values[0], ref[name], rtol=1e-5, atol=1e-4, err_msg=name)


@pytest.mark.parametrize("method", ["auto", "numba"])
def test_era5_dask_equals_eager(method):
    ds_pl, ds_sfc = make_era5(times=("2026-08-08T12:00", "2026-08-08T13:00"), nlat=6, nlon=8)
    eager = era5.compute_severe_parameters(ds_pl, ds_sfc, method=method)
    lazy = era5.compute_severe_parameters(ds_pl, ds_sfc, method=method,
                                          chunks={"valid_time": 1, "latitude": 3, "longitude": 4})
    assert all(isinstance(v.data, dsa.Array) for v in lazy.data_vars.values())
    assert all(isinstance(v.data, np.ndarray) for v in eager.data_vars.values())
    _same(lazy.load(), eager)
    auto = era5.compute_severe_parameters(ds_pl, ds_sfc, method=method, chunks="auto")
    _same(auto.load(), eager)


def test_bbox_subset_and_input_layout(era5_pair):
    ds_pl, ds_sfc = era5_pair
    full = era5.compute_severe_parameters(ds_pl, ds_sfc)
    box = dict(north=19.75, west=118.5, south=19.0, east=119.5)
    sub = era5.compute_severe_parameters(ds_pl, ds_sfc, **box)
    assert sub.sizes == {"time": 1, "lat": 4, "lon": 5}
    _same(sub, full.sel(lat=slice(19.75, 19.0), lon=slice(118.5, 119.5)))
    # level order, 0..360 longitudes and Pa levels in the files do not matter
    shuffled = ds_pl.sortby("pressure_level").assign_coords(pressure_level=np.sort(LEVELS) * 100.0)
    again = era5.compute_severe_parameters(shuffled, ds_sfc)
    _same(again, full)
    with pytest.raises(ValueError):
        era5.compute_severe_parameters(ds_pl, ds_sfc, north=19.0)                 # incomplete box
    with pytest.raises(ValueError):
        era5.subset_bbox(ds_sfc, north=50, west=0, south=40, east=10)             # no overlap
    with pytest.raises(ValueError):
        era5.compute_severe_parameters(ds_pl, ds_sfc.isel(longitude=slice(1, None)))  # grids differ


def test_build_requests():
    reqs = era5.build_requests(["2026-08-08T14", "2026-08-09T00:00", "2026-08-09T06:00"],
                               north=30, west=100, south=-10, east=145)
    assert len(reqs) == 2
    pl, sfc = reqs[0]["pressure"], reqs[0]["surface"]
    assert pl["filename"] == "era5_pl_2026080814.nc" and sfc["filename"] == "era5_sfc_2026080814.nc"
    assert pl["dataset"] == "reanalysis-era5-pressure-levels"
    r = pl["request"]
    assert r["area"] == [30.0, 100.0, -10.0, 145.0] and r["grid"] == [0.25, 0.25]
    assert (r["year"], r["month"], r["day"], r["time"]) == (["2026"], ["08"], ["08"], ["14:00"])
    assert len(r["pressure_level"]) == 37 and r["data_format"] == "netcdf"
    assert set(sfc["request"]["variable"]) == {
        "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature",
        "2m_dewpoint_temperature", "surface_pressure", "geopotential"}
    assert reqs[1]["pressure"]["request"]["time"] == ["00:00", "06:00"]
    assert reqs[1]["pressure"]["filename"] == "era5_pl_20260809_h0006.nc"
    assert len(era5.time_range("2026-08-08T00", "2026-08-08T23")) == 24
    for bad in (dict(north=10, south=20, west=0, east=10), dict(north=20, south=10, west=10, east=0)):
        with pytest.raises(ValueError):
            era5.build_requests("2026-08-08T14", **bad)
    with pytest.raises(ValueError):
        era5.build_requests("2026-08-08T14:30", north=30, west=100, south=-10, east=145)


def test_download_compute_save_roundtrip(tmp_path):
    times = ["2026-08-08T12:00", "2026-08-08T13:00", "2026-08-09T00:00"]
    client = FakeCDS(times)
    box = dict(north=20.0, west=118.0, south=18.5, east=120.0)
    out_file = tmp_path / "out" / "params.nc"
    ds = era5.era5_severe_parameters(times, **box, cache_dir=tmp_path / "cache", client=client,
                                     output=out_file, verbose=False)
    assert len(client.calls) == 4                       # 2 days x (pressure + surface)
    assert ds.sizes == {"time": 3, "lat": 7, "lon": 9}
    assert set(ds.data_vars) == EXPECTED

    # the file on disk has the layout of the original scripts
    with xr.open_dataset(out_file) as back:
        assert back.sizes == {"time": 3, "lat": 7, "lon": 9}
        assert back["SFC_CAPE"].dtype == np.float32 and back["SFC_CAPE"].encoding["zlib"]
        assert back["time"].encoding["units"].startswith("hours since 1900-01-01")
        assert back["lat"].attrs["units"] == "degrees_north"
        np.testing.assert_allclose(back["STP"].values, ds["STP"].values, rtol=1e-6, atol=1e-7)
        assert back.attrs["Conventions"] == "CF-1.8" and "xcaper" in back.attrs["history"]

    # second run: everything comes from the cache
    again = era5.era5_severe_parameters(times, **box, cache_dir=tmp_path / "cache",
                                        client=client, verbose=False, chunks=None)
    assert len(client.calls) == 4
    xr.testing.assert_allclose(again, ds, rtol=1e-6, atol=1e-6)

    # a cached file of another region is refused instead of being silently reused
    with pytest.raises(ValueError, match="cached file"):
        era5.download(times[:2], north=20.0, west=118.0, south=19.0, east=120.0,
                      cache_dir=tmp_path / "cache", client=client, verbose=False)


def test_command_line_with_local_files(tmp_path, capsys):
    ds_pl, ds_sfc = make_era5()
    ds_pl.to_netcdf(tmp_path / "pl.nc")
    ds_sfc.to_netcdf(tmp_path / "sfc.nc")
    out = tmp_path / "cli.nc"
    rc = era5.main(["--pl-file", str(tmp_path / "pl.nc"), "--sfc-file", str(tmp_path / "sfc.nc"),
                    "--area", "19.5", "118.25", "19.0", "119.0", "--storm-motion", "bunkers-shallow",
                    "-o", str(out)])
    assert rc == 0 and "Successfully saved" in capsys.readouterr().out
    with xr.open_dataset(out) as ds:
        assert ds.sizes == {"time": 1, "lat": 3, "lon": 4}
        assert "MEAN03_U" in ds and "STP_CIN" in ds
    with pytest.raises(SystemExit):
        era5.main(["--time", "2026-08-08T14"])          # download without --area


def test_xarray_wrappers(era5_pair):
    a = era5_to_arrays(*era5_pair)
    dims3, dims2 = ("y", "x", "lev"), ("y", "x")
    p = xr.DataArray(a["p"], dims="lev")
    prof = {k: xr.DataArray(a[k], dims=dims3) for k in ("t", "td", "u", "v", "z_agl")}
    sfc = {k: xr.DataArray(a[k], dims=dims2, coords={"y": np.arange(7), "x": np.arange(9)})
           for k in ("ps", "ts", "tds", "us", "vs")}
    ds = xx.calc_cape(p, prof["t"], prof["td"], sfc["ps"], sfc["ts"], sfc["tds"],
                      vertical_dim="lev", source="most-unstable")
    ref = calc_cape(a["p"], a["t"], a["td"], a["ps"], a["ts"], a["tds"], source="most-unstable",
                    vertical_lev="pressure")
    assert list(ds.data_vars) == ["cape", "cin", "mulev", "zmulev"] and ds.cape.dims == dims2
    np.testing.assert_array_equal(ds.cape.values, ref[0])
    # dimension order of the inputs is irrelevant, dask stays lazy
    lazy = xx.calc_cape(p, prof["t"].transpose("lev", "y", "x").chunk({"y": 3}),
                        prof["td"].transpose("lev", "y", "x").chunk({"y": 3}),
                        sfc["ps"].chunk({"y": 3}), sfc["ts"].chunk({"y": 3}), sfc["tds"].chunk({"y": 3}),
                        vertical_dim="lev", source="most-unstable")
    assert isinstance(lazy.cape.data, dsa.Array)
    np.testing.assert_array_equal(lazy.cape.values, ref[0])
    srh = xx.calc_srh(p, prof["t"], prof["td"], prof["u"], prof["v"], sfc["ps"], sfc["ts"],
                      sfc["tds"], sfc["us"], sfc["vs"], vertical_dim="lev", depth=1000,
                      output_var="all")
    assert len(srh.data_vars) == 8 and "1000 m" in srh.srh_rm.attrs["long_name"]
    kin = xx.calc_kinematics(prof["z_agl"], prof["u"], prof["v"], sfc["us"], sfc["vs"],
                             vertical_dim="lev")
    assert kin["S06"].attrs["long_name"] == "0-6 km bulk wind difference"
    with pytest.raises(ValueError):
        xx.calc_cape(p, prof["t"], prof["td"], sfc["ps"], sfc["ts"], sfc["tds"], vertical_dim="z")


def test_package_namespace():
    assert xcaper.__version__
    assert xcaper.calc_cape is calc_cape and callable(xcaper.calc_srh) and callable(xcaper.calc_severe)
    assert xcaper.era5 is era5
    with pytest.raises(AttributeError):
        xcaper.does_not_exist


def test_effective_outputs_consistent(era5_pair):
    from xcaper import composites as cp
    a = era5_to_arrays(*era5_pair)
    args = [a[k] for k in ("p", "t", "td", "u", "v", "z_agl", "ps", "ts", "tds", "us", "vs")]
    out = calc_severe(*args)
    np.testing.assert_allclose(out["ESCP"], cp.effective_supercell_composite(
        out["MU_CAPE"], out["ESRH_RM"], out["EBWD"], out["MU_CIN"]), rtol=1e-6)
    assert np.all(out["ESTP"][out["EIL_BASE"] > 0] == 0)
    assert np.all(out["ESTP"][np.isnan(out["EIL_BASE"])] == 0)
    assert np.all(out["ESRH_RM"][np.isnan(out["EIL_BASE"])] == 0)
    np.testing.assert_array_equal(out["ESRH"], np.maximum(np.abs(out["ESRH_RM"]), np.abs(out["ESRH_LM"])))
    # skipping the effective part leaves every other field untouched
    lean = calc_severe(*args, effective=False)
    assert not set(lean) & {"ESRH", "EIL_BASE", "DCAPE", "MU_EL"}
    for name in lean:
        np.testing.assert_array_equal(lean[name], out[name], err_msg=name)
    eff = calc_severe(*args, storm_motion="bunkers-effective")
    assert {"BUNKERS_MEAN_U", "BUNKERS_EFF_FLAG"} <= set(eff)
    with pytest.raises(ValueError):
        calc_severe(*args, storm_motion="bunkers-effective", effective=False)
