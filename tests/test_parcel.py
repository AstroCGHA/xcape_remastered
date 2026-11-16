"""LFC, EL, effective inflow layer, DCAPE and the effective-layer kinematics."""
import dask.array as dsa
import numpy as np
import pytest

import xcaper.parcel as P
from xcaper import composites as cp
from xcaper.cape_numba import _setup_profile, getcape
from xcaper.kinematics import calc_kinematics
from xcaper.parcel import PARCEL_NAMES, calc_parcel_diagnostics

from .conftest import era5_to_arrays, make_era5


@pytest.fixture(scope="module")
def columns():
    """Synthetic ERA5-like columns (vertical axis last, bottom to top)."""
    a = era5_to_arrays(*make_era5(nlat=6, nlon=8, seed=3))
    return a


def _thermo(a):
    return a["p"], a["t"], a["td"], a["ps"], a["ts"], a["tds"]


def _column(a, j, i):
    keep = a["p"] < a["ps"][j, i]
    return (np.concatenate(([a["ps"][j, i]], a["p"][keep])),
            np.concatenate(([a["ts"][j, i]], a["t"][j, i][keep])),
            np.concatenate(([a["tds"][j, i]], a["td"][j, i][keep])))


# ---------------------------------------------------------------------------
# parcel ascent
# ---------------------------------------------------------------------------
def test_ascent_reproduces_getcape(columns):
    a = columns
    res = np.empty(6)
    for j in range(a["ps"].shape[0]):
        for i in range(a["ps"].shape[1]):
            p, t, td = _column(a, j, i)
            w = np.empty((8, p.size))
            nk = _setup_profile(p[1:], t[1:], td[1:], p[0], t[0], td[0], w)
            for source in (1, 2, 3):
                ref = getcape(p[1:], t[1:], td[1:], p[0], t[0], td[0], 500.0, source, 500.0, 1,
                              np.empty((8, p.size)))
                P._lift_source(w, nk, source, 500.0, 500.0, 1, np.inf, res)
                assert res[0] == ref[0] and res[1] == ref[1]


def test_lfc_el_consistency(columns):
    a = columns
    out = calc_parcel_diagnostics(*_thermo(a), vertical_lev="pressure")
    assert list(out) == list(PARCEL_NAMES)
    from xcaper.core import calc_cape, calc_lcl
    for tag, source in (("SFC", "surface"), ("ML", "mixed-layer"), ("MU", "most-unstable")):
        cape = calc_cape(*_thermo(a), source=source, vertical_lev="pressure", method="numba")[0]
        lfc, el = out[f"{tag}_LFC"], out[f"{tag}_EL"]
        has = np.isfinite(lfc)
        assert has.any()
        assert np.all(cape[has] > 0)                       # an LFC implies CAPE
        assert np.all(np.isfinite(el[has]))
        assert np.all(el[has] > lfc[has])
        assert np.all(out[f"{tag}_EL_P"][has] < out[f"{tag}_LFC_P"][has])
    # the surface LFC is at or above the surface LCL
    _, p_lcl = calc_lcl(*_thermo(a), source="surface", vertical_lev="pressure")
    has = np.isfinite(out["SFC_LFC_P"])
    assert np.all(out["SFC_LFC_P"][has] <= p_lcl[has] + 0.05)


def test_stable_column_has_no_lfc_el_or_layer():
    p = np.array([1000., 950, 900, 850, 800, 700, 600, 500, 400, 300, 200, 100])
    z = 44330.8 * (1.0 - (p / 1013.25) ** 0.190263)
    t = 30.0 - 2.0 * z / 1000.0                      # very stable: 2 K/km
    td = t - 25.0
    out = calc_parcel_diagnostics(p, t[None], td[None], np.array([1010.0]), np.array([25.0]),
                                  np.array([5.0]), vertical_lev="pressure")
    for name in ("SFC_LFC", "SFC_EL", "ML_LFC", "MU_EL", "EIL_BASE", "EIL_TOP"):
        assert np.isnan(out[name]).all(), name
    assert np.isfinite(out["DCAPE"]).all()


def test_missing_surface_data_gives_nan(columns):
    a = dict(columns)
    a["ts"] = a["ts"].copy()
    a["ts"][0, 0] = np.nan
    out = calc_parcel_diagnostics(*_thermo(a), vertical_lev="pressure", return_valid=True)
    assert out["_VALID"][0, 0] == 0 and out["_VALID"][1, 1] == 1
    for name in PARCEL_NAMES:
        assert np.isnan(out[name][0, 0]), name


# ---------------------------------------------------------------------------
# effective inflow layer vs a brute-force search
# ---------------------------------------------------------------------------
def _brute_force_eil(p, t, td, refine_hpa=None, search_top=500.0):
    if refine_hpa:
        pr, tr, tdr = [p[0]], [t[0]], [td[0]]
        for k in range(1, p.size):
            if p[k - 1] > search_top and p[k - 1] - p[k] > refine_hpa:
                n = int(np.ceil((p[k - 1] - p[k]) / refine_hpa))
                pm = p[k - 1] - np.arange(1, n) * (p[k - 1] - p[k]) / n
                f = np.log(p[k - 1] / pm) / np.log(p[k - 1] / p[k])
                pr += list(pm)
                tr += list(t[k - 1] + f * (t[k] - t[k - 1]))
                tdr += list(td[k - 1] + f * (td[k] - td[k - 1]))
            pr.append(p[k]); tr.append(t[k]); tdr.append(td[k])
        p, t, td = np.array(pr), np.array(tr), np.array(tdr)

    def cc(k):
        r = getcape(p[k + 1:], t[k + 1:], td[k + 1:], p[k], t[k], td[k], 500.0, 1, 500.0, 1,
                    np.empty((8, p.size - k)))
        return r[0] >= 100.0 and r[1] <= 250.0

    base = next((k for k in range(p.size) if p[k] >= search_top and cc(k)), None)
    if base is None:
        return np.nan, np.nan
    top = base
    for k in range(base + 1, p.size):
        if not cc(k):
            break
        top = k
    return p[base], p[top]


@pytest.mark.parametrize("eil_dp", [None, 10.0])
def test_effective_layer_matches_brute_force(eil_dp):
    a = era5_to_arrays(*make_era5(nlat=5, nlon=6, seed=11))
    out = calc_parcel_diagnostics(*_thermo(a), vertical_lev="pressure", eil_dp=eil_dp)
    found = 0
    for j in range(5):
        for i in range(6):
            p, t, td = _column(a, j, i)
            mu = getcape(p[1:], t[1:], td[1:], p[0], t[0], td[0], 500.0, 2, 500.0, 1,
                         np.empty((8, p.size)))
            if mu[0] >= 100.0 and mu[1] <= 250.0:
                ref = _brute_force_eil(p, t, td, refine_hpa=eil_dp)
            else:
                ref = (np.nan, np.nan)
            np.testing.assert_allclose([out["EIL_BASE_P"][j, i], out["EIL_TOP_P"][j, i]], ref,
                                       rtol=1e-5, err_msg=f"column {j},{i}")
            found += np.isfinite(ref[0])
    assert found >= 5          # the synthetic field must actually exercise the search


def test_refined_search_is_deeper_or_equal_and_validated():
    a = era5_to_arrays(*make_era5(nlat=5, nlon=6, seed=11))
    native = calc_parcel_diagnostics(*_thermo(a), vertical_lev="pressure", eil_dp=None)
    fine = calc_parcel_diagnostics(*_thermo(a), vertical_lev="pressure", eil_dp=10.0)
    np.testing.assert_array_equal(np.isfinite(native["EIL_BASE"]), np.isfinite(fine["EIL_BASE"]))
    with pytest.raises(ValueError):
        calc_parcel_diagnostics(*_thermo(a), vertical_lev="pressure", eil_dp=-5.0)
    assert np.isfinite(calc_parcel_diagnostics(*_thermo(a), vertical_lev="pressure",
                                               eil_dp=0)["DCAPE"]).all()


# ---------------------------------------------------------------------------
# DCAPE
# ---------------------------------------------------------------------------
def test_dcape_against_metpy():
    mc = pytest.importorskip("metpy.calc")
    from metpy.units import units
    a = era5_to_arrays(*make_era5(nlat=3, nlon=4, seed=5))
    out = calc_parcel_diagnostics(*_thermo(a), vertical_lev="pressure")
    for j in range(3):
        for i in range(4):
            p, t, td = _column(a, j, i)
            ref = mc.downdraft_cape(p * units.hPa, t * units.degC, td * units.degC)[0].m
            assert out["DCAPE"][j, i] == pytest.approx(ref, rel=0.02, abs=5.0)


def test_dcape_source_layer_and_high_terrain():
    a = era5_to_arrays(*make_era5(nlat=3, nlon=4, seed=5))
    out = calc_parcel_diagnostics(*_thermo(a), vertical_lev="pressure")
    assert np.all((out["DCAPE_SRC_P"] <= 700.0) & (out["DCAPE_SRC_P"] >= 500.0))
    ps = a["ps"].copy()
    ps[0, 0] = 480.0                                    # surface above the source layer
    out = calc_parcel_diagnostics(a["p"], a["t"], a["td"], ps, a["ts"], a["tds"],
                                  vertical_lev="pressure")
    assert np.isnan(out["DCAPE"][0, 0]) and np.isfinite(out["DCAPE"][1, 1])


def test_dask_matches_numpy(columns):
    a = columns
    eager = calc_parcel_diagnostics(*_thermo(a), vertical_lev="pressure")
    t = dsa.from_array(a["t"], chunks=(3, 4, 10))
    td = dsa.from_array(a["td"], chunks=(3, 4, 10))
    sfc = [dsa.from_array(a[k], chunks=(3, 4)) for k in ("ps", "ts", "tds")]
    lazy = calc_parcel_diagnostics(a["p"], t, td, *sfc, vertical_lev="pressure")
    for name in PARCEL_NAMES:
        assert isinstance(lazy[name], dsa.Array)
        np.testing.assert_array_equal(lazy[name].compute(), eager[name], err_msg=name)


# ---------------------------------------------------------------------------
# effective-layer kinematics
# ---------------------------------------------------------------------------
def _linear_shear(a=0.004, top=16000.0):
    z = np.arange(250.0, top + 1.0, 250.0)[None, :]
    return z, a * z, np.zeros_like(z), np.zeros(1), np.zeros(1)


def test_effective_kinematics_analytic():
    a = 0.004
    z, u, v, us, vs = _linear_shear(a)
    res = calc_kinematics(z, u, v, us, vs, eil_base=np.array([0.0]), eil_top=np.array([1000.0]),
                          el_height=np.array([12000.0]))
    # straight hodograph: SRH = deviation * shear * depth for the right mover
    np.testing.assert_allclose(res["ESRH_RM"], 7.5 * a * 1000.0, rtol=1e-5)
    np.testing.assert_allclose(res["ESRH_LM"], -7.5 * a * 1000.0, rtol=1e-5)
    np.testing.assert_allclose(res["EBWD"], a * 6000.0, rtol=1e-5)          # base to EL/2
    np.testing.assert_allclose(res["E_SWV"], res["SWV01"], rtol=1e-6)      # same 0-1 km layer
    np.testing.assert_allclose(res["E_SWNE"], res["SWNESS01"], rtol=1e-6)
    # elevated layer, bounds off the 100 m grid
    res = calc_kinematics(z, u, v, us, vs, eil_base=np.array([437.0]),
                          eil_top=np.array([1663.0]), el_height=np.array([10437.0]))
    np.testing.assert_allclose(res["ESRH_RM"], 7.5 * a * (1663.0 - 437.0), rtol=1e-5)
    np.testing.assert_allclose(res["EBWD"], a * 0.5 * (10437.0 - 437.0), rtol=1e-5)


def test_no_effective_layer():
    z, u, v, us, vs = _linear_shear()
    res = calc_kinematics(z, u, v, us, vs, eil_base=np.array([np.nan]),
                          eil_top=np.array([np.nan]), el_height=np.array([np.nan]))
    assert res["ESRH_RM"][0] == 0 and res["ESRH_LM"][0] == 0 and res["EBWD"][0] == 0
    assert np.isnan(res["E_SWV"][0]) and np.isnan(res["E_SWNE"][0])
    assert np.isfinite(res["SRH1_RM"][0])
    with pytest.raises(ValueError):
        calc_kinematics(z, u, v, us, vs, eil_base=np.array([0.0]))


def test_effective_bunkers_motion():
    a = 0.004
    z, u, v, us, vs = _linear_shear(a)
    kw = dict(eil_base=np.array([0.0, np.nan]), eil_top=np.array([800.0, np.nan]),
              el_height=np.array([12000.0, np.nan]))
    z2, u2, v2 = (np.repeat(x, 2, axis=0) for x in (z, u, v))
    us2, vs2 = np.zeros(2), np.zeros(2)
    eff = calc_kinematics(z2, u2, v2, us2, vs2, storm_motion="bunkers-effective", **kw)
    std = calc_kinematics(z2, u2, v2, us2, vs2, storm_motion="bunkers", **kw)
    # column 0: mean wind of 0 - 0.65 * 12 km = a * 3.9 km, shear eastward -> RM to the south
    np.testing.assert_allclose([eff["BUNKERS_RM_U"][0], eff["BUNKERS_RM_V"][0]], [a * 3900.0, -7.5],
                               rtol=1e-5)
    np.testing.assert_allclose(eff["BUNKERS_MEAN_U"][0], a * 3900.0, rtol=1e-5)
    assert eff["BUNKERS_EFF_FLAG"][0] == 1
    # column 1: no effective layer -> standard 0-6 km fallback
    assert eff["BUNKERS_EFF_FLAG"][1] == 0
    np.testing.assert_allclose(eff["BUNKERS_RM_U"][1], std["BUNKERS_RM_U"][1], rtol=1e-6)
    with pytest.raises(ValueError):
        calc_kinematics(z, u, v, us, vs, storm_motion="bunkers-effective")


def test_effective_fields_above_the_data_top():
    a = 0.004
    z, u, v, us, vs = _linear_shear(a, top=9000.0)
    res = calc_kinematics(z, u, v, us, vs, storm_motion="bunkers-effective",
                          eil_base=np.array([0.0]), eil_top=np.array([1000.0]),
                          el_height=np.array([16000.0]))
    # 0.65 * 16 km = 10.4 km is above the data: fall back; EBWD top (8 km) is fine
    assert res["BUNKERS_EFF_FLAG"][0] == 0
    np.testing.assert_allclose(res["EBWD"], a * 8000.0, rtol=1e-5)
    assert np.isfinite(res["S06"][0])


# ---------------------------------------------------------------------------
# effective composites
# ---------------------------------------------------------------------------
def test_effective_composites():
    x = lambda *v: np.array(v, dtype="f8")  # noqa: E731
    # ESCP
    assert cp.effective_supercell_composite(x(1000), x(50), x(20), x(0))[0] == pytest.approx(1.0)
    assert cp.effective_supercell_composite(x(2000), x(100), x(25), x(80))[0] == pytest.approx(2 * 2 * 1 * 0.5)
    assert cp.effective_supercell_composite(x(2000), x(100), x(9), x(0))[0] == 0.0
    # ESTP: surface-based layer only
    args = (x(1500, 1500, 1500), x(800, 800, 800), x(150, 150, 150), x(20, 20, 20), x(25, 25, 25))
    estp = cp.effective_significant_tornado(*args, x(0.0, 120.0, np.nan))
    np.testing.assert_allclose(estp, [1.0, 0.0, 0.0])
    assert cp.effective_significant_tornado(x(1500), x(800), x(150), x(20), x(125), x(0))[0] == pytest.approx(0.5)


def test_saturated_start_has_lfc_and_el(columns):
    """Regression: a (super)saturated source parcel starts at its own LCL.

    ERA5 dewpoints derived from specific humidity can exceed the temperature;
    round-off in the ln(p) interpolation used to drop the LFC (and EL) of
    such parcels although they had CAPE.
    """
    a = dict(columns)
    from xcaper.core import calc_cape
    for offset in (0.0, 0.4):
        tds = a["ts"] + offset
        out = calc_parcel_diagnostics(a["p"], a["t"], a["td"], a["ps"], a["ts"], tds,
                                      vertical_lev="pressure")
        cape = calc_cape(a["p"], a["t"], a["td"], a["ps"], a["ts"], tds, source="surface",
                         vertical_lev="pressure", method="numba")[0]
        big = cape > 50.0
        assert big.any()
        assert np.isfinite(out["SFC_LFC"][big]).all() and np.isfinite(out["SFC_EL"][big]).all()
        assert np.all(out["SFC_LFC_P"][big] <= a["ps"][big] + 1e-3)
