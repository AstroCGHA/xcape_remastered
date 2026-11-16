import dask.array as dsa
import numpy as np
import pytest
import xarray as xr

from xcaper import composites as cp
from xcaper.thermo import calc_lapse_rates, dewpoint_from_specific_humidity


def test_dewpoint_roundtrip():
    p = np.array([1000.0, 850.0, 500.0, 200.0])
    td = np.array([295.0, 285.0, 250.0, 215.0])
    es = 6.112 * np.exp(17.67 * (td - 273.15) / (td - 29.65))      # hPa
    r = 0.62197 * es / (p - es)
    q = r / (1.0 + r)
    np.testing.assert_allclose(dewpoint_from_specific_humidity(q, p), td, rtol=1e-10)
    # non-physical humidity does not produce NaN
    assert np.isfinite(dewpoint_from_specific_humidity(np.array([0.0, -1e-6]), 500.0)).all()
    # dask and xarray pass through lazily / with coordinates
    assert isinstance(dewpoint_from_specific_humidity(dsa.from_array(q), p), dsa.Array)
    da_ = dewpoint_from_specific_humidity(xr.DataArray(q, dims="level"), xr.DataArray(p, dims="level"))
    assert isinstance(da_, xr.DataArray)


def test_lapse_rates_constant_lapse_atmosphere():
    gamma = 6.5                                   # K/km
    p = np.array([1000, 925, 850, 700, 600, 500, 400, 300], dtype="f8")
    z_msl = 44330.8 * (1.0 - (p / 1013.25) ** 0.190263)
    for zsfc, ps in ((0.0, 1013.25), (1200.0, 877.0), (3500.0, 657.0)):
        z_agl = (z_msl - zsfc)[None, :]
        t = 30.0 - gamma * z_msl[None, :] / 1000.0
        ts = np.array([30.0 - gamma * zsfc / 1000.0])
        lr03, lr75 = calc_lapse_rates(p, t, z_agl, np.array([ps]), ts)
        np.testing.assert_allclose(lr03, gamma, rtol=1e-5)
        if ps >= 700.0:
            np.testing.assert_allclose(lr75, gamma, rtol=1e-5)
        else:
            assert np.isnan(lr75).all()           # 700 hPa is below the ground
    lazy = calc_lapse_rates(p, dsa.from_array(t), dsa.from_array(z_agl), dsa.from_array(np.array([ps])),
                            dsa.from_array(ts))
    np.testing.assert_array_equal(lazy[0].compute(), lr03)
    with pytest.raises(ValueError):
        calc_lapse_rates(p, t, z_agl, np.array([ps]), ts, pressure_layer=(500.0, 700.0))


def test_significant_tornado():
    a = lambda x: np.array([x], dtype="f8")
    assert cp.significant_tornado(a(1500), a(1000), a(150), a(20))[0] == pytest.approx(1.0)
    assert cp.significant_tornado(a(3000), a(500), a(300), a(40))[0] == pytest.approx(2 * 1 * 2 * 1.5)
    assert cp.significant_tornado(a(3000), a(2500), a(300), a(40))[0] == 0.0       # high LCL
    assert cp.significant_tornado(a(3000), a(1500), a(300), a(20))[0] == pytest.approx(2 * 0.5 * 2 * 1)
    assert cp.significant_tornado(a(3000), a(500), a(300), a(12.0))[0] == 0.0      # weak shear
    assert cp.significant_tornado(a(1500), a(1000), a(-150), a(20))[0] == pytest.approx(-1.0)
    # CIN version: xcaper CIN is a positive number
    assert cp.significant_tornado_cin(a(1500), a(1000), a(150), a(20), a(30))[0] == pytest.approx(1.0)
    assert cp.significant_tornado_cin(a(1500), a(1000), a(150), a(20), a(125))[0] == pytest.approx(0.5)
    assert cp.significant_tornado_cin(a(1500), a(1000), a(150), a(20), a(250))[0] == 0.0


def test_supercell_composite_and_others():
    a = lambda x: np.array([x], dtype="f8")
    assert cp.supercell_composite(a(1000), a(50), a(20))[0] == pytest.approx(1.0)
    assert cp.supercell_composite(a(3000), a(200), a(35))[0] == pytest.approx(3 * 4 * 1.0)
    assert cp.supercell_composite(a(3000), a(200), a(15))[0] == pytest.approx(3 * 4 * 0.75)
    assert cp.supercell_composite(a(3000), a(200), a(9.9))[0] == 0.0
    assert cp.supercell_composite(a(1000), a(50), a(20), cin=a(20))[0] == pytest.approx(1.0)
    assert cp.supercell_composite(a(1000), a(50), a(20), cin=a(80))[0] == pytest.approx(0.5)
    assert cp.supercell_composite(a(1000), a(50), a(20), cin=a(0))[0] == pytest.approx(1.0)
    assert cp.energy_helicity_index(a(2000), a(160))[0] == pytest.approx(2.0)
    assert cp.wmaxshear(a(2000), a(20))[0] == pytest.approx(np.sqrt(4000.0) * 20.0)


def test_composites_keep_container_type_and_nan():
    x = np.array([[1500.0, np.nan], [3000.0, 0.0]])
    lcl, srh, shr = np.full_like(x, 800.0), np.full_like(x, 150.0), np.full_like(x, 20.0)
    ref = cp.significant_tornado(x, lcl, srh, shr)
    assert np.isnan(ref[0, 1]) and ref[1, 1] == 0.0
    lazy = cp.significant_tornado(*(dsa.from_array(a, chunks=1) for a in (x, lcl, srh, shr)))
    assert isinstance(lazy, dsa.Array)
    np.testing.assert_array_equal(lazy.compute(), ref)
    xa = cp.significant_tornado(*(xr.DataArray(a, dims=("y", "x")) for a in (x, lcl, srh, shr)))
    assert isinstance(xa, xr.DataArray)
    np.testing.assert_array_equal(xa.values, ref)


def test_virtual_lapse_rates():
    """Virtual-temperature lapse rates (SHARPpy / SounderPy convention)."""
    p = np.array([1000, 925, 850, 700, 600, 500, 400, 300], dtype="f8")
    z = 44330.8 * (1.0 - (p / 1013.25) ** 0.190263)
    t = (30.0 - 6.0 * z / 1000.0)[None, :]
    td = t - np.linspace(2.0, 25.0, p.size)[None, :]          # moist below, dry aloft
    ps, ts, tds = np.array([1013.25]), np.array([30.0]), np.array([28.0])
    plain = calc_lapse_rates(p, t, z[None, :], ps, ts)
    virt = calc_lapse_rates(p, t, z[None, :], ps, ts, td=td, tds=tds)
    np.testing.assert_allclose(plain[0], 6.0, rtol=1e-6)
    # moisture decreasing with height makes the virtual lapse rate larger
    assert virt[0][0] > plain[0][0] + 0.2 and virt[1][0] > plain[1][0]

    def tv(tc, tdc, pp):          # SHARPpy thermo.virtemp
        e = 6.112 * np.exp(17.67 * tdc / (tdc + 243.5))
        w = 0.62197 * e / (pp - e)
        return (tc + 273.15) * (1 + w / 0.62197) / (1 + w) - 273.15
    tvs = np.concatenate((tv(ts, tds, ps), tv(t[0], td[0], p)))
    zz = np.concatenate(([0.0], z))
    ref = -(np.interp(3000.0, zz, tvs) - tvs[0]) / 3.0
    np.testing.assert_allclose(virt[0][0], ref, rtol=1e-6)
    lazy = calc_lapse_rates(p, dsa.from_array(t), dsa.from_array(z[None, :]), dsa.from_array(ps),
                            dsa.from_array(ts), td=dsa.from_array(td), tds=dsa.from_array(tds))
    np.testing.assert_array_equal(lazy[0].compute(), virt[0])
    with pytest.raises(ValueError):
        calc_lapse_rates(p, t, z[None, :], ps, ts, td=td)
