"""Wind-profile diagnostics: analytic cases and a brute-force numpy reference."""
import dask.array as dsa
import numpy as np
import pytest

from xcaper.kinematics import StormMotion, calc_kinematics, kinematic_names, layer_tag


# ---------------------------------------------------------------------------
# independent numpy reference (the algorithm of the original ERA5 scripts)
# ---------------------------------------------------------------------------
def reference(z, u, v, us, vs, sm, srh_tops, shear_tops, swv_tops, dz=100.0):
    top = max((sm.top,) + tuple(srh_tops) + tuple(shear_tops) + tuple(swv_tops))
    zg = np.arange(0.0, top + dz * 0.5, dz)
    ok = np.isfinite(z) & np.isfinite(u) & np.isfinite(v) & (z > 0.0)
    if not (np.isfinite(us) and np.isfinite(vs)) or ok.sum() < 2:
        return None
    za = np.concatenate(([0.0], z[ok])); ua = np.concatenate(([us], u[ok])); va = np.concatenate(([vs], v[ok]))
    order = np.argsort(za, kind="stable"); za, ua, va = za[order], ua[order], va[order]
    if za[-1] < top:
        return None
    up, vp = np.interp(zg, za, ua), np.interp(zg, za, va)

    def mean(a, lay):
        sel = (zg >= lay[0] - 1e-6) & (zg <= lay[1] + 1e-6)
        zs, a = zg[sel], a[sel]
        return np.sum(0.5 * (a[1:] + a[:-1]) * np.diff(zs)) / (zs[-1] - zs[0])

    mu, mv = mean(up, sm.mean_layer), mean(vp, sm.mean_layer)
    su = mean(up, sm.head_layer) - mean(up, sm.tail_layer)
    sv = mean(vp, sm.head_layer) - mean(vp, sm.tail_layer)
    mag = np.hypot(su, sv)
    du, dv = (sm.deviation * sv / mag, -sm.deviation * su / mag) if mag > 0 else (0.0, 0.0)
    out = {"mean": (mu, mv), "rm": (mu + du, mv + dv), "lm": (mu - du, mv - dv)}
    for name, (cu, cv) in (("rm", out["rm"]), ("lm", out["lm"])):
        for t in srh_tops:
            sel = zg <= t + 1e-6
            usr, vsr = up[sel] - cu, vp[sel] - cv
            out[f"srh_{name}_{t:g}"] = np.sum(usr[1:] * vsr[:-1] - usr[:-1] * vsr[1:])
    for t in shear_tops:
        i = int(round(t / dz))
        out[f"shear_{t:g}"] = np.hypot(up[i] - up[0], vp[i] - vp[0])
    cu, cv = out["rm"]
    for t in swv_tops:
        sel = zg <= t + 1e-6
        zz, uu, vv = zg[sel], up[sel], vp[sel]
        dudz = np.empty_like(uu); dvdz = np.empty_like(vv)
        dudz[1:-1] = (uu[2:] - uu[:-2]) / (zz[2:] - zz[:-2]); dvdz[1:-1] = (vv[2:] - vv[:-2]) / (zz[2:] - zz[:-2])
        dudz[0], dudz[-1], dvdz[0], dvdz[-1] = dudz[1], dudz[-2], dvdz[1], dvdz[-2]
        ox, oy = -dvdz, dudz
        usr, vsr = uu - cu, vv - cv
        spd = np.hypot(usr, vsr)
        sw = np.abs((usr * ox + vsr * oy) / spd)
        out[f"swv_{t:g}"] = np.mean(sw)
        out[f"swness_{t:g}"] = 100.0 * np.mean(sw / np.hypot(ox, oy))
    return out


def random_columns(n=60, nlev=30, seed=1):
    rng = np.random.default_rng(seed)
    # pressure-level like heights: some levels below ground (negative AGL)
    base = np.linspace(-800.0, 16000.0, nlev)
    z = base[None, :] + rng.uniform(-900.0, 900.0, (n, 1)) + rng.normal(0, 15.0, (n, nlev))
    u = rng.normal(0, 8, (n, nlev)).cumsum(axis=1) * 0.4
    v = rng.normal(0, 8, (n, nlev)).cumsum(axis=1) * 0.4
    return z, u, v, rng.normal(0, 4, n), rng.normal(0, 4, n)


@pytest.mark.parametrize("motion", ["bunkers", "bunkers-shallow",
                                    StormMotion(mean_layer=(0, 4000), tail_layer=(0, 1000),
                                                head_layer=(3000, 4000), deviation=6.0)])
def test_against_numpy_reference(motion):
    z, u, v, us, vs = random_columns()
    srh, shr, swv = (500.0, 1000.0, 3000.0), (1000.0, 6000.0), (500.0, 1000.0)
    res = calc_kinematics(z, u, v, us, vs, storm_motion=motion, srh_layers=srh,
                          shear_layers=shr, swv_layers=swv)
    sm = StormMotion.from_any(motion)
    mean = f"MEAN{layer_tag(sm.mean_layer[1], pad=True)}"
    for i in range(z.shape[0]):
        ref = reference(z[i], u[i], v[i], us[i], vs[i], sm, srh, shr, swv)
        assert ref is not None
        tol = dict(rtol=2e-5, atol=2e-4)
        np.testing.assert_allclose([res[f"{mean}_U"][i], res[f"{mean}_V"][i]], ref["mean"], **tol)
        np.testing.assert_allclose([res["BUNKERS_RM_U"][i], res["BUNKERS_RM_V"][i]], ref["rm"], **tol)
        np.testing.assert_allclose([res["BUNKERS_LM_U"][i], res["BUNKERS_LM_V"][i]], ref["lm"], **tol)
        for t in srh:
            np.testing.assert_allclose(res[f"SRH{layer_tag(t)}_RM"][i], ref[f"srh_rm_{t:g}"], rtol=1e-4, atol=2e-3)
            np.testing.assert_allclose(res[f"SRH{layer_tag(t)}_LM"][i], ref[f"srh_lm_{t:g}"], rtol=1e-4, atol=2e-3)
        for t in shr:
            np.testing.assert_allclose(res[f"S{layer_tag(t, True)}"][i], ref[f"shear_{t:g}"], **tol)
        for t in swv:
            np.testing.assert_allclose(res[f"SWV{layer_tag(t, True)}"][i], ref[f"swv_{t:g}"], rtol=1e-4, atol=1e-8)
            np.testing.assert_allclose(res[f"SWNESS{layer_tag(t, True)}"][i], ref[f"swness_{t:g}"], rtol=1e-4)


def test_analytic_unidirectional_shear():
    """u = a z, v = 0: everything is known in closed form."""
    a = 0.005                                   # 30 m/s over 6 km
    z = np.arange(250.0, 12001.0, 250.0)[None, :]
    u, v = a * z, np.zeros_like(z)
    res = calc_kinematics(z, u, v, np.zeros(1), np.zeros(1))
    np.testing.assert_allclose(res["MEAN06_U"], 15.0, atol=1e-5)
    np.testing.assert_allclose(res["MEAN06_V"], 0.0, atol=1e-6)
    # right mover is 7.5 m/s to the right of the (westerly) shear: southward
    np.testing.assert_allclose([res["BUNKERS_RM_U"][0], res["BUNKERS_RM_V"][0]], [15.0, -7.5], atol=1e-5)
    np.testing.assert_allclose([res["BUNKERS_LM_U"][0], res["BUNKERS_LM_V"][0]], [15.0, 7.5], atol=1e-5)
    # SRH = 7.5 * a * depth for the right mover, opposite for the left mover
    np.testing.assert_allclose(res["SRH1_RM"], 7.5 * a * 1000.0, rtol=1e-5)
    np.testing.assert_allclose(res["SRH3_RM"], 7.5 * a * 3000.0, rtol=1e-5)
    np.testing.assert_allclose(res["SRH3_LM"], -7.5 * a * 3000.0, rtol=1e-5)
    np.testing.assert_allclose([res["S01"][0], res["S03"][0], res["S06"][0]], [5.0, 15.0, 30.0], rtol=1e-5)
    # streamwise vorticity: |omega| = a, storm-relative wind (u - 15, 7.5)
    zg = np.arange(0.0, 1001.0, 100.0)
    spd = np.hypot(a * zg - 15.0, 7.5)
    np.testing.assert_allclose(res["SWV01"], np.mean(7.5 * a / spd), rtol=1e-5)
    np.testing.assert_allclose(res["SWNESS01"], 100.0 * np.mean(7.5 / spd), rtol=1e-5)
    # 0-500 m shear points east, surface inflow of the RM is (15, -7.5)
    np.testing.assert_allclose(res["CRITICAL_ANGLE"], np.degrees(np.arctan2(7.5, 15.0)), rtol=1e-5)


def test_purely_streamwise_and_crosswise():
    z = np.arange(100.0, 8001.0, 100.0)[None, :]
    # storm-relative wind along -x at all levels would need a known storm motion;
    # use a custom zero-deviation motion so the storm moves with the mean wind
    sm = StormMotion(deviation=0.0)
    # crosswise: wind and shear both along x -> vorticity along y, SR wind along x
    res = calc_kinematics(z, 0.004 * z, np.zeros_like(z), np.zeros(1), np.zeros(1), storm_motion=sm)
    np.testing.assert_allclose(res["SWNESS01"], 0.0, atol=1e-4)
    np.testing.assert_allclose(res["SRH3_RM"], 0.0, atol=1e-4)


def test_missing_and_shallow_columns_are_nan():
    z, u, v, us, vs = random_columns(n=4)
    z[0] = np.linspace(100.0, 5000.0, z.shape[1])     # does not reach 6 km
    us[1] = np.nan                                     # no surface wind
    u[2, 10:] = np.nan                                 # profile ends too low
    res = calc_kinematics(z, u, v, us, vs)
    for name, a in res.items():
        assert np.isnan(a[:3]).all(), name
        if name != "CRITICAL_ANGLE":
            assert np.isfinite(a[3]), name


def test_vertical_order_does_not_matter():
    z, u, v, us, vs = random_columns(n=10)
    a = calc_kinematics(z, u, v, us, vs)
    b = calc_kinematics(z[:, ::-1], u[:, ::-1], v[:, ::-1], us, vs)
    for name in a:
        np.testing.assert_allclose(a[name], b[name], rtol=1e-6, atol=1e-6, err_msg=name)


def test_dask_matches_numpy_and_shapes():
    z, u, v, us, vs = random_columns(n=24)
    shape = (4, 6)
    z3, u3, v3 = (a.reshape(shape + (-1,)) for a in (z, u, v))
    us2, vs2 = us.reshape(shape), vs.reshape(shape)
    eager = calc_kinematics(z3, u3, v3, us2, vs2, storm_motion="bunkers-shallow")
    lazy = calc_kinematics(dsa.from_array(z3, chunks=(2, 3, 7)), dsa.from_array(u3, chunks=(2, 3, 7)),
                           dsa.from_array(v3, chunks=(2, 3, 7)), dsa.from_array(us2, chunks=(2, 3)),
                           dsa.from_array(vs2, chunks=(2, 3)), storm_motion="bunkers-shallow")
    assert list(eager) == list(lazy) == kinematic_names("bunkers-shallow")
    for name in eager:
        assert eager[name].shape == shape and eager[name].dtype == np.float32
        assert isinstance(lazy[name], dsa.Array)
        np.testing.assert_array_equal(lazy[name].compute(), eager[name], err_msg=name)


def test_names_and_validation():
    assert layer_tag(1000) == "1" and layer_tag(6000, pad=True) == "06" and layer_tag(500) == "500M"
    names = kinematic_names("bunkers-shallow", (500, 1000), (6000,), (1000,))
    assert names == ["MEAN03_U", "MEAN03_V", "BUNKERS_RM_U", "BUNKERS_RM_V", "BUNKERS_LM_U",
                     "BUNKERS_LM_V", "CRITICAL_ANGLE", "SRH500M_RM", "SRH500M_LM", "SRH1_RM",
                     "SRH1_LM", "S06", "SWV01", "SWNESS01"]
    z, u, v, us, vs = random_columns(n=2)
    with pytest.raises(ValueError):
        calc_kinematics(z, u, v, us, vs, srh_layers=(1050.0,))       # not a multiple of dz
    with pytest.raises(ValueError):
        calc_kinematics(z, u, v, us, vs, storm_motion="corfidi")
    with pytest.raises(ValueError):
        calc_kinematics(z, u, v, us[:1], vs[:1])
