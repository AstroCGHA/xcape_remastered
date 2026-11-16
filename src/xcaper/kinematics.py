"""Wind-profile diagnostics for severe convection (numba + dask).

Everything here is computed from **one common wind profile per column**:

* the near-surface (10 m) wind is used as the wind at 0 m above ground;
* all levels that lie above the local surface are added (heights above
  ground must be supplied, e.g. from geopotential);
* the profile is interpolated linearly to a uniform height grid
  (``dz`` = 100 m by default) from the ground to the highest requested layer.

From that profile:

* storm motion with the Bunkers et al. (2000) internal-dynamics method, with
  configurable layers (the standard 0-6 km version, a shallow 0-3 km version
  for low-topped / tropical supercells, or any custom layers);
* storm relative helicity (SRH) of right and left movers for any layers;
* bulk wind difference ("shear") for any layers;
* layer-mean streamwise vorticity and streamwiseness for any layers;
* the critical angle (Esterheld and Giuliano 2008).

numpy inputs are processed with a ``numba.prange`` loop over all cores;
dask inputs are processed lazily, chunk by chunk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import dask.array as da
import numpy as np
from numba import njit, prange

from ._parallel import run_loop

__all__ = ["StormMotion", "calc_kinematics", "kinematic_names", "layer_tag"]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StormMotion:
    """Layers of the Bunkers et al. (2000) internal-dynamics storm motion.

    ``V_RM = V_mean + D (S x k)/|S|`` and ``V_LM = V_mean - D (S x k)/|S|``
    where ``V_mean`` is the height-weighted (non pressure-weighted) mean
    wind of ``mean_layer`` and ``S`` joins the mean wind of ``tail_layer``
    to the mean wind of ``head_layer``.  All bounds are metres above ground.
    """

    mean_layer: tuple[float, float] = (0.0, 6000.0)
    tail_layer: tuple[float, float] = (0.0, 500.0)
    head_layer: tuple[float, float] = (5500.0, 6000.0)
    deviation: float = 7.5
    name: str = "bunkers"
    #: use the effective-layer method of Bunkers et al. (2014) where an
    #: effective inflow layer exists (the layers above are the fallback)
    effective: bool = False
    #: fraction of the effective-base-to-MU-EL depth used by that method
    effective_fraction: float = 0.65

    @classmethod
    def bunkers(cls) -> "StormMotion":
        """Standard Bunkers et al. (2000): 0-6 km mean wind and shear."""
        return cls()

    @classmethod
    def bunkers_shallow(cls) -> "StormMotion":
        """Bunkers layers scaled down to 0-3 km.

        A non-standard adaptation meant for shallow / low-topped (tropical,
        nocturnal, tropical-cyclone embedded mini-) supercells.
        """
        return cls(mean_layer=(0.0, 3000.0), tail_layer=(0.0, 500.0),
                   head_layer=(2500.0, 3000.0), name="bunkers-shallow")

    @classmethod
    def bunkers_effective(cls) -> "StormMotion":
        """Effective-layer Bunkers motion (Bunkers et al. 2014).

        Mean wind and shear of the layer from the effective inflow base to
        65% of the way up to the equilibrium level of the most-unstable
        parcel; where there is no effective inflow layer, the standard
        0-6 km method is used instead.  The depth adapts to the depth of the
        convection, which suits low-topped storms.
        """
        return cls(effective=True, name="bunkers-effective")

    @classmethod
    def from_any(cls, value) -> "StormMotion":
        if isinstance(value, cls):
            return value
        presets = {"bunkers": cls.bunkers, "bunkers-shallow": cls.bunkers_shallow,
                   "bunkers_shallow": cls.bunkers_shallow,
                   "bunkers-effective": cls.bunkers_effective,
                   "bunkers_effective": cls.bunkers_effective}
        if isinstance(value, str) and value in presets:
            return presets[value]()
        raise ValueError(
            "storm_motion must be 'bunkers', 'bunkers-shallow', 'bunkers-effective' or a "
            f"StormMotion instance; got {value!r}"
        )

    @property
    def top(self) -> float:
        return max(self.mean_layer[1], self.tail_layer[1], self.head_layer[1])

    def describe(self) -> str:
        fixed = (
            "Bunkers et al. (2000) internal-dynamics method: non-pressure-weighted "
            "mean wind %g-%g m AGL; shear vector from the %g-%g m mean wind to the "
            "%g-%g m mean wind; deviation %.1f m/s"
            % (self.mean_layer + self.tail_layer + self.head_layer + (self.deviation,))
        )
        if not self.effective:
            return fixed
        return (
            "Bunkers et al. (2014) effective-layer method where an effective inflow layer "
            "exists: non-pressure-weighted mean wind and bulk shear from the effective "
            "inflow base to %g%% of the depth up to the most-unstable EL, deviation %.1f "
            "m/s; elsewhere %s (BUNKERS_EFF_FLAG tells which)"
            % (100 * self.effective_fraction, self.deviation, fixed)
        )


def layer_tag(top: float, pad: bool = False) -> str:
    """Short tag of a 0-``top`` layer used in variable names.

    Whole kilometres give ``'1'`` (or ``'01'`` with ``pad``); anything else
    gives metres, e.g. ``'500M'``.
    """
    km = top / 1000.0
    if abs(km - round(km)) < 1e-9:
        return f"{int(round(km)):02d}" if pad else f"{int(round(km))}"
    return f"{int(round(top))}M"


#: effective-layer outputs of :func:`calc_kinematics`, in kernel order
EFFECTIVE_NAMES = ("ESRH_RM", "ESRH_LM", "EBWD", "E_SWV", "E_SWNE")


def kinematic_names(storm_motion="bunkers", srh_layers=(500.0, 1000.0, 3000.0),
                    shear_layers=(1000.0, 3000.0, 6000.0), swv_layers=(1000.0,),
                    effective=False):
    """Names of the fields returned by :func:`calc_kinematics`, in kernel order."""
    sm = StormMotion.from_any(storm_motion)
    if sm.effective:
        mean = "BUNKERS_MEAN"
    else:
        mean = f"MEAN{layer_tag(sm.mean_layer[1], pad=True)}"
    names = [f"{mean}_U", f"{mean}_V", "BUNKERS_RM_U", "BUNKERS_RM_V",
             "BUNKERS_LM_U", "BUNKERS_LM_V", "CRITICAL_ANGLE"]
    for top in srh_layers:
        names += [f"SRH{layer_tag(top)}_RM", f"SRH{layer_tag(top)}_LM"]
    for top in shear_layers:
        names.append(f"S{layer_tag(top, pad=True)}")
    for top in swv_layers:
        names += [f"SWV{layer_tag(top, pad=True)}", f"SWNESS{layer_tag(top, pad=True)}"]
    if effective:
        names += list(EFFECTIVE_NAMES)
    if sm.effective:
        names.append("BUNKERS_EFF_FLAG")
    return names


# ---------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _layer_mean(a, i0, i1):
    """Trapezoidal mean of ``a`` between grid indices ``i0`` and ``i1``."""
    if i1 <= i0:
        return a[i0]
    s = 0.5 * (a[i0] + a[i1])
    for k in range(i0 + 1, i1):
        s += a[k]
    return s / (i1 - i0)


@njit(cache=True, nogil=True)
def _at(a, h, dz, nz):
    """Linear interpolation of the uniform-grid profile ``a`` to height ``h``."""
    x = h / dz
    if not (x >= 0.0) or x > nz - 1:
        return np.nan
    k = int(x)
    if k >= nz - 1:
        return a[nz - 1]
    return a[k] + (a[k + 1] - a[k]) * (x - k)


@njit(cache=True, nogil=True)
def _mean_exact(a, h0, h1, dz, nz):
    """Height-weighted mean of ``a`` between h0 and h1 (exact trapezoid)."""
    if not (h1 > h0):
        return _at(a, h0, dz, nz)
    k0 = int(math.floor(h0 / dz)) + 1
    k1 = int(math.ceil(h1 / dz)) - 1
    za = h0
    aa = _at(a, h0, dz, nz)
    s = 0.0
    for k in range(k0, k1 + 1):
        zb = k * dz
        s += 0.5 * (aa + a[k]) * (zb - za)
        za = zb
        aa = a[k]
    s += 0.5 * (aa + _at(a, h1, dz, nz)) * (h1 - za)
    return s / (h1 - h0)


@njit(cache=True, nogil=True)
def _srh_exact(u, v, h0, h1, cu, cv, dz, nz):
    """SRH between h0 and h1 of the piecewise-linear profile (exact end points)."""
    if not (h1 > h0):
        return 0.0
    k0 = int(math.floor(h0 / dz)) + 1
    k1 = int(math.ceil(h1 / dz)) - 1
    ua = _at(u, h0, dz, nz) - cu
    va = _at(v, h0, dz, nz) - cv
    s = 0.0
    for k in range(k0, k1 + 1):
        ub = u[k] - cu
        vb = v[k] - cv
        s += ub * va - ua * vb
        ua = ub
        va = vb
    ub = _at(u, h1, dz, nz) - cu
    vb = _at(v, h1, dz, nz) - cv
    s += ub * va - ua * vb
    return s


@njit(cache=True, nogil=True)
def _swv_layer(u, v, i0, i1, rm_u, rm_v, dz, nz):
    """Mean streamwise vorticity (1/s) and streamwiseness (%) on grid points i0..i1.

    Centred differences; for layers spanning at least two grid intervals the
    end points reuse the nearest interior derivative (the convention of the
    fixed 0-x km layers), thinner layers use the full-profile derivative.
    """
    sum_swv = 0.0
    n_swv = 0
    sum_frac = 0.0
    n_frac = 0
    for k in range(i0, i1 + 1):
        if i1 - i0 >= 2:
            kc = min(max(k, i0 + 1), i1 - 1)
        else:
            kc = min(max(k, 1), nz - 2)
        dudz = (u[kc + 1] - u[kc - 1]) / (2.0 * dz)
        dvdz = (v[kc + 1] - v[kc - 1]) / (2.0 * dz)
        omega_x = -dvdz
        omega_y = dudz
        u_sr = u[k] - rm_u
        v_sr = v[k] - rm_v
        spd = math.hypot(u_sr, v_sr)
        if spd > 0.0:
            sw = abs((u_sr * omega_x + v_sr * omega_y) / spd)
            sum_swv += sw
            n_swv += 1
            omega = math.hypot(omega_x, omega_y)
            if omega > 0.0:
                sum_frac += sw / omega
                n_frac += 1
    swv = sum_swv / n_swv if n_swv > 0 else np.nan
    swness = 100.0 * sum_frac / n_frac if n_frac > 0 else np.nan
    return swv, swness


@njit(cache=True, nogil=True)
def _kin_point(i, z2d, u2d, v2d, us, vs, eb, et, el, nz, nz_req, dz, bunk_idx, deviation,
               i500, srh_idx, shr_idx, swv_idx, do_eff, eff_motion, eff_frac, out):
    nout = out.shape[1]
    for j in range(nout):
        out[i, j] = np.nan
    if not (np.isfinite(us[i]) and np.isfinite(vs[i])):
        return

    # ---- gather the levels above ground, surface wind first ----
    nlev = z2d.shape[1]
    zc = np.empty(nlev + 1)
    uc = np.empty(nlev + 1)
    vc = np.empty(nlev + 1)
    zc[0] = 0.0
    uc[0] = us[i]
    vc[0] = vs[i]
    n = 1
    for k in range(nlev):
        zk = z2d[i, k]
        uk = u2d[i, k]
        vk = v2d[i, k]
        if np.isfinite(zk) and np.isfinite(uk) and np.isfinite(vk) and zk > 0.0:
            zc[n] = zk
            uc[n] = uk
            vc[n] = vk
            n += 1
    if n < 3:
        return

    # ---- sort by height (insertion sort: input is (nearly) sorted) ----
    for a in range(2, n):
        za = zc[a]
        ua = uc[a]
        va = vc[a]
        b = a - 1
        while b >= 1 and zc[b] > za:
            zc[b + 1] = zc[b]
            uc[b + 1] = uc[b]
            vc[b + 1] = vc[b]
            b -= 1
        zc[b + 1] = za
        uc[b + 1] = ua
        vc[b + 1] = va

    if zc[n - 1] < (nz_req - 1) * dz:
        return

    # ---- linear interpolation to the uniform grid (NaN above the data) ----
    u = np.empty(nz)
    v = np.empty(nz)
    u[0] = uc[0]
    v[0] = vc[0]
    j = 0
    for k in range(1, nz):
        zt = k * dz
        if zt > zc[n - 1]:
            u[k] = np.nan
            v[k] = np.nan
            continue
        while j < n - 2 and zc[j + 1] < zt:
            j += 1
        dzc = zc[j + 1] - zc[j]
        if dzc > 0.0:
            f = (zt - zc[j]) / dzc
        else:
            f = 0.0
        u[k] = uc[j] + (uc[j + 1] - uc[j]) * f
        v[k] = vc[j] + (vc[j + 1] - vc[j]) * f

    # ---- Bunkers storm motion ----
    used_eff = False
    if eff_motion:
        # Bunkers et al. (2014): effective inflow base to a fraction of the
        # depth up to the most-unstable EL
        hb = eb[i]
        he = el[i]
        if np.isfinite(hb) and np.isfinite(he) and he > hb:
            htop = hb + eff_frac * (he - hb)
            mean_u = _mean_exact(u, hb, htop, dz, nz)
            mean_v = _mean_exact(v, hb, htop, dz, nz)
            shr_u = _at(u, htop, dz, nz) - _at(u, hb, dz, nz)
            shr_v = _at(v, htop, dz, nz) - _at(v, hb, dz, nz)
            used_eff = (np.isfinite(mean_u) and np.isfinite(mean_v)
                        and np.isfinite(shr_u) and np.isfinite(shr_v))
    if not used_eff:
        mean_u = _layer_mean(u, bunk_idx[0], bunk_idx[1])
        mean_v = _layer_mean(v, bunk_idx[0], bunk_idx[1])
        shr_u = _layer_mean(u, bunk_idx[4], bunk_idx[5]) - _layer_mean(u, bunk_idx[2], bunk_idx[3])
        shr_v = _layer_mean(v, bunk_idx[4], bunk_idx[5]) - _layer_mean(v, bunk_idx[2], bunk_idx[3])
    shr_mag = math.hypot(shr_u, shr_v)
    if shr_mag > 0.0:
        # (S x k) / |S| = (S_v, -S_u) / |S|
        dev_u = deviation * shr_v / shr_mag
        dev_v = -deviation * shr_u / shr_mag
    else:
        dev_u = 0.0
        dev_v = 0.0
    rm_u = mean_u + dev_u
    rm_v = mean_v + dev_v
    lm_u = mean_u - dev_u
    lm_v = mean_v - dev_v

    out[i, 0] = mean_u
    out[i, 1] = mean_v
    out[i, 2] = rm_u
    out[i, 3] = rm_v
    out[i, 4] = lm_u
    out[i, 5] = lm_v

    # ---- critical angle: 0-500 m shear vs. surface storm-relative inflow ----
    if i500 > 0:
        ax = u[i500] - u[0]
        ay = v[i500] - v[0]
        bx = rm_u - u[0]
        by = rm_v - v[0]
        den = math.hypot(ax, ay) * math.hypot(bx, by)
        if den > 0.0:
            c = (ax * bx + ay * by) / den
            c = max(-1.0, min(1.0, c))
            out[i, 6] = math.degrees(math.acos(c))

    col = 7
    # ---- storm relative helicity ----
    for m in range(srh_idx.shape[0]):
        s_rm = 0.0
        s_lm = 0.0
        for k in range(srh_idx[m]):
            s_rm += (u[k + 1] - rm_u) * (v[k] - rm_v) - (u[k] - rm_u) * (v[k + 1] - rm_v)
            s_lm += (u[k + 1] - lm_u) * (v[k] - lm_v) - (u[k] - lm_u) * (v[k + 1] - lm_v)
        out[i, col] = s_rm
        out[i, col + 1] = s_lm
        col += 2

    # ---- bulk wind difference ----
    for m in range(shr_idx.shape[0]):
        out[i, col] = math.hypot(u[shr_idx[m]] - u[0], v[shr_idx[m]] - v[0])
        col += 1

    # ---- streamwise vorticity and streamwiseness (right mover) ----
    for m in range(swv_idx.shape[0]):
        it = swv_idx[m]
        sum_swv = 0.0
        n_swv = 0
        sum_frac = 0.0
        n_frac = 0
        for k in range(it + 1):
            # centred differences; the end points reuse the nearest
            # interior derivative
            kc = min(max(k, 1), it - 1)
            dudz = (u[kc + 1] - u[kc - 1]) / (2.0 * dz)
            dvdz = (v[kc + 1] - v[kc - 1]) / (2.0 * dz)
            omega_x = -dvdz
            omega_y = dudz
            u_sr = u[k] - rm_u
            v_sr = v[k] - rm_v
            spd = math.hypot(u_sr, v_sr)
            if spd > 0.0:
                sw = abs((u_sr * omega_x + v_sr * omega_y) / spd)
                sum_swv += sw
                n_swv += 1
                omega = math.hypot(omega_x, omega_y)
                if omega > 0.0:
                    sum_frac += sw / omega
                    n_frac += 1
        if n_swv > 0:
            out[i, col] = sum_swv / n_swv
        if n_frac > 0:
            out[i, col + 1] = 100.0 * sum_frac / n_frac
        col += 2

    # ---- effective inflow layer ----
    if do_eff:
        hb = eb[i]
        ht = et[i]
        if np.isfinite(hb) and np.isfinite(ht):
            out[i, col] = _srh_exact(u, v, hb, ht, rm_u, rm_v, dz, nz)
            out[i, col + 1] = _srh_exact(u, v, hb, ht, lm_u, lm_v, dz, nz)
            he = el[i]
            if np.isfinite(he) and he > hb:
                # bulk wind difference: effective base to half of the MU EL depth
                htop = hb + 0.5 * (he - hb)
                out[i, col + 2] = math.hypot(_at(u, htop, dz, nz) - _at(u, hb, dz, nz),
                                             _at(v, htop, dz, nz) - _at(v, hb, dz, nz))
            i0 = min(int(round(hb / dz)), nz - 1)
            i1 = min(max(int(round(ht / dz)), i0), nz - 1)
            swv, swness = _swv_layer(u, v, i0, i1, rm_u, rm_v, dz, nz)
            out[i, col + 3] = swv
            out[i, col + 4] = swness
        else:
            # no effective inflow layer: ESRH and EBWD are zero (SPC convention)
            out[i, col] = 0.0
            out[i, col + 1] = 0.0
            out[i, col + 2] = 0.0
        col += 5
    if eff_motion:
        out[i, col] = 1.0 if used_eff else 0.0


@njit(cache=True, nogil=True)
def _loop_kin_serial(z2d, u2d, v2d, us, vs, eb, et, el, nz, nz_req, dz, bunk_idx, deviation,
                     i500, srh_idx, shr_idx, swv_idx, do_eff, eff_motion, eff_frac, nout):
    out = np.empty((z2d.shape[0], nout), dtype=np.float64)
    for i in range(z2d.shape[0]):
        _kin_point(i, z2d, u2d, v2d, us, vs, eb, et, el, nz, nz_req, dz, bunk_idx, deviation,
                   i500, srh_idx, shr_idx, swv_idx, do_eff, eff_motion, eff_frac, out)
    return out


@njit(cache=True, nogil=True, parallel=True)
def _loop_kin_parallel(z2d, u2d, v2d, us, vs, eb, et, el, nz, nz_req, dz, bunk_idx, deviation,
                     i500, srh_idx, shr_idx, swv_idx, do_eff, eff_motion, eff_frac, nout):
    out = np.empty((z2d.shape[0], nout), dtype=np.float64)
    for i in prange(z2d.shape[0]):
        _kin_point(i, z2d, u2d, v2d, us, vs, eb, et, el, nz, nz_req, dz, bunk_idx, deviation,
                   i500, srh_idx, shr_idx, swv_idx, do_eff, eff_motion, eff_frac, out)
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def _grid_index(height, dz, what):
    idx = height / dz
    if height < 0 or abs(idx - round(idx)) > 1e-9:
        raise ValueError(f"{what} ({height} m) must be a non-negative multiple of dz ({dz} m)")
    return int(round(idx))


#: top (m AGL) of the uniform profile used for the effective-layer fields;
#: EBWD reaches half and the effective Bunkers layer 65% of the MU EL height
EFFECTIVE_PROFILE_TOP = 16000.0


def _prepare(storm_motion, srh_layers, shear_layers, swv_layers, dz, effective=False):
    sm = StormMotion.from_any(storm_motion)
    dz = float(dz)
    if dz <= 0:
        raise ValueError("dz must be positive")
    srh_layers = tuple(float(x) for x in srh_layers)
    shear_layers = tuple(float(x) for x in shear_layers)
    swv_layers = tuple(float(x) for x in swv_layers)
    for name, layers in (("srh_layers", srh_layers), ("shear_layers", shear_layers),
                         ("swv_layers", swv_layers)):
        if any(x <= 0 for x in layers):
            raise ValueError(f"{name} must contain positive layer tops (m AGL)")
    for lay in (sm.mean_layer, sm.tail_layer, sm.head_layer):
        if not lay[1] >= lay[0] >= 0:
            raise ValueError(f"invalid storm-motion layer {lay}")

    bunk_idx = np.array(
        [_grid_index(x, dz, "storm-motion layer bound")
         for lay in (sm.mean_layer, sm.tail_layer, sm.head_layer) for x in lay],
        dtype=np.int64)
    srh_idx = np.array([_grid_index(x, dz, "SRH layer top") for x in srh_layers], dtype=np.int64)
    shr_idx = np.array([_grid_index(x, dz, "shear layer top") for x in shear_layers], dtype=np.int64)
    swv_idx = np.array([_grid_index(x, dz, "streamwise-vorticity layer top") for x in swv_layers],
                       dtype=np.int64)
    if np.any(swv_idx < 2):
        raise ValueError("streamwise-vorticity layers must span at least 2*dz")
    if not 0.0 < sm.effective_fraction <= 1.0:
        raise ValueError("effective_fraction must be in (0, 1]")
    top = max((sm.top,) + srh_layers + shear_layers + swv_layers)
    nz_req = _grid_index(top, dz, "profile top") + 1
    nz = nz_req
    if effective or sm.effective:
        nz = max(nz_req, int(math.ceil(EFFECTIVE_PROFILE_TOP / dz)) + 1)
    i500 = int(round(500.0 / dz)) if abs(500.0 / dz - round(500.0 / dz)) < 1e-9 else -1
    if i500 > nz_req - 1:
        i500 = -1
    names = kinematic_names(sm, srh_layers, shear_layers, swv_layers, effective=effective)
    return sm, dz, nz, nz_req, bunk_idx, i500, srh_idx, shr_idx, swv_idx, names


def _kinematics_numpy(z_agl, u, v, us, vs, eb, et, el, *, nz, nz_req, dz, bunk_idx, deviation,
                      i500, srh_idx, shr_idx, swv_idx, do_eff, eff_motion, eff_frac, nout,
                      _parallel=True):
    z_agl, u, v = (np.asarray(a) for a in (z_agl, u, v))
    us, vs = np.asarray(us), np.asarray(vs)
    if not (z_agl.shape == u.shape == v.shape):
        raise ValueError("z_agl, u and v must have the same shape")
    if us.shape != u.shape[:-1] or vs.shape != u.shape[:-1]:
        raise ValueError("us and vs must have the shape of u without its last (vertical) axis")
    nlev = u.shape[-1]
    ncol = int(np.prod(u.shape[:-1], dtype=np.int64))
    as2d = lambda a: np.ascontiguousarray(a.reshape(-1, nlev), dtype=np.float64)  # noqa: E731

    def as1d(a):
        return np.ascontiguousarray(np.broadcast_to(np.asarray(a, dtype=np.float64),
                                                    u.shape[:-1]).reshape(ncol))

    out = run_loop(_loop_kin_parallel, _loop_kin_serial, _parallel,
                   as2d(z_agl), as2d(u), as2d(v), as1d(us), as1d(vs), as1d(eb), as1d(et),
                   as1d(el), nz, nz_req, dz, bunk_idx, float(deviation), i500, srh_idx,
                   shr_idx, swv_idx, bool(do_eff), bool(eff_motion), float(eff_frac), nout)
    return out.astype(np.float32).reshape(u.shape[:-1] + (nout,))


def calc_kinematics(z_agl, u, v, us, vs, *, storm_motion="bunkers",
                    srh_layers=(500.0, 1000.0, 3000.0),
                    shear_layers=(1000.0, 3000.0, 6000.0),
                    swv_layers=(1000.0,), dz=100.0,
                    eil_base=None, eil_top=None, el_height=None):
    r"""
    Storm motion, SRH, bulk shear and streamwise vorticity from wind profiles.

    Parameters
    ----------
    z_agl : array-like, shape ``(..., nlev)``
        Height above ground level (m) of every level.  Levels at or below
        the ground (``z_agl <= 0``) and non-finite values are ignored, so
        pressure-level data can be passed as they are.  Any vertical
        ordering is accepted.
    u, v : array-like, shape ``(..., nlev)``
        Wind components (m/s).
    us, vs : array-like, shape ``(...)``
        Near-surface (10 m) wind components (m/s), used as the wind at 0 m.
    storm_motion : {'bunkers', 'bunkers-shallow', 'bunkers-effective'} or StormMotion
        'bunkers' is the standard 0-6 km method of Bunkers et al. (2000);
        'bunkers-shallow' scales its layers down to 0-3 km;
        'bunkers-effective' is the effective-layer method of Bunkers et al.
        (2014), which needs ``eil_base`` and ``el_height`` and falls back to
        the 0-6 km method where there is no effective inflow layer.  Pass a
        :class:`StormMotion` for custom layers.
    srh_layers, shear_layers, swv_layers : sequence of float
        Layer tops (m AGL, layers start at the ground) for storm relative
        helicity, bulk wind difference and streamwise vorticity.
    dz : float
        Spacing (m) of the uniform height grid.  All layer bounds must be
        multiples of it.
    eil_base, eil_top : array-like, shape ``(...)``, optional
        Effective inflow layer bounds (m AGL; NaN where there is none), e.g.
        from :func:`xcaper.parcel.calc_parcel_diagnostics`.  Giving them adds
        the effective-layer outputs.
    el_height : array-like, shape ``(...)``, optional
        Equilibrium level (m AGL) of the most-unstable parcel; needed for
        EBWD and for ``storm_motion='bunkers-effective'``.

    Returns
    -------
    dict of str -> array
        float32 arrays of shape ``(...)`` (dask arrays for dask input):

        ``MEANxx_U/V``
            mean wind of the storm-motion mean layer (m/s);
        ``BUNKERS_RM_U/V``, ``BUNKERS_LM_U/V``
            right / left mover motion (m/s);
        ``SRH<layer>_RM``, ``SRH<layer>_LM``
            storm relative helicity (m2/s2), e.g. ``SRH1_RM``, ``SRH500M_RM``;
        ``S<layer>``
            bulk wind difference magnitude (m/s), e.g. ``S06``;
        ``SWV<layer>``
            layer-mean magnitude of the streamwise vorticity relative to the
            right mover (1/s), e.g. ``SWV01``;
        ``SWNESS<layer>``
            layer-mean streamwiseness, the share of the horizontal vorticity
            that is streamwise (%), e.g. ``SWNESS01``;
        ``CRITICAL_ANGLE``
            angle (degrees) between the 0-500 m shear vector and the surface
            storm-relative inflow of the right mover.
        ``ESRH_RM``, ``ESRH_LM`` (with ``eil_base``/``eil_top``)
            SRH of the effective inflow layer (m2/s2); 0 where there is no
            effective inflow layer.
        ``EBWD`` (with ``eil_base``/``eil_top``/``el_height``)
            effective bulk wind difference (m/s): from the effective inflow
            base to half of the depth up to the most-unstable EL; 0 where
            there is no effective inflow layer.
        ``E_SWV``, ``E_SWNE`` (with ``eil_base``/``eil_top``)
            streamwise vorticity (1/s) and streamwiseness (%) averaged over
            the effective inflow layer; NaN where there is none.
        ``BUNKERS_EFF_FLAG`` (``storm_motion='bunkers-effective'`` only)
            1 where the effective-layer method was used, 0 for the fallback.

        Columns whose profile does not reach the highest fixed layer are
        NaN.  The effective-layer fields use the profile up to 16 km AGL
        (or the top of the data, if lower).

    Notes
    -----
    SRH uses the discrete form

    .. math:: SRH = \sum_k (u_{k+1}-c_u)(v_k-c_v) - (u_k-c_u)(v_{k+1}-c_v)

    With :math:`\omega_h = (-\partial v/\partial z, \partial u/\partial z)`
    and the storm-relative wind :math:`V_{sr}`, the streamwise vorticity is
    :math:`\omega_s = |V_{sr} \cdot \omega_h| / |V_{sr}|` and the
    streamwiseness is :math:`\omega_s / |\omega_h|`; both are averaged over
    the grid points of the layer (centred differences, as in SounderPy).
    """
    do_eff = eil_base is not None or eil_top is not None
    if do_eff and (eil_base is None or eil_top is None):
        raise ValueError("give both eil_base and eil_top")
    sm, dz, nz, nz_req, bunk_idx, i500, srh_idx, shr_idx, swv_idx, names = _prepare(
        storm_motion, srh_layers, shear_layers, swv_layers, dz, effective=do_eff)
    if sm.effective and (eil_base is None or el_height is None):
        raise ValueError("storm_motion='bunkers-effective' needs eil_base, eil_top and el_height")
    kwargs = dict(nz=nz, nz_req=nz_req, dz=dz, bunk_idx=bunk_idx, deviation=sm.deviation,
                  i500=i500, srh_idx=srh_idx, shr_idx=shr_idx, swv_idx=swv_idx,
                  do_eff=do_eff, eff_motion=sm.effective, eff_frac=sm.effective_fraction,
                  nout=len(names))

    nan = np.float32(np.nan)
    eb = nan if eil_base is None else eil_base
    et = nan if eil_top is None else eil_top
    el = nan if el_height is None else el_height
    args = (z_agl, u, v, us, vs, eb, et, el)
    if any(isinstance(a, da.Array) for a in args):
        out = da.apply_gufunc(_kinematics_numpy, "(k),(k),(k),(),(),(),(),()->(n)", *args,
                              output_dtypes="f4", output_sizes={"n": len(names)},
                              vectorize=False, allow_rechunk=True, _parallel=False,
                              **kwargs)
    else:
        out = _kinematics_numpy(*args, **kwargs)
    return {name: out[..., j] for j, name in enumerate(names)}
