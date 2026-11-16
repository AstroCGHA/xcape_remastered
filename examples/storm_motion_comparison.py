#!/usr/bin/env python3
"""
How the choice of Bunkers storm motion changes SRH and the composites.

Storm-relative quantities (SRH, ESRH, streamwise vorticity, SCP, STP, ESCP,
ESTP, ...) depend on the storm motion. This script computes ERA5 parameters
three times, once for each storm-motion method of xcaper, and compares them:

* 'bunkers'            standard 0-6 km method (Bunkers et al. 2000)
* 'bunkers-shallow'    the same, layers scaled to 0-3 km (low-topped or
                       tropical supercells in weak deep-layer shear)
* 'bunkers-effective'  effective-inflow-layer method (Bunkers et al. 2014),
                       falls back to 0-6 km where there is no effective layer

Thermodynamic parameters (CAPE, CIN, LCL, ...) and the bulk shear do not
depend on the storm motion and are identical in all three runs.

    python storm_motion_comparison.py                              # download from the CDS
    python storm_motion_comparison.py era5_pl_2026080814.nc era5_sfc_2026080814.nc

Prints domain statistics and saves a 3 x 3 figure (rows: 0-1 km SRH,
effective SRH, ESCP; columns: the three methods).
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from xcaper import era5

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.6,
})

# ----------------------------------------------------------------------------
# What to compute
# ----------------------------------------------------------------------------
TIME = "2026-08-08T14:00"                       # UTC (used when downloading)
AREA = dict(north=22, west=110, south=4, east=135)
METHODS = {"bunkers": "Bunkers 0–6 km",
           "bunkers-shallow": "Bunkers 0–3 km (shallow)",
           "bunkers-effective": "Bunkers effective (2014)"}
PL_FILE = sys.argv[1] if len(sys.argv) > 2 else None
SFC_FILE = sys.argv[2] if len(sys.argv) > 2 else None

# ----------------------------------------------------------------------------
# Compute: the same input, three storm motions
# ----------------------------------------------------------------------------
if PL_FILE:
    ds_pl, ds_sfc = era5.open_files(PL_FILE, SFC_FILE)
else:
    ds_pl, ds_sfc = era5.open_files(*era5.download([TIME], cache_dir="era5_cache", **AREA))

runs = {}
for method in METHODS:
    print(f"computing with storm_motion='{method}' ...")
    runs[method] = era5.compute_severe_parameters(ds_pl, ds_sfc, storm_motion=method,
                                                  chunks="auto", **AREA).isel(time=0).load()

first = runs["bunkers"]
when = first["time"].values.astype("datetime64[h]").astype(object)

# ----------------------------------------------------------------------------
# Domain statistics
# ----------------------------------------------------------------------------
has_layer = np.isfinite(first["EIL_BASE"].values)
print(f"\n{when:%Y-%m-%d %H} UTC, {first.lat.size} x {first.lon.size} grid points, "
      f"{has_layer.mean():.0%} with an effective inflow layer\n")

header = f"{'':28s}" + "".join(f"{m:>20s}" for m in METHODS)
print(header)
print("-" * len(header))


def row(label, values):
    print(f"{label:28s}" + "".join(f"{v:20.2f}" for v in values))


speed = [np.hypot(r["BUNKERS_RM_U"], r["BUNKERS_RM_V"]).values for r in runs.values()]
row("RM storm speed, median (m/s)", [np.nanmedian(s) for s in speed])
for name, label in [("SRH1_RM", "0-1 km SRH (RM)"), ("SRH3_RM", "0-3 km SRH (RM)"),
                    ("ESRH_RM", "Effective SRH (RM)")]:
    vals = [r[name].values[has_layer] if name == "ESRH_RM" else r[name].values
            for r in runs.values()]
    row(f"{label}, median", [np.nanmedian(v) for v in vals])
    row(f"{label}, 95th pct", [np.nanpercentile(v, 95) for v in vals])
for name in ("SCP", "STP", "ESCP", "ESTP"):
    row(f"{name}, % of points >= 1", [100 * np.nanmean(r[name].values >= 1) for r in runs.values()])
    row(f"{name}, maximum", [np.nanmax(r[name].values) for r in runs.values()])

if "BUNKERS_EFF_FLAG" in runs["bunkers-effective"]:
    flag = runs["bunkers-effective"]["BUNKERS_EFF_FLAG"].values
    print(f"\n'bunkers-effective': effective method at {np.nanmean(flag == 1):.1%} of the "
          f"points, 0-6 km fallback (no effective layer) at {np.nanmean(flag == 0):.1%}.")
same = all(np.allclose(r["MU_CAPE"], first["MU_CAPE"], equal_nan=True) for r in runs.values())
print(f"MU_CAPE identical in all runs (storm motion does not affect it): {same}")

# ----------------------------------------------------------------------------
# Figure: rows A-C (parameters), columns 1-3 (storm motions)
# ----------------------------------------------------------------------------
ROWS = [("SRH1_RM", np.arange(-100, 401, 50), "RdYlBu_r", r"0–1 km SRH (RM) [m$^2$ s$^{-2}$]"),
        ("ESRH_RM", np.arange(-100, 401, 50), "RdYlBu_r", r"Effective SRH (RM) [m$^2$ s$^{-2}$]"),
        ("ESCP", np.arange(0, 8.1, 1), "YlOrRd", "ESCP")]

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader

    shpreader.natural_earth(resolution="50m", category="physical", name="coastline")
    projection, transform = ccrs.PlateCarree(), dict(transform=ccrs.PlateCarree())
except Exception as err:                         # no cartopy, or no coastline data
    print(f"Plotting without coastlines ({type(err).__name__}: {err})")
    ccrs = None
    projection, transform = None, {}

lon, lat = first["lon"].values, first["lat"].values
extent = (AREA["west"], AREA["east"], AREA["south"], AREA["north"])
fig, axes = plt.subplots(3, 3, figsize=(11.0, 8.0), sharex=True, sharey=True,
                         subplot_kw=dict(projection=projection), layout="constrained")
for i, (name, levels, cmap, label) in enumerate(ROWS):
    norm = matplotlib.colors.BoundaryNorm(levels, plt.get_cmap(cmap).N, extend="both")
    for j, (method, title) in enumerate(METHODS.items()):
        ax = axes[i, j]
        mesh = ax.pcolormesh(lon, lat, runs[method][name].values, cmap=cmap, norm=norm,
                             shading="auto", rasterized=True, **transform)
        if ccrs is not None:
            ax.coastlines(linewidth=0.4, color="k")
            ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="grey")
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.gridlines(linestyle="--", linewidth=0.4, alpha=0.6)
        else:
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
            ax.grid(True)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            if i == len(ROWS) - 1:
                ax.set_xlabel("Longitude [°E]")
            if j == 0:
                ax.set_ylabel("Latitude [°N]")
        if i == 0:
            ax.set_title(title, fontsize=9)
        ax.text(0.97, 0.05, f"{'ABC'[i]}{j + 1}", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="black", linewidth=0.6, pad=2.5))
    cb = fig.colorbar(mesh, ax=axes[i, :], orientation="vertical", pad=0.01, fraction=0.03,
                      aspect=18, extend="both")
    cb.ax.tick_params(labelsize=7, direction="in")
    cb.set_label(label, fontsize=8)

fig.suptitle(f"ERA5 {when:%Y-%m-%d %H} UTC: effect of the storm-motion method", fontsize=10)
figfile = f"storm_motion_comparison_{when:%Y%m%d_%H}.png"
fig.savefig(figfile, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {figfile}")
