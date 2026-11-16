#!/usr/bin/env python3
"""
ERA5 severe-weather parameters with xcaper >= 1.0.

Successor of the stand-alone scripts ``ERA5_Severe_Params_with_SWV.py`` and
``xcape_3km.py``: the download, the combination of surface and
pressure-level data, the shallow (0-3 km) Bunkers storm motion, SRH, shear
and streamwise vorticity now live in the package, so this script only
states *what* to compute.

Run it as it is to download from the Copernicus Climate Data Store (see the
one-time set-up in ``xcaper/era5.py``), or point PL_FILE / SFC_FILE to ERA5
files that are already on disk:

    python era5_severe_params.py
    python era5_severe_params.py era5_pl_2026080814.nc era5_sfc_2026080814.nc
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from xcaper import era5

# ----------------------------------------------------------------------------
# Global figure style (set first so that it takes effect everywhere)
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.titlesize": 9,
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
TIMES = ["2026-08-08T14:00"]                    # UTC, on the hour; several are fine
AREA = dict(north=30, west=100, south=-10, east=145)
GRID = 0.25
PLOT_EXTENT = (110, 135, 4, 22)                 # lon_min, lon_max, lat_min, lat_max

# 'bunkers'          standard 0-6 km Bunkers et al. (2000) storm motion
# 'bunkers-shallow'  layers scaled to 0-3 km (low-topped / tropical supercells)
STORM_MOTION = "bunkers-shallow"

# Local ERA5 files (None -> download from the CDS into CACHE_DIR)
PL_FILE = sys.argv[1] if len(sys.argv) > 2 else None
SFC_FILE = sys.argv[2] if len(sys.argv) > 2 else None
CACHE_DIR = "."

OPTIONS = dict(
    storm_motion=STORM_MOTION,
    srh_layers=(500.0, 1000.0, 3000.0),         # m AGL
    shear_layers=(1000.0, 3000.0, 6000.0),
    swv_layers=(1000.0,),
    ml_depth=500.0,                             # m
    pinc=500.0,                                 # Pa
    method="auto",                              # 'fortran', 'numba' or 'auto'
)

# ----------------------------------------------------------------------------
# Compute
# ----------------------------------------------------------------------------
if PL_FILE and SFC_FILE:
    ds_pl, ds_sfc = era5.open_files(PL_FILE, SFC_FILE)
    params = era5.compute_severe_parameters(ds_pl, ds_sfc, chunks="auto", **OPTIONS).load()
else:
    params = era5.era5_severe_parameters(TIMES, **AREA, grid=GRID, cache_dir=CACHE_DIR,
                                         **OPTIONS)

first = params["time"].values[0].astype("datetime64[h]").astype(object)
tag = "B03" if STORM_MOTION == "bunkers-shallow" else "B06"
outfile = f"ERA5_Severe_Params_{tag}_{first:%Y%m%d_%H}.nc"
era5.save(params, outfile)
print(f"Successfully saved: {outfile}")

for name in ("SFC_CAPE", "ML_CAPE", "MU_CAPE", "DCAPE", "MU_EL", "SRH1", "SRH3", "S06", "SWV01",
             "SWNESS01", "EIL_BASE", "EIL_TOP", "ESRH", "EBWD", "E_SWV", "E_SWNE", "SCP", "STP",
             "STP_CIN", "ESCP", "ESTP"):
    field = params[name]
    print(f"{name:9s} min / max = {float(field.min()):10.4f} / {float(field.max()):10.4f}  "
          f"[{field.attrs['units']}]")

# ----------------------------------------------------------------------------
# Quick-look figure: rows A-D, panels labelled A1 ... D3
# ----------------------------------------------------------------------------
PANELS = [
    [("SFC_CAPE", np.arange(0, 4001, 500), "YlOrRd", r"SBCAPE [J kg$^{-1}$]"),
     ("ML_CAPE", np.arange(0, 2501, 250), "YlOrRd", r"MLCAPE [J kg$^{-1}$]"),
     ("DCAPE", np.arange(0, 1601, 200), "PuBu", r"DCAPE [J kg$^{-1}$]")],
    [("S06", np.arange(0, 33, 4), "YlOrRd", r"0–6 km shear [m s$^{-1}$]"),
     ("SRH1_RM", np.arange(-100, 401, 50), "RdYlBu_r", r"0–1 km SRH (RM) [m$^2$ s$^{-2}$]"),
     ("EBWD", np.arange(0, 33, 4), "YlOrRd", r"EBWD [m s$^{-1}$]")],
    [("SWV01", np.arange(0, 0.0221, 0.002), "YlOrRd", r"0–1 km streamwise vorticity [s$^{-1}$]"),
     ("E_SWNE", np.arange(0, 101, 10), "viridis", "Effective-layer streamwiseness [%]"),
     ("ESRH_RM", np.arange(-100, 401, 50), "RdYlBu_r", r"Effective SRH (RM) [m$^2$ s$^{-2}$]")],
    [("STP", np.arange(0, 3.1, 0.25), "YlOrRd", "STP (fixed layer)"),
     ("ESTP", np.arange(0, 3.1, 0.25), "YlOrRd", "ESTP (effective layer)"),
     ("ESCP", np.arange(0, 8.1, 1), "YlOrRd", "ESCP (effective layer)")],
]

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader

    # coastlines are downloaded on first use; without them (offline machine)
    # fall back to plain lon / lat axes instead of failing at savefig
    shpreader.natural_earth(resolution="50m", category="physical", name="coastline")
    shpreader.natural_earth(resolution="110m", category="cultural", name="admin_0_boundary_lines_land")
    projection, transform = ccrs.PlateCarree(), dict(transform=ccrs.PlateCarree())
except Exception as err:                         # no cartopy, or no coastline data
    print(f"Plotting without coastlines ({type(err).__name__}: {err})")
    ccrs = cfeature = None
    projection, transform = None, {}

snap = params.isel(time=0)
lon, lat = snap["lon"].values, snap["lat"].values

fig, axes = plt.subplots(4, 3, figsize=(11.0, 10.0), sharex=True, sharey=True,
                         subplot_kw=dict(projection=projection), layout="constrained")
fig.get_layout_engine().set(h_pad=0.03, w_pad=0.03, hspace=0.0, wspace=0.0)   # tight rows
for i, row in enumerate(PANELS):
    for j, (name, levels, cmap, title) in enumerate(row):
        ax = axes[i, j]
        norm = matplotlib.colors.BoundaryNorm(levels, plt.get_cmap(cmap).N, extend="both")
        mesh = ax.pcolormesh(lon, lat, snap[name].values, cmap=cmap, norm=norm,
                             shading="auto", rasterized=True, **transform)
        if ccrs is not None:
            ax.coastlines(linewidth=0.4, color="k")
            ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="grey")
            ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
            ax.gridlines(linestyle="--", linewidth=0.4, alpha=0.6)
        else:
            ax.set_xlim(PLOT_EXTENT[:2])
            ax.set_ylim(PLOT_EXTENT[2:])
            ax.grid(True)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            if i == len(PANELS) - 1:
                ax.set_xlabel("Longitude [°E]")
            if j == 0:
                ax.set_ylabel("Latitude [°N]")
        cb = fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.02, fraction=0.05,
                          aspect=22, extend="both")
        cb.ax.tick_params(labelsize=7, direction="in")
        cb.set_label(title, fontsize=8)
        ax.text(0.97, 0.05, f"{'ABCD'[i]}{j + 1}", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="black", linewidth=0.6, pad=2.5))

fig.suptitle(f"ERA5 severe-weather parameters, {first:%Y-%m-%d %H} UTC "
             f"(storm motion: {STORM_MOTION})", fontsize=10)
figfile = f"ERA5_Severe_Params_{tag}_{first:%Y%m%d_%H}.png"
fig.savefig(figfile, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved: {figfile}")
