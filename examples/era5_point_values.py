#!/usr/bin/env python3
"""
Severe-weather parameters at chosen locations from ERA5, as a table.

For every point the nearest ERA5 grid point is used (0.25 degree grid). Only a
small box around the points is computed, so this is quick even when the ERA5
files cover a large domain. The table is printed and saved as CSV.

    python era5_point_values.py                                    # download from the CDS
    python era5_point_values.py era5_pl_2026080814.nc era5_sfc_2026080814.nc

Edit POINTS, TIMES and STORM_MOTION below. Downloading needs the one-time
Copernicus Climate Data Store set-up described in the README.
"""
import sys

import numpy as np
import pandas as pd

from xcaper import era5
from xcaper.severe import variable_attrs

# ----------------------------------------------------------------------------
# What to compute
# ----------------------------------------------------------------------------
POINTS = {                                  # name: (latitude, longitude)
    "Point A": (10.772, 122.579),
    "Point B": (14.600, 121.000),
    "Point C": (7.100, 125.600),
}
TIMES = ["2026-08-08T14:00"]                # UTC (used when downloading)
STORM_MOTION = "bunkers"                    # or 'bunkers-shallow', 'bunkers-effective'
MARGIN = 0.5                                # degrees around the points
OUTPUT_CSV = "era5_point_values.csv"

# Rows of the printed table (every one of the 64 variables goes to the CSV)
SHOW = ["SFC_CAPE", "ML_CAPE", "MU_CAPE", "SFC_CIN", "ML_CIN", "ML_LCL", "MU_EL",
        "DCAPE", "LR03", "LR75", "S06", "SRH1_RM", "SRH3_RM", "EIL_BASE", "EIL_TOP",
        "ESRH_RM", "EBWD", "SWV01", "SWNESS01", "SCP", "STP", "ESCP", "ESTP"]

PL_FILE = sys.argv[1] if len(sys.argv) > 2 else None
SFC_FILE = sys.argv[2] if len(sys.argv) > 2 else None


def main():
    lats = [lat for lat, _ in POINTS.values()]
    lons = [lon for _, lon in POINTS.values()]
    box = dict(north=max(lats) + MARGIN, south=min(lats) - MARGIN,
               west=min(lons) - MARGIN, east=max(lons) + MARGIN)

    if PL_FILE:
        ds_pl, ds_sfc = era5.open_files(PL_FILE, SFC_FILE)
        ds = era5.compute_severe_parameters(ds_pl, ds_sfc, storm_motion=STORM_MOTION, **box)
    else:
        ds = era5.era5_severe_parameters(TIMES, storm_motion=STORM_MOTION,
                                         cache_dir="era5_cache", **box)
    ds = ds.load()

    rows = []
    for name, (lat, lon) in POINTS.items():
        at = ds.sel(lat=lat, lon=lon, method="nearest")
        for t in np.atleast_1d(at.time.values):
            col = at.sel(time=t)
            row = {"point": name, "lat": lat, "lon": lon,
                   "grid_lat": float(col.lat), "grid_lon": float(col.lon),
                   "time": pd.Timestamp(t).strftime("%Y-%m-%d %H UTC")}
            row.update({k: float(col[k]) for k in ds.data_vars})
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_CSV, index=False, float_format="%.6g")

    # printed view: one column per point (and time), one row per parameter
    view = table.set_index(["point", "time"])[SHOW].T
    view.columns = [f"{p} {t[-6:-4]}Z" if table.time.nunique() > 1 else p for p, t in view.columns]
    units = [variable_attrs(k, STORM_MOTION).get("units", "") for k in SHOW]
    view.insert(0, "units", ["" if u == "1" else u for u in units])
    print("\nNearest grid points:")
    for _, r in table.drop_duplicates("point").iterrows():
        print(f"  {r.point:10s} {r.lat:7.3f}N {r.lon:8.3f}E  ->  {r.grid_lat:6.2f}N {r.grid_lon:7.2f}E")
    print(f"\nStorm motion: {STORM_MOTION}; CIN positive = inhibition; "
          "EIL NaN = no effective inflow layer\n")
    fmt = lambda x: f"{x:.0f}" if abs(x) >= 1000 else f"{x:.4g}"
    with pd.option_context("display.float_format", fmt,
                           "display.width", 160, "display.max_columns", 20):
        print(view)
    print(f"\nAll {len(ds.data_vars)} parameters saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
