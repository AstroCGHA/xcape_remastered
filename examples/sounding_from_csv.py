#!/usr/bin/env python3
"""
Severe-weather parameters of a single sounding read from a CSV file.

Works for a radiosonde ascent, a model column, or any profile you can write
as a table. No download is needed: run it as it is to use the ERA5 sample
column shipped in ``examples/data``, or pass your own file:

    python sounding_from_csv.py
    python sounding_from_csv.py my_sounding.csv
    python sounding_from_csv.py my_sounding.csv --storm-motion bunkers-shallow
    python sounding_from_csv.py my_sounding.csv --knots      # wind speed in kt

CSV format (comma separated, lines starting with '#' are ignored):

    pressure_hPa,height_m,temperature_C,dewpoint_C,wind_dir_deg,wind_speed_ms
    998.40,92.3,27.08,24.79,228,6.6        <- first row: the surface (2 m / 10 m)
    975.00,301.8,24.93,22.75,233,10.3
    ...                                     <- then upwards, pressure decreasing

* ``height_m`` may be above sea level or above ground: heights are taken
  relative to the first (surface) row.
* Wind direction is the meteorological convention (direction the wind blows
  from, degrees).
* Missing values can be left empty or written as NaN; such levels are dropped.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from xcaper.severe import calc_severe, variable_attrs

SAMPLE = Path(__file__).parent / "data" / "sounding_era5_10.75N_122.50E_2026080814.csv"

COLUMNS = ("pressure_hPa", "height_m", "temperature_C", "dewpoint_C",
           "wind_dir_deg", "wind_speed_ms")

# Output grouped for printing: (heading, [variables])
GROUPS = [
    ("Thermodynamics", ["SFC_CAPE", "SFC_CIN", "ML_CAPE", "ML_CIN", "MU_CAPE", "MU_CIN",
                        "SFC_LCL", "ML_LCL", "SFC_LFC", "SFC_EL", "MU_LFC", "MU_EL",
                        "DCAPE", "LR03", "LR36", "LR75"]),
    ("Effective inflow layer", ["EIL_BASE", "EIL_TOP", "EIL_BASE_P", "EIL_TOP_P"]),
    ("Storm motion and shear", ["BUNKERS_RM_U", "BUNKERS_RM_V", "S01", "S03", "S06",
                                "EBWD", "CRITICAL_ANGLE"]),
    ("Helicity and streamwise vorticity", ["SRH500M_RM", "SRH1_RM", "SRH3_RM", "ESRH_RM",
                                           "SWV01", "SWNESS01", "E_SWV", "E_SWNE"]),
    ("Composite indices", ["SCP", "STP", "STP_CIN", "ESCP", "ESTP", "EHI1", "EHI3",
                           "WMAXSHEAR"]),
]


def read_sounding(path, knots=False):
    """Read the CSV into a dict of 1-D float arrays (surface first)."""
    with open(path, newline="") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    header = [h.strip() for h in (reader.fieldnames or [])]
    missing = [c for c in COLUMNS if c not in header]
    if missing:
        sys.exit(f"{path}: missing column(s) {', '.join(missing)}; "
                 f"expected {', '.join(COLUMNS)}")

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan                       # empty cell, 'NaN', '-', ...

    rows = [{k.strip(): v for k, v in r.items() if k is not None} for r in reader]
    cols = {c: np.array([num(r.get(c)) for r in rows]) for c in COLUMNS}
    if knots:
        cols["wind_speed_ms"] = cols["wind_speed_ms"] * 0.514444
    return cols


def to_xcaper_inputs(s):
    """Split a sounding into the surface values and the levels above it.

    xcaper wants the surface (2 m temperature and dewpoint, 10 m wind,
    surface pressure) separately from the profile above the ground, with the
    vertical axis last. A single sounding is a field of one column, so every
    profile gets the shape (1, nlev) and every surface value the shape (1,).
    """
    sfc = {k: v[0] for k, v in s.items()}
    if not np.all(np.isfinite([sfc[c] for c in COLUMNS])):
        sys.exit("The first (surface) row must be complete.")

    above = {k: v[1:] for k, v in s.items()}
    ok = np.all([np.isfinite(above[c]) for c in COLUMNS], axis=0)
    ok &= above["pressure_hPa"] < sfc["pressure_hPa"]
    above = {k: v[ok] for k, v in above.items()}
    order = np.argsort(-above["pressure_hPa"])          # bottom to top
    above = {k: v[order] for k, v in above.items()}
    if above["pressure_hPa"].size < 5:
        sys.exit("Need at least 5 complete levels above the surface.")

    def uv(direction, speed):
        rad = np.radians(direction)
        return -speed * np.sin(rad), -speed * np.cos(rad)

    u, v = uv(above["wind_dir_deg"], above["wind_speed_ms"])
    us, vs = uv(sfc["wind_dir_deg"], sfc["wind_speed_ms"])
    col = lambda a: np.asarray(a, dtype=float)[None, :]   # (1, nlev)
    one = lambda x: np.array([x], dtype=float)            # (1,)
    return dict(
        p=col(above["pressure_hPa"]),
        t=col(above["temperature_C"]),
        td=col(above["dewpoint_C"]),
        u=col(u), v=col(v),
        z_agl=col(above["height_m"] - sfc["height_m"]),
        ps=one(sfc["pressure_hPa"]),
        ts=one(sfc["temperature_C"]),
        tds=one(sfc["dewpoint_C"]),
        us=one(us), vs=one(vs),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="?", default=str(SAMPLE), help="sounding CSV (default: sample)")
    ap.add_argument("--storm-motion", default="bunkers",
                    choices=("bunkers", "bunkers-shallow", "bunkers-effective"))
    ap.add_argument("--knots", action="store_true", help="wind speed column is in knots")
    args = ap.parse_args(argv)

    inputs = to_xcaper_inputs(read_sounding(args.csv, knots=args.knots))
    nlev = inputs["p"].shape[1]

    # vertical_lev='sigma': the pressure is given per column (shape (1, nlev)),
    # as for any sounding whose levels are not fixed pressure levels.
    res = calc_severe(**inputs, vertical_lev="sigma", storm_motion=args.storm_motion)

    print(f"\n{Path(args.csv).name}: surface {inputs['ps'][0]:.1f} hPa, "
          f"{inputs['ts'][0]:.1f} / {inputs['tds'][0]:.1f} C, {nlev} levels above ground, "
          f"storm motion '{args.storm_motion}'")
    for heading, names in GROUPS:
        print(f"\n{heading}")
        print("-" * len(heading))
        for name in names:
            if name not in res:
                continue
            value = float(np.asarray(res[name])[0])
            attrs = variable_attrs(name, args.storm_motion)
            units = attrs.get("units", "")
            units = "" if units == "1" else units
            if np.isnan(value):
                text = "       n/a"
            elif value != 0 and abs(value) < 0.1:
                text = f"{value:10.5f}"                 # vorticity (1/s)
            else:
                text = f"{value:10.2f}"
            print(f"  {name:16s}{text}  {units:9s} {attrs.get('long_name', '')}")
    print("\nCIN is positive (inhibition). NaN levels (n/a) mean the parcel has no "
          "LFC/EL, or there is no effective inflow layer.")
    return res


if __name__ == "__main__":
    main()
