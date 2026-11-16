#!/usr/bin/env python3
"""
Process a folder of ERA5 files: one output file per time step.

Looks for pairs of pressure-level / single-level files named like the files
that ``xcaper.era5`` downloads (``era5_pl_<stamp>.nc`` and
``era5_sfc_<stamp>.nc``; one or several time steps per file), computes all
parameters with dask in parallel chunks, and writes one compressed NetCDF per
time step. Time steps whose output already exists are skipped, so the script
can be re-run after an interruption or when new files arrive.

    python batch_local_files.py era5_cache/
    python batch_local_files.py era5_cache/ --out params/ --storm-motion bunkers-shallow
    python batch_local_files.py era5_cache/ --area 22 110 4 135        # N W S E subset
    python batch_local_files.py era5_cache/ --workers 8

To fill the folder for a period first:

    xcaper-era5 --area 30 100 -10 145 --start 2026-08-08T00 --end 2026-08-08T23 \\
                --cache-dir era5_cache
"""
import argparse
import sys
import time
from pathlib import Path

import dask
import numpy as np

from xcaper import era5


def find_pairs(folder, pl_prefix="era5_pl_", sfc_prefix="era5_sfc_"):
    """Pressure-level files with their matching single-level file."""
    pairs, missing = [], []
    for pl in sorted(Path(folder).glob(f"{pl_prefix}*.nc")):
        sfc = pl.with_name(sfc_prefix + pl.name[len(pl_prefix):])
        (pairs if sfc.exists() else missing).append((pl, sfc))
    for pl, sfc in missing:
        print(f"  skipping {pl.name}: {sfc.name} not found")
    return pairs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", help="folder with era5_pl_*.nc and era5_sfc_*.nc files")
    ap.add_argument("--out", default="params", help="output folder (default: params/)")
    ap.add_argument("--storm-motion", default="bunkers",
                    choices=("bunkers", "bunkers-shallow", "bunkers-effective"))
    ap.add_argument("--area", nargs=4, type=float, metavar=("N", "W", "S", "E"),
                    help="compute only this box")
    ap.add_argument("--workers", type=int, default=None,
                    help="dask threads (default: all cores)")
    ap.add_argument("--overwrite", action="store_true", help="recompute existing outputs")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    box = dict(zip(("north", "west", "south", "east"), args.area)) if args.area else {}
    tag = {"bunkers": "B06", "bunkers-shallow": "B03", "bunkers-effective": "BEFF"}[args.storm_motion]

    pairs = find_pairs(args.folder)
    if not pairs:
        sys.exit(f"No era5_pl_*.nc / era5_sfc_*.nc pairs in {args.folder}")
    print(f"{len(pairs)} file pair(s) in {args.folder}; storm motion '{args.storm_motion}'")

    n_done = n_skipped = 0
    t_start = time.perf_counter()
    # dask runs the chunks on a thread pool; numba / Fortran release the GIL
    with dask.config.set(scheduler="threads", num_workers=args.workers):
        for pl, sfc in pairs:
            ds_pl, ds_sfc = era5.open_files(pl, sfc)
            # one lazy (dask) computation for all time steps of the file
            params = era5.compute_severe_parameters(ds_pl, ds_sfc, chunks="auto",
                                                    storm_motion=args.storm_motion, **box)
            for t in params["time"].values:
                stamp = np.datetime64(t, "h").astype(object)
                target = out / f"ERA5_Severe_Params_{tag}_{stamp:%Y%m%d_%H}.nc"
                if target.exists() and not args.overwrite:
                    n_skipped += 1
                    continue
                tic = time.perf_counter()
                step = params.sel(time=[t]).load()      # compute this time step
                era5.save(step, target)
                n_done += 1
                print(f"  {stamp:%Y-%m-%d %H} UTC -> {target.name} "
                      f"({time.perf_counter() - tic:.1f} s, max MU_CAPE "
                      f"{float(step['MU_CAPE'].max()):.0f} J/kg)")
            ds_pl.close()
            ds_sfc.close()

    print(f"\nDone: {n_done} written, {n_skipped} already present, "
          f"{time.perf_counter() - t_start:.0f} s in total. Outputs in {out}/")
    if n_done + n_skipped > 1:
        print(f"Open them together with: xarray.open_mfdataset('{out}/ERA5_Severe_Params_{tag}_*.nc')")


if __name__ == "__main__":
    main()
