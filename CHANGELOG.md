# Changelog

## 1.0.0 (first release of xcaper)

xcaper is a modernised and extended derivative of xcape 0.1.4 (Lepore, Allen
and Abernathey; doi:10.5281/zenodo.5270332). This entry lists every change
with respect to xcape 0.1.4.

### Packaging and build

* Package renamed to **xcaper** (`import xcaper`, command `xcaper-era5`).
* Python >= 3.10 (tested 3.10-3.14; the 3.15 pre-release passes),
  NumPy 1.24 - 2.x.
* `numpy.distutils` / `setup.py` / `versioneer` replaced by `pyproject.toml` +
  `meson-python`; the f2py wrappers are generated from the `.pyf` files at
  build time. (xcape 0.1 cannot be built on Python >= 3.12, where
  `numpy.distutils` no longer exists.)
* The Fortran backend is optional (`-Csetup-args=-Dfortran=disabled`): without
  it everything runs on numba with the same results.
* Version from package metadata (`xcaper.__version__`).

### Fortran

* The four kernels (`getcape`, `stdheight`, Bunkers, SRH) existed twice (model
  levels and pressure levels). They now live once in the module
  `xcaper_kernels.f90` (explicit kinds, `pure` procedures, `intent`s); the
  eight extension modules are thin drivers with the same names and Python
  signatures as in xcape 0.1.
* Fixed: the Bunkers accumulators (mean wind, shear sums) were never
  initialised.
* Fixed: `STOP` / `PRINT` inside the kernels (could kill the interpreter or
  garble output from threads) replaced by a zero return; `kmax` is initialised
  (NaN profiles could index out of bounds); the mixed-layer integration cannot
  read past the last level; `start_3d` is an integer everywhere.
* Every driver is `threadsafe` (releases the GIL).
* Legacy intrinsics removed (`alog`, `float`, `real*8`); `-fno-range-check` is
  no longer needed.

### Core API (`xcaper.core`)

* `calc_cape` / `calc_srh`: same signatures and results as xcape 0.1. New
  `method="auto"` (default) and `method="numba"`. Keyword defaults now work
  (xcape 0.1 raised `KeyError` when `source` or `vertical_lev` were omitted).
* Fixed: dask + pressure levels. The gufunc signature declared seven inputs for
  six arrays, so every dask call with `vertical_lev="pressure"` failed on
  current dask. The vertical dimension is now rechunked automatically.
* New `calc_lcl`.
* `xcaper.xarray` is a working xarray API (in xcape 0.1 it was an unfinished
  copy of xhistogram that could not be imported).

### New numba backend

* Ports of all four kernels; parallel over grid points for numpy inputs,
  serial and GIL-free inside dask tasks. Parallel numba loops are serialised
  with a lock, so concurrent calls from several threads are safe.

### New parameters

* `xcaper.kinematics.calc_kinematics`: storm motion (Bunkers standard 0-6 km,
  shallow 0-3 km, effective-layer Bunkers et al. 2014 with 0-6 km fallback, or
  custom layers), SRH for any layers, bulk wind difference, critical angle,
  layer-mean streamwise vorticity and streamwiseness; with an effective inflow
  layer also ESRH, EBWD, E_SWV and E_SWNE.
* `xcaper.parcel.calc_parcel_diagnostics`: LFC and equilibrium level of the
  surface, mixed-layer and most-unstable parcels (height and pressure); the
  effective inflow layer (Thompson et al. 2007; searched on a profile refined
  to 10 hPa by default, `eil_dp`); DCAPE (MetPy recipe). The ascent is the
  same as `calc_cape`'s, so CAPE and CIN are identical.
* `xcaper.thermo`: dewpoint from specific humidity, lapse rates of the virtual
  temperature (SHARPpy / SPC convention; `virtual_lapse_rate=False` for the
  temperature) over height and pressure layers.
* `xcaper.composites`: SCP, STP (fixed layer), STP with CIN term, ESCP, ESTP
  (0 when the effective layer is elevated), EHI, WMAXSHEAR.
* `xcaper.severe.calc_severe`: all of the above in one call, 64 output
  variables with units and descriptions.

### ERA5 workflow (`xcaper.era5`, command `xcaper-era5`)

* Builds Climate Data Store requests for a bounding box and times, downloads
  and caches the pressure-level and single-level files (refusing cached files
  that do not cover the request), combines them into columns, computes the
  parameters (optionally in parallel dask chunks) and writes compressed,
  CF-style NetCDF atomically.

### Validation

ERA5 2026-08-08 14 UTC, 29,141 columns over the Philippine region:

* Fortran kernels reproduce xcape 0.1.4 bit for bit for CAPE, CIN, heights
  and Bunkers motion (SRH to 1e-13); the original 0.1.4 test suite passes.
* numba and Fortran CAPE agree to 0.25 J/kg; CIN agrees to 0.1 J/kg except
  in 2 columns where the parcel buoyancy touches zero within single-precision
  resolution (see README, "Backends and parallelism").
* Effective inflow layer identical to a brute-force search (full getcape
  ascent from every level) in 300/300 columns, for both search modes.
* DCAPE within 0.3% (median) of MetPy `downdraft_cape`.
* LFC / EL within 0.24 / 0.92 hPa (median) of the same rule applied to MetPy's
  virtual-temperature parcel.
* ESRH, EBWD, E_SWV, E_SWNE and the effective Bunkers motion identical to an
  independent numpy implementation (<= 1e-5); ESRH agrees with MetPy
  `storm_relative_helicity` on the raw levels to well under 1 m2/s2 (median).
* Lapse rates match SounderPy (0-3 km 5.81 K/km on the test column).
* Parcel-code hydrostatic heights match ERA5 geopotential heights within 3 m.
* The new ERA5 workflow reproduces the stand-alone scripts it replaces
  (`xcape_3km.py`, `ERA5_Severe_Params_with_SWV.py`) to float32 round-off.
* Identical results on Python 3.10-3.15 and with the oldest and newest
  supported dependency versions.

Bugs found and fixed during development: the LFC (and EL) of saturated or
supersaturated source parcels was lost through `exp(log(p))` round-off (about
2,000 ERA5 grid boxes in the test case have Td > T).
