# xcaper 1.0.1

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24273035.svg)](https://doi.org/10.5281/zenodo.24273035)

**xcaper** computes convective and severe-weather parameters for gridded data
(numpy, dask, xarray), with a Fortran and a numba backend and a ready-made
ERA5 workflow. It is a modernised and extended derivative of
[xcape](https://github.com/xgcm/xcape) 0.1.4 by Chiara Lepore, John Allen and
Ryan Abernathey ([doi:10.5281/zenodo.5270332](https://doi.org/10.5281/zenodo.5270332)).

* **CAPE / CIN** of surface, mixed-layer and most-unstable parcels (George
  Bryan's `getcape`), **LCL**, **LFC** and **equilibrium level**, **DCAPE**,
  lapse rates (virtual temperature)
* **Storm motion** (Bunkers: standard 0-6 km, shallow 0-3 km, effective-layer
  2014, or custom layers), **SRH**, **bulk shear**, **streamwise vorticity**,
  **streamwiseness**, critical angle
* **Effective inflow layer** (Thompson et al. 2007) and its parameters:
  **ESRH, EBWD, ESCP, ESTP**, effective-layer streamwise vorticity and
  streamwiseness (**E_SWV, E_SWNE**)
* **Composites**: SCP, STP (fixed layer), STP with CIN term, ESCP, ESTP, EHI,
  WMAXSHEAR
* **ERA5**: query a bounding box from the Copernicus Climate Data Store (or
  read local files) and compute 64 parameters in one call or one shell command

## Install

**Quick start** (Linux or macOS, Python 3.10 or newer):

```bash
python3 -m venv xcaper-env
source xcaper-env/bin/activate
pip install "xcaper[era5] @ git+https://github.com/AstroCGHA/xcape_remastered.git@v1.0.1"
python -c "import xcaper; print(xcaper.__version__)"
```

The step-by-step guide below explains each of these commands and the
alternatives.

### Step by step

**1. Check Python.** You need Python 3.10 or newer:

```bash
python3 --version
```

**2. Install a Fortran compiler (optional, recommended).** With gfortran, xcaper
builds its Fortran kernels, which reproduce xcape 0.1 exactly. Without it,
xcaper installs anyway and uses its numba backend instead, which gives the
same results to round-off (see [Backends and parallelism](#backends-and-parallelism)).
Nothing else needs a compiler.

| System | Command |
|---|---|
| Ubuntu / Debian / WSL | `sudo apt install gfortran` |
| Fedora / RHEL | `sudo dnf install gcc-gfortran` |
| conda (any system) | `conda install -c conda-forge compilers` |
| macOS (not tested) | `brew install gcc` |
| Windows (not tested) | use WSL (Ubuntu) and the line above, or skip: xcaper then uses numba |

**3. Create an environment.** This keeps xcaper and its dependencies separate
from the rest of your system. With the standard `venv`:

```bash
python3 -m venv xcaper-env
source xcaper-env/bin/activate        # Windows: xcaper-env\Scripts\activate
python -m pip install --upgrade pip
```

or with conda:

```bash
conda create -n xcaper -c conda-forge python=3.13 compilers
conda activate xcaper
```

Activate the environment again (`source xcaper-env/bin/activate` or
`conda activate xcaper`) every time you open a new terminal.

**4. Install xcaper.** Choose one:

```bash
# a) the 1.0.1 release from GitHub (needs git)
pip install "xcaper[era5] @ git+https://github.com/AstroCGHA/xcape_remastered.git@v1.0.1"

# b) the latest version on GitHub (needs git)
pip install "xcaper[era5] @ git+https://github.com/AstroCGHA/xcape_remastered.git"

# c) a downloaded release file (from Zenodo or the GitHub release page)
pip install "xcaper-1.0.1.tar.gz[era5]"
```

* `[era5]` also installs `cdsapi` and `netCDF4`, which you need to download ERA5
  and to read and write NetCDF files. Leave it out for the core package only.
* pip compiles the package, which takes 1-2 minutes. A C and Fortran compiler
  is used when found. To skip the Fortran backend on purpose, add
  `-Csetup-args=-Dfortran=disabled`.

**5. Check the installation.**

```bash
python -c "import xcaper; from xcaper.core import HAS_FORTRAN; print(xcaper.__version__, 'Fortran:', HAS_FORTRAN)"
xcaper-era5 --version
```

This prints `1.0.1 Fortran: True` (or `False` without a Fortran compiler,
which is fine) and `xcaper 1.0.1`.

**6. Try it.** The example scripts are in the [`examples`](examples/) folder of
the repository (download it, or clone the repository as in the next
section). The first one needs no data:

```bash
python examples/sounding_from_csv.py
```

The first run takes some extra seconds: numba compiles the kernels once and
caches them.

**7. Set up ERA5 downloads (only for downloading).** Register at the Copernicus
Climate Data Store and create `~/.cdsapirc`, see [ERA5 in one call](#era5-in-one-call).
ERA5 files already on disk need no set-up.

### For development

To edit the code, clone the repository and install it in editable mode:
changes to the Python files take effect at the next import, and changes to the
Fortran files are recompiled automatically.

```bash
git clone https://github.com/AstroCGHA/xcape_remastered.git
cd xcape_remastered
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install meson-python meson ninja numpy       # build tools, needed first
pip install --no-build-isolation -e ".[dev]"     # xcaper + pytest, metpy, cdsapi, netCDF4
pytest                                            # 116 tests, about a minute
```

### Updating and uninstalling

```bash
pip install --upgrade --force-reinstall --no-cache-dir "xcaper[era5] @ git+https://github.com/AstroCGHA/xcape_remastered.git"
pip uninstall xcaper
```

### Troubleshooting

| Problem | Solution |
|---|---|
| `No module named 'xcaper'` | The environment is not active: run `source xcaper-env/bin/activate` (or `conda activate xcaper`) first. |
| `Fortran: False` although you want the Fortran backend | Install gfortran (step 2), then reinstall with `pip install --force-reinstall --no-cache-dir ...`, because pip otherwise reuses the package it built before. |
| `git: command not found` | Install git, or download the release file and use option c of step 4. |
| Build error mentioning `meson` or `numpy` in an editable install | Install the build tools first: `pip install meson-python meson ninja numpy`. |
| A `.nc` file cannot be opened or saved | Install the extras: `pip install netCDF4` (or reinstall with `[era5]`). |
| Downloading ERA5 fails with an authentication or licence error | Check `~/.cdsapirc` and accept the dataset licences on the CDS website (step 7). |

### Tested platforms

Linux x86_64, gfortran 13, built from the source distribution in a fresh
environment each time. Every combination passes the full test suite, the
original xcape 0.1.4 test suite, and reproduces the reference ERA5 output
bit for bit.

| Python | NumPy | numba | Status |
|---|---|---|---|
| 3.10 | 1.24.4 (oldest supported) and 2.2.6 | 0.59.1 and 0.68.1 | tested |
| 3.11 | 2.4.6 | 0.68.1 | tested |
| 3.12 | 2.5.5 | 0.68.1 | tested |
| 3.13 | 2.5.5 | 0.68.1 | tested |
| 3.14 | 2.5.5 | 0.68.1 | tested |
| 3.15 | 2.5.5 | 0.68.1 | pre-release (3.15.0b4) passes |

macOS and Windows have not been tested.

## ERA5 in one call

```python
from xcaper import era5

ds = era5.era5_severe_parameters(
    "2026-08-08T14:00",                       # one time, a list, or era5.time_range(...)
    north=30, west=100, south=-10, east=145,  # bounding box
    storm_motion="bunkers-shallow",           # or "bunkers" (standard 0-6 km), "bunkers-effective"
    cache_dir="era5_cache",
    output="ERA5_Severe_Params_B03_20260808_14.nc",
)
```

or from the shell:

```bash
xcaper-era5 --area 30 100 -10 145 --time 2026-08-08T14 --storm-motion bunkers-shallow
xcaper-era5 --area 30 100 -10 145 --start 2026-08-08T00 --end 2026-08-08T23 --step 3
# files already on disk, optionally cut to a smaller box
xcaper-era5 --pl-file era5_pl_2026080814.nc --sfc-file era5_sfc_2026080814.nc \
            --area 22 110 4 135
```

Local files without the download:

```python
ds_pl, ds_sfc = era5.open_files("era5_pl_2026080814.nc", "era5_sfc_2026080814.nc")
ds = era5.compute_severe_parameters(ds_pl, ds_sfc, chunks="auto",      # dask, parallel
                                    north=22, west=110, south=4, east=135)
era5.save(ds.load(), "params.nc")
```

How surface and pressure-level data are combined: 2 m temperature / dewpoint
and surface pressure are the lowest thermodynamic level; the 10 m wind is the
wind at 0 m AGL; heights AGL are `(z - z_sfc) / g`; pressure levels below the
ground are dropped; dewpoint on pressure levels comes from specific humidity.

CDS set-up (once): `pip install "cdsapi>=0.7.2"`, register at the Copernicus
Climate Data Store, accept the terms of `reanalysis-era5-single-levels` and
`reanalysis-era5-pressure-levels`, and create `~/.cdsapirc`. Downloads are
cached as `era5_pl_YYYYMMDDHH.nc` / `era5_sfc_YYYYMMDDHH.nc`; a cached file
that does not cover the requested box or times is refused rather than
silently reused.

## Output variables

| Variable | Units | Meaning |
|---|---|---|
| `SFC_CAPE`, `ML_CAPE`, `MU_CAPE` | J/kg | CAPE of the surface, mixed-layer (500 m by default) and most-unstable parcel |
| `SFC_CIN`, `ML_CIN`, `MU_CIN` | J/kg | CIN, **as a positive number** |
| `SFC_LCL`, `ML_LCL` | m AGL | LCL height (Bolton 1980) |
| `SFC_LFC`, `ML_LFC`, `MU_LFC` (+`_P`) | m AGL (hPa) | level of free convection: lowest point at or above the LCL where the parcel is buoyant (virtual temperature) |
| `SFC_EL`, `ML_EL`, `MU_EL` (+`_P`) | m AGL (hPa) | equilibrium level: top of the highest buoyant layer (top of the data if still buoyant there) |
| `DCAPE` | J/kg | downdraft CAPE, MetPy recipe: min theta-e 700-500 hPa, wet-bulb start, pseudoadiabatic descent |
| `DCAPE_SRC_P` | hPa | pressure of the DCAPE source parcel |
| `LR03`, `LR36`, `LR75` | K/km | 0-3 km, 3-6 km AGL and 700-500 hPa lapse rates of the virtual temperature (SHARPpy/SPC convention) |
| `MEANxx_U/V` | m/s | mean wind of the storm-motion layer (`MEAN06`, `MEAN03`; `BUNKERS_MEAN` for the effective method) |
| `BUNKERS_RM_U/V`, `BUNKERS_LM_U/V` | m/s | right / left mover motion |
| `SRH500M_RM/LM`, `SRH1_RM/LM`, `SRH3_RM/LM` | m2/s2 | storm relative helicity, signed |
| `SRH500M`, `SRH1`, `SRH3` | m2/s2 | max(\|RM\|, \|LM\|) |
| `S01`, `S03`, `S06` | m/s | bulk wind difference |
| `SWV01` | 1/s | 0-1 km mean streamwise vorticity magnitude (right mover) |
| `SWNESS01` | % | 0-1 km mean streamwiseness (streamwise share of the horizontal vorticity) |
| `CRITICAL_ANGLE` | degree | 0-500 m shear vs. surface storm-relative inflow |
| `EIL_BASE`, `EIL_TOP` (+`_P`) | m AGL (hPa) | effective inflow layer: CAPE >= 100 J/kg and CIN <= 250 J/kg; NaN = no layer, base 0 = surface based |
| `ESRH_RM`, `ESRH_LM`, `ESRH` | m2/s2 | SRH of the effective inflow layer (0 where there is none) |
| `EBWD` | m/s | bulk wind difference from the effective base to half the depth up to the MU EL (0 where no layer) |
| `E_SWV`, `E_SWNE` | 1/s, % | streamwise vorticity and streamwiseness averaged over the effective layer (NaN where none) |
| `ESCP` | - | (MUCAPE/1000)(ESRH/50)(EBWD term)(40/MUCIN, max 1) |
| `ESTP` | - | (MLCAPE/1500)((2000-MLLCL)/1000)(ESRH/150)(EBWD term)((200-MLCIN)/150); 0 when the layer is elevated |
| `SCP` | - | (MUCAPE/1000)(SRH3/50)(S06/20, 0 below 10 m/s, max 1) |
| `STP` | - | (SBCAPE/1500)((2000-SBLCL)/1000)(SRH1/150)(S06/20) with the usual limits |
| `STP_CIN` | - | (MLCAPE/1500)((2000-MLLCL)/1000)(SRH1/150)(S06/20)((200-MLCIN)/150) |
| `EHI1`, `EHI3` | - | MLCAPE x SRH / 160000 |
| `WMAXSHEAR` | m2/s2 | sqrt(2 MLCAPE) x S06 |

Layers are options (`srh_layers`, `shear_layers`, `swv_layers`, in m AGL);
names follow the layer (`SRH500M`, `S03`, `SWV01`, ...). Composites use the
right-mover SRH and keep its sign. With `storm_motion="bunkers-effective"` the
output also has `BUNKERS_EFF_FLAG` (1 = effective method used, 0 = 0-6 km
fallback).

## Options

```python
ds = era5.compute_severe_parameters(ds_pl, ds_sfc,
                                    storm_motion="bunkers-effective",  # Bunkers et al. (2014)
                                    eil_dp=10.0,          # refined search spacing (hPa), None = input levels
                                    cape_min=100., cin_max=250.,
                                    dcape_layer=(700., 500.),
                                    ml_depth=500.,        # mixed-layer depth (m)
                                    virtual_lapse_rate=True)
ds = era5.compute_severe_parameters(ds_pl, ds_sfc, effective=False)  # skip LFC/EL/DCAPE/effective layer, ~2x faster
```

`eil_dp` matters for coarse data: on ERA5's 25 hPa levels, a layer whose top
lies between the surface and the next level would otherwise have zero depth
and therefore zero ESRH, ESCP and ESTP. With the default 10 hPa refinement
(temperature and dewpoint interpolated linearly in ln p) this happened in
2,095 instead of 3,582 of 26,364 effective-layer columns of the 2026-08-08
14 UTC test case, at no extra cost. The remaining zero-depth layers are
genuine: only the 2 m parcel qualifies.

`storm_motion="bunkers-effective"` uses the layer from the effective inflow
base to 65% of the way up to the most-unstable EL (height-weighted mean wind,
bulk shear across the layer), and the standard 0-6 km method where there is
no effective layer. SPC/SHARPpy use a pressure-weighted mean wind for this
method.

## Examples

The [`examples`](examples/) folder has runnable scripts: every parameter of a
single sounding from a CSV file (sample included, no download needed), the
individual functions step by step, a parameter table at chosen points, domain
maps, a comparison of the storm-motion methods, and batch processing of a
folder of ERA5 files. See [`examples/README.md`](examples/README.md).

## The building blocks

```python
from xcaper.core import calc_cape, calc_srh, calc_lcl     # xcape API + LCL
from xcaper.kinematics import calc_kinematics, StormMotion
from xcaper.parcel import calc_parcel_diagnostics          # LFC, EL, EIL, DCAPE
from xcaper.thermo import calc_lapse_rates, dewpoint_from_specific_humidity
from xcaper import composites
from xcaper.severe import calc_severe                     # everything, arrays in
from xcaper import xarray as xcx                          # the same for DataArrays

cape, cin = calc_cape(p, t, td, ps, ts, tds, source="mixed-layer", ml_depth=500.,
                      vertical_lev="pressure", method="auto")

kin = calc_kinematics(z_agl, u, v, u10, v10,
                      storm_motion=StormMotion(mean_layer=(0, 4000), tail_layer=(0, 500),
                                               head_layer=(3500, 4000), deviation=7.5),
                      srh_layers=(500, 1000, 3000), swv_layers=(500, 1000))
```

Conventions: vertical axis last, bottom to top; hPa, Celsius, m/s, m.
numpy in -> numpy out (computed at once); dask in -> dask out (lazy, chunked,
parallel; the vertical dimension is gathered into one chunk automatically).

## Backends and parallelism

| | Fortran (`method="fortran"`) | numba (`method="numba"`) |
|---|---|---|
| precision | single (as xcape 0.1) | double |
| numpy input | one core | all cores (`numba.prange`) |
| dask input | one task per chunk, GIL released | one task per chunk, GIL released |
| needs compiler | at install time | no |

The two backends agree to round-off: on the ERA5 test field (29,141 columns)
CAPE differs by at most 0.22 J/kg and 50 of the 64 output variables are
identical. The exception is a parcel whose buoyancy touches zero within
single-precision resolution (about 1e-6 m/s2): there the double-precision
numba ascent may count the negative area below that point as CIN and the
single-precision Fortran ascent may not, or the other way round. This
happened in 2 of the 29,141 columns (CIN 2.7 and 6.9 J/kg vs 0). Results
are always reproducible for a given backend; `method="fortran"` gives xcape
0.1 results bit for bit.

The newer diagnostics (kinematics, LCL, LFC/EL, effective layer, DCAPE, lapse
rates) are numba only. Inside dask tasks the serial numba loops are used, so
dask threads and numba threads never nest. Control the thread counts with
`dask.config.set(num_workers=...)` and `numba.set_num_threads(...)`.

## Coming from xcape 0.1

Replace `import xcape` with `import xcaper`. `calc_cape` and `calc_srh` keep
their signatures and results: the original xcape 0.1.4 test suite passes, and
on a real ERA5 field the Fortran kernels reproduce xcape 0.1.4 bit for bit for
CAPE, CIN, heights and Bunkers motion, and to 1e-13 for SRH. See
`CHANGELOG.md` for everything that changed.

Two flavours of Bunkers motion / SRH coexist on purpose:

* `calc_srh` - the xcape 0.1 algorithm: heights from the hypsometric
  equation, the profile sampled every 500 m;
* `calc_kinematics` - heights supplied by you (geopotential), layer means on a
  100 m grid, configurable layers. This is what the ERA5 workflow uses.

They agree to a few tenths of a m/s where the shear is well defined and differ
where the 0-6 km shear vector is weak (its direction is then ill-defined).

## Known limitations

* The effective inflow layer uses Bryan's getcape parcel. MetPy's parcel gives
  10-30 J/kg more CAPE near the 100 J/kg threshold, so on ERA5 the layer top
  differs by one level in about 1 column in 9 (bases agree).
* The most-unstable parcel is the max theta-e below 500 hPa (xcape heritage);
  SPC uses the lowest 300 hPa.
* The mixed layer is 500 m deep by default (xcape heritage); SPC uses the
  lowest 100 hPa. This also affects MLCAPE / MLCIN / MLLCL in ESTP. Set
  `ml_depth` to change it.
* Composites use right-mover SRH (Northern Hemisphere convention).
* Behaviours inherited from xcape 0.1 and kept for compatibility: columns with
  a surface temperature <= 0 degC return zero CAPE/CIN; a parcel ascent that
  does not converge returns zero; `zMUlev` is the height of the top of the
  parcel ascent, not of the most-unstable level.

## Comparing with SounderPy

SounderPy and xcaper agree on the thermodynamics, lapse rates and SRH
for the same column and storm motion, but SounderPy's effective-layer SCP and
STP are not the SPC / SHARPpy definitions:

* its effective shear is the bulk shear *across the effective inflow layer*
  (base to top), not EBWD (base to half the most-unstable EL height, as in
  SHARPpy's own `profile.py` and Thompson et al. 2007);
* that shear is in knots but enters `scp()` / `stp_cin()`, which expect m/s;
* its effective STP uses the most-unstable parcel's CAPE, LCL and CIN instead
  of the mixed-layer parcel's;
* its default storm motion is SHARPpy's effective-layer Bunkers (2014) motion
  (`storm_motion="bunkers-effective"` in xcaper), not the 0-6 km Bunkers motion.

Expect SounderPy's effective SCP / STP to be larger than xcaper's ESCP / ESTP
wherever the low-level shear is strong and the deep-layer shear weak.

## Citing

If you use xcaper, please cite this release (see `CITATION.cff`) and the
original xcape:

* Capuli, G. (2026): xcaper v1.0.1 (software). Zenodo.
  [doi:10.5281/zenodo.24273035](https://doi.org/10.5281/zenodo.24273035)
  (version 1.0.0: [doi:10.5281/zenodo.24269592](https://doi.org/10.5281/zenodo.24269592))
* Lepore, C., Allen, J. and Abernathey, R. (2021): xcape v0.1.4 (software).
  Zenodo. [doi:10.5281/zenodo.5270332](https://doi.org/10.5281/zenodo.5270332)

The parcel ascent is George H. Bryan's `getcape` (Bryan and Fritsch 2004).
Parameter definitions follow Bunkers et al. (2000, 2014), Thompson et al.
(2003, 2004, 2007, 2012), Hart and Korotky (1991) and Taszarek et al. (2020).

## Data attribution

Results computed from ERA5 contain modified Copernicus Climate Change Service
information. ERA5 is distributed under the Creative Commons Attribution 4.0
licence (CC-BY 4.0). When you publish or share such results, credit the source
("Contains modified Copernicus Climate Change Service information [year]"),
state the licence, and note that neither the European Commission nor ECMWF is
responsible for any use that may be made of the Copernicus information or data
it contains. Cite ERA5 (Hersbach et al. 2020) and the DOIs listed on the
dataset pages of the Climate Data Store.

## Licence

MIT. See `LICENSE` for the copyright notices of UCAR (getcape), the xcape
authors and xcaper.
