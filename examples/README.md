# xcaper examples

Each script runs on its own. Start with the first two: they need no
download.

| Script | What it shows | Input |
|---|---|---|
| [`sounding_from_csv.py`](sounding_from_csv.py) | Every parameter of one sounding (radiosonde, model column, ...) read from a CSV file, printed as a table | the bundled sample, or your own CSV |
| [`building_blocks.py`](building_blocks.py) | The individual functions step by step: CAPE/CIN of three parcels, LCL/LFC/EL, effective inflow layer, DCAPE, lapse rates, kinematics with a custom storm motion, composites; checked against `calc_severe` | the bundled sample, or your own CSV |
| [`era5_point_values.py`](era5_point_values.py) | Parameter table at chosen latitude/longitude points (nearest grid point), saved to CSV | ERA5 files, or download |
| [`era5_severe_params.py`](era5_severe_params.py) | A domain map: computes all parameters, saves NetCDF and a 12-panel figure | ERA5 files, or download |
| [`storm_motion_comparison.py`](storm_motion_comparison.py) | Standard vs shallow vs effective Bunkers storm motion: domain statistics of SRH, ESRH, SCP, STP, ESCP, ESTP, and a 3 x 3 figure | ERA5 files, or download |
| [`batch_local_files.py`](batch_local_files.py) | A folder of ERA5 files: one output per time step, parallel with dask, skips finished steps | a folder of ERA5 files |

## Running them

```bash
cd examples
python sounding_from_csv.py                         # sample sounding
python sounding_from_csv.py my_sounding.csv --storm-motion bunkers-shallow
python building_blocks.py

# with ERA5 files already on disk (pressure levels, then single levels)
python era5_point_values.py        era5_pl_2026080814.nc era5_sfc_2026080814.nc
python era5_severe_params.py       era5_pl_2026080814.nc era5_sfc_2026080814.nc
python storm_motion_comparison.py  era5_pl_2026080814.nc era5_sfc_2026080814.nc
python batch_local_files.py era5_cache/ --out params/

# without file arguments the ERA5 scripts download from the Copernicus Climate
# Data Store (one-time set-up: see the main README)
```

The ERA5 scripts read their settings (times, area, points, storm motion) from
the block near the top of each file; edit them there.

Figures need `matplotlib`; coastlines are drawn when `cartopy` is installed and
its coastline data can be downloaded, otherwise the maps use plain
latitude/longitude axes.

## Sounding CSV format

```
pressure_hPa,height_m,temperature_C,dewpoint_C,wind_dir_deg,wind_speed_ms
998.40,92.3,27.08,24.79,228,6.6        <- first row: the surface (2 m T/Td, 10 m wind)
975.00,301.8,24.93,22.75,233,10.3
...                                     <- upwards, pressure decreasing
```

Heights may be above sea level or above ground (they are taken relative to
the first row); the wind direction is the direction the wind blows from;
`--knots` reads the speed in knots; lines starting with `#` are ignored and
missing values may be empty.

## Sample data

[`data/sounding_era5_10.75N_122.50E_2026080814.csv`](data/sounding_era5_10.75N_122.50E_2026080814.csv)
is the ERA5 column at 10.75°N, 122.50°E (Panay, Philippines) on
2026-08-08 at 14 UTC. It contains modified Copernicus Climate Change Service
information 2026 (ERA5, CC-BY 4.0); neither the European Commission nor ECMWF
is responsible for any use that may be made of it.
