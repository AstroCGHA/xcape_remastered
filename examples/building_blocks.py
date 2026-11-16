#!/usr/bin/env python3
"""
The individual xcaper functions, step by step.

``calc_severe`` / ``era5.compute_severe_parameters`` compute everything in one
call. This script does the same work piece by piece, so you can see what
each function takes and returns and use only the parts you need, with your
own layers, parcels or storm motion. It runs on the sample sounding in
``examples/data`` (or on your own CSV, same format as sounding_from_csv.py):

    python building_blocks.py
    python building_blocks.py my_sounding.csv

All functions accept numpy arrays of any shape (..., nlev), dask arrays
(lazy, parallel) or, through ``xcaper.xarray``, DataArrays. The vertical axis
is last and runs bottom to top; units are hPa, Celsius, m/s and m.
"""
import sys
from pathlib import Path

import numpy as np

from xcaper import composites
from xcaper.core import calc_cape, calc_lcl
from xcaper.kinematics import StormMotion, calc_kinematics
from xcaper.parcel import calc_parcel_diagnostics
from xcaper.severe import calc_severe
from xcaper.thermo import calc_lapse_rates

# the CSV reader of the sounding example (same folder)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sounding_from_csv import SAMPLE, read_sounding, to_xcaper_inputs  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else SAMPLE
x = to_xcaper_inputs(read_sounding(path))
p, t, td, u, v, z = x["p"], x["t"], x["td"], x["u"], x["v"], x["z_agl"]
ps, ts, tds, us, vs = x["ps"], x["ts"], x["tds"], x["us"], x["vs"]
thermo = (p, t, td, ps, ts, tds)
# A sounding has its own pressure per level: vertical_lev='sigma'. For data on
# fixed pressure levels (ERA5), pass p as 1-D and use vertical_lev='pressure'.
LEV = "sigma"
first = lambda a: float(np.asarray(a).ravel()[0])          # value of the one column

# ----------------------------------------------------------------------------
# 1. CAPE and CIN (George Bryan's getcape; xcape-compatible API)
# ----------------------------------------------------------------------------
print("1. CAPE / CIN  (calc_cape)")
sb_cape, sb_cin = calc_cape(*thermo, source="surface", vertical_lev=LEV)
ml_cape, ml_cin = calc_cape(*thermo, source="mixed-layer", ml_depth=500.0, vertical_lev=LEV)
# the most-unstable parcel also returns its level index and the ascent top
mu_cape, mu_cin, mu_lev, _ = calc_cape(*thermo, source="most-unstable", vertical_lev=LEV)
for name, cape, cin in [("surface", sb_cape, sb_cin), ("mixed layer 500 m", ml_cape, ml_cin),
                        ("most unstable", mu_cape, mu_cin)]:
    print(f"   {name:18s} CAPE {first(cape):7.1f} J/kg   CIN {first(cin):5.1f} J/kg")

# the two backends: Fortran (single precision, as xcape 0.1) and numba (double)
try:
    cape_f = first(calc_cape(*thermo, source="surface", vertical_lev=LEV, method="fortran")[0])
    cape_n = first(calc_cape(*thermo, source="surface", vertical_lev=LEV, method="numba")[0])
    print(f"   backends: Fortran {cape_f:.2f}  numba {cape_n:.2f} J/kg")
except ImportError:
    print("   backends: Fortran not installed, numba used")

# ----------------------------------------------------------------------------
# 2. LCL, LFC, EL, effective inflow layer, DCAPE
# ----------------------------------------------------------------------------
print("\n2. Parcel levels  (calc_lcl, calc_parcel_diagnostics)")
z_lcl, p_lcl = calc_lcl(*thermo, source="surface", vertical_lev=LEV)
print(f"   surface-parcel LCL {first(z_lcl):.0f} m AGL ({first(p_lcl):.1f} hPa)")

diag = calc_parcel_diagnostics(*thermo, vertical_lev=LEV,
                               cape_min=100.0, cin_max=250.0,     # Thompson et al. (2007)
                               eil_dp=10.0,                       # refined search, hPa
                               dcape_layer=(700.0, 500.0))
print(f"   returns: {', '.join(sorted(diag))}")
print(f"   MU parcel: LFC {first(diag['MU_LFC']):.0f} m, EL {first(diag['MU_EL']):.0f} m "
      f"({first(diag['MU_EL_P']):.0f} hPa)")
print(f"   effective inflow layer: {first(diag['EIL_BASE']):.0f}-{first(diag['EIL_TOP']):.0f} m AGL "
      f"({first(diag['EIL_BASE_P']):.0f}-{first(diag['EIL_TOP_P']):.0f} hPa)")
print(f"   DCAPE {first(diag['DCAPE']):.0f} J/kg from {first(diag['DCAPE_SRC_P']):.0f} hPa")

# ----------------------------------------------------------------------------
# 3. Lapse rates (virtual temperature, as SPC / SHARPpy)
# ----------------------------------------------------------------------------
print("\n3. Lapse rates  (calc_lapse_rates)")
lr_z, lr_p = calc_lapse_rates(p, t, z, ps, ts, td=td, tds=tds,
                              height_layer=(0.0, 3000.0), pressure_layer=(700.0, 500.0))
lr_36, _ = calc_lapse_rates(p, t, z, ps, ts, td=td, tds=tds, height_layer=(3000.0, 6000.0))
print(f"   0-3 km {first(lr_z):.2f}   3-6 km {first(lr_36):.2f}   700-500 hPa {first(lr_p):.2f} K/km")

# ----------------------------------------------------------------------------
# 4. Kinematics with a custom storm motion
# ----------------------------------------------------------------------------
print("\n4. Kinematics  (calc_kinematics)")
# Any Bunkers-type motion: mean wind of mean_layer, shear from the tail_layer
# mean to the head_layer mean, deviation (m/s) perpendicular to that shear.
# Here a 0-4 km version; 'bunkers', 'bunkers-shallow' and 'bunkers-effective'
# are the built-in choices.
custom = StormMotion(mean_layer=(0.0, 4000.0), tail_layer=(0.0, 500.0),
                     head_layer=(3500.0, 4000.0), deviation=7.5, name="bunkers-0-4km")
kin = calc_kinematics(z, u, v, us, vs, storm_motion=custom,
                      srh_layers=(500.0, 1000.0, 3000.0),   # m AGL, multiples of dz
                      shear_layers=(1000.0, 3000.0, 6000.0),
                      swv_layers=(500.0, 1000.0),
                      dz=100.0,
                      # effective-layer outputs need the layer and the MU EL:
                      eil_base=diag["EIL_BASE"], eil_top=diag["EIL_TOP"],
                      el_height=diag["MU_EL"])
print(f"   returns: {', '.join(sorted(kin))}")
rm = np.hypot(first(kin["BUNKERS_RM_U"]), first(kin["BUNKERS_RM_V"]))
print(f"   right mover {rm:.1f} m/s;  SRH 0-1 km {first(kin['SRH1_RM']):.0f}, "
      f"0-3 km {first(kin['SRH3_RM']):.0f}, effective {first(kin['ESRH_RM']):.0f} m2/s2")
print(f"   shear 0-6 km {first(kin['S06']):.1f} m/s, EBWD {first(kin['EBWD']):.1f} m/s;  "
      f"0-500 m streamwiseness {first(kin['SWNESS500M']):.0f} %")

# ----------------------------------------------------------------------------
# 5. Composite indices from the pieces
# ----------------------------------------------------------------------------
print("\n5. Composites  (xcaper.composites)")
scp = composites.supercell_composite(mu_cape, kin["SRH3_RM"], kin["S06"])
stp = composites.significant_tornado(sb_cape, z_lcl, kin["SRH1_RM"], kin["S06"])
escp = composites.effective_supercell_composite(mu_cape, kin["ESRH_RM"], kin["EBWD"], mu_cin)
estp = composites.effective_significant_tornado(ml_cape, calc_lcl(*thermo, source="mixed-layer",
                                                                  vertical_lev=LEV)[0],
                                                kin["ESRH_RM"], kin["EBWD"], ml_cin,
                                                diag["EIL_BASE"])
ehi3 = composites.energy_helicity_index(ml_cape, kin["SRH3_RM"])
print(f"   SCP {first(scp):.2f}  STP {first(stp):.2f}  ESCP {first(escp):.2f}  "
      f"ESTP {first(estp):.2f}  EHI3 {first(ehi3):.2f}   (0-4 km storm motion)")

# ----------------------------------------------------------------------------
# 6. Check: the same with calc_severe and the built-in 0-6 km storm motion
# ----------------------------------------------------------------------------
print("\n6. All at once  (calc_severe, storm_motion='bunkers')")
res = calc_severe(p, t, td, u, v, z, ps, ts, tds, us, vs, vertical_lev=LEV, storm_motion="bunkers")
kin06 = calc_kinematics(z, u, v, us, vs, storm_motion="bunkers", eil_base=diag["EIL_BASE"],
                        eil_top=diag["EIL_TOP"], el_height=diag["MU_EL"])
pieces = {
    "MU_CAPE": mu_cape, "ML_CIN": ml_cin, "MU_EL": diag["MU_EL"], "DCAPE": diag["DCAPE"],
    "LR03": lr_z, "SRH3_RM": kin06["SRH3_RM"], "ESRH_RM": kin06["ESRH_RM"],
    "SCP": composites.supercell_composite(mu_cape, kin06["SRH3_RM"], kin06["S06"]),
    "ESCP": composites.effective_supercell_composite(mu_cape, kin06["ESRH_RM"],
                                                     kin06["EBWD"], mu_cin),
}
worst = max(abs(first(res[k]) - first(val)) for k, val in pieces.items())
print(f"   {len(res)} parameters; largest difference from the pieces above: {worst:.2g}")
