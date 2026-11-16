python_sources = [
  '__init__.py',
  '_parallel.py',
  '_version.py',
  'cape_fortran.py',
  'cape_numba.py',
  'composites.py',
  'core.py',
  'duck_array_ops.py',
  'era5.py',
  'kinematics.py',
  'parcel.py',
  'severe.py',
  'srh.py',
  'srh_numba.py',
  'stdheight.py',
  'thermo.py',
  'xarray.py',
]

py.install_sources(python_sources, subdir: 'xcaper')

py.install_sources('fortran/__init__.py', subdir: 'xcaper/fortran')

if build_fortran
  subdir('fortran')
endif
