project(
  'xcaper',
  version: '1.0.0',
  license: 'MIT',
  meson_version: '>=1.1.0',
  default_options: [
    'buildtype=release',
    'c_std=c99',
    'warning_level=1',
  ],
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# ---------------------------------------------------------------------------
# The Fortran backend is optional (-Dfortran=enabled|auto|disabled).  Without
# it the package is pure Python and every function runs on the numba backend.
# ---------------------------------------------------------------------------
build_fortran = add_languages('c', 'fortran', native: false, required: get_option('fortran'))

subdir('src/xcaper')

if build_fortran
  summary('Fortran backend', 'yes', section: 'xcaper')
else
  summary('Fortran backend', 'no (numba backend only)', section: 'xcaper')
endif
