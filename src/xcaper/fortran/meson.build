# ---------------------------------------------------------------------------
# Fortran extension modules
#
# All single-column kernels live in xcaper_kernels.f90, compiled once into a
# static library.  Each extension module is a thin driver (loops over the
# grid) plus the C wrapper that f2py generates from its .pyf signature file.
# ---------------------------------------------------------------------------
fc = meson.get_compiler('fortran')
cc = meson.get_compiler('c')

# NumPy / f2py headers.  Meson refuses absolute paths that point into the
# source tree, which is where they point when the virtual environment lives
# inside the project folder (e.g. ./.venv): give those relative to this
# directory, everything else (the usual case) as an absolute path.
_header_dir = '''
import os, sys
import numpy, numpy.f2py
path = numpy.get_include() if sys.argv[1] == "numpy" else numpy.f2py.get_include()
path = os.path.realpath(path)
root = os.path.realpath(sys.argv[2])
here = os.path.realpath(sys.argv[3])
inside = os.path.commonpath([path, root]) == root
print(os.path.relpath(path, here) if inside else path)
'''
incdir_numpy = run_command(
  py, ['-c', _header_dir, 'numpy', meson.project_source_root(), meson.current_source_dir()],
  check: true,
).stdout().strip()

incdir_f2py = run_command(
  py, ['-c', _header_dir, 'f2py', meson.project_source_root(), meson.current_source_dir()],
  check: true,
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)
fortranobject_c = incdir_f2py / 'fortranobject.c'

numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'

# Silence the (harmless) warnings of f2py generated C code.
c_ignore_warnings = cc.get_supported_arguments(
  '-Wno-unused-function',
  '-Wno-unused-variable',
  '-Wno-unused-but-set-variable',
  '-Wno-misleading-indentation',
  '-Wno-maybe-uninitialized',
)

# The f2py wrappers are fixed form, the kernels are free form; both are
# recognised from the file extension, no flag is needed.
fortran_args = fc.get_supported_arguments('-Wno-unused-dummy-argument')

kernels_lib = static_library(
  'xcaper_kernels',
  'xcaper_kernels.f90',
  fortran_args: fortran_args,
  pic: true,
)

# name -> does the signature file expose Fortran *functions*?  f2py writes an
# extra fixed-form wrapper file (<name>-f2pywrappers.f) for those.
extensions = {
  'CAPE_CODE_model_lev': true,
  'CAPE_CODE_pressure_lev': true,
  'Bunkers_model_lev': false,
  'Bunkers_pressure_lev': false,
  'SREH_model_lev': true,
  'SREH_pressure_lev': true,
  'stdheight_2D_model_lev': false,
  'stdheight_2D_pressure_lev': false,
}

foreach name, has_functions : extensions
  outputs = [name + 'module.c']
  if has_functions
    outputs += [name + '-f2pywrappers.f']
  endif

  wrappers = custom_target(
    name + '_f2py',
    input: name + '.pyf',
    output: outputs,
    command: [py, '-m', 'numpy.f2py', '@INPUT@', '--build-dir', '@OUTDIR@'],
  )

  py.extension_module(
    name,
    [wrappers, name + '.f90', fortranobject_c],
    include_directories: inc_np,
    dependencies: py_dep,
    link_with: kernels_lib,
    c_args: [numpy_nodepr_api] + c_ignore_warnings,
    fortran_args: fortran_args,
    install: true,
    subdir: 'xcaper/fortran',
  )
endforeach
