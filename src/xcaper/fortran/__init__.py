"""Compiled Fortran kernels of xcaper (f2py extension modules).

The extension modules are imported lazily by :mod:`xcaper.cape_fortran`,
:mod:`xcaper.srh` and :mod:`xcaper.stdheight`.  When xcaper was installed
without a Fortran compiler they are simply absent and the numba backend
is used instead.
"""
