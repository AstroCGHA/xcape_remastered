"""xcaper: fast convective and severe-weather parameters for numpy, dask and xarray.

Sub-modules
-----------
``xcaper.core``        ``calc_cape`` and ``calc_srh`` (the original xcaper API)
``xcaper.kinematics``  storm motion, SRH, bulk shear, streamwise vorticity
``xcaper.thermo``      dewpoint, LCL height, lapse rates
``xcaper.parcel``      LFC, EL, effective inflow layer, DCAPE
``xcaper.composites``  SCP, STP, ESCP, ESTP, EHI, WMAXSHEAR
``xcaper.severe``      all of the above in one call (numpy / dask arrays)
``xcaper.xarray``      the same for xarray objects
``xcaper.era5``        ERA5 download for a bounding box + parameter computation
"""
from ._version import __version__

__all__ = [
    "__version__",
    "core",
    "kinematics",
    "thermo",
    "composites",
    "parcel",
    "severe",
    "xarray",
    "era5",
    "calc_cape",
    "calc_srh",
    "calc_severe",
]

_lazy_functions = {
    "calc_cape": "core",
    "calc_srh": "core",
    "calc_severe": "severe",
}
_submodules = {"core", "kinematics", "thermo", "composites", "parcel", "severe", "xarray", "era5"}


def __getattr__(name):
    # Lazy imports keep ``import xcaper`` cheap (numba and dask are only
    # imported when something that needs them is first used).
    import importlib

    if name in _submodules:
        return importlib.import_module(f".{name}", __name__)
    if name in _lazy_functions:
        module = importlib.import_module(f".{_lazy_functions[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
