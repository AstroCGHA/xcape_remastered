"""Compatibility module defining operations on duck numpy-arrays.

Dispatches to dask for dask array inputs and to numpy otherwise.
"""
from __future__ import annotations

from functools import reduce

import numpy as np

try:
    import dask.array as dsa

    has_dask = True
except ImportError:  # pragma: no cover - dask is a hard dependency of xcaper
    dsa = None
    has_dask = False


def _dask_or_eager_func(name, eager_module=np, list_of_args=False, n_array_args=1):
    """Create a function that dispatches to dask for dask array inputs."""
    if has_dask:

        def f(*args, **kwargs):
            dispatch_args = args[0] if list_of_args else args
            if any(isinstance(a, dsa.Array) for a in dispatch_args[:n_array_args]):
                module = dsa
            else:
                module = eager_module
            return getattr(module, name)(*args, **kwargs)

    else:

        def f(data, *args, **kwargs):
            return getattr(eager_module, name)(data, *args, **kwargs)

    f.__name__ = name
    return f


reshape = _dask_or_eager_func("reshape")
concatenate = _dask_or_eager_func("concatenate", list_of_args=True)
broadcast_arrays = _dask_or_eager_func("broadcast_arrays", n_array_args=2**31)


def any_dask_array(*args) -> bool:
    """True when at least one of the arguments is a dask array."""
    return has_dask and any(isinstance(a, dsa.Array) for a in args)


def ravel_multi_index(multi_index, dims, order="C"):
    """Dask-friendly clone of :func:`numpy.ravel_multi_index`."""
    # implementation from https://stackoverflow.com/a/20266350/3266235
    if len(multi_index) != len(dims):
        raise ValueError("multi_index and dims must have the same length")
    if order == "C":
        offset_factors = [reduce(lambda x, y: x * y, dims[n:]) for n in range(1, len(dims))] + [1]
    elif order == "F":
        offset_factors = [1] + [reduce(lambda x, y: x * y, dims[:n]) for n in range(1, len(dims))]
    else:
        raise ValueError("order must be 'C' or 'F'")
    full_index = [of * ix for of, ix in zip(offset_factors, multi_index)]
    return sum(full_index)
