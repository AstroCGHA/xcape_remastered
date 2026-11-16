"""Dispatch between the parallel and the serial flavour of a numba grid loop."""
from __future__ import annotations

import threading

# numba's default ``workqueue`` threading layer must not be entered from two
# Python threads at once (the process is terminated when that happens).  The
# parallel loops are therefore serialised with a lock: concurrent callers
# simply queue up, each call still using every numba thread.  Inside dask
# tasks xcaper uses the serial, GIL-free loops instead, so dask worker threads
# never wait for this lock.
_PARALLEL_LOCK = threading.Lock()


def run_loop(parallel_loop, serial_loop, parallel, *args):
    """Call ``parallel_loop(*args)`` under the lock, or ``serial_loop(*args)``."""
    if parallel:
        with _PARALLEL_LOCK:
            return parallel_loop(*args)
    return serial_loop(*args)
