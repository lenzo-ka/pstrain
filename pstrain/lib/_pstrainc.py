"""Low-level CFFI bindings to libpstrainc.

This module provides direct access to C functions in libpstrainc.  In the
``contained-3`` phase, only ``prune_tree``, ``make_quests``, and
``mdef_gen_ci`` are routed through the persistent native worker.  All other
operations retain their existing in-process path.
For most use cases, prefer the higher-level Python wrappers in pstrain.lib.

Implementation is organized into submodules:
- _cffi.cdef: C type definitions
- _cffi.core: FFI initialization and helpers
- _cffi.io: Model file I/O
- _cffi.logmath: Log-domain math wrapper
"""

from __future__ import annotations

from typing import Any

# Re-export from submodules
from pstrain.lib._cffi.cdef import CDEF
from pstrain.lib._cffi.core import _find_library, _init, get_ffi, get_lib, path_or_null
from pstrain.lib._cffi.io import (
    read_dnom,
    read_gau,
    read_mixw,
    read_tmat,
    write_dnom,
    write_gau,
    write_mixw,
    write_tmat,
)
from pstrain.lib._cffi.logmath import LogMath

__all__ = [
    # CDEF
    "CDEF",
    # Core
    "_init",
    "_find_library",
    "get_ffi",
    "get_lib",
    "path_or_null",
    # I/O
    "read_mixw",
    "write_mixw",
    "read_tmat",
    "write_tmat",
    "read_gau",
    "write_gau",
    "read_dnom",
    "write_dnom",
    # LogMath
    "LogMath",
]


# Convenience: allow direct attribute access to C functions
def __getattr__(name: str) -> Any:
    """Allow pstrainc.function_name() syntax."""
    lib = get_lib()
    if hasattr(lib, name):
        return getattr(lib, name)
    raise AttributeError(f"module 'pstrain.lib._pstrainc' has no attribute '{name}'")
