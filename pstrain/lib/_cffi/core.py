"""Core FFI initialization and helpers.

This module handles library discovery, FFI initialization, and common utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cffi import FFI

from pstrain.lib._cffi.cdef import CDEF

_ffi: FFI | None = None
_lib: Any = None
PSTRAIN_ABI_VERSION = 1


def _find_library() -> Path:
    """Find the pstrainc shared library.

    Uses pstrain.lib.paths for discovery, which checks:
    1. Environment variables (PSTRAIN_LIB_PATH)
    2. Bundled in wheel (pstrain/_lib/lib/)
    3. Development build (build/lib/)
    4. System library paths
    """
    # Import here to avoid circular import
    from pstrain.lib.paths import get_lib_path

    lib_path = get_lib_path()
    if lib_path:
        return lib_path

    raise RuntimeError(
        "Could not find libpstrainc. Options:\n"
        "  - Build: cmake -S . -B build && cmake --build build\n"
        "  - Install: pip install pstrain (from wheel with bundled library)\n"
        "  - Set PSTRAIN_LIB_PATH environment variable"
    )


def _init() -> tuple[FFI, Any]:
    """Initialize FFI and load library."""
    global _ffi, _lib

    if _ffi is not None and _lib is not None:
        return _ffi, _lib

    _ffi = FFI()
    _ffi.cdef(CDEF)

    lib_path = _find_library()
    lib = _ffi.dlopen(str(lib_path))
    try:
        actual_abi = int(lib.pstrain_abi_version())
    except AttributeError as exc:
        _ffi = None
        _lib = None
        raise RuntimeError(
            "libpstrainc ABI mismatch: Python expects ABI version "
            f"{PSTRAIN_ABI_VERSION}, but the library has no ABI version handshake "
            "(pre-handshake library)"
        ) from exc
    if actual_abi != PSTRAIN_ABI_VERSION:
        _ffi = None
        _lib = None
        raise RuntimeError(
            "libpstrainc ABI mismatch: Python expects ABI version "
            f"{PSTRAIN_ABI_VERSION}, library reports {actual_abi}"
        )
    _lib = lib

    return _ffi, _lib


def get_ffi() -> FFI:
    """Get the FFI instance."""
    ffi, _ = _init()
    return ffi


def get_lib() -> Any:
    """Get the loaded library with all C functions."""
    _, lib = _init()
    return lib


def path_or_null(path: Path | str | None) -> bytes | Any:
    """Convert a path to bytes, or return NULL if None.

    Helper to reduce boilerplate when passing optional path arguments to C.

    Args:
        path: Path to encode, or None

    Returns:
        Encoded path bytes, or ffi.NULL if path is None
    """
    if path is None:
        return get_ffi().NULL
    return str(path).encode()
