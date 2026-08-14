"""Regression gates for the shared-library export boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from cffi import FFI

from pstrain.lib._cffi.cdef import CDEF
from pstrain.lib._cffi.core import _find_library
from tests.conftest import requires_c_library

pytestmark = requires_c_library


def _exports(path: Path) -> set[str]:
    if sys.platform == "darwin":
        command = ["nm", "-gU", str(path)]
    else:
        command = ["nm", "-D", "--defined-only", str(path)]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    return {line.split()[-1].removeprefix("_").split("@", 1)[0] for line in output.splitlines()}


def _cffi_functions() -> set[str]:
    ffi = FFI()
    ffi.cdef(CDEF)
    return {
        key.removeprefix("function ")
        for key in ffi._parser._declarations  # type: ignore[attr-defined]
        if key.startswith("function ")
    }


def _pocketsphinx_library(pstrain_library: Path) -> Path:
    candidates = sorted(pstrain_library.parent.glob("libpocketsphinx*"))
    candidates = [path for path in candidates if path.is_file() and not path.name.endswith(".a")]
    assert candidates, f"PocketSphinx library not found beside {pstrain_library}"
    return candidates[0]


def test_exports_are_exactly_the_cffi_surface() -> None:
    """No undeclared native implementation symbol may escape the library."""
    assert _exports(_find_library()) == _cffi_functions()


def test_no_pocketsphinx_symbol_collisions() -> None:
    """Fail if libpstrainc again participates in PocketSphinx symbol resolution."""
    pstrain_library = _find_library()
    collisions = _exports(pstrain_library) & _exports(_pocketsphinx_library(pstrain_library))
    assert collisions == set(), f"exported PocketSphinx collisions: {sorted(collisions)}"
