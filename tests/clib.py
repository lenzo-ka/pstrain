"""Single source of truth for libpstrainc availability in the test suite.

Detection is based on whether the library can actually be *loaded* through
the same discovery path the package uses at runtime
(``pstrain.lib._pstrainc.get_lib`` → ``pstrain.lib.paths``), not on hardcoded ``build/``
locations. The old per-file checks only probed ``build/libpstrainc.{dylib,so}``
and so silently skipped every CFFI test whenever the library lived in
``build/lib/`` (e.g. the CI root-configured build on macOS).

Set ``PSTRAIN_REQUIRE_CLIB=1`` to turn "library missing" from a skip into a hard
collection error, so CI asserts the C library was actually exercised instead
of quietly degrading to a Python-only run. See ``conftest.py`` for the gate.
"""

from __future__ import annotations

import functools
import os

import pytest


@functools.lru_cache(maxsize=1)
def c_library_available() -> bool:
    """Return True if libpstrainc can be loaded via the real discovery path."""
    try:
        from pstrain.lib._pstrainc import get_lib

        get_lib()
    except Exception:
        return False
    return True


def require_clib_env() -> bool:
    """True if the environment demands the C library be present (CI gate)."""
    return os.environ.get("PSTRAIN_REQUIRE_CLIB", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


C_LIBRARY_AVAILABLE = c_library_available()

requires_c_library = pytest.mark.skipif(
    not C_LIBRARY_AVAILABLE,
    reason="libpstrainc not loadable (build it: 'make build-c'). "
    "Set PSTRAIN_REQUIRE_CLIB=1 to make this a hard failure instead of a skip.",
)
"""Skip a test when libpstrainc is unavailable (unless PSTRAIN_REQUIRE_CLIB gates it)."""
