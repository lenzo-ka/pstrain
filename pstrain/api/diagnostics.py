"""Public API for environment diagnostics."""

from pstrain.lib._cffi.core import _find_library
from pstrain.lib.commands import PSTRAIN_BINARIES, resolve_binary
from pstrain.lib.runtime import fp_contract_policy

__all__ = [
    "PSTRAIN_BINARIES",
    "fp_contract_policy",
    "native_library_available",
    "resolve_binary",
]


def native_library_available() -> bool:
    """Return whether the native pstrain library can be loaded."""
    try:
        _find_library()
        return True
    except RuntimeError:
        return False
