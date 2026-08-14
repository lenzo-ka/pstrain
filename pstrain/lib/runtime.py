"""Contained queries about the loaded native runtime."""

from __future__ import annotations

from typing import cast

from pstrain.lib._cffi.core import get_ffi, get_lib
from pstrain.lib.native_worker import contained


@contained
def fp_contract_policy() -> str:
    """Return the floating-point contraction policy declared by the native build."""
    return cast(str, get_ffi().string(get_lib().pstrain_fp_contract_policy()).decode("ascii"))
