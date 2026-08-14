"""Tied-state to codebook mapping.

Creates and manages the mapping between tied states and Gaussian codebooks.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pstrain.lib import _pstrainc
from pstrain.lib.native_worker import contained

if TYPE_CHECKING:
    import numpy.typing as npt


class TyingType(Enum):
    """Gaussian tying types."""

    SEMI = "semi"  # Semi-continuous: all states share one codebook
    CONT = "cont"  # Continuous: each state has its own codebook
    PTM = "ptm"  # Phone-tied mixture: states share by phone


@contained
def create_ts2cb(
    n_tied_state: int,
    tying_type: TyingType | str = TyingType.CONT,
    mdef_path: Path | None = None,
) -> tuple[npt.NDArray[np.uint32], int]:
    """Create a tied-state to codebook mapping.

    Args:
        n_tied_state: Number of tied states.
        tying_type: Type of Gaussian tying.
        mdef_path: Model definition path (required for PTM tying).

    Returns:
        Tuple of (ts2cb array, n_codebooks).

    Raises:
        ValueError: If tying_type is PTM but mdef_path not provided.
    """
    if isinstance(tying_type, str):
        tying_type = TyingType(tying_type)

    lib = _pstrainc.get_lib()

    if tying_type == TyingType.SEMI:
        ptr = lib.semi_ts2cb(n_tied_state)
        n_cb = 1
    elif tying_type == TyingType.CONT:
        ptr = lib.cont_ts2cb(n_tied_state)
        n_cb = n_tied_state
    elif tying_type == TyingType.PTM:
        if mdef_path is None:
            raise ValueError("PTM tying requires mdef_path")
        # For PTM, we need to read the mdef and get the mapping
        # This requires the model_def_t structure which is complex
        raise NotImplementedError("PTM tying not yet implemented via CFFI")
    else:
        raise ValueError(f"Unknown tying type: {tying_type}")

    # Convert C array to numpy
    ts2cb = np.zeros(n_tied_state, dtype=np.uint32)
    for i in range(n_tied_state):
        ts2cb[i] = ptr[i]

    # Free the C-allocated memory
    lib.pstrain_cffi_ckd_free(ptr)

    return ts2cb, n_cb


@contained
def read_ts2cb(path: Path) -> tuple[npt.NDArray[np.uint32], int]:
    """Read a tied-state to codebook mapping from a file.

    Args:
        path: Path to the ts2cb file.

    Returns:
        Tuple of (ts2cb array, n_codebooks).

    Raises:
        RuntimeError: If reading fails.
    """
    lib = _pstrainc.get_lib()
    ffi = _pstrainc.get_ffi()

    out_ts2cb = ffi.new("uint32 **")
    out_n_ts = ffi.new("uint32 *")
    out_n_cb = ffi.new("uint32 *")

    ret = lib.s3ts2cb_read(str(path).encode(), out_ts2cb, out_n_ts, out_n_cb)
    if ret != 0:
        raise RuntimeError(f"Failed to read ts2cb file: {path}")

    n_ts = out_n_ts[0]
    n_cb = out_n_cb[0]
    ptr = out_ts2cb[0]

    # Convert to numpy
    ts2cb = np.zeros(n_ts, dtype=np.uint32)
    for i in range(n_ts):
        ts2cb[i] = ptr[i]

    # Free C memory
    lib.pstrain_cffi_ckd_free(ptr)

    return ts2cb, n_cb


@contained
def write_ts2cb(
    path: Path,
    ts2cb: npt.NDArray[np.uint32],
    n_cb: int | None = None,
) -> None:
    """Write a tied-state to codebook mapping to a file.

    Args:
        path: Output path.
        ts2cb: Tied-state to codebook mapping array.
        n_cb: Number of codebooks (if None, computed from max(ts2cb) + 1).

    Raises:
        RuntimeError: If writing fails.
    """
    lib = _pstrainc.get_lib()
    ffi = _pstrainc.get_ffi()

    ts2cb = np.ascontiguousarray(ts2cb, dtype=np.uint32)
    n_ts = len(ts2cb)

    if n_cb is None:
        n_cb = int(ts2cb.max()) + 1

    ptr = ffi.cast("uint32 *", ts2cb.ctypes.data)

    ret = lib.s3ts2cb_write(str(path).encode(), ptr, n_ts, n_cb)
    if ret != 0:
        raise RuntimeError(f"Failed to write ts2cb file: {path}")


def _parse_mdef_n_tied_state(mdef_path: Path) -> int:
    """Parse the n_tied_state from an mdef file.

    Args:
        mdef_path: Path to model definition file.

    Returns:
        Number of tied states.

    Raises:
        RuntimeError: If mdef file is invalid.
    """
    with mdef_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "n_tied_state":
                return int(parts[0])

    raise RuntimeError(f"Could not find n_tied_state in mdef: {mdef_path}")


@contained
def create_and_write_ts2cb(
    mdef_path: Path,
    output_path: Path,
    tying_type: TyingType | str = TyingType.CONT,
) -> tuple[npt.NDArray[np.uint32], int]:
    """Create a ts2cb mapping from an mdef file and write it.

    This is a convenience function that combines reading the mdef,
    creating the mapping, and writing it to a file.

    Args:
        mdef_path: Path to model definition file.
        output_path: Path to write ts2cb file.
        tying_type: Type of Gaussian tying.

    Returns:
        Tuple of (ts2cb array, n_codebooks).

    Raises:
        RuntimeError: If mdef parsing or ts2cb writing fails.
    """
    # Parse mdef to get n_tied_state
    n_tied_state = _parse_mdef_n_tied_state(mdef_path)

    # Create mapping
    ts2cb_arr, n_cb = create_ts2cb(n_tied_state, tying_type)

    # Write to file
    write_ts2cb(output_path, ts2cb_arr, n_cb)

    return ts2cb_arr, n_cb
