"""Deleted interpolation for mixture weight smoothing.

This module provides CFFI-based wrappers for deleted interpolation, which
smooths context-dependent mixture weights by interpolating between CD, CI,
and uniform distributions using held-out data.

Example usage:
    from pstrain.lib.delint import deleted_interpolation

    # Smooth mixture weights
    deleted_interpolation(
        mdef_path="model/mdef",
        output_path="model/mixw_smoothed",
        accum_dirs=["bwaccum/part1", "bwaccum/part2"],
        ci_lambda=0.9
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from . import _pstrainc

__all__ = ["deleted_interpolation"]


def deleted_interpolation(
    mdef_path: Path | str,
    output_path: Path | str,
    accum_dirs: Sequence[Path | str],
    ci_lambda: float = 0.9,
    max_iter: int = 100,
) -> int:
    """Perform deleted interpolation to smooth mixture weights.

    Deleted interpolation is an EM-style algorithm that interpolates between:
    - Context-dependent (CD) distributions
    - Context-independent (CI) distributions
    - Uniform distributions

    The interpolation weights are estimated from held-out data to prevent
    overfitting, especially for rarely-seen triphones.

    Args:
        mdef_path: Path to model definition file
        output_path: Path to write smoothed mixture weights
        accum_dirs: List of accumulator directories from BW training.
                    Must be an even number (for cross-validation).
        ci_lambda: CI interpolation weight (0.0-1.0, default 0.9)
        max_iter: Maximum iterations for convergence (default 100)

    Returns:
        0 on success, non-zero on error

    Raises:
        ValueError: If fewer than 2 accumulator directories specified
        RuntimeError: If the C library call fails

    Note:
        This is typically the final step in semi-continuous model training.
        For continuous models, mixture weight smoothing is usually not needed.

        The accumulator directories must contain mixw_counts files from
        separate partitions of the training data.
    """
    if len(accum_dirs) < 2:
        raise ValueError("Must specify at least 2 accumulator directories")

    if len(accum_dirs) % 2 != 0:
        raise ValueError("Must specify an even number of accumulator directories")

    ffi = _pstrainc.get_ffi()
    lib = _pstrainc.get_lib()

    # Create NULL-terminated array of strings
    accum_dirs_c = ffi.new("char*[]", len(accum_dirs) + 1)
    for i, d in enumerate(accum_dirs):
        accum_dirs_c[i] = ffi.new("char[]", str(d).encode())
    accum_dirs_c[len(accum_dirs)] = ffi.NULL

    result: int = lib.pstrain_delint(
        str(mdef_path).encode(),
        str(output_path).encode(),
        accum_dirs_c,
        ci_lambda,
        max_iter,
    )

    if result != 0:
        raise RuntimeError(f"Deleted interpolation failed for {mdef_path}")

    return result
