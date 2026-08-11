"""KD-tree construction for fast Gaussian selection.

This module provides CFFI-based wrappers for building KD-trees from acoustic
model parameters, which are used for fast Gaussian selection during decoding.

Example usage:
    from pstrain.lib.kdtree import build_kdtree

    # Build KD-tree from trained model
    build_kdtree(
        mean_path="model/means",
        var_path="model/variances",
        output_path="model/kdtree",
        threshold=0.2,
        depth=6
    )
"""

from __future__ import annotations

from pathlib import Path

from . import _pstrainc
from .native_worker import contained

__all__ = ["build_kdtree"]


@contained
def build_kdtree(
    mean_path: Path | str,
    var_path: Path | str,
    output_path: Path | str | None = None,
    threshold: float = 0.2,
    depth: int = 6,
    absolute: bool = False,
) -> int:
    """Build KD-trees for fast Gaussian selection.

    Creates KD-tree data structures from the means and variances of a
    semi-continuous acoustic model. These trees enable fast approximate
    nearest neighbor search during decoding.

    Args:
        mean_path: Path to Gaussian means file (S3 format)
        var_path: Path to Gaussian variances file (S3 format)
        output_path: Path to write KD-tree file (None to skip writing)
        threshold: Gaussian box threshold (0.0-1.0, default 0.2)
        depth: Number of tree levels (default 6)
        absolute: Use absolute threshold calculation (default False)

    Returns:
        0 on success, non-zero on error

    Raises:
        RuntimeError: If the C library call fails

    Note:
        KD-trees are primarily used with semi-continuous models where
        all states share a single Gaussian codebook.
    """
    lib = _pstrainc.get_lib()

    result: int = lib.pstrain_kdtree_build(
        str(mean_path).encode(),
        str(var_path).encode(),
        _pstrainc.path_or_null(output_path),
        threshold,
        depth,
        1 if absolute else 0,
    )

    if result != 0:
        raise RuntimeError(f"Failed to build KD-tree from {mean_path}")

    return result
