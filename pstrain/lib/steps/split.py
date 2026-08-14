"""Gaussian splitting step function.

Doubles the number of Gaussian components in a model by splitting.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np

from pstrain.lib import _pstrainc
from pstrain.lib.model import MODEL_FILES_REQUIRED
from pstrain.lib.split import double_gaussians
from pstrain.lib.validate import validate_files_exist

logger = logging.getLogger(__name__)

# Default count per Gaussian for uniform initialization (used when no training counts available)
DEFAULT_UNIFORM_COUNT = 1000.0

__all__ = ["run_split", "create_uniform_counts", "DEFAULT_UNIFORM_COUNT"]


def create_uniform_counts(
    mixw_path: Path,
    output_path: Path,
    count_per_gaussian: float = DEFAULT_UNIFORM_COUNT,
) -> None:
    """Create uniform density counts for Gaussian splitting.

    When no training counts are available (e.g., for flat models),
    this creates uniform counts that will cause all Gaussians to be
    split equally.

    Args:
        mixw_path: Path to mixture weights file (to determine dimensions)
        output_path: Output path for density counts file
        count_per_gaussian: Count value per Gaussian (default 1000)
    """
    # Read mixture weights to get dimensions
    mixw_raw, n_mixw, n_feat, n_density = _pstrainc.read_mixw_counts(str(mixw_path))

    # Create uniform counts: (n_cb, n_feat, n_density)
    # For CI models, n_cb = n_mixw (one codebook per senone)
    dnom = np.full((n_mixw, n_feat, n_density), count_per_gaussian, dtype=np.float32)

    # Write counts file
    _pstrainc.write_dnom(str(output_path), dnom)
    logger.info("Created uniform counts: %s (shape: %s)", output_path, dnom.shape)


def run_split(
    input_model_dir: Path,
    output_model_dir: Path,
    dcount_path: Path | None = None,
    create_counts_if_missing: bool = True,
) -> dict[str, Path]:
    """Double Gaussian density in a model.

    Splits each Gaussian component into two, doubling the model capacity.
    Uses density counts from BW training to determine which Gaussians to split.

    Args:
        input_model_dir: Directory containing input model (means, variances,
            mixture_weights, mdef, transition_matrices)
        output_model_dir: Directory to write split model
        dcount_path: Path to density counts file from BW training.
            If None, uses input_model_dir/gauden_counts
        create_counts_if_missing: If True and dcount_path doesn't exist,
            create uniform counts (default True)

    Returns:
        Dict mapping file type to output path

    Raises:
        FileNotFoundError: If required files are missing
        RuntimeError: If splitting fails
    """
    input_model_dir = Path(input_model_dir)
    output_model_dir = Path(output_model_dir)

    # Default density counts location
    if dcount_path is None:
        dcount_path = input_model_dir / "gauden_counts"

    # Validate inputs (except dcount which may be created)
    validate_files_exist(
        [input_model_dir / f for f in MODEL_FILES_REQUIRED],
        context="Gaussian splitting",
    )

    # Create output directory
    output_model_dir.mkdir(parents=True, exist_ok=True)

    # Create uniform counts if needed
    if not dcount_path.exists():
        if create_counts_if_missing:
            logger.info("Density counts not found, creating uniform counts")
            create_uniform_counts(
                mixw_path=input_model_dir / "mixture_weights",
                output_path=dcount_path,
            )
        else:
            raise FileNotFoundError(f"Density counts not found: {dcount_path}")

    logger.info("Splitting Gaussians: %s -> %s", input_model_dir, output_model_dir)

    # Split means, variances, mixture weights
    result = double_gaussians(
        model_dir=input_model_dir,
        dcount_path=dcount_path,
        output_dir=output_model_dir,
    )

    logger.info("Split complete: %s", result)

    # Copy unchanged files
    shutil.copy(input_model_dir / "mdef", output_model_dir / "mdef")
    shutil.copy(
        input_model_dir / "transition_matrices",
        output_model_dir / "transition_matrices",
    )

    return {f: output_model_dir / f for f in MODEL_FILES_REQUIRED}
