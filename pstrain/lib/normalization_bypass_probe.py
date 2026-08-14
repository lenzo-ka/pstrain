"""Deliberately unsafe probability consumer used by the normalization lane probe.

This module is intentionally temporary.  It models the smallest plausible new
consumer: load the serialized arrays through the public low-level readers and
use their values directly as probabilities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pstrain.lib import _pstrainc


def unnormalized_path_score(model_dir: str | Path) -> float:
    """Return a deliberately invalid score computed directly from raw counts."""
    model_dir = Path(model_dir)
    mixw = _pstrainc.read_mixw(str(model_dir / "mixture_weights"))[0]
    tmat = _pstrainc.read_tmat(str(model_dir / "transition_matrices"))[0]
    return float(np.log(mixw[0, 0, 0]) + np.log(tmat[0, 0, 0]))
