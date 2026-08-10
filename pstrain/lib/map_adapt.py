"""MAP (Maximum A Posteriori) adaptation.

This module provides CFFI-based wrappers for MAP adaptation, which updates
acoustic model parameters to maximize the posterior probability given
adaptation data and a prior distribution derived from the baseline model.

Example usage:
    from pstrain.lib.map_adapt import map_adapt

    # Adapt a speaker-independent model
    map_adapt(
        mean_path="si_model/means",
        var_path="si_model/variances",
        mixw_path="si_model/mixw",
        output_mean_path="adapted/means",
        accum_dirs=["adapt_bwaccum"]
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from . import _pstrainc

__all__ = ["map_adapt"]


def map_adapt(
    mean_path: Path | str,
    var_path: Path | str,
    accum_dirs: Sequence[Path | str],
    output_mean_path: Path | str,
    mixw_path: Path | str | None = None,
    tmat_path: Path | str | None = None,
    output_var_path: Path | str | None = None,
    output_mixw_path: Path | str | None = None,
    output_tmat_path: Path | str | None = None,
    mdef_path: Path | str | None = None,
    ts2cb_path: Path | str | None = None,
    tau: float = 10.0,
    fixed_tau: bool = False,
    bayes_mean: bool = True,
    mw_floor: float = 1e-5,
    var_floor: float = 1e-5,
    tp_floor: float = 1e-4,
) -> int:
    """Perform MAP adaptation of acoustic model parameters.

    MAP adaptation updates model parameters using a Bayesian framework where
    the prior distribution is derived from the speaker-independent (or other
    baseline) model. This allows adaptation even with limited data.

    Args:
        mean_path: Path to baseline Gaussian means (required)
        var_path: Path to baseline Gaussian variances (required)
        accum_dirs: List of accumulator directories from BW on adaptation data
        output_mean_path: Path to write adapted means (required)
        mixw_path: Path to baseline mixture weights (required if adapting mixw)
        tmat_path: Path to baseline transition matrices (required if adapting tmat)
        output_var_path: Path to write adapted variances (None to skip)
        output_mixw_path: Path to write adapted mixture weights (None to skip)
        output_tmat_path: Path to write adapted transition matrices (None to skip)
        mdef_path: Model definition file (required for tied-state models)
        ts2cb_path: Tied-state to codebook mapping file (or ".semi", ".cont", ".ptm")
        tau: Prior weight hyperparameter (default 10.0)
        fixed_tau: Use fixed tau value (True) or estimate from data (False)
        bayes_mean: Use Bayesian mean estimation (True) or MAP (False)
        mw_floor: Mixture weight floor (default 1e-5)
        var_floor: Variance floor (default 1e-5)
        tp_floor: Transition probability floor (default 1e-4)

    Returns:
        0 on success, non-zero on error

    Raises:
        ValueError: If required parameters are missing
        RuntimeError: If the C library call fails

    Note:
        - bayes_mean=True (default) uses simple Bayesian updating which ignores
          tau and typically works better for most cases
        - Variance adaptation (output_var_path) doesn't work with -2passvar and
          can sometimes degrade accuracy; use with caution
        - For tied-state models, mdef_path and ts2cb_path are needed for mixture
          weight adaptation
    """
    if not accum_dirs:
        raise ValueError("Must specify at least one accumulator directory")

    ffi = _pstrainc.get_ffi()
    lib = _pstrainc.get_lib()

    # Create NULL-terminated array of accumulator directories
    accum_dirs_c = ffi.new("char*[]", len(accum_dirs) + 1)
    for i, d in enumerate(accum_dirs):
        accum_dirs_c[i] = ffi.new("char[]", str(d).encode())
    accum_dirs_c[len(accum_dirs)] = ffi.NULL

    result: int = lib.pstrain_map_adapt(
        str(mean_path).encode(),
        str(var_path).encode(),
        _pstrainc.path_or_null(mixw_path),
        _pstrainc.path_or_null(tmat_path),
        accum_dirs_c,
        str(output_mean_path).encode(),
        _pstrainc.path_or_null(output_var_path),
        _pstrainc.path_or_null(output_mixw_path),
        _pstrainc.path_or_null(output_tmat_path),
        _pstrainc.path_or_null(mdef_path),
        _pstrainc.path_or_null(ts2cb_path),
        tau,
        1 if fixed_tau else 0,
        1 if bayes_mean else 0,
        mw_floor,
        var_floor,
        tp_floor,
    )

    if result != 0:
        raise RuntimeError(f"MAP adaptation failed for {mean_path}")

    return result
