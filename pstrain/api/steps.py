"""Public API for training steps.

This module exposes step functions from lib.steps through the public API.
CLI and web clients should use these functions, not call lib.steps directly.

Steps can be run via CLI: pstrain step ci_hmm [args]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pstrain.lib.steps import (
    CDHMMUntiedStep,
    CIHMMStep,
    FeaturesStep,
    Step,
    StepContext,
    StepDefinition,
    cd_hmm_untied_step,
    ci_hmm_step,
    features_step,
)

# Re-export base classes
__all__ = [
    "Step",
    "StepContext",
    "StepDefinition",
    # Step instances
    "features_step",
    "ci_hmm_step",
    "cd_hmm_untied_step",
    # Step classes
    "FeaturesStep",
    "CIHMMStep",
    "CDHMMUntiedStep",
    # Convenience functions
    "step_features",
    "run_step_features",
    "step_ci_hmm",
    "run_step_ci_hmm",
    "step_cd_hmm_untied",
    "run_step_cd_hmm_untied",
]


# =============================================================================
# Feature extraction
# =============================================================================


def step_features(
    project_dir: Path | str,
    experiment: str = "default",
    config: str = "baseline",
    **params: Any,
) -> dict[str, Any]:
    """Get the rule definition for feature extraction.

    Args:
        project_dir: Project directory path
        experiment: Experiment name
        config: Configuration name
        **params: Additional parameters

    Returns:
        StepDefinition with rule metadata
    """
    ctx = StepContext(Path(project_dir), experiment, config)
    return features_step.to_dict(ctx, **params)


def run_step_features(
    project_dir: Path | str,
    experiment: str = "default",
    config: str = "baseline",
    dry_run: bool = False,
    **params: Any,
) -> int:
    """Run feature extraction step.

    Args:
        project_dir: Project directory path
        experiment: Experiment name
        config: Configuration name
        dry_run: If True, show what would be done without executing
        **params: Additional parameters

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    return features_step.run(project_dir, experiment, config, dry_run=dry_run, **params)


# =============================================================================
# CI HMM training
# =============================================================================


def step_ci_hmm(
    project_dir: Path | str,
    experiment: str = "default",
    config: str = "baseline",
    **params: Any,
) -> dict[str, Any]:
    """Get the rule definition for CI HMM training.

    Args:
        project_dir: Project directory path
        experiment: Experiment name
        config: Model configuration name
        **params: Training parameters

    Returns:
        StepDefinition with rule metadata
    """
    ctx = StepContext(Path(project_dir), experiment, config)
    return ci_hmm_step.to_dict(ctx, **params)


def run_step_ci_hmm(
    project_dir: Path | str,
    experiment: str = "default",
    config: str = "baseline",
    dry_run: bool = False,
    **params: Any,
) -> int:
    """Run CI HMM training.

    Args:
        project_dir: Project directory path
        experiment: Experiment name
        config: Model configuration name
        dry_run: If True, show what would be done without executing
        **params: Training parameters

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    return ci_hmm_step.run(project_dir, experiment, config, dry_run=dry_run, **params)


# =============================================================================
# CD HMM untied training
# =============================================================================


def step_cd_hmm_untied(
    project_dir: Path | str,
    experiment: str = "default",
    config: str = "baseline",
    **params: Any,
) -> dict[str, Any]:
    """Get the rule definition for CD HMM untied training.

    Args:
        project_dir: Project directory path
        experiment: Experiment name
        config: Model configuration name
        **params: Training parameters

    Returns:
        StepDefinition with rule metadata
    """
    ctx = StepContext(Path(project_dir), experiment, config)
    return cd_hmm_untied_step.to_dict(ctx, **params)


def run_step_cd_hmm_untied(
    project_dir: Path | str,
    experiment: str = "default",
    config: str = "baseline",
    dry_run: bool = False,
    **params: Any,
) -> int:
    """Run CD HMM untied training.

    Args:
        project_dir: Project directory path
        experiment: Experiment name
        config: Model configuration name
        dry_run: If True, show what would be done without executing
        **params: Training parameters

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    return cd_hmm_untied_step.run(project_dir, experiment, config, dry_run=dry_run, **params)
