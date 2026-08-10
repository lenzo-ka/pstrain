"""Training steps (internal library).

Numbered training stages:
- 00.setup - Project setup (dictionary, phoneset, config)
- features - Feature extraction (before numbered steps)
- 20.ci_hmm - Context-independent HMM training
- 30.cd_hmm_untied - Context-dependent untied HMM training
- 40.buildtrees - Decision tree building
- 45.prunetree - State tying
- 50.cd_hmm_tied - Context-dependent tied HMM training

The pipeline runner (pstrain.lib.pipeline) determines execution order from
declared inputs/outputs.

These are internal library modules. Use pstrain.api.steps for the public API.
"""

from pstrain.lib.steps.base import Step, StepContext, StepDefinition
from pstrain.lib.steps.cd_hmm_untied import CDHMMUntiedStep, cd_hmm_untied_step
from pstrain.lib.steps.cd_pipeline import (
    run_build_trees,
    run_init_cd_tied,
    run_init_cd_untied,
    run_make_questions,
    run_prune_trees,
    run_tiestate,
    run_untied_mdef,
)
from pstrain.lib.steps.ci_hmm import CIHMMStep, ci_hmm_step
from pstrain.lib.steps.features import FeaturesStep, features_step
from pstrain.lib.steps.package import create_noisedict, package_model
from pstrain.lib.steps.split import run_split
from pstrain.lib.steps.train import TrainingIteration, TrainingResult, run_bw_training

__all__ = [
    # Base classes
    "Step",
    "StepContext",
    "StepDefinition",
    # Step classes
    "FeaturesStep",
    "CIHMMStep",
    "CDHMMUntiedStep",
    # Step instances
    "features_step",
    "ci_hmm_step",
    "cd_hmm_untied_step",
    # Step functions
    "run_bw_training",
    "run_split",
    "TrainingResult",
    "TrainingIteration",
    # CD pipeline functions
    "run_untied_mdef",
    "run_init_cd_untied",
    "run_make_questions",
    "run_build_trees",
    "run_prune_trees",
    "run_tiestate",
    "run_init_cd_tied",
    # Packaging functions
    "package_model",
    "create_noisedict",
]
