"""Feature extraction step.

Extracts acoustic features from audio files using sphinx_fe.
Features are shared across experiments in shared/features/{feature_set_id}/.

Note: This runs before numbered stages. The pipeline runner determines order
from declared inputs/outputs.

Usage:
    Library: from pstrain.lib.steps.features import FeaturesStep
    CLI: python -m pstrain.lib.steps.features [args]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pstrain.lib.config import DEFAULT_FEAT_PARAMS
from pstrain.lib.steps.base import Step, StepContext


class FeaturesStep(Step):
    """Feature extraction step."""

    name = "features"
    description = "Extract acoustic features from audio using sphinx_fe"
    script = "sphinx_fe"

    default_params: dict[str, Any] = {
        k: v for k, v in DEFAULT_FEAT_PARAMS.items() if k != "feat_type"
    }

    def get_inputs(self, ctx: StepContext) -> list[Path]:
        """Get input files for feature extraction."""
        return [
            ctx.project_dir / "audio",
            ctx.experiment_dir / "etc" / "train.fileids",
            ctx.experiment_dir / "etc" / "test.fileids",
        ]

    def get_outputs(self, ctx: StepContext) -> list[Path]:
        """Get output files from feature extraction."""
        feature_dir = ctx.shared_dir / "features" / "default"
        return [
            feature_dir,
            feature_dir / "feat.params",
        ]

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add feature extraction specific arguments."""
        super().add_arguments(parser)
        parser.add_argument(
            "--feature-set-id",
            type=str,
            default="default",
            help="Feature set ID (default: default)",
        )
        parser.add_argument(
            "-j",
            "--jobs",
            type=int,
            default=None,
            help="Parallel jobs (default: auto, bounded by CPU count and batch size)",
        )

    def execute(self, ctx: StepContext, **params: Any) -> int:
        """Execute feature extraction via the pipeline runner.

        This is the single-step variant of `pstrain build features`. It builds a
        pipeline scoped to the current project/experiment/config and runs the
        "features" target.
        """
        from pstrain.lib.pipeline import PipelineContext
        from pstrain.lib.pipeline.tasks import build_pipeline

        jobs = params.get("jobs")
        config_name = params.get("feature_set_id", "default")

        pipeline_ctx = PipelineContext.from_config(
            ctx.project_dir,
            experiment=ctx.experiment,
            config_name=config_name,
        )
        ctx.log(f"Feature extraction: {pipeline_ctx.features_dir}")
        ctx.log(f"  Jobs: {jobs if jobs is not None else 'auto'}")
        pipeline = build_pipeline(pipeline_ctx)
        return pipeline.run("features", dry_run=ctx.dry_run, jobs=jobs)


# Singleton instance
features_step = FeaturesStep()

# Convenience aliases
step_features = features_step.to_dict
run_step_features = features_step.run

if __name__ == "__main__":
    sys.exit(features_step.main())
