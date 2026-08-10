"""Step 20: CI HMM training (context-independent models).

Trains context-independent HMM models using Baum-Welch algorithm.

Usage:
    Library: from pstrain.lib.steps.ci_hmm import CIHMMStep
    CLI: python -m pstrain.lib.steps.ci_hmm [args]
    pstrain CLI: pstrain step 20 [args]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pstrain.lib.steps.base import Step, StepContext


class CIHMMStep(Step):
    """CI HMM training step."""

    name = "ci_hmm"
    description = "Train context-independent HMM models using Baum-Welch"
    script = "bw"

    # Training parameters belong to PipelineContext/TrainParams.  This legacy
    # step delegates to the pipeline and therefore must not advertise a second,
    # ignored parameter surface.
    default_params: dict[str, Any] = {}

    def get_inputs(self, ctx: StepContext) -> list[Path]:
        """Get input files for CI HMM training."""
        flat = ctx.flat_dir("ci")
        return [
            flat / "mdef",
            flat / "means",
            flat / "variances",
            flat / "mixture_weights",
            flat / "transition_matrices",
            ctx.shared_dir / "dictionary.dict",
            ctx.experiment_dir / "etc" / "train.fileids",
            ctx.experiment_dir / "etc" / "train.transcription",
            ctx.shared_dir / "features" / "default",
            ctx.shared_dir / "features" / "default" / "feat.params",
        ]

    def get_outputs(self, ctx: StepContext) -> list[Path]:
        """Get output files from CI HMM training."""
        hmm = ctx.hmm_dir("ci")
        return [
            hmm / "mdef",
            hmm / "means",
            hmm / "variances",
            hmm / "mixture_weights",
            hmm / "transition_matrices",
            hmm / "feat.params",
        ]

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add CI HMM specific arguments."""
        super().add_arguments(parser)

    def get_params_from_args(self, args: argparse.Namespace) -> dict[str, Any]:
        """Extract training parameters from args."""
        del args
        return {}

    def execute(self, ctx: StepContext, **params: Any) -> int:
        """Execute CI HMM training by delegating to the pipeline runner.

        The pipeline does the right thing for stale dependencies: if `flat`
        or features are missing/older than their inputs, those tasks will
        run too. Use `pstrain build ci-1g` for the same effect from the CLI.
        """
        from pstrain.lib.pipeline import PipelineContext
        from pstrain.lib.pipeline.tasks import build_pipeline

        pl_ctx = PipelineContext.from_config(
            ctx.project_dir,
            experiment=ctx.experiment,
            config_name=ctx.config,
        )
        ctx.comment(f"Step {self.name}: {self.description}")
        ctx.comment(f"  Experiment: {ctx.experiment}, Config: {ctx.config}")
        return build_pipeline(pl_ctx).run("ci-1g", dry_run=ctx.dry_run)


ci_hmm_step = CIHMMStep()


def main() -> int:
    """CLI entry point."""
    return ci_hmm_step.main()


if __name__ == "__main__":
    sys.exit(main())
