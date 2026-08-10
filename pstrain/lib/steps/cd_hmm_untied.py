"""Step 30: CD HMM untied training (context-dependent untied models).

Stage 30: context-dependent untied HMM training.
Trains context-dependent untied HMM models using Baum-Welch algorithm.

Usage:
    Library: from pstrain.lib.steps.cd_hmm_untied import CDHMMUntiedStep
    CLI: python -m pstrain.lib.steps.cd_hmm_untied [args]
    pstrain CLI: pstrain step 30 [args]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pstrain.lib.steps.base import Step, StepContext


class CDHMMUntiedStep(Step):
    """CD HMM untied training step."""

    name = "cd_hmm_untied"
    description = "Train context-dependent untied HMM models using Baum-Welch"
    script = "bw"

    default_params: dict[str, Any] = {}

    def get_inputs(self, ctx: StepContext) -> list[Path]:
        """Get input files for CD HMM untied training."""
        ci_hmm = ctx.hmm_dir("ci")
        return [
            ci_hmm / "mdef",
            ci_hmm / "means",
            ci_hmm / "variances",
            ci_hmm / "mixture_weights",
            ci_hmm / "transition_matrices",
            ctx.shared_dir / "dictionary.dict",
            ctx.experiment_dir / "etc" / "train.fileids",
            ctx.experiment_dir / "etc" / "train.transcription",
            ctx.shared_dir / "features" / "default",
        ]

    def get_outputs(self, ctx: StepContext) -> list[Path]:
        """Get output files from CD HMM untied training."""
        cd_hmm = ctx.hmm_dir("cd_untied")
        return [
            cd_hmm / "mdef",
            cd_hmm / "means",
            cd_hmm / "variances",
            cd_hmm / "mixture_weights",
            cd_hmm / "transition_matrices",
        ]

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add CD HMM untied specific arguments."""
        super().add_arguments(parser)

    def execute(self, ctx: StepContext, **params: Any) -> int:
        """Execute CD HMM untied training by delegating to the pipeline runner.

        Upstream tasks (flat, ci-1g, cd-untied-init) will run automatically
        if their outputs are stale. Use `pstrain build cd-untied` from the CLI.
        """
        from pstrain.lib.pipeline import PipelineContext
        from pstrain.lib.pipeline.tasks import build_pipeline

        pl_ctx = PipelineContext.from_config(
            ctx.project_dir,
            experiment=ctx.experiment,
            config_name=ctx.config,
        )
        ctx.log(f"CD HMM untied training: {pl_ctx.model_dir('cd-untied')}")
        return build_pipeline(pl_ctx).run("cd-untied", dry_run=ctx.dry_run)


# Singleton instance
cd_hmm_untied_step = CDHMMUntiedStep()

# Convenience aliases
step_30_cd_hmm_untied = cd_hmm_untied_step.to_dict
run_step_30 = cd_hmm_untied_step.run

if __name__ == "__main__":
    sys.exit(cd_hmm_untied_step.main())
