"""CLI command for feature extraction.

Drives the pipeline runner's "features" target, which fans out one task per
audio file and runs them in a process pool.
"""

from __future__ import annotations

import argparse

from pstrain.cli.base import CommandContext, CommandResult, ProjectCommand
from pstrain.lib.pipeline import PipelineContext, UnknownTargetError
from pstrain.lib.pipeline.tasks import build_pipeline


class FeaturesCommand(ProjectCommand):
    """Extract acoustic features from audio files."""

    name = "features"
    help = "Extract acoustic features from audio files"
    description = "Extract MFCC features from audio files found recursively under audio/."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            default="default",
            dest="config_name",
            help="Named config from etc/configs.yaml (default: default)",
        )
        parser.add_argument(
            "-j",
            "--jobs",
            type=int,
            default=None,
            help="Parallel workers (default: CPU count minus 2; explicit N may use full machine)",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Re-extract all features, even if up to date",
        )

    def execute(self, ctx: CommandContext) -> CommandResult:
        try:
            pipeline_ctx = PipelineContext.from_config(
                ctx.project_dir,
                experiment=ctx.experiment,
                config_name=ctx.args.config_name,
            )
        except ValueError as exc:
            return CommandResult.fail(str(exc))

        fileids = pipeline_ctx.audio_fileids()

        ctx.log_action("Extract features", str(pipeline_ctx.features_dir))
        ctx.log(f"  Audio:  {pipeline_ctx.audio_dir}")
        ctx.log(f"  Output: {pipeline_ctx.features_dir}")
        ctx.log(f"  Files:  {len(fileids)}")
        ctx.log(f"  Jobs:   {ctx.args.jobs if ctx.args.jobs is not None else 'auto'}")

        try:
            pipeline = build_pipeline(pipeline_ctx)
        except ValueError as exc:
            return CommandResult.fail(str(exc))
        try:
            rc = pipeline.run(
                "features",
                dry_run=ctx.dry_run,
                force=ctx.args.force,
                jobs=ctx.args.jobs,
            )
        except UnknownTargetError as exc:
            return CommandResult.fail(f"unknown target: {exc}")

        if rc == 0:
            return CommandResult.ok()
        return CommandResult.fail(f"feature extraction failed (rc={rc})", exit_code=rc)


features_command = FeaturesCommand()
