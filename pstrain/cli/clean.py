"""CLI command for cleaning step outputs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pstrain.cli.base import CommandContext, CommandResult, ProjectCommand


class CleanCommand(ProjectCommand):
    """Clean outputs from training steps."""

    name = "clean"
    help = "Clean outputs from training steps"
    description = "Remove output files from specific training steps"

    # Map step names to their output directories
    STEP_OUTPUTS = {
        "features": ["shared/features/{feature_set_id}"],
        "flat": ["experiments/{experiment}/models/ci/{config}/model/flat"],
        "ci": ["experiments/{experiment}/models/ci/{config}/model/hmm"],
        "cd_untied": ["experiments/{experiment}/models/cd_untied/{config}/model/hmm"],
        "split": [
            "experiments/{experiment}/etc/train.transcription",
            "experiments/{experiment}/etc/test.transcription",
            "experiments/{experiment}/etc/train.fileids",
            "experiments/{experiment}/etc/test.fileids",
        ],
    }

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command-specific arguments."""
        parser.add_argument(
            "step",
            nargs="?",
            choices=list(self.STEP_OUTPUTS.keys()) + ["all"],
            help="Step to clean (features, flat, ci, cd_untied, split, or all)",
        )
        parser.add_argument(
            "--feature-set-id",
            type=str,
            default="default",
            help="Feature set ID to clean (default: default)",
        )
        parser.add_argument(
            "--config",
            type=str,
            default="baseline",
            help="Model config to clean (default: baseline)",
        )
        parser.add_argument(
            "--list",
            action="store_true",
            dest="list_steps",
            help="List available steps and their outputs",
        )
        parser.add_argument(
            "-f",
            "--force",
            action="store_true",
            help="Don't ask for confirmation",
        )

    def execute(self, ctx: CommandContext) -> CommandResult:
        """Execute clean command."""
        project_dir = ctx.project_dir

        if ctx.args.list_steps:
            ctx.log("Available steps to clean:")
            for step_name, step_paths in self.STEP_OUTPUTS.items():
                ctx.log(f"  {step_name}:")
                for p in step_paths:
                    ctx.log(f"    {p}")
            return CommandResult.ok()

        if not ctx.args.step:
            ctx.error("Please specify a step to clean (or --list to see options)")
            return CommandResult.fail("No step specified")

        # Determine which steps to clean
        if ctx.args.step == "all":
            steps_to_clean = list(self.STEP_OUTPUTS.keys())
        else:
            steps_to_clean = [ctx.args.step]

        # Resolve paths
        paths_to_clean: list[Path] = []
        for step in steps_to_clean:
            for path_template in self.STEP_OUTPUTS[step]:
                resolved = path_template.format(
                    experiment=ctx.experiment,
                    config=ctx.args.config,
                    feature_set_id=ctx.args.feature_set_id,
                )
                full_path = project_dir / resolved
                if full_path.exists():
                    paths_to_clean.append(full_path)

        # Type narrowing for mypy
        paths: list[Path] = paths_to_clean

        if not paths:
            ctx.log(f"Nothing to clean for step '{ctx.args.step}'")
            return CommandResult.ok()

        ctx.log_action("Clean", f"step '{ctx.args.step}'")
        ctx.log(f"  Experiment: {ctx.experiment}")
        if ctx.args.step in ["flat", "ci", "cd_untied"]:
            ctx.log(f"  Config: {ctx.args.config}")
        if ctx.args.step == "features":
            ctx.log(f"  Feature set: {ctx.args.feature_set_id}")

        ctx.log("")
        ctx.log("Will remove:")
        for path in paths:
            if path.is_dir():
                # Count files in directory
                file_count = sum(1 for f in path.rglob("*") if f.is_file())
                ctx.log(f"  {path}/ ({file_count} files)")
            else:
                ctx.log(f"  {path}")

        if ctx.dry_run:
            ctx.emit_blank()
            for path in paths:
                if path.is_dir():
                    ctx.comment(f"rm -rf {path}")
                else:
                    ctx.comment(f"rm {path}")
            return CommandResult.ok("Dry run complete")

        # Confirm unless --force
        if not ctx.args.force:
            ctx.log("")
            ctx.log("Use --force to skip confirmation, or --dry-run to preview.")
            return CommandResult.fail("Confirmation required (use -f to force)")

        # Actually delete
        for path in paths:
            if path.is_dir():
                shutil.rmtree(path)
                ctx.log(f"  Removed directory: {path}")
            else:
                path.unlink()
                ctx.log(f"  Removed file: {path}")

        return CommandResult.ok(f"Cleaned {len(paths)} path(s)")


# Singleton instance for registration
clean_command = CleanCommand()
