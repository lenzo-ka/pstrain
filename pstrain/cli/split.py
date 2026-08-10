"""CLI command for data splitting."""

from __future__ import annotations

import argparse
from pathlib import Path

from pstrain.cli.base import CommandContext, CommandResult, ProjectCommand
from pstrain.lib.corpus import train_test_split


class SplitCommand(ProjectCommand):
    """Split corpus into training and test sets."""

    name = "split"
    help = "Split corpus into train/test sets"
    description = "Split corpus into training and test sets for a given experiment"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command-specific arguments."""
        parser.add_argument(
            "--transcription-file",
            type=str,
            help="Input transcription file (default: {project_dir}/etc/all.transcription)",
        )
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "--train-ratio",
            type=float,
            default=None,
            help="Training set ratio (e.g., 0.95 = 95%% train, 5%% test)",
        )
        group.add_argument(
            "--test-count",
            type=int,
            default=None,
            help="Exact number of test samples (e.g., 200)",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for reproducibility (default: 42)",
        )

    def execute(self, ctx: CommandContext) -> CommandResult:
        """Execute split command."""
        project_dir = ctx.project_dir
        etc_dir = project_dir / "experiments" / ctx.experiment / "etc"
        transcription_file = (
            Path(ctx.args.transcription_file)
            if ctx.args.transcription_file
            else project_dir / "etc" / "all.transcription"
        )

        if not transcription_file.exists():
            return CommandResult.fail(f"Transcription file not found: {transcription_file}")

        if ctx.dry_run:
            ctx.log_action("Split", str(transcription_file))
            ctx.mkdir(etc_dir)
            ctx.log(f"# Would write train/test fileids + transcriptions to {etc_dir}")
            return CommandResult.ok("Dry run complete")

        try:
            result = train_test_split(
                transcription_file,
                etc_dir,
                train_ratio=ctx.args.train_ratio,
                test_count=ctx.args.test_count,
                seed=ctx.args.seed,
            )
        except (FileNotFoundError, ValueError) as exc:
            return CommandResult.fail(str(exc))

        total = result.n_train + result.n_test
        ctx.log(f"Split {total} utterances into:")
        ctx.log(f"  Train: {result.n_train} ({result.n_train / total * 100:.1f}%)")
        ctx.log(f"    {result.train_transcription}")
        ctx.log(f"    {result.train_fileids}")
        ctx.log(f"  Test:  {result.n_test} ({result.n_test / total * 100:.1f}%)")
        ctx.log(f"    {result.test_transcription}")
        ctx.log(f"    {result.test_fileids}")
        return CommandResult.ok()


# Singleton instance for registration
split_command = SplitCommand()
