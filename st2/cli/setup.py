"""Setup command for ST2."""

from __future__ import annotations

import argparse
from pathlib import Path

from st2.api import validate_project
from st2.cli.base import Command, CommandContext, CommandResult
from st2.lib.setup import setup_project


class SetupCommand(Command):
    """Set up a new ST2 project."""

    name = "setup"
    help = "Set up a new ST2 project"
    description = "Initialize a new ST2 project with required files and directory structure"
    needs_project_dir = False  # We handle project_dir specially as positional arg
    needs_experiment = False

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add setup-specific arguments."""
        parser.add_argument(
            "project_dir",
            nargs="?",
            type=str,
            help="Project directory (default: current directory)",
        )
        parser.add_argument(
            "--transcription",
            type=str,
            help="Path to transcription file (fileid + words)",
        )
        parser.add_argument(
            "--audio",
            type=str,
            help="Path to audio files directory or file",
        )
        parser.add_argument(
            "--dictionary",
            type=str,
            help="Path to pronunciation dictionary file",
        )
        parser.add_argument(
            "--phoneset",
            type=str,
            help="Path to phoneset file (or extract from dictionary)",
        )
        parser.add_argument(
            "--filler-dict",
            type=str,
            help="Path to filler dictionary (optional)",
        )
        parser.add_argument(
            "--config",
            type=str,
            help="Path to config file (or create default)",
        )
        parser.add_argument(
            "--link",
            action="store_true",
            help="Symlink audio directory instead of copying (only if --audio is provided)",
        )
        parser.add_argument(
            "--clobber",
            action="store_true",
            help="Overwrite existing files (default: skip existing files)",
        )
        parser.add_argument(
            "--validate",
            action="store_true",
            help="Run validation after setup",
        )

    def execute(self, ctx: CommandContext) -> CommandResult:
        """Set up a project through the public library API."""
        # Resolve project directory
        if ctx.args.project_dir:
            project_dir = Path(ctx.args.project_dir).resolve()
        else:
            project_dir = Path.cwd()

        # Resolve input paths
        transcription_path = (
            Path(ctx.args.transcription).resolve() if ctx.args.transcription else None
        )
        audio_path = Path(ctx.args.audio).resolve() if ctx.args.audio else None
        dictionary_path = Path(ctx.args.dictionary).resolve() if ctx.args.dictionary else None
        phoneset_path = Path(ctx.args.phoneset).resolve() if ctx.args.phoneset else None
        filler_dict_path = Path(ctx.args.filler_dict).resolve() if ctx.args.filler_dict else None

        ctx.comment(f"Setup project: {project_dir}")
        ctx.blank()

        if ctx.dry_run:
            ctx.comment("Would create the project and install the requested setup files")
            return CommandResult.ok()

        setup_project(
            project_dir=project_dir,
            transcription_path=transcription_path,
            audio_path=audio_path,
            dictionary_path=dictionary_path,
            phoneset_path=phoneset_path,
            filler_dict_path=filler_dict_path,
            config_path=Path(ctx.args.config).resolve() if ctx.args.config else None,
            link_audio=ctx.args.link,
            clobber=ctx.args.clobber,
        )

        # Validation
        if ctx.args.validate:
            ctx.comment("Validate project")
            report = validate_project(project_dir)
            if not report.is_valid:
                return CommandResult.fail(
                    f"Validation failed with {len(report.errors)} error(s)"
                )
            ctx.blank()

        ctx.comment("Done. To work in this project:")
        ctx.comment(f"  cd {project_dir}")

        return CommandResult.ok()


# Singleton instance for registration
setup_command = SetupCommand()
