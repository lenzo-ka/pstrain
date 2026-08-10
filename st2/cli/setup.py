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
        config_path = Path(ctx.args.config).resolve() if ctx.args.config else None

        named_sources = [
            ("Transcription", transcription_path),
            ("Audio", audio_path),
            ("Dictionary", dictionary_path),
            ("Phoneset", phoneset_path),
            ("Filler dictionary", filler_dict_path),
            ("Config file", config_path),
        ]
        for label, source in named_sources:
            if source is not None and not source.exists():
                return CommandResult.fail(f"{label} does not exist: {source}")
        if ctx.args.link and audio_path is not None:
            source_in_project = audio_path == project_dir or audio_path.is_relative_to(
                project_dir
            )
            source_contains_project = project_dir.is_relative_to(audio_path)
            if source_in_project or source_contains_project:
                return CommandResult.fail(
                    "Cannot link project audio from or around the project directory: "
                    f"{audio_path}"
                )

        ctx.comment(f"Setup project: {project_dir}")
        ctx.blank()

        if ctx.dry_run:
            audio_dir = project_dir / "audio"

            def describe_install(label: str, source: str, destination: Path) -> None:
                if destination.exists() and not ctx.args.clobber:
                    ctx.comment(f"Skip existing {label}: {destination}")
                else:
                    action = f"Replace {label}" if destination.exists() else label
                    ctx.comment(f"{action}: {source} -> {destination}")

            directories = [
                project_dir,
                project_dir / "etc",
                project_dir / "shared",
                project_dir / "shared" / "features",
                project_dir / "experiments" / "default" / "etc",
            ]
            if not (ctx.args.link and audio_path is not None):
                directories.append(audio_dir)
            for directory in directories:
                ctx.comment(f"Create directory: {directory}")
            if ctx.args.clobber:
                ctx.comment("Clobber enabled: replace existing destination files")
            else:
                ctx.comment("Clobber disabled: keep existing destination files")
            if transcription_path:
                describe_install(
                    "Copy transcription",
                    str(transcription_path),
                    project_dir / "etc" / "all.transcription",
                )
            if audio_path:
                if ctx.args.link:
                    if (audio_dir.exists() or audio_dir.is_symlink()) and ctx.args.clobber:
                        ctx.comment(f"Remove existing audio destination: {audio_dir}")
                    describe_install("Link audio", str(audio_path), audio_dir)
                elif audio_path.is_dir():
                    for source in sorted(path for path in audio_path.rglob("*") if path.is_file()):
                        describe_install(
                            "Copy audio",
                            str(source),
                            audio_dir / source.relative_to(audio_path),
                        )
                else:
                    describe_install("Copy audio", str(audio_path), audio_dir / audio_path.name)
            if dictionary_path:
                describe_install(
                    "Copy dictionary",
                    str(dictionary_path),
                    project_dir / "shared" / "dictionary.dict",
                )
            if filler_dict_path:
                describe_install(
                    "Copy filler dictionary",
                    str(filler_dict_path),
                    project_dir / "shared" / "filler.dict",
                )
            else:
                describe_install(
                    "Copy filler dictionary",
                    "packaged filler.dict",
                    project_dir / "shared" / "filler.dict",
                )
            if phoneset_path:
                describe_install(
                    "Copy phoneset",
                    str(phoneset_path),
                    project_dir / "shared" / "phoneset.txt",
                )
            elif dictionary_path:
                describe_install(
                    "Extract phoneset",
                    "installed dictionaries",
                    project_dir / "shared" / "phoneset.txt",
                )
            if config_path:
                describe_install(
                    "Write config.yaml", str(config_path), project_dir / "etc" / "config.yaml"
                )
            else:
                describe_install(
                    "Write default config.yaml",
                    "generated defaults",
                    project_dir / "etc" / "config.yaml",
                )
            describe_install(
                "Write configs.yaml profiles",
                "built-in profiles",
                project_dir / "etc" / "configs.yaml",
            )
            if ctx.args.validate:
                ctx.comment(f"Validate project after setup: {project_dir}")
            return CommandResult.ok()

        setup_project(
            project_dir=project_dir,
            transcription_path=transcription_path,
            audio_path=audio_path,
            dictionary_path=dictionary_path,
            phoneset_path=phoneset_path,
            filler_dict_path=filler_dict_path,
            config_path=config_path,
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
