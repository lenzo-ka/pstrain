"""Validate command for ST2."""

from __future__ import annotations

import argparse
from pathlib import Path

from st2.api import validate_project
from st2.cli.base import Command, CommandContext, CommandResult


class ValidateCommand(Command):
    """Validate an ST2 project."""

    name = "validate-project"
    help = "Validate an ST2 project"
    description = "Validate project structure, files, and data consistency"
    needs_project_dir = False  # We handle it ourselves as positional arg

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command-specific arguments."""
        parser.add_argument(
            "project_dir",
            nargs="?",
            type=str,
            help="Project directory (default: current directory)",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Output JSON to stdout instead of summary",
        )
        parser.add_argument(
            "--output",
            "-o",
            type=str,
            help="Write JSON report to file (default: etc/validation.json)",
        )

    def execute(self, ctx: CommandContext) -> CommandResult:
        """Execute validate command."""
        # Resolve project directory
        project_dir = Path(ctx.args.project_dir).resolve() if ctx.args.project_dir else Path.cwd()

        if not project_dir.exists():
            return CommandResult.fail(f"Project directory does not exist: {project_dir}")

        ctx.log_action("Validate", str(project_dir))

        if ctx.dry_run:
            ctx.log("# Would validate project structure and files")
            return CommandResult.ok("Dry run complete")

        report = validate_project(project_dir)

        # Write JSON report file
        if ctx.args.output:
            json_path = Path(ctx.args.output)
        else:
            # Default location in experiment etc/
            json_path = project_dir / "experiments" / "default" / "etc" / "validation.json"

        json_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_json(json_path)
        ctx.log(f"Report saved: {json_path}")

        # Output
        if ctx.args.json:
            ctx.log(report.to_json())
        else:
            ctx.log(report.summary())

        if report.is_valid:
            return CommandResult.ok(f"Project validation passed: {project_dir}")

        return CommandResult.fail(
            f"Validation failed with {len(report.errors)} error(s)", exit_code=1
        )


# Singleton instance for registration
validate_command = ValidateCommand()
