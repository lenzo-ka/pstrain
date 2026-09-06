"""CLI command for copying the bundled tutorial notebook."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from pstrain.cli.base import Command, CommandContext, CommandResult


class TutorialCommand(Command):
    """Write the tutorial notebook to a reader-controlled destination."""

    name = "tutorial"
    help = "Copy the HMM-GMM tutorial notebook"
    description = "Copy the bundled Arctic HMM-GMM tutorial notebook"
    needs_project_dir = False
    supports_json_output = True

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-o",
            "--output",
            type=Path,
            default=None,
            help="Output file or existing directory (default: current directory)",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Replace an existing destination file",
        )

    def execute(self, ctx: CommandContext) -> CommandResult:
        """Copy the notebook through the public API."""
        from pstrain.api import TUTORIAL_FILENAME, TutorialExistsError, copy_tutorial

        output = ctx.args.output or Path.cwd() / TUTORIAL_FILENAME
        try:
            result = copy_tutorial(output, force=ctx.args.force, dry_run=ctx.dry_run)
        except TutorialExistsError as error:
            if ctx.json_output:
                print(
                    ctx.format_json(
                        {"status": "error", "path": str(error.path), "error": str(error)}
                    )
                )
                return CommandResult.fail("")
            return CommandResult.fail(str(error))

        if ctx.json_output:
            print(ctx.format_json(result))
        elif ctx.dry_run:
            print(f"Would write tutorial to: {result['path']}")
        else:
            path = result["path"]
            print(f"Wrote tutorial to: {path}")
            print("To open it with Jupyter, install it if needed: python -m pip install jupyter")
            print(f"Launch it with: jupyter notebook {shlex.quote(path)}")
            print("The notebook installs its remaining dependencies on first run.")
        return CommandResult.ok(data=dict(result))


tutorial_command = TutorialCommand()
