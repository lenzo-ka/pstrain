"""Inspect persisted pipeline timing artifacts."""

from __future__ import annotations

import argparse

from pstrain.cli.base import Command, CommandContext, CommandResult
from pstrain.lib.pipeline.timings import format_summary, load_document


class TimingsCommand(Command):
    name = "timings"
    help = "Show timing rollup for a pipeline run"
    description = "Pretty-print the latest, or a named, pipeline timing artifact."
    supports_dry_run = False

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("run_id", nargs="?", help="Run ID (default: latest)")

    def execute(self, ctx: CommandContext) -> CommandResult:
        try:
            path, document = load_document(ctx.project_dir, ctx.args.run_id)
        except (FileNotFoundError, ValueError):
            return CommandResult.fail("no pipeline timing runs found")
        print(f"Run: {document['run_id']} ({path})")
        print(f"Target: {document['target']}")
        print(format_summary(document))
        return CommandResult.ok()


timings_command = TimingsCommand()
