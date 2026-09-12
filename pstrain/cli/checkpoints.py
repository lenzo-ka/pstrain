"""Inspect retained updates or explicitly restore one selected checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pstrain.api.checkpoints import list_checkpoints, restore_checkpoint
from pstrain.cli.base import Command, CommandContext, CommandResult


class CheckpointsCommand(Command):
    name = "checkpoints"
    help = "Inspect retained BW updates or restore an explicitly selected checkpoint"
    needs_project_dir = False
    supports_json_output = True

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("model_dir", type=Path, help="Model directory containing iterations/")
        parser.add_argument(
            "--restore",
            type=int,
            metavar="N",
            help="Restore update N with backup; stop concurrent training first",
        )

    def execute(self, ctx: CommandContext) -> CommandResult:
        result: dict[str, Any]
        if ctx.args.restore is None:
            result = {"checkpoints": list_checkpoints(ctx.args.model_dir)}
        else:
            result = restore_checkpoint(ctx.args.model_dir, ctx.args.restore, dry_run=ctx.dry_run)
        if ctx.json_output:
            print(ctx.format_json(result))
        elif ctx.args.restore is None:
            for row in result["checkpoints"]:
                print(
                    f"{row['checkpoint']:02d}\tevaluation pass {row['evaluation_pass']}\t{row['status']}"
                )
        else:
            print(f"{'Would restore' if ctx.dry_run else 'Restored'} checkpoint {ctx.args.restore}")
            print(f"Evaluation evidence: {result['selected']['status']}")
            if "backup" in result:
                print(f"Previous model retained: {result['backup']}")
        return CommandResult.ok()


checkpoints_command = CheckpointsCommand()
