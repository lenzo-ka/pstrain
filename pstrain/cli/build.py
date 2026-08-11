"""CLI command for building model targets.

Drives `pstrain.lib.pipeline` directly (no subprocess). Targets are defined in
`pstrain/lib/pipeline/tasks.py::TARGETS`.

Usage:
    pstrain build cd-8g                       # Build cd-8g model
    pstrain build cd-8g --dry-run             # Print plan without running
    pstrain build cd-8g -j 8                  # 8 workers for feature fan-out
    pstrain build --list                      # List available targets
    pstrain build cd-8g --force               # Force rebuild from scratch
"""

from __future__ import annotations

import argparse
import sys

from pstrain.cli.base import Command, CommandContext, CommandResult
from pstrain.lib.pipeline import PipelineContext, UnknownTargetError
from pstrain.lib.pipeline.tasks import TARGETS, build_pipeline


def list_targets() -> None:
    print("Available targets:")
    print()
    groups: dict[str, list[str]] = {}
    for spec in TARGETS:
        groups.setdefault(spec.kind, []).append(f"  {spec.name:18} {spec.description}")

    labels = {
        "split": "Corpus",
        "features": "Features",
        "ci": "CI Models (Context-Independent)",
        "cd": "CD Models (Context-Dependent)",
        "lm": "Language Model",
        "test": "Testing / WER",
        "package": "Distribution Packages",
    }
    for kind in ["split", "features", "ci", "cd", "lm", "test", "package"]:
        if kind in groups:
            print(f"  {labels[kind]}:")
            for line in groups[kind]:
                print(line)
            print()

    print("Usage:")
    print("  pstrain build cd-8g                     # Build with default config")
    print("  pstrain build cd-8g --config telephone  # Build with telephone config")
    print("  pstrain build cd-8g -j 8                # 8 workers for feature fan-out")
    print("  pstrain build --dry-run cd-8g           # Print plan, don't run")


class BuildCommand(Command):
    """Build a model target via the in-process pipeline runner."""

    name = "build"
    help = "Build a model target (e.g., ci-1g, cd-8g)"
    description = """Build acoustic models using predefined targets.

The pipeline runner determines the required tasks and their order from
file dependencies, runs only what's stale, and parallelizes the feature
extraction fan-out across workers.

Use --dry-run to see the plan before running. Use --force to rebuild
everything, even if up to date.
"""
    needs_project_dir = True
    supports_dry_run = True

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "target",
            nargs="?",
            help="Target to build (e.g., ci-8g, cd-8g)",
        )
        parser.add_argument(
            "--list",
            action="store_true",
            help="List available targets",
        )
        parser.add_argument(
            "-j",
            "--jobs",
            type=int,
            default=None,
            help="Parallel workers (default: CPU count minus 2; explicit N may use full machine)",
        )
        parser.add_argument(
            "--experiment",
            type=str,
            default="default",
            help="Experiment name (default: default)",
        )
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            default="default",
            dest="config_name",
            help="Named config from etc/configs.yaml (default, wideband, telephone)",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Rebuild all tasks reachable from target, even if up to date",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Always print the pipeline timing summary",
        )

    def execute(self, ctx: CommandContext) -> CommandResult:
        if ctx.args.list:
            list_targets()
            return CommandResult.ok()

        if not ctx.args.target:
            print("Error: no target specified.", file=sys.stderr)
            print("Use 'pstrain build --list' to see available targets.", file=sys.stderr)
            return CommandResult.fail("no target specified")

        project_dir = ctx.project_dir
        try:
            pipeline_ctx = PipelineContext.from_config(
                project_dir,
                experiment=ctx.args.experiment,
                config_name=ctx.args.config_name,
                cli_overrides=(
                    {"runner": {"jobs": ctx.args.jobs}} if ctx.args.jobs is not None else None
                ),
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return CommandResult.fail(str(exc))

        try:
            pipeline = build_pipeline(pipeline_ctx)
        except ValueError as exc:
            return CommandResult.fail(str(exc))

        try:
            rc = pipeline.run(
                ctx.args.target,
                dry_run=ctx.dry_run,
                force=ctx.args.force,
                jobs=ctx.args.jobs,
                verbose=ctx.verbose,
            )
        except UnknownTargetError as exc:
            print(f"Error: unknown target {exc!s}", file=sys.stderr)
            print("Use 'pstrain build --list' to see available targets.", file=sys.stderr)
            return CommandResult.fail(f"unknown target: {exc}")

        if rc == 0:
            return CommandResult.ok()
        return CommandResult.fail(f"pipeline exited with code {rc}", exit_code=rc)


build_command = BuildCommand()
