"""Command-line interface for pstrain.

Thin wrapper around pstrain.api - all business logic lives in the library.
"""

import argparse
import sys

from pstrain import __version__
from pstrain.cli.base import add_dry_run_argument, add_json_argument


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="pstrain",
        description="pstrain - Acoustic model training toolkit",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"pstrain {__version__}",
    )
    add_json_argument(parser)
    add_dry_run_argument(parser)

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Import command instances
    from pstrain.cli.align import align_command
    from pstrain.cli.build import build_command
    from pstrain.cli.clean import clean_command
    from pstrain.cli.compare import compare_command
    from pstrain.cli.config import register_config_command
    from pstrain.cli.features import features_command
    from pstrain.cli.flat import flat_command
    from pstrain.cli.info import info_command
    from pstrain.cli.setup import setup_command
    from pstrain.cli.split import split_command
    from pstrain.cli.step import register_step_command
    from pstrain.cli.test import test_command
    from pstrain.cli.validate import validate_command

    # Register Command-based commands
    commands = [
        setup_command,
        build_command,
        split_command,
        features_command,
        flat_command,
        clean_command,
        validate_command,
        test_command,
        align_command,
        info_command,
        compare_command,
    ]
    for cmd in commands:
        cmd.register(subparsers)

    # Register commands with subcommands
    register_config_command(subparsers)
    register_step_command(subparsers)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Execute command
    from pstrain.cli.base import execute_command

    return execute_command(args)


if __name__ == "__main__":
    sys.exit(main())
