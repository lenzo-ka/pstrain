"""Command-line interface for pstrain.

Thin wrapper around pstrain.api - all business logic lives in the library.
"""

import argparse
import sys

from pstrain import __version__
from pstrain.cli.base import add_dry_run_argument, add_json_argument

_JSON_OPTIONS = {"--json", "--json-indent", "--json-ascii"}


def _leaf_commands(
    parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], argparse.ArgumentParser]]:
    subcommands = next(
        (action for action in parser._actions if isinstance(action, argparse._SubParsersAction)),
        None,
    )
    if subcommands is None:
        return [(prefix, parser)]
    return [
        leaf
        for name, child in subcommands.choices.items()
        for leaf in _leaf_commands(child, (*prefix, name))
    ]


def _audit_json_capabilities(parser: argparse.ArgumentParser) -> None:
    """Reject JSON capability metadata that disagrees with finished leaf help."""
    for path, command_parser in _leaf_commands(parser):
        command = " ".join(path)
        if command_parser.get_default("json_command") != command:
            raise RuntimeError(f"missing JSON capability metadata for pstrain {command}")
        options = {option for action in command_parser._actions for option in action.option_strings}
        advertised = options & _JSON_OPTIONS
        supported = bool(command_parser.get_default("supports_json_output"))
        if supported and advertised != _JSON_OPTIONS:
            missing = ", ".join(sorted(_JSON_OPTIONS - advertised))
            raise RuntimeError(f"pstrain {command} JSON help is missing: {missing}")
        if not supported and advertised:
            unexpected = ", ".join(sorted(advertised))
            raise RuntimeError(f"pstrain {command} advertises unsupported JSON help: {unexpected}")


def create_parser() -> argparse.ArgumentParser:
    """Build the complete command-line parser."""
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
    from pstrain.cli.package import package_command
    from pstrain.cli.setup import setup_command
    from pstrain.cli.split import split_command
    from pstrain.cli.step import register_step_command
    from pstrain.cli.test import test_command
    from pstrain.cli.timings import timings_command
    from pstrain.cli.train import train_command
    from pstrain.cli.tutorial import tutorial_command
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
        package_command,
        align_command,
        info_command,
        compare_command,
        timings_command,
        train_command,
        tutorial_command,
    ]
    for cmd in commands:
        cmd.register(subparsers)

    # Register commands with subcommands
    register_config_command(subparsers)
    register_step_command(subparsers)

    _audit_json_capabilities(parser)
    return parser


def main() -> int:
    parser = create_parser()

    args = parser.parse_args()

    if args.command is None:
        if args.json:
            print("Error: --json requires a command", file=sys.stderr)
            return 2
        parser.print_help()
        return 0

    # Execute command
    from pstrain.cli.base import execute_command

    return execute_command(args)


if __name__ == "__main__":
    sys.exit(main())
