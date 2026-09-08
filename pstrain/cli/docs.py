"""Generate CLI reference documentation from the command parser."""

from __future__ import annotations

import argparse

from pstrain.cli.cli import create_parser

_JSON_OPTIONS = {"--json", "--json-indent", "--json-ascii"}


def _leaf_commands(
    parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()
) -> list[tuple[str, argparse.ArgumentParser]]:
    subcommands = next(
        (action for action in parser._actions if isinstance(action, argparse._SubParsersAction)),
        None,
    )
    if subcommands is None:
        command = " ".join(prefix)
        if parser.get_default("json_command") != command:
            raise RuntimeError(f"missing JSON capability metadata for pstrain {command}")
        return [(command, parser)]
    return [
        leaf
        for name, child in subcommands.choices.items()
        for leaf in _leaf_commands(child, (*prefix, name))
    ]


def _format_commands(commands: list[str]) -> str:
    return "\n".join(f"* ``{command}``" for command in commands)


def generate_cli_reference() -> str:
    """Render the CLI reference from parser capability metadata."""
    parser = create_parser()
    global_options = {option for action in parser._actions for option in action.option_strings}
    if not global_options >= _JSON_OPTIONS:
        missing = ", ".join(sorted(_JSON_OPTIONS - global_options))
        raise RuntimeError(f"global JSON help is missing: {missing}")

    supported: list[str] = []
    rejected: list[str] = []
    for command, command_parser in _leaf_commands(parser):
        if command_parser.get_default("supports_json_output"):
            command_options = {
                option for action in command_parser._actions for option in action.option_strings
            }
            if not command_options >= _JSON_OPTIONS:
                missing = ", ".join(sorted(_JSON_OPTIONS - command_options))
                raise RuntimeError(f"pstrain {command} JSON help is missing: {missing}")
            supported.append(command)
        else:
            rejected.append(command)

    heading_rule = "=" * len("CLI API")
    return f"""CLI API
{heading_rule}

Command-line interface.

JSON output
-----------

``--json`` produces a single machine-readable result on standard output for:

{_format_commands(supported)}

The formatting controls ``--json-indent`` and ``--json-ascii`` apply to these
commands. Human-readable progress that accompanies a JSON result is written to
standard error.

The following commands do not yet have a stable machine-readable result
contract and reject ``--json`` with a nonzero exit status instead of silently
producing human output:

{_format_commands(rejected)}

Place the global flag before the command name when checking support. Commands
that support JSON also advertise it in their command help.

.. automodule:: pstrain.cli
   :members:
   :undoc-members:
   :show-inheritance:
"""


__all__ = ["generate_cli_reference"]
