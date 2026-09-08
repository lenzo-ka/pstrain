"""Cross-command JSON output contract tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from pstrain.cli.cli import create_parser, main

FIXTURE = Path(__file__).parent / "fixtures" / "mini_arctic"


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


def _probe_arguments(
    command: tuple[str, ...], parser: argparse.ArgumentParser, tmp_path: Path
) -> list[str]:
    values = {
        "audio": str(FIXTURE / "wav"),
        "dictionary": str(FIXTURE / "dictionary.dict"),
        "key": "runner.jobs",
        "project_dir": str(tmp_path / "project"),
        "prompts": str(FIXTURE / "transcription.txt"),
    }
    arguments = ["--json", "--dry-run", *command]
    for action in parser._actions:
        if action.option_strings:
            if action.required:
                arguments.extend([action.option_strings[0], values.get(action.dest, "probe")])
            continue
        if action.dest == "help":
            continue
        if action.nargs in (None, "+") or action.dest in values:
            arguments.append(values.get(action.dest, "probe"))
    return arguments


def test_every_command_exposing_json_emits_json_or_refuses_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = create_parser()
    commands = _leaf_commands(parser)
    assert commands

    for command, command_parser in commands:
        arguments = _probe_arguments(command, command_parser, tmp_path / "-".join(command))
        parsed = parser.parse_args(arguments)
        assert parsed.json is True

        monkeypatch.setattr(sys, "argv", ["pstrain", *arguments])
        return_code = main()
        captured = capsys.readouterr()
        if return_code == 0:
            json.loads(captured.out)
        else:
            assert return_code == 2
            assert captured.out == ""
            assert captured.err.strip() == (
                f"Error: --json is not supported by 'pstrain {' '.join(command)}'"
            )


@pytest.mark.parametrize(
    "selector",
    ["--bin-dir", "--lib-path", "--include-dir", "--cflags", "--ldflags", "--version"],
)
def test_info_selectors_honor_json(
    selector: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["pstrain", "info", selector, "--json"])

    return_code = main()
    captured = capsys.readouterr()
    if return_code == 0:
        assert isinstance(json.loads(captured.out), str)
    else:
        assert captured.out == ""


def test_json_without_a_command_is_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["pstrain", "--json"])

    assert main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == "Error: --json requires a command"


def test_config_schema_rejects_conflicting_json_format(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["pstrain", "config", "schema", "--json", "--format", "markdown"],
    )

    assert main() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == ("Error: --json cannot be combined with a non-JSON --format")
