"""Cross-command JSON output contract tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pstrain.cli.base import Command, CommandContext, CommandResult, add_json_argument
from pstrain.cli.cli import create_parser, main
from pstrain.lib.validate import ValidationReport

FIXTURE = Path(__file__).parent / "fixtures" / "mini_arctic"


class _MismatchedJsonCommand(Command):
    name = "mismatched"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        add_json_argument(parser, suppress_defaults=True)

    def execute(self, ctx: CommandContext) -> CommandResult:
        return CommandResult.ok()


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
    command: tuple[str, ...],
    parser: argparse.ArgumentParser,
    tmp_path: Path,
    *,
    dry_run: bool,
) -> list[str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    if command == ("validate-project",):
        (tmp_path / "project").mkdir(parents=True)
    values = {
        "audio": str(FIXTURE / "wav" if dry_run else tmp_path / "missing-audio"),
        "dictionary": str(FIXTURE / "dictionary.dict"),
        "key": "runner.jobs",
        "project_dir": str(tmp_path / "project"),
        "prompts": str(FIXTURE / "transcription.txt"),
    }
    arguments = ["--json"]
    if dry_run:
        arguments.append("--dry-run")
    arguments.extend(command)
    for action in parser._actions:
        if action.option_strings:
            if action.required:
                arguments.extend([action.option_strings[0], values.get(action.dest, "probe")])
            continue
        if action.dest == "help":
            continue
        if action.nargs in (None, "+") or action.dest in values:
            arguments.append(values.get(action.dest, "probe"))
    if not dry_run and any("--output" in action.option_strings for action in parser._actions):
        arguments.extend(["--output", str(tmp_path / "output")])
    return arguments


def test_every_command_exposing_json_emits_json_or_refuses_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = create_parser()
    commands = _leaf_commands(parser)
    assert commands

    for dry_run in (False, True):
        for command, command_parser in commands:
            probe_path = tmp_path / ("dry" if dry_run else "normal") / "-".join(command)
            arguments = _probe_arguments(command, command_parser, probe_path, dry_run=dry_run)
            parsed = parser.parse_args(arguments)
            assert parsed.json is True

            monkeypatch.setattr(sys, "argv", ["pstrain", *arguments])
            return_code = main()
            captured = capsys.readouterr()
            if parsed.supports_json_output:
                assert return_code in (0, 1)
                json.loads(captured.out)
            else:
                assert return_code == 2
                assert captured.out == ""
                assert captured.err.strip() == (
                    f"Error: --json is not supported by 'pstrain {' '.join(command)}'"
                )


def test_config_schema_json_output_is_written_and_emitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "schema.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["pstrain", "config", "schema", "--json", "--output", str(output)],
    )

    assert main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == json.loads(output.read_text())


def test_config_get_requires_a_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["pstrain", "config", "get", "--json"])

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "the following arguments are required: key" in captured.err


def test_config_get_invalid_key_error_is_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["pstrain", "config", "get", "not.a.key", "--json"])

    assert main() == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "status": "error",
        "message": "unknown or invalid config key 'not.a.key': 'not.a.key'",
    }


def test_command_registration_rejects_json_help_metadata_disagreement() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    with pytest.raises(RuntimeError, match="JSON help and supports_json_output disagree"):
        _MismatchedJsonCommand().register(subparsers)


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
        assert json.loads(captured.out)["status"] == "error"
        assert captured.err == ""


def test_validate_project_emits_only_report_on_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import pstrain.cli.validate as validate_cli

    project = tmp_path / "project"
    project.mkdir()
    report = ValidationReport(total_utterances=3, train_utterances=2, test_utterances=1)
    monkeypatch.setattr(validate_cli, "validate_project", lambda path: report)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pstrain", "validate-project", str(project), "--json", "--json-indent", "0"],
    )

    assert main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == report.to_dict()
    assert captured.out.count("\n") == 1
    assert f"Validate: {project}" in captured.err
    assert "Report saved:" in captured.err
    assert "Project validation passed:" in captured.err


def test_model_test_emits_only_report_on_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import pstrain.api.testing as testing_api

    project = tmp_path / "project"
    model = project / "shared" / "models" / "ci-1g" / "default"
    model.mkdir(parents=True)
    (project / "shared" / "dictionary.dict").write_text("WORD W ER D\n")
    (project / "audio").mkdir()
    transcripts = project / "experiments" / "default" / "etc"
    transcripts.mkdir(parents=True)
    (transcripts / "test.decoder.transcription").write_text("<s> WORD </s> (utt)\n")
    result = SimpleNamespace(wer=0.0, n_decoded=1, n_utterances=1)
    payload = {"metrics": {"wer": 0.0}, "counts": {"decoded": 1, "utterances": 1}}
    report = SimpleNamespace(to_dict=lambda: payload, description="")
    monkeypatch.setattr(testing_api, "check_pocketsphinx", lambda: (True, ""))
    monkeypatch.setattr(testing_api, "load_transcripts", lambda path: {"utt": "WORD"})
    monkeypatch.setattr(testing_api, "test_model", lambda **kwargs: result)
    monkeypatch.setattr(testing_api, "create_report", lambda **kwargs: report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pstrain",
            "test",
            "ci-1g",
            "--project-dir",
            str(project),
            "--no-lm",
            "--json",
            "--json-indent",
            "0",
        ],
    )

    assert main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
    assert captured.out.count("\n") == 1
    assert f"Test: {model}" in captured.err
    assert "WER: 0.00% (1/1 decoded)" in captured.err


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

    assert main() == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "status": "error",
        "message": "--json cannot be combined with a non-JSON --format",
    }
