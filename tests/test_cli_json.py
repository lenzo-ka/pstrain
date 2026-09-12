"""Cross-command JSON output contract tests."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from pstrain.cli.base import (
    UNSUPPORTED_JSON_EXIT_CODE,
    add_json_argument,
)
from pstrain.cli.cli import _audit_json_capabilities, create_parser, main
from pstrain.lib.validate import ValidationReport

FIXTURE = Path(__file__).parent / "fixtures" / "mini_arctic"
ScenarioFactory = Callable[[Path, pytest.MonkeyPatch], list[str]]
OutputScenarioFactory = Callable[[Path, pytest.MonkeyPatch], tuple[list[str], Path]]


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


def _one_json_document(output: str) -> object:
    document, end = json.JSONDecoder().raw_decode(output.lstrip())
    assert output.lstrip()[end:].strip() == ""
    return document


def _validate_success(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import pstrain.cli.validate as validate_cli

    project = root / "project"
    project.mkdir(parents=True)
    report = ValidationReport(total_utterances=1, train_utterances=1)
    monkeypatch.setattr(validate_cli, "validate_project", lambda path: report)
    return ["validate-project", str(project)]


def _validate_failure(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    return ["validate-project", str(root / "missing-project")]


def _model_test_success(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import pstrain.api.testing as testing_api

    project = root / "project"
    model = project / "shared" / "models" / "ci-1g" / "default"
    model.mkdir(parents=True)
    (project / "shared" / "dictionary.dict").write_text("WORD W ER D\n")
    (project / "audio").mkdir()
    transcripts = project / "experiments" / "default" / "etc"
    transcripts.mkdir(parents=True)
    (transcripts / "test.decoder.transcription").write_text("<s> WORD </s> (utt)\n")
    result = SimpleNamespace(wer=0.0, n_decoded=1, n_utterances=1)
    payload = {"metrics": {"wer": 0.0}, "counts": {"decoded": 1, "utterances": 1}}

    def save_json(path: Path) -> None:
        Path(path).write_text(json.dumps(payload) + "\n", encoding="utf-8")

    report = SimpleNamespace(
        to_dict=lambda: payload,
        description="",
        save_json=save_json,
    )
    monkeypatch.setattr(testing_api, "check_pocketsphinx", lambda: (True, ""))
    monkeypatch.setattr(testing_api, "load_transcripts", lambda path: {"utt": "WORD"})
    monkeypatch.setattr(testing_api, "test_model", lambda **kwargs: result)
    monkeypatch.setattr(testing_api, "create_report", lambda **kwargs: report)
    return ["test", "ci-1g", "--project-dir", str(project), "--no-lm"]


def _model_test_failure(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import pstrain.api.testing as testing_api

    monkeypatch.setattr(testing_api, "check_pocketsphinx", lambda: (True, ""))
    return ["test", "ci-1g", "--project-dir", str(root / "missing-project"), "--no-lm"]


class _SuccessfulPipeline:
    def run(self, *args: object, **kwargs: object) -> int:
        return 0


def _train_success(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import pstrain.cli.train as train_cli

    monkeypatch.setattr(train_cli, "build_pipeline", lambda context: _SuccessfulPipeline())
    return [
        "train",
        str(root / "project"),
        "--audio",
        str(FIXTURE / "wav"),
        "--prompts",
        str(FIXTURE / "transcription.txt"),
        "--dictionary",
        str(FIXTURE / "dictionary.dict"),
        "--phoneset",
        str(FIXTURE / "phoneset.txt"),
        "--filler-dict",
        str(FIXTURE / "filler.dict"),
        "--target",
        "ci-1g",
        "--jobs",
        "1",
    ]


def _train_failure(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    return [
        "train",
        str(root / "project"),
        "--audio",
        str(root / "missing-audio"),
        "--prompts",
        str(FIXTURE / "transcription.txt"),
        "--dictionary",
        str(FIXTURE / "dictionary.dict"),
    ]


def _info_success(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    return ["info", "--version"]


def _info_failure(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import pstrain.cli.info as info_cli

    monkeypatch.setattr(info_cli, "get_paths", lambda: SimpleNamespace(bin_dir=None))
    return ["info", "--bin-dir"]


def _tutorial_success(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    _patch_tutorial_copy(monkeypatch)
    monkeypatch.chdir(root)
    return ["tutorial"]


def _tutorial_failure(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    _patch_tutorial_copy(monkeypatch)
    output = root / "tutorial.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("occupied", encoding="utf-8")
    return ["tutorial", "--output", str(output)]


def _patch_tutorial_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    import pstrain.api as api

    def copy_tutorial(
        output: str | Path, *, force: bool = False, dry_run: bool = False
    ) -> dict[str, str]:
        destination = Path(output).absolute()
        if destination.exists() and not force:
            raise api.TutorialExistsError(destination)
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("{}\n", encoding="utf-8")
        return {
            "status": "dry-run" if dry_run else "written",
            "path": str(destination),
        }

    monkeypatch.setattr(api, "copy_tutorial", copy_tutorial)


def _config_explain_success(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    return ["config", "explain", "runner.jobs", "--project-dir", str(root)]


def _config_explain_failure(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    return ["config", "explain", "not.a.key", "--project-dir", str(root)]


def _config_profiles_success(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    return ["config", "profiles", "--project-dir", str(root)]


def _config_profiles_failure(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import pstrain.cli.config as config_cli

    def fail(path: Path) -> list[dict[str, object]]:
        raise ValueError("invalid profiles")

    monkeypatch.setattr(config_cli, "list_profiles", fail)
    return ["config", "profiles", "--project-dir", str(root)]


def _config_show_success(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    return ["config", "show", "--project-dir", str(root)]


def _config_show_failure(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import pstrain.cli.config as config_cli

    def fail(*args: object, **kwargs: object) -> object:
        raise ValueError("invalid configuration")

    monkeypatch.setattr(config_cli, "resolve_config", fail)
    return ["config", "show", "--project-dir", str(root)]


def _config_get_success(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    return ["config", "get", "runner.jobs", "--project-dir", str(root)]


def _config_get_failure(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    return ["config", "get", "not.a.key", "--project-dir", str(root)]


def _config_schema_success(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    return ["config", "schema"]


def _config_schema_failure(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    return ["config", "schema", "--format", "markdown"]


def _config_list_success(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    return ["config", "list", "--section", "runner"]


def _config_list_failure(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    return ["config", "list", "--section", "not-a-section"]


@pytest.fixture
def json_success_scenarios() -> dict[tuple[str, ...], ScenarioFactory]:
    return {
        ("validate-project",): _validate_success,
        ("test",): _model_test_success,
        ("info",): _info_success,
        ("checkpoints",): _checkpoints_success,
        ("train",): _train_success,
        ("tutorial",): _tutorial_success,
        ("config", "explain"): _config_explain_success,
        ("config", "profiles"): _config_profiles_success,
        ("config", "show"): _config_show_success,
        ("config", "get"): _config_get_success,
        ("config", "schema"): _config_schema_success,
        ("config", "list"): _config_list_success,
    }


@pytest.fixture
def json_failure_scenarios() -> dict[tuple[str, ...], ScenarioFactory]:
    return {
        ("validate-project",): _validate_failure,
        ("test",): _model_test_failure,
        ("info",): _info_failure,
        ("checkpoints",): _checkpoints_failure,
        ("train",): _train_failure,
        ("tutorial",): _tutorial_failure,
        ("config", "explain"): _config_explain_failure,
        ("config", "profiles"): _config_profiles_failure,
        ("config", "show"): _config_show_failure,
        ("config", "get"): _config_get_failure,
        ("config", "schema"): _config_schema_failure,
        ("config", "list"): _config_list_failure,
    }


def _validate_output_scenario(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], Path]:
    arguments = _validate_success(root, monkeypatch)
    output = root / "validation.json"
    return [*arguments, "--output", str(output)], output


def _model_test_output_scenario(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], Path]:
    arguments = _model_test_success(root, monkeypatch)
    output = root / "test-report.json"
    return [*arguments, "--output", str(output)], output


def _tutorial_output_scenario(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], Path]:
    _patch_tutorial_copy(monkeypatch)
    output = root / "tutorial.ipynb"
    return ["tutorial", "--output", str(output)], output


def _config_schema_output_scenario(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], Path]:
    output = root / "schema.json"
    return ["config", "schema", "--output", str(output)], output


@pytest.fixture
def json_output_scenarios() -> dict[tuple[str, ...], OutputScenarioFactory]:
    return {
        ("validate-project",): _validate_output_scenario,
        ("test",): _model_test_output_scenario,
        ("tutorial",): _tutorial_output_scenario,
        ("config", "schema"): _config_schema_output_scenario,
    }


def test_json_scenario_inventory_matches_supported_parser_inventory(
    json_success_scenarios: dict[tuple[str, ...], ScenarioFactory],
    json_failure_scenarios: dict[tuple[str, ...], ScenarioFactory],
    json_output_scenarios: dict[tuple[str, ...], OutputScenarioFactory],
) -> None:
    parser = create_parser()
    leaves = _leaf_commands(parser)
    supported = {
        command
        for command, command_parser in leaves
        if command_parser.get_default("supports_json_output")
    }
    supported_with_output = {
        command
        for command, command_parser in leaves
        if command in supported
        and any("--output" in action.option_strings for action in command_parser._actions)
    }

    assert set(json_success_scenarios) == supported
    assert set(json_failure_scenarios) == supported
    assert set(json_output_scenarios) == supported_with_output


@pytest.mark.parametrize("dry_run", [False, True], ids=["normal", "dry-run"])
def test_every_supported_json_command_has_a_succeeding_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    json_success_scenarios: dict[tuple[str, ...], ScenarioFactory],
    dry_run: bool,
) -> None:
    for command, prepare in json_success_scenarios.items():
        root = tmp_path / ("dry" if dry_run else "normal") / "-".join(command)
        root.mkdir(parents=True)
        with monkeypatch.context() as scenario_patch:
            arguments = prepare(root, scenario_patch)
            if dry_run:
                arguments = ["--dry-run", *arguments]
            scenario_patch.setattr(sys, "argv", ["pstrain", *arguments, "--json"])
            assert main() == 0, command
        captured = capsys.readouterr()
        _one_json_document(captured.out)


def test_every_supported_json_command_has_a_failing_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    json_failure_scenarios: dict[tuple[str, ...], ScenarioFactory],
) -> None:
    for command, prepare in json_failure_scenarios.items():
        root = tmp_path / "failure" / "-".join(command)
        root.mkdir(parents=True)
        with monkeypatch.context() as scenario_patch:
            arguments = prepare(root, scenario_patch)
            scenario_patch.setattr(sys, "argv", ["pstrain", *arguments, "--json"])
            assert main() != 0, command
        captured = capsys.readouterr()
        _one_json_document(captured.out)


def test_every_supported_json_output_path_is_exercised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    json_output_scenarios: dict[tuple[str, ...], OutputScenarioFactory],
) -> None:
    for command, prepare in json_output_scenarios.items():
        root = tmp_path / "output" / "-".join(command)
        root.mkdir(parents=True)
        with monkeypatch.context() as scenario_patch:
            arguments, output = prepare(root, scenario_patch)
            scenario_patch.setattr(sys, "argv", ["pstrain", *arguments, "--json"])
            assert main() == 0, command
        captured = capsys.readouterr()
        stdout_document = _one_json_document(captured.out)
        assert output.is_file(), command
        assert output.stat().st_size > 0, command
        output_document = json.loads(output.read_text(encoding="utf-8"))
        if output.suffix == ".json":
            assert output_document == stdout_document, command
            assert output.read_bytes() == captured.out.encode("utf-8"), command


def test_every_unsupported_json_command_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = create_parser()
    for command, command_parser in _leaf_commands(parser):
        if command_parser.get_default("supports_json_output"):
            continue
        arguments = ["--json", *command, *_required_arguments(command_parser)]
        monkeypatch.setattr(sys, "argv", ["pstrain", *arguments])
        assert main() == UNSUPPORTED_JSON_EXIT_CODE
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.strip() == (
            f"Error: --json is not supported by 'pstrain {' '.join(command)}'"
        )


def _required_arguments(parser: argparse.ArgumentParser) -> list[str]:
    arguments: list[str] = []
    for action in parser._actions:
        if action.dest == "help":
            continue
        value = str(next(iter(action.choices))) if action.choices else "probe"
        if action.option_strings:
            if action.required:
                arguments.extend((action.option_strings[0], value))
            continue
        count = action.nargs if isinstance(action.nargs, int) else 1
        if action.nargs not in ("?", "*"):
            arguments.extend(value for _ in range(count))
    return arguments


@pytest.mark.parametrize("indent", [None, 0], ids=["default-indent", "compact"])
@pytest.mark.parametrize("ascii_output", [False, True], ids=["unicode", "ascii"])
def test_config_schema_json_output_file_is_byte_identical_to_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    indent: int | None,
    ascii_output: bool,
) -> None:
    output = tmp_path / "schema.json"
    arguments = ["pstrain", "config", "schema", "--json", "--output", str(output)]
    if indent is not None:
        arguments.extend(("--json-indent", str(indent)))
    if ascii_output:
        arguments.append("--json-ascii")
    monkeypatch.setattr(
        sys,
        "argv",
        arguments,
    )

    assert main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.encode("utf-8") == output.read_bytes()
    json.loads(captured.out)


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


def test_finished_parser_rejects_json_help_metadata_disagreement() -> None:
    parser = create_parser()
    migrate = dict(_leaf_commands(parser))[("config", "migrate")]
    add_json_argument(migrate, suppress_defaults=True)

    with pytest.raises(
        RuntimeError,
        match=(
            "pstrain config migrate advertises unsupported JSON help: "
            "--json, --json-ascii, --json-indent"
        ),
    ):
        _audit_json_capabilities(parser)


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


def test_unsupported_json_and_parser_errors_have_distinct_statuses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["pstrain", "--json", "build", "ci-1g"])
    assert main() == UNSUPPORTED_JSON_EXIT_CODE
    unsupported = capsys.readouterr()
    assert unsupported.out == ""
    assert unsupported.err.strip() == "Error: --json is not supported by 'pstrain build'"

    monkeypatch.setattr(sys, "argv", ["pstrain", "build", "ci-1g", "--json"])
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 2
    malformed = capsys.readouterr()
    assert malformed.out == ""
    assert "unrecognized arguments: --json" in malformed.err


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


def _checkpoints_success(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    return ["checkpoints", str(root)]


def _checkpoints_failure(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    return ["checkpoints", str(root), "--restore", "1"]
