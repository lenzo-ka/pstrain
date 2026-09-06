"""Tests for the model packaging command."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pstrain.cli.base import add_dry_run_argument, add_json_argument
from pstrain.cli.cli import main
from pstrain.cli.config import register_config_command
from pstrain.cli.package import package_command
from pstrain.cli.test import test_command
from pstrain.cli.validate import validate_command

FIXTURE = Path(__file__).parent / "fixtures" / "multipron_final_state"
ARTIFACT_KEYS = {
    "mdef",
    "means",
    "variances",
    "mixture_weights",
    "transition_matrices",
    "feat_params",
    "noisedict",
    "dictionary",
    "filler_dict",
    "manifest",
    "readme",
}


def _project_with_model(tmp_path: Path, *, profile: str = "default") -> tuple[Path, Path]:
    project = tmp_path / "project"
    model = project / "shared" / "models" / "ci-1g" / profile
    model.parent.mkdir(parents=True)
    shutil.copytree(FIXTURE / "model", model)
    shutil.copy(FIXTURE / "dictionary.dict", project / "shared" / "dictionary.dict")
    shutil.copy(FIXTURE / "filler.dict", project / "shared" / "filler.dict")
    return project, model


def _run(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["pstrain", "package", *args])
    return main()


def _write_unmarked_package(path: Path) -> Path:
    acoustic = path / "acoustic"
    acoustic.mkdir(parents=True)
    for source in (FIXTURE / "model").iterdir():
        shutil.copy(source, acoustic / source.name)
    (acoustic / "noisedict").write_text("<sil> SIL\n")
    (path / "README.txt").write_text("hand-assembled decoder model\n")
    marker = path / "keep"
    marker.write_text("unrelated content\n")
    return marker


def test_package_argument_defaults_and_overrides() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    package_command.register(subparsers)

    defaults = parser.parse_args(["package", "ci-1g"])
    overrides = parser.parse_args(
        [
            "package",
            "./model",
            "--project-dir",
            "project",
            "--config",
            "wideband",
            "--out",
            "out",
            "--name",
            "release",
            "--dict",
            "words.dict",
            "--filler-dict",
            "fillers.dict",
            "--no-dict",
            "--overwrite",
        ]
    )

    assert defaults.command == "package"
    assert defaults.command_instance is package_command
    assert defaults.project_dir is None
    assert defaults.dry_run is False
    assert defaults.json is False
    assert defaults.json_indent == 2
    assert defaults.json_ascii is False
    assert defaults.target == "ci-1g"
    assert defaults.config is None
    assert defaults.out is None
    assert defaults.name is None
    assert defaults.dict is None
    assert defaults.filler_dict is None
    assert defaults.no_dict is False
    assert defaults.overwrite is False
    assert overrides.config == "wideband"
    assert overrides.out == Path("out")
    assert overrides.name == "release"
    assert overrides.dict == Path("words.dict")
    assert overrides.filler_dict == Path("fillers.dict")
    assert overrides.no_dict is True
    assert overrides.overwrite is True


@pytest.mark.parametrize(
    ("register", "command"),
    [
        (package_command.register, ["package", "ci-1g"]),
        (test_command.register, ["test", "model"]),
        (validate_command.register, ["validate-project"]),
        (register_config_command, ["config", "profiles"]),
    ],
)
def test_global_option_values_survive_every_command_parser(
    register: Callable[[Any], Any], command: list[str]
) -> None:
    parser = argparse.ArgumentParser()
    add_json_argument(parser)
    add_dry_run_argument(parser)
    subparsers = parser.add_subparsers(dest="command")
    register(subparsers)

    args = parser.parse_args(
        ["--json", "--json-indent", "0", "--json-ascii", "--dry-run", *command]
    )

    assert args.json is True
    assert args.json_indent == 0
    assert args.json_ascii is True
    assert args.dry_run is True


def test_package_help_is_registered(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["pstrain", "package", "--help"])

    with pytest.raises(SystemExit) as raised:
        main()

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    normalized_help = " ".join(help_text.split())
    assert "--config NAME" in help_text
    assert "explicit model path" in normalized_help
    assert "relative paths resolve from the current directory" in normalized_help


def test_package_trained_model_with_project_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Package the repository's cached complete trained-model fixture."""
    project, model = _project_with_model(tmp_path)
    output = tmp_path / "output"

    assert (
        _run(
            monkeypatch,
            "ci-1g",
            "--project-dir",
            str(project),
            "--out",
            str(output),
        )
        == 0
    )

    package = output / "ci-1g"
    assert (package / "acoustic" / "mdef").read_bytes() == (model / "mdef").read_bytes()
    assert (package / "dict" / "cmudict.dict").read_bytes() == (
        project / "shared" / "dictionary.dict"
    ).read_bytes()
    lines = capsys.readouterr().out.splitlines()
    assert {line.split("\t", 1)[0] for line in lines} == ARTIFACT_KEYS
    assert f"mdef\t{package / 'acoustic' / 'mdef'}" in lines
    assert f"dictionary\t{package / 'dict' / 'cmudict.dict'}" in lines


def test_package_dry_run_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    project, model = _project_with_model(tmp_path)
    output = tmp_path / "absent-output"

    assert (
        _run(
            monkeypatch,
            "ci-1g",
            "-n",
            "--project-dir",
            str(project),
            "--out",
            str(output),
        )
        == 0
    )

    lines = capsys.readouterr().out.splitlines()
    assert f"source\t{model}" in lines
    assert f"destination\t{output / 'ci-1g'}" in lines
    assert f"dictionary\t{project / 'shared' / 'dictionary.dict'}" in lines
    assert f"mdef\t{output / 'ci-1g' / 'acoustic' / 'mdef'}" in lines
    assert not output.exists()
    assert not list(output.glob(".ci-1g-*"))


def test_package_no_dict_documents_external_dictionary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _ = _project_with_model(tmp_path)
    output = tmp_path / "output"

    assert (
        _run(
            monkeypatch,
            "ci-1g",
            "--project-dir",
            str(project),
            "--out",
            str(output),
            "--no-dict",
        )
        == 0
    )

    package = output / "ci-1g"
    readme = (package / "README.txt").read_text()
    assert not (package / "dict").exists()
    assert "dict/" not in readme
    assert "A pronunciation dictionary is not included" in readme
    assert "-dict /path/to/dictionary.dict" in readme


def test_package_missing_default_dictionary_names_remedies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    project, _ = _project_with_model(tmp_path)
    dictionary = project / "shared" / "dictionary.dict"
    dictionary.unlink()

    assert _run(monkeypatch, "ci-1g", "--project-dir", str(project)) == 1

    assert capsys.readouterr().err.strip() == (
        f"Error: Dictionary not found: {dictionary}. Provide --dict PATH or use --no-dict."
    )
    assert not (project / "packages").exists()


def test_package_explicit_relative_path_and_custom_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _ = _project_with_model(tmp_path)
    explicit = tmp_path / "trained"
    shutil.copytree(FIXTURE / "model", explicit)
    monkeypatch.chdir(tmp_path)

    assert (
        _run(
            monkeypatch,
            "./trained",
            "--project-dir",
            str(project),
            "--name",
            "release",
        )
        == 0
    )

    assert (project / "packages" / "release" / "acoustic" / "mdef").read_bytes() == (
        explicit / "mdef"
    ).read_bytes()


def test_bare_target_ignores_matching_cwd_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, model = _project_with_model(tmp_path)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "ci-1g").mkdir()
    monkeypatch.chdir(cwd)

    assert _run(monkeypatch, "ci-1g", "--project-dir", str(project)) == 0

    packaged = project / "packages" / "ci-1g" / "acoustic" / "mdef"
    assert packaged.read_bytes() == (model / "mdef").read_bytes()


def test_package_requires_config_when_target_has_multiple_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    project, _ = _project_with_model(tmp_path, profile="telephone")
    shutil.copytree(FIXTURE / "model", project / "shared" / "models" / "ci-1g" / "wideband")

    assert _run(monkeypatch, "ci-1g", "--project-dir", str(project)) == 1
    assert "multiple profiles: telephone, wideband" in capsys.readouterr().err

    assert _run(monkeypatch, "ci-1g", "--project-dir", str(project), "--config", "wideband") == 0


def test_package_automatically_selects_only_nondefault_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, model = _project_with_model(tmp_path, profile="telephone")

    assert _run(monkeypatch, "ci-1g", "--project-dir", str(project)) == 0

    packaged = project / "packages" / "ci-1g" / "acoustic" / "mdef"
    assert packaged.read_bytes() == (model / "mdef").read_bytes()


def test_package_zero_profiles_uses_default_path_in_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    dictionary = tmp_path / "shared" / "dictionary.dict"
    dictionary.parent.mkdir(parents=True)
    dictionary.write_text("WORD W ER D\n")

    assert _run(monkeypatch, "ci-1g", "--project-dir", str(tmp_path)) == 1

    expected = tmp_path / "shared" / "models" / "ci-1g" / "default"
    assert f"complete model directory {expected}" in capsys.readouterr().err


def test_package_rejects_name_that_escapes_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    project, _ = _project_with_model(tmp_path)
    output = tmp_path / "output"
    outside = tmp_path / "important-directory"
    outside.mkdir()
    marker = outside / "keep"
    marker.write_text("unrelated")

    assert (
        _run(
            monkeypatch,
            "ci-1g",
            "--project-dir",
            str(project),
            "--out",
            str(output),
            "--name",
            "../important-directory",
        )
        == 1
    )

    error = capsys.readouterr().err
    assert str(outside) in error
    assert "one ordinary path component" in error
    assert marker.read_text() == "unrelated"


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_package_refuses_unrecognized_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, kind: str
) -> None:
    project, _ = _project_with_model(tmp_path)
    destination = project / "packages" / "ci-1g"
    destination.parent.mkdir(parents=True)
    if kind == "file":
        destination.write_text("unrelated")
    else:
        destination.mkdir()
        (destination / "keep").write_text("unrelated")

    assert _run(monkeypatch, "ci-1g", "--project-dir", str(project)) == 1

    error = capsys.readouterr().err
    assert str(destination) in error
    assert "not a recognizable pstrain package" in error


def test_package_rejects_source_equal_to_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    project, _ = _project_with_model(tmp_path)
    model = tmp_path / "model"
    shutil.copytree(FIXTURE / "model", model)

    assert (
        _run(
            monkeypatch,
            str(model),
            "--project-dir",
            str(project),
            "--out",
            str(tmp_path),
            "--name",
            "model",
            "--no-dict",
        )
        == 1
    )

    error = capsys.readouterr().err
    assert str(model) in error
    assert "overlaps source model" in error
    assert (model / "mdef").is_file()


def test_package_rejects_destination_containing_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    project, _ = _project_with_model(tmp_path)
    destination = tmp_path / "container"
    source = destination / "model"
    destination.mkdir()
    shutil.copytree(FIXTURE / "model", source)

    assert (
        _run(
            monkeypatch,
            str(source),
            "--project-dir",
            str(project),
            "--out",
            str(tmp_path),
            "--name",
            "container",
            "--no-dict",
            "--dry-run",
        )
        == 1
    )

    captured = capsys.readouterr()
    assert str(destination) in captured.err
    assert str(source) in captured.err
    assert "overlaps source model" in captured.err
    assert captured.out == ""
    assert (source / "mdef").is_file()


def test_package_reports_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    project, _ = _project_with_model(tmp_path)

    assert _run(monkeypatch, "ci-1g", "--project-dir", str(project)) == 0
    capsys.readouterr()
    stale = project / "packages" / "ci-1g" / "stale"
    stale.write_text("old")

    assert _run(monkeypatch, "ci-1g", "--project-dir", str(project)) == 0

    package = project / "packages" / "ci-1g"
    assert not stale.exists()
    assert capsys.readouterr().out.splitlines()[0] == f"replaced\t{package}"
    assert json.loads((package / "pstrain-package.json").read_text()) == {
        "format_version": 1,
        "generator": "pstrain",
    }


def test_package_refuses_unmarked_package_shaped_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    project, _ = _project_with_model(tmp_path)
    destination = project / "packages" / "ci-1g"
    marker = _write_unmarked_package(destination)

    assert _run(monkeypatch, "ci-1g", "--project-dir", str(project)) == 1

    error = capsys.readouterr().err
    assert error.strip() == (
        f"Error: Refusing to replace {destination}: existing package has no pstrain package "
        "marker. Use --overwrite to replace the entire existing directory."
    )
    assert marker.read_text() == "unrelated content\n"


def test_package_explicit_overwrite_replaces_unmarked_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _ = _project_with_model(tmp_path)
    destination = project / "packages" / "ci-1g"
    marker = _write_unmarked_package(destination)

    assert _run(monkeypatch, "ci-1g", "--project-dir", str(project), "--overwrite") == 0

    assert not marker.exists()
    assert (destination / "pstrain-package.json").is_file()


@pytest.mark.parametrize("version", [2, True, 1.0])
def test_package_refuses_invalid_or_unsupported_marker_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    version: object,
) -> None:
    project, _ = _project_with_model(tmp_path)
    destination = project / "packages" / "ci-1g"
    assert _run(monkeypatch, "ci-1g", "--project-dir", str(project)) == 0
    capsys.readouterr()
    marker = destination / "pstrain-package.json"
    marker.write_text(json.dumps({"format_version": version, "generator": "pstrain"}))

    assert _run(monkeypatch, "ci-1g", "--project-dir", str(project), "--overwrite") == 1

    assert capsys.readouterr().err.strip() == (
        f"Error: Refusing to replace {destination}: existing pstrain package marker is "
        "invalid or uses an unsupported format version. An overwrite replaces the entire "
        "existing directory, so the marker must be understood before replacement."
    )
    assert json.loads(marker.read_text())["format_version"] == version


def test_package_rejects_identity_overlap_with_different_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    project, _ = _project_with_model(tmp_path)
    destination = tmp_path / "Package"
    source = destination / "model"
    destination.mkdir()
    shutil.copytree(FIXTURE / "model", source)
    _write_unmarked_package(destination)
    alternate_destination = tmp_path / "package"
    if not alternate_destination.exists():
        pytest.skip("filesystem is case-sensitive")

    assert (
        _run(
            monkeypatch,
            str(source),
            "--project-dir",
            str(project),
            "--out",
            str(tmp_path),
            "--name",
            "package",
            "--no-dict",
        )
        == 1
    )

    captured = capsys.readouterr()
    assert "overlaps source model" in captured.err
    assert captured.out == ""
    assert (source / "mdef").is_file()


def test_package_rejects_empty_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    project, _ = _project_with_model(tmp_path)

    assert _run(monkeypatch, "ci-1g", "--project-dir", str(project), "--name", "") == 1

    error = capsys.readouterr().err
    assert "package name '' must be exactly one ordinary path component" in error
    assert not (project / "packages").exists()


def test_package_missing_model_reports_complete_model_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    dictionary = tmp_path / "shared" / "dictionary.dict"
    dictionary.parent.mkdir()
    dictionary.write_text("WORD W ER D\n")

    assert _run(monkeypatch, "missing", "--project-dir", str(tmp_path)) == 1

    expected = tmp_path / "shared" / "models" / "missing" / "default"
    error = capsys.readouterr().err
    assert "Missing feat.params" in error
    assert f"complete model directory {expected}" in error
