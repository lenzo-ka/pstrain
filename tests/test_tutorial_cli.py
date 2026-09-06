"""Tests for the tutorial notebook API and CLI command."""

from __future__ import annotations

import json
import os
import sys
from importlib import import_module
from pathlib import Path

import pytest

from pstrain.api import TUTORIAL_FILENAME, copy_tutorial
from pstrain.cli.cli import main

SOURCE_NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / TUTORIAL_FILENAME
REPOSITORY_ROOT = SOURCE_NOTEBOOK.parents[1]


@pytest.fixture(autouse=True)
def _source_checkout_resource(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make source-tree API tests stand in for the build backend's wheel mapping."""
    tutorial_api = import_module("pstrain.api.tutorial")
    monkeypatch.setattr(tutorial_api, "files", lambda package: REPOSITORY_ROOT)


def _run(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["pstrain", "tutorial", *args])
    return main()


def test_tutorial_default_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert _run(monkeypatch) == 0
    assert (tmp_path / TUTORIAL_FILENAME).read_bytes() == SOURCE_NOTEBOOK.read_bytes()


def test_tutorial_explicit_output_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "nested" / "lesson.ipynb"

    assert _run(monkeypatch, "--output", str(output)) == 0
    assert output.read_bytes() == SOURCE_NOTEBOOK.read_bytes()


def test_tutorial_explicit_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "lessons"
    output.mkdir()

    assert _run(monkeypatch, "-o", str(output)) == 0
    assert (output / TUTORIAL_FILENAME).read_bytes() == SOURCE_NOTEBOOK.read_bytes()


def test_tutorial_refuses_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "lesson.ipynb"
    output.write_text("reader edits", encoding="utf-8")

    assert _run(monkeypatch, "--output", str(output)) == 1
    assert capsys.readouterr().err.strip() == (
        f"Error: Refusing to replace {output}: destination already exists. "
        "Use --force to replace it."
    )
    assert output.read_text(encoding="utf-8") == "reader edits"


def test_tutorial_force_replaces_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "lesson.ipynb"
    output.write_text("reader edits", encoding="utf-8")

    assert _run(monkeypatch, "--output", str(output), "--force") == 0
    assert output.read_bytes() == SOURCE_NOTEBOOK.read_bytes()


def test_tutorial_refuses_dangling_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "lesson.ipynb"
    output.symlink_to(tmp_path / "missing.ipynb")

    assert _run(monkeypatch, "--output", str(output)) == 1
    assert f"Refusing to replace {output}" in capsys.readouterr().err
    assert output.is_symlink()


def test_tutorial_no_force_publish_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "lesson.ipynb"
    original_link = os.link

    def create_racing_destination(source: Path, destination: Path) -> None:
        destination.write_text("racing writer", encoding="utf-8")
        original_link(source, destination)

    tutorial_api = import_module("pstrain.api.tutorial")
    monkeypatch.setattr(tutorial_api.os, "link", create_racing_destination)

    with pytest.raises(FileExistsError, match=f"Refusing to replace {output}"):
        copy_tutorial(output)
    assert output.read_text(encoding="utf-8") == "racing writer"


def test_tutorial_reports_unsupported_hard_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "lesson.ipynb"
    tutorial_api = import_module("pstrain.api.tutorial")

    def reject_hard_link(source: Path, destination: Path) -> None:
        raise OSError("hard links unavailable")

    monkeypatch.setattr(tutorial_api.os, "link", reject_hard_link)

    with pytest.raises(OSError, match="filesystem does not support atomic no-clobber"):
        copy_tutorial(output)
    assert not output.exists()


def test_tutorial_ignores_temporary_cleanup_failure_after_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "lesson.ipynb"
    original_unlink = Path.unlink

    def fail_temporary_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path.name.startswith(f".{output.name}."):
            raise OSError("temporary cleanup failed")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_temporary_unlink)

    assert copy_tutorial(output)["status"] == "written"
    assert output.read_bytes() == SOURCE_NOTEBOOK.read_bytes()


def test_tutorial_dry_run_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "missing" / "lesson.ipynb"

    assert _run(monkeypatch, "--output", str(output), "--dry-run") == 0
    assert capsys.readouterr().out.strip() == f"Would write tutorial to: {output}"
    assert not output.parent.exists()


def test_tutorial_json_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "lesson.ipynb"

    assert _run(monkeypatch, "--output", str(output), "--json") == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "written",
        "path": str(output),
    }


def test_tutorial_global_json_ordering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "lesson.ipynb"
    monkeypatch.setattr(
        sys,
        "argv",
        ["pstrain", "--json", "tutorial", "--dry-run", "--output", str(output)],
    )

    assert main() == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "dry-run",
        "path": str(output),
    }
    assert not output.exists()


def test_tutorial_json_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "lesson.ipynb"
    output.write_text("reader edits", encoding="utf-8")

    assert _run(monkeypatch, "--json", "--output", str(output)) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "status": "error",
        "path": str(output),
        "error": (
            f"Refusing to replace {output}: destination already exists. Use --force to replace it."
        ),
    }


def test_tutorial_api_result_is_json_serializable(tmp_path: Path) -> None:
    result = copy_tutorial(tmp_path)

    assert json.loads(json.dumps(result)) == {
        "status": "written",
        "path": str(tmp_path / TUTORIAL_FILENAME),
    }


def test_tutorial_has_one_canonical_wheel_mapping() -> None:
    configuration = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    mapping = (
        '"notebooks/arctic_hmm_gmm_tutorial.ipynb" = '
        '"pstrain/data/notebooks/arctic_hmm_gmm_tutorial.ipynb"'
    )

    assert configuration.count(mapping) == 1
    assert not (REPOSITORY_ROOT / "pstrain/data/notebooks" / TUTORIAL_FILENAME).exists()
