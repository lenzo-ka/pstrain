"""Tests for resolving the tutorial resource in supported and hostile layouts."""

from importlib import import_module
from pathlib import Path

import pytest

from pstrain.api import TUTORIAL_FILENAME, copy_tutorial


def test_tutorial_resolves_package_resource_in_source_checkout(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    source = repository_root / "pstrain" / "data" / "notebooks" / TUTORIAL_FILENAME
    output = tmp_path / TUTORIAL_FILENAME

    assert source.is_file()
    assert copy_tutorial(output)["status"] == "written"
    assert output.read_bytes() == source.read_bytes()


def test_installed_package_does_not_use_adjacent_checkout_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_root = tmp_path / "target"
    fake_package = install_root / "pstrain"
    fake_module = fake_package / "api" / "tutorial.py"
    adjacent_notebook = install_root / "notebooks" / TUTORIAL_FILENAME
    output = tmp_path / "copied.ipynb"
    fake_module.parent.mkdir(parents=True)
    adjacent_notebook.parent.mkdir()
    (install_root / "pyproject.toml").write_text("[project]\nname='decoy'\n")
    adjacent_notebook.write_text("not the packaged notebook", encoding="utf-8")

    tutorial_api = import_module("pstrain.api.tutorial")
    monkeypatch.setattr(tutorial_api, "__file__", str(fake_module))
    monkeypatch.setattr(tutorial_api, "files", lambda package: fake_package / "data")

    with pytest.raises(FileNotFoundError, match="missing from this installation"):
        copy_tutorial(output)
    assert not output.exists()
