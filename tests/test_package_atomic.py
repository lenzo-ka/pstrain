"""Regression tests for complete package installation."""

import os
import shutil
from pathlib import Path

import pytest

import pstrain.lib.steps.package as package_step
from pstrain.lib.model import MODEL_FILES_REQUIRED
from pstrain.lib.pipeline.context import FeatParams
from pstrain.lib.pipeline.feat_params import write_feat_params
from pstrain.lib.steps.package import package_model


def _write_complete_model(model_dir: Path) -> None:
    model_dir.mkdir()
    for filename in MODEL_FILES_REQUIRED:
        (model_dir / filename).write_text(filename)
    write_feat_params(model_dir / "feat.params", FeatParams())


def test_packaging_copy_failure_leaves_no_partial_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "model"
    _write_complete_model(model_dir)
    package_dir = tmp_path / "dist" / "test-model"
    original_copy = shutil.copy2
    expected = OSError("injected copy failure")
    copy_count = 0

    def fail_second_copy(src: Path, dst: Path) -> str:
        nonlocal copy_count
        copy_count += 1
        if copy_count == 2:
            raise expected
        return original_copy(src, dst)

    monkeypatch.setattr("pstrain.lib.steps.package.shutil.copy", fail_second_copy)
    monkeypatch.setattr("pstrain.lib.steps.package.shutil.copy2", fail_second_copy)

    with pytest.raises(OSError) as raised:
        package_model(model_dir, tmp_path / "dist", model_name="test-model")

    assert raised.value is expected
    assert not package_dir.exists()


def test_packaging_success_tree_is_unchanged(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_complete_model(model_dir)
    package_dir = tmp_path / "dist" / "test-model"

    package_model(model_dir, tmp_path / "dist", model_name="test-model")

    assert sorted(
        path.relative_to(package_dir).as_posix() for path in package_dir.rglob("*")
    ) == sorted(
        [
            "README.txt",
            "acoustic",
            "acoustic/feat.params",
            "acoustic/noisedict",
            *(f"acoustic/{filename}" for filename in MODEL_FILES_REQUIRED),
            "dict",
        ]
    )


def test_packaging_overwrite_removes_stale_files(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_complete_model(model_dir)
    package_dir = tmp_path / "dist" / "test-model"
    stale_file = package_dir / "acoustic" / "stale"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("old package")

    package_model(model_dir, tmp_path / "dist", model_name="test-model")

    assert not stale_file.exists()
    assert (package_dir / "acoustic" / MODEL_FILES_REQUIRED[0]).is_file()


def test_unnamed_package_preserves_unrelated_output(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_complete_model(model_dir)
    output_dir = tmp_path / "dist"
    output_dir.mkdir()
    stray_file = output_dir / "keep.txt"
    stray_file.write_text("unrelated")

    package_model(model_dir, output_dir)

    assert stray_file.read_text() == "unrelated"
    assert (output_dir / "acoustic" / MODEL_FILES_REQUIRED[0]).is_file()
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "README.txt",
        "acoustic",
        "dict",
        "keep.txt",
    ]


def test_unnamed_package_rolls_back_all_paths_after_install_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_model = tmp_path / "old-model"
    new_model = tmp_path / "new-model"
    _write_complete_model(old_model)
    _write_complete_model(new_model)
    (new_model / MODEL_FILES_REQUIRED[0]).write_text("new model")
    output_dir = tmp_path / "dist"
    dictionary = tmp_path / "dictionary"
    dictionary.write_text("WORD W ER D\n")
    package_model(old_model, output_dir, dictionary_path=dictionary)
    old_files = {
        path.relative_to(output_dir).as_posix(): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    original_replace = os.replace
    install_count = 0

    def fail_second_install(src: Path, dst: Path) -> None:
        nonlocal install_count
        if Path(src).parent.name.startswith(".dist-"):
            install_count += 1
            if install_count == 2:
                raise OSError("injected second-subtree install failure")
        original_replace(src, dst)

    monkeypatch.setattr(package_step.os, "replace", fail_second_install)

    with pytest.raises(OSError, match="second-subtree"):
        package_model(new_model, output_dir, dictionary_path=dictionary)

    restored_files = {
        path.relative_to(output_dir).as_posix(): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    assert restored_files == old_files
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "README.txt",
        "acoustic",
        "dict",
    ]


def test_installed_package_survives_backup_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    model_dir = tmp_path / "model"
    _write_complete_model(model_dir)
    package_dir = tmp_path / "dist" / "test-model"
    package_model(model_dir, tmp_path / "dist", model_name="test-model")
    original_rmtree = shutil.rmtree

    def fail_backup_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if "-old-" in Path(path).name:
            raise OSError("injected cleanup failure")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(package_step.shutil, "rmtree", fail_backup_cleanup)

    package_model(model_dir, tmp_path / "dist", model_name="test-model")

    backups = list((tmp_path / "dist").glob(".test-model-old-*"))
    assert (package_dir / "README.txt").is_file()
    assert len(backups) == 1
    assert str(backups[0]) in caplog.text


def test_restore_failure_reports_original_error_and_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "model"
    _write_complete_model(model_dir)
    package_dir = tmp_path / "dist" / "test-model"
    package_model(model_dir, tmp_path / "dist", model_name="test-model")
    original_replace = os.replace
    install_error = OSError("injected install failure")
    replace_count = 0

    def fail_install_and_restore(src: Path, dst: Path) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count > 1:
            raise OSError("injected restore failure") if replace_count == 3 else install_error
        original_replace(src, dst)

    monkeypatch.setattr(package_step.os, "replace", fail_install_and_restore)

    with pytest.raises(RuntimeError, match=r"\.test-model-old-") as raised:
        package_model(model_dir, tmp_path / "dist", model_name="test-model")

    assert raised.value.__cause__ is install_error
    assert not package_dir.exists()
    assert len(list((tmp_path / "dist").glob(".test-model-old-*"))) == 1
