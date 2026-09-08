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
            "pstrain-package.json",
            "acoustic",
            "acoustic/feat.params",
            "acoustic/noisedict",
            *(f"acoustic/{filename}" for filename in MODEL_FILES_REQUIRED),
            "dict",
        ]
    )


def test_readme_only_documents_dictionary_artifacts_that_exist(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_complete_model(model_dir)
    dictionary = tmp_path / "dictionary.dict"
    dictionary.write_text("WORD W ER D\n")

    result = package_model(
        model_dir,
        tmp_path / "dist",
        model_name="test-model",
        dictionary_path=dictionary,
    )

    readme = result["readme"].read_text()
    assert "cmudict.dict" in readme
    assert "filler.dict   -" not in readme


def test_readme_without_dictionary_does_not_claim_one(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_complete_model(model_dir)

    result = package_model(model_dir, tmp_path / "dist", include_dict=False)

    readme = result["readme"].read_text()
    assert "dict/" not in readme
    assert "A pronunciation dictionary is not included" in readme
    assert "-dict /path/to/dictionary.dict" in readme


def test_packaging_overwrite_removes_stale_files(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_complete_model(model_dir)
    package_dir = tmp_path / "dist" / "test-model"
    package_model(model_dir, tmp_path / "dist", model_name="test-model")
    stale_file = package_dir / "acoustic" / "stale"
    stale_file.write_text("old package")

    package_model(model_dir, tmp_path / "dist", model_name="test-model")

    assert not stale_file.exists()
    assert (package_dir / "acoustic" / MODEL_FILES_REQUIRED[0]).is_file()


def test_destination_introduced_after_final_validation_is_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "model"
    _write_complete_model(model_dir)
    output_dir = tmp_path / "dist"
    package_dir = output_dir / "test-model"
    unrelated = package_dir / "belongs-to-nobody.txt"
    original_rename = package_step._rename_noreplace_at
    injected = False

    def inject_before_install(
        source_parent_fd: int,
        source: str,
        destination_parent_fd: int,
        destination: str,
    ) -> None:
        nonlocal injected
        if (
            destination == package_dir.name
            and source.startswith(f".{package_dir.name}-")
            and not injected
        ):
            injected = True
            package_dir.mkdir()
            unrelated.write_text("racing destination")
        original_rename(source_parent_fd, source, destination_parent_fd, destination)

    monkeypatch.setattr(package_step, "_rename_noreplace_at", inject_before_install)

    with pytest.raises(FileExistsError):
        package_model(model_dir, output_dir, model_name="test-model")

    assert injected
    assert unrelated.read_text() == "racing destination"


def test_packaging_fails_closed_without_atomic_noreplace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "model"
    _write_complete_model(model_dir)
    package_dir = tmp_path / "dist" / "test-model"

    def unsupported(
        _source_parent_fd: int,
        _source: str,
        _destination_parent_fd: int,
        _destination: str,
    ) -> None:
        raise OSError("atomic no-replace rename unavailable")

    monkeypatch.setattr(package_step, "_rename_noreplace_at", unsupported)

    with pytest.raises(OSError, match="atomic no-replace rename unavailable"):
        package_model(model_dir, tmp_path / "dist", model_name="test-model")

    assert not package_dir.exists()
    assert (model_dir / "mdef").is_file()


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
        "pstrain-package.json",
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
    original_rename = package_step._rename_noreplace_at
    rename_count = 0

    def fail_second_install(
        source_parent_fd: int,
        source: str,
        destination_parent_fd: int,
        destination: str,
    ) -> None:
        nonlocal rename_count
        rename_count += 1
        if rename_count == 6:
            raise OSError("injected second-subtree install failure")
        original_rename(source_parent_fd, source, destination_parent_fd, destination)

    monkeypatch.setattr(package_step, "_rename_noreplace_at", fail_second_install)

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
        "pstrain-package.json",
    ]


def test_installed_package_survives_backup_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "model"
    _write_complete_model(model_dir)
    package_dir = tmp_path / "dist" / "test-model"
    package_model(model_dir, tmp_path / "dist", model_name="test-model")
    original_remove = package_step._remove_open_entry

    def fail_backup_cleanup(
        parent_fd: int,
        name: str,
        entry: object,
        display_path: Path,
    ) -> None:
        if name == "retained":
            raise OSError("injected cleanup failure")
        original_remove(parent_fd, name, entry, display_path)

    monkeypatch.setattr(package_step, "_remove_open_entry", fail_backup_cleanup)

    with pytest.raises(OSError, match="injected cleanup failure"):
        package_model(model_dir, tmp_path / "dist", model_name="test-model")

    backups = list((tmp_path / "dist").glob(".test-model-old-*"))
    assert (package_dir / "README.txt").is_file()
    assert len(backups) == 1
    assert (backups[0] / "retained" / "README.txt").is_file()


def test_restore_failure_reports_original_error_and_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "model"
    _write_complete_model(model_dir)
    package_dir = tmp_path / "dist" / "test-model"
    package_model(model_dir, tmp_path / "dist", model_name="test-model")
    original_rename = package_step._rename_noreplace_at
    install_error = OSError("injected install failure")
    replace_count = 0

    def fail_install_and_restore(
        source_parent_fd: int,
        source: str,
        destination_parent_fd: int,
        destination: str,
    ) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count > 1:
            raise OSError("injected restore failure") if replace_count == 3 else install_error
        original_rename(source_parent_fd, source, destination_parent_fd, destination)

    monkeypatch.setattr(package_step, "_rename_noreplace_at", fail_install_and_restore)

    with pytest.raises(OSError, match="injected install failure") as raised:
        package_model(model_dir, tmp_path / "dist", model_name="test-model")

    assert raised.value is install_error
    assert any(
        "could not restore retained package entry" in note for note in raised.value.__notes__
    )
    assert not package_dir.exists()
    assert len(list((tmp_path / "dist").glob(".test-model-old-*"))) == 1


def test_retained_package_swap_after_rename_refuses_without_deleting_either_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_model = tmp_path / "old-model"
    new_model = tmp_path / "new-model"
    substitute_model = tmp_path / "substitute-model"
    _write_complete_model(old_model)
    _write_complete_model(new_model)
    _write_complete_model(substitute_model)
    (old_model / "mdef").write_text("old destination")
    (substitute_model / "mdef").write_text("substituted package")
    output = tmp_path / "dist"
    destination = output / "release"
    substitute = tmp_path / "substitute"
    package_model(old_model, output, model_name="release", include_dict=False)
    package_model(substitute_model, tmp_path, model_name="substitute", include_dict=False)
    original_rename = package_step._rename_noreplace_at
    injected = False

    def swap_after_rename(
        source_parent_fd: int,
        source: str,
        destination_parent_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal injected
        original_rename(source_parent_fd, source, destination_parent_fd, destination_name)
        if source == "release" and destination_name == "retained" and not injected:
            os.rename(
                "retained",
                "real-retained",
                src_dir_fd=destination_parent_fd,
                dst_dir_fd=destination_parent_fd,
            )
            os.rename(substitute, "retained", dst_dir_fd=destination_parent_fd)
            injected = True

    monkeypatch.setattr(package_step, "_rename_noreplace_at", swap_after_rename)

    with pytest.raises(RuntimeError, match="neither rename endpoint"):
        package_model(new_model, output, model_name="release", include_dict=False)

    backup = next(output.glob(".release-old-*"))
    assert injected
    assert not destination.exists()
    assert (backup / "real-retained" / "acoustic" / "mdef").read_text() == "old destination"
    assert (backup / "retained" / "acoustic" / "mdef").read_text() == "substituted package"


def test_backup_root_swap_during_cleanup_refuses_without_deleting_substitute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_model = tmp_path / "old-model"
    new_model = tmp_path / "new-model"
    _write_complete_model(old_model)
    _write_complete_model(new_model)
    output = tmp_path / "dist"
    package_model(old_model, output, model_name="release", include_dict=False)
    original_remove = package_step._remove_open_entry
    displaced = tmp_path / "open-backup"
    unrelated: Path | None = None

    def swap_before_descriptor_cleanup(
        parent_fd: int,
        name: str,
        entry: object,
        display_path: Path,
    ) -> None:
        nonlocal unrelated
        if name == "retained" and unrelated is None:
            backup_root = display_path.parent
            backup_root.rename(displaced)
            backup_root.mkdir()
            unrelated = backup_root / "belongs-to-nobody.txt"
            unrelated.write_text("unrelated file survived")
        original_remove(parent_fd, name, entry, display_path)

    monkeypatch.setattr(package_step, "_remove_open_entry", swap_before_descriptor_cleanup)

    with pytest.raises(RuntimeError, match="changed identity"):
        package_model(new_model, output, model_name="release", include_dict=False)

    assert unrelated is not None
    assert unrelated.read_text() == "unrelated file survived"
    assert (output / "release" / "acoustic" / "mdef").is_file()


def test_unnamed_interruption_after_successful_rename_reconciles_and_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_model = tmp_path / "old-model"
    new_model = tmp_path / "new-model"
    _write_complete_model(old_model)
    _write_complete_model(new_model)
    output = tmp_path / "dist"
    package_model(old_model, output, include_dict=False)
    old_files = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    original_rename = package_step._rename_noreplace_at
    interruption = RuntimeError("injected after successful rename")
    injected = False

    def interrupt_after_rename(
        source_parent_fd: int,
        source: str,
        destination_parent_fd: int,
        destination: str,
    ) -> None:
        nonlocal injected
        original_rename(source_parent_fd, source, destination_parent_fd, destination)
        if source == "acoustic" and destination == "acoustic" and not injected:
            injected = True
            raise interruption

    monkeypatch.setattr(package_step, "_rename_noreplace_at", interrupt_after_rename)

    with pytest.raises(RuntimeError, match="injected after successful rename") as raised:
        package_model(new_model, output, include_dict=False)

    restored_files = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file() and ".pstrain-package-old-" not in path.as_posix()
    }
    assert raised.value is interruption
    assert injected
    assert restored_files == old_files
    assert (output / "acoustic").is_dir()
    assert (output / "README.txt").is_file()
    assert (output / "pstrain-package.json").is_file()


def test_named_interruption_after_successful_install_reconciles_and_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_model = tmp_path / "old-model"
    new_model = tmp_path / "new-model"
    _write_complete_model(old_model)
    _write_complete_model(new_model)
    (old_model / "mdef").write_text("old destination")
    (new_model / "mdef").write_text("new destination")
    output = tmp_path / "dist"
    package_model(old_model, output, model_name="release", include_dict=False)
    original_rename = package_step._rename_noreplace_at
    interruption = RuntimeError("injected after named install")
    injected = False

    def interrupt_after_install(
        source_parent_fd: int,
        source: str,
        destination_parent_fd: int,
        destination: str,
    ) -> None:
        nonlocal injected
        original_rename(source_parent_fd, source, destination_parent_fd, destination)
        if source.startswith(".release-") and destination == "release" and not injected:
            injected = True
            raise interruption

    monkeypatch.setattr(package_step, "_rename_noreplace_at", interrupt_after_install)

    with pytest.raises(RuntimeError, match="injected after named install") as raised:
        package_model(new_model, output, model_name="release", include_dict=False)

    assert raised.value is interruption
    assert injected
    assert (output / "release" / "acoustic" / "mdef").read_text() == "old destination"


def test_first_publish_file_exists_remains_primary_when_cleanup_also_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    _write_complete_model(model)
    output = tmp_path / "dist"
    original_rename = package_step._rename_noreplace_at
    original_empty = package_step._empty_open_directory
    injected = False

    def introduce_destination(
        source_parent_fd: int,
        source: str,
        destination_parent_fd: int,
        destination: str,
    ) -> None:
        nonlocal injected
        if source == "acoustic" and destination == "acoustic" and not injected:
            injected = True
            (output / "acoustic").mkdir()
            (output / "acoustic" / "belongs-to-nobody.txt").write_text("unrelated")
        original_rename(source_parent_fd, source, destination_parent_fd, destination)

    def fail_staging_cleanup(directory_fd: int, display_path: Path) -> None:
        if display_path.name.startswith(".dist-"):
            raise OSError("injected cleanup failure")
        original_empty(directory_fd, display_path)

    monkeypatch.setattr(package_step, "_rename_noreplace_at", introduce_destination)
    monkeypatch.setattr(package_step, "_empty_open_directory", fail_staging_cleanup)

    with pytest.raises(FileExistsError) as raised:
        package_model(model, output, include_dict=False)

    assert injected
    assert "acoustic -> acoustic" in str(raised.value)
    assert (output / "acoustic" / "belongs-to-nobody.txt").read_text() == "unrelated"
