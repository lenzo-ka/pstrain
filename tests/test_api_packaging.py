"""Tests for package safety at the public API boundary."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import pstrain.lib.steps.package as package_step
from pstrain.api import package_model

FIXTURE = Path(__file__).parent / "fixtures" / "multipron_final_state" / "model"


def _copy_model(path: Path) -> Path:
    shutil.copytree(FIXTURE, path)
    return path


@pytest.mark.parametrize("overwrite", [False, True])
def test_api_refuses_unrecognized_destination_without_destroying_it(
    tmp_path: Path, overwrite: bool
) -> None:
    model = _copy_model(tmp_path / "model")
    destination = tmp_path / "dist" / "release"
    destination.mkdir(parents=True)
    unrelated = destination / "belongs-to-nobody.txt"
    unrelated.write_text("unrelated\n")

    with pytest.raises(ValueError, match="not a recognizable pstrain package"):
        package_model(
            model,
            tmp_path / "dist",
            model_name="release",
            include_dict=False,
            overwrite=overwrite,
        )

    assert unrelated.read_text() == "unrelated\n"
    assert not (destination / "pstrain-package.json").exists()


def test_api_replaces_supported_package_without_opt_in(tmp_path: Path) -> None:
    model = _copy_model(tmp_path / "model")
    destination = tmp_path / "dist" / "release"
    package_model(model, tmp_path / "dist", model_name="release", include_dict=False)
    stale = destination / "stale.txt"
    stale.write_text("old package\n")

    package_model(model, tmp_path / "dist", model_name="release", include_dict=False)

    assert not stale.exists()
    assert json.loads((destination / "pstrain-package.json").read_text()) == {
        "format_version": 1,
        "generator": "pstrain",
    }


def test_api_requires_opt_in_to_replace_unmarked_legacy_package(tmp_path: Path) -> None:
    model = _copy_model(tmp_path / "model")
    destination = tmp_path / "dist" / "release"
    package_model(model, tmp_path / "dist", model_name="release", include_dict=False)
    (destination / "pstrain-package.json").unlink()
    legacy_file = destination / "legacy-extra.txt"
    legacy_file.write_text("legacy package\n")

    with pytest.raises(ValueError, match="existing package has no pstrain package marker"):
        package_model(model, tmp_path / "dist", model_name="release", include_dict=False)

    assert legacy_file.read_text() == "legacy package\n"

    package_model(
        model,
        tmp_path / "dist",
        model_name="release",
        include_dict=False,
        overwrite=True,
    )

    assert not legacy_file.exists()
    assert (destination / "pstrain-package.json").is_file()


@pytest.mark.parametrize(
    "marker_contents",
    [
        "{not-json\n",
        json.dumps({"format_version": 2, "generator": "pstrain"}),
    ],
    ids=["invalid", "unsupported"],
)
def test_api_refuses_invalid_and_unsupported_markers_even_with_opt_in(
    tmp_path: Path, marker_contents: str
) -> None:
    model = _copy_model(tmp_path / "model")
    destination = tmp_path / "dist" / "release"
    package_model(model, tmp_path / "dist", model_name="release", include_dict=False)
    marker = destination / "pstrain-package.json"
    marker.write_text(marker_contents)
    retained = destination / "retain.txt"
    retained.write_text("old package\n")

    with pytest.raises(ValueError, match="invalid or uses an unsupported format version"):
        package_model(
            model,
            tmp_path / "dist",
            model_name="release",
            include_dict=False,
            overwrite=True,
        )

    assert marker.read_text() == marker_contents
    assert retained.read_text() == "old package\n"


@pytest.mark.parametrize("model_name", ["", ".", "..", "nested/release"])
def test_api_refuses_names_that_are_not_one_component(tmp_path: Path, model_name: str) -> None:
    model = _copy_model(tmp_path / "model")

    with pytest.raises(ValueError, match="must be exactly one ordinary path component"):
        package_model(
            model,
            tmp_path / "dist",
            model_name=model_name,
            include_dict=False,
        )

    assert not (tmp_path / "dist").exists()
    assert (model / "mdef").is_file()


def test_api_refuses_name_that_would_escape_output(tmp_path: Path) -> None:
    model = _copy_model(tmp_path / "model")
    outside = tmp_path / "important"
    outside.mkdir()
    unrelated = outside / "keep.txt"
    unrelated.write_text("unrelated\n")

    with pytest.raises(ValueError, match="must be exactly one ordinary path component"):
        package_model(
            model,
            tmp_path / "dist",
            model_name="../important",
            include_dict=False,
        )

    assert unrelated.read_text() == "unrelated\n"


@pytest.mark.parametrize(
    "relationship",
    ["equal", "destination-contains-source", "source-contains-destination", "symlink-alias"],
)
def test_api_refuses_source_and_destination_overlap(tmp_path: Path, relationship: str) -> None:
    if relationship == "equal":
        model = _copy_model(tmp_path / "model")
        output = tmp_path
        name = "model"
    elif relationship == "destination-contains-source":
        model = _copy_model(tmp_path / "container" / "model")
        output = tmp_path
        name = "container"
    elif relationship == "source-contains-destination":
        model = _copy_model(tmp_path / "model")
        output = model
        name = "package"
    else:
        actual_destination = tmp_path / "actual"
        model = _copy_model(actual_destination / "model")
        (tmp_path / "alias").symlink_to(actual_destination, target_is_directory=True)
        output = tmp_path
        name = "alias"
    original = (model / "mdef").read_bytes()

    with pytest.raises(ValueError, match="overlaps source model"):
        package_model(model, output, model_name=name, include_dict=False)

    assert (model / "mdef").read_bytes() == original
    assert not list(tmp_path.rglob(".*-old-*"))


def test_api_refuses_source_identity_found_beneath_destination(tmp_path: Path) -> None:
    old_model = _copy_model(tmp_path / "old-model")
    source = _copy_model(tmp_path / "source")
    output = tmp_path / "dist"
    destination = output / "release"
    package_model(old_model, output, model_name="release", include_dict=False)
    (destination / "source-alias").symlink_to(source, target_is_directory=True)
    original = (source / "mdef").read_bytes()

    with pytest.raises(ValueError, match="overlaps source model"):
        package_model(source, output, model_name="release", include_dict=False)

    assert (source / "mdef").read_bytes() == original
    assert (destination / "source-alias").is_symlink()


def test_api_fails_closed_when_destination_identity_walk_is_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_model = _copy_model(tmp_path / "old-model")
    source = _copy_model(tmp_path / "source")
    output = tmp_path / "dist"
    destination = output / "release"
    package_model(old_model, output, model_name="release", include_dict=False)
    marker = destination / "pstrain-package.json"
    original_scandir = package_step.os.scandir

    def deny_destination(path: Path) -> object:
        if Path(path) == destination:
            raise PermissionError("identity walk denied")
        return original_scandir(path)

    monkeypatch.setattr(package_step.os, "scandir", deny_destination)

    with pytest.raises(ValueError, match="Cannot safely inspect.*identity walk denied"):
        package_model(source, output, model_name="release", include_dict=False)

    assert marker.is_file()
    assert (source / "mdef").is_file()


def test_api_permits_ordinary_name_and_separate_destination(tmp_path: Path) -> None:
    model = _copy_model(tmp_path / "model")

    artifacts = package_model(
        model,
        tmp_path / "dist",
        model_name="release",
        include_dict=False,
    )

    assert artifacts["mdef"].read_bytes() == (model / "mdef").read_bytes()
    assert artifacts["manifest"].is_file()


def test_api_unnamed_package_refuses_unrecognized_generated_entry(tmp_path: Path) -> None:
    model = _copy_model(tmp_path / "model")
    output = tmp_path / "dist"
    acoustic = output / "acoustic"
    acoustic.mkdir(parents=True)
    unrelated = acoustic / "keep.txt"
    unrelated.write_text("not a package\n")

    with pytest.raises(ValueError, match="not a recognizable pstrain package"):
        package_model(model, output, include_dict=False, overwrite=True)

    assert unrelated.read_text() == "not a package\n"


def test_api_unnamed_package_preserves_unmanaged_output_entries(tmp_path: Path) -> None:
    model = _copy_model(tmp_path / "model")
    output = tmp_path / "dist"
    output.mkdir()
    unrelated = output / "keep.txt"
    unrelated.write_text("unmanaged\n")

    package_model(model, output, include_dict=False)

    assert unrelated.read_text() == "unmanaged\n"
    assert (output / "acoustic" / "mdef").is_file()
