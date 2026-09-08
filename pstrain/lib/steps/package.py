"""Model packaging for distribution.

Creates distributable model packages compatible with PocketSphinx,
Sphinx3, and other Sphinx-based decoders.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import tempfile
from pathlib import Path

from pstrain.lib.model import MODEL_FILES_REQUIRED, require_complete_model

logger = logging.getLogger(__name__)

__all__ = ["package_model", "create_noisedict", "validate_package_destination"]

PACKAGE_MANIFEST_NAME = "pstrain-package.json"
PACKAGE_FORMAT_VERSION = 1


def create_noisedict(
    output_path: Path,
    filler_dict_path: Path | None = None,
) -> Path:
    """Create noisedict file for Sphinx decoders.

    This is the filler dictionary used during decoding.

    Args:
        output_path: Output file path
        filler_dict_path: Source filler dictionary (optional)

    Returns:
        Path to created file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if filler_dict_path and Path(filler_dict_path).exists():
        # Copy existing filler dict
        shutil.copy(filler_dict_path, output_path)
    else:
        # Create minimal noisedict (matches pstrain/data/filler.dict)
        with output_path.open("w") as f:
            f.write("<sil> SIL\n")
            f.write("<s> SIL\n")
            f.write("</s> SIL\n")

    logger.info("Created noisedict: %s", output_path)
    return output_path


def package_model(
    model_dir: Path,
    output_dir: Path,
    model_name: str | None = None,
    dictionary_path: Path | None = None,
    filler_dict_path: Path | None = None,
    include_dict: bool = True,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Package a trained model for distribution.

    Creates a complete, self-contained model directory that can be
    used directly with PocketSphinx and other Sphinx decoders.

    Args:
        model_dir: Source model directory
        output_dir: Output directory for packaged model
        model_name: Name for the model (used in output path)
        dictionary_path: Path to pronunciation dictionary
        filler_dict_path: Path to filler dictionary
        include_dict: Whether to include dictionary in package
        overwrite: Allow replacement of a recognizable package without a marker

    Returns:
        Dict mapping file types to output paths

    Notes:
        A named package is fully built before its destination is changed, so its
        path holds the old package, the new package, or nothing, never a partially
        copied package. Replacing a populated directory requires moving the old one
        aside first. A supported package marker permits replacement by default; a
        recognizable legacy package without a marker requires ``overwrite=True``.
        Unrecognized destinations and invalid or unsupported markers are never
        replaced. If the process stops between the renames, the old package can be
        recovered from a sibling named ``.<name>-old-*``. With no model name,
        ``acoustic``, ``dict``, ``README.txt``, and ``pstrain-package.json`` are
        replaced as a transaction: success installs all new paths, and a handled
        failure restores all old paths, never a mixed or partially copied result.
        Individual paths can be absent during the sequence of renames. Unrelated
        entries in ``output_dir`` are preserved.

    Example output structure::

        dist/models/my-model/
        ├── acoustic/
        │   ├── feat.params
        │   ├── mdef
        │   ├── means
        │   ├── variances
        │   ├── mixture_weights
        │   ├── transition_matrices
        │   └── noisedict
        ├── dict/
        │   ├── cmudict.dict
        │   └── filler.dict
        └── README.txt
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    package_dir = validate_package_destination(
        model_dir,
        output_dir,
        model_name,
        include_dict=include_dict,
        overwrite=overwrite,
    )
    source_feat_params = require_complete_model(model_dir)

    staging_parent = package_dir.parent if model_name is not None else package_dir
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{package_dir.name}-", dir=staging_parent))

    try:
        result = _build_package(
            model_dir=model_dir,
            package_dir=package_dir,
            staging_dir=staging_dir,
            source_feat_params=source_feat_params,
            model_name=model_name,
            dictionary_path=dictionary_path,
            filler_dict_path=filler_dict_path,
            include_dict=include_dict,
        )
        # Recheck after staging so a destination introduced or changed while the
        # package was being built cannot bypass the replacement policy.
        validate_package_destination(
            model_dir,
            output_dir,
            model_name,
            include_dict=include_dict,
            overwrite=overwrite,
        )
        if model_name is not None:
            _replace_paths([(staging_dir, package_dir)])
        else:
            generated_names = ["acoustic", "README.txt"]
            if include_dict:
                generated_names.insert(1, "dict")
            generated_names.append(PACKAGE_MANIFEST_NAME)
            _replace_paths([(staging_dir / name, package_dir / name) for name in generated_names])
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    logger.info("Packaged model to: %s", package_dir)
    return result


def validate_package_destination(
    model_dir: Path,
    output_dir: Path,
    model_name: str | None = None,
    *,
    include_dict: bool = True,
    overwrite: bool = False,
) -> Path:
    """Validate a package destination without changing the filesystem.

    Named packages replace their complete destination. Unnamed packages replace
    only their generated entries and preserve unrelated entries in ``output_dir``.
    Existing generated entries must belong to a recognizable package in either
    case. A supported package marker establishes ownership; replacing a legacy
    package without one requires explicit opt-in.

    Args:
        model_dir: Source model directory
        output_dir: Output directory for packaged model
        model_name: Name for the model, as one ordinary path component
        include_dict: Whether packaging will replace the generated dictionary directory
        overwrite: Allow replacement of a recognizable package without a marker

    Returns:
        The package directory that packaging will write

    Raises:
        ValueError: If the name, source relationship, or existing destination is unsafe
    """
    model_dir = Path(model_dir).resolve()
    output_dir = Path(output_dir)
    if model_name is None:
        package_dir = output_dir
    else:
        package_dir = output_dir / model_name
        destination = package_dir.resolve()
        resolved_output = output_dir.resolve()
        if (
            not model_name
            or model_name in {".", ".."}
            or len(Path(model_name).parts) != 1
            or not destination.is_relative_to(resolved_output)
        ):
            raise ValueError(
                f"Invalid package destination {destination}: package name {model_name!r} "
                "must be exactly one ordinary path component."
            )

    destination = package_dir.resolve()
    if _paths_overlap(model_dir, destination):
        raise ValueError(
            f"Package destination {destination} overlaps source model {model_dir}; "
            "choose a separate output directory and package name."
        )

    if model_name is None:
        destination_exists = any(
            (package_dir / name).exists() or (package_dir / name).is_symlink()
            for name in _generated_names(include_dict)
        )
    else:
        destination_exists = package_dir.exists() or package_dir.is_symlink()
    if not destination_exists:
        return package_dir

    if package_dir.is_symlink() or not _has_package_structure(package_dir):
        raise ValueError(
            f"Refusing to replace {package_dir}: existing destination "
            "is not a recognizable pstrain package."
        )
    marker_status = _package_marker_status(package_dir)
    if marker_status == "absent" and not overwrite:
        raise ValueError(
            f"Refusing to replace {package_dir}: existing package has no pstrain package "
            "marker. Use --overwrite to replace the entire existing directory."
        )
    if marker_status == "invalid":
        raise ValueError(
            f"Refusing to replace {package_dir}: existing pstrain package marker is invalid "
            "or uses an unsupported format version. An overwrite replaces the entire "
            "existing directory, so the marker must be understood before replacement."
        )
    return package_dir


def _generated_names(include_dict: bool) -> tuple[str, ...]:
    """Return the output entries replaced by unnamed packaging."""
    names = ["acoustic", "README.txt", PACKAGE_MANIFEST_NAME]
    if include_dict:
        names.append("dict")
    return tuple(names)


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either resolved path contains the other."""
    return (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
        or _same_as_existing_ancestor(first, second)
        or _same_as_existing_ancestor(second, first)
    )


def _same_as_existing_ancestor(path: Path, other: Path) -> bool:
    """Compare one existing path with every existing ancestor of another."""
    if not path.exists():
        return False
    for ancestor in (other, *other.parents):
        if ancestor.exists() and path.samefile(ancestor):
            return True
    return False


def _has_package_structure(path: Path) -> bool:
    """Return whether a destination has the historical package structure."""
    if not path.is_dir() or not (path / "README.txt").is_file():
        return False
    acoustic = path / "acoustic"
    required = (*MODEL_FILES_REQUIRED, "feat.params", "noisedict")
    return acoustic.is_dir() and all((acoustic / name).is_file() for name in required)


def _package_marker_status(path: Path) -> str:
    """Classify a package marker as supported, absent, or invalid."""
    marker_path = path / PACKAGE_MANIFEST_NAME
    try:
        marker_mode = marker_path.lstat().st_mode
    except FileNotFoundError:
        return "absent"
    except OSError:
        return "invalid"
    if not stat.S_ISREG(marker_mode):
        return "invalid"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "invalid"
    supported = (
        isinstance(marker, dict)
        and type(marker.get("format_version")) is int
        and marker["format_version"] == PACKAGE_FORMAT_VERSION
        and marker.get("generator") == "pstrain"
    )
    return "supported" if supported else "invalid"


def _build_package(
    *,
    model_dir: Path,
    package_dir: Path,
    staging_dir: Path,
    source_feat_params: Path,
    model_name: str | None,
    dictionary_path: Path | None,
    filler_dict_path: Path | None,
    include_dict: bool,
) -> dict[str, Path]:
    """Build a complete package in a private staging directory."""
    acoustic_dir = staging_dir / "acoustic"
    acoustic_dir.mkdir()
    final_acoustic_dir = package_dir / "acoustic"

    result: dict[str, Path] = {}

    # Copy acoustic model files
    for fname in MODEL_FILES_REQUIRED:
        src = model_dir / fname
        dst = acoustic_dir / fname
        shutil.copy2(src, dst)
        result[fname] = final_acoustic_dir / fname
        logger.debug("Copied %s -> %s", src, dst)

    feat_path = acoustic_dir / "feat.params"
    shutil.copyfile(source_feat_params, feat_path)
    result["feat_params"] = final_acoustic_dir / "feat.params"

    # Create noisedict
    noisedict_path = create_noisedict(
        acoustic_dir / "noisedict",
        filler_dict_path,
    )
    result["noisedict"] = final_acoustic_dir / noisedict_path.name

    # Copy dictionary files if requested
    if include_dict:
        dict_dir = staging_dir / "dict"
        dict_dir.mkdir()
        final_dict_dir = package_dir / "dict"

        if dictionary_path and Path(dictionary_path).exists():
            dict_dst = dict_dir / "cmudict.dict"
            shutil.copy2(dictionary_path, dict_dst)
            result["dictionary"] = final_dict_dir / dict_dst.name
            logger.debug("Copied dictionary: %s", dict_dst)

        if filler_dict_path and Path(filler_dict_path).exists():
            filler_dst = dict_dir / "filler.dict"
            shutil.copy2(filler_dict_path, filler_dst)
            result["filler_dict"] = final_dict_dir / filler_dst.name
            logger.debug("Copied filler dict: %s", filler_dst)

    manifest_path = staging_dir / PACKAGE_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(
            {"format_version": PACKAGE_FORMAT_VERSION, "generator": "pstrain"},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result["manifest"] = package_dir / PACKAGE_MANIFEST_NAME

    # Create README
    readme_path = staging_dir / "README.txt"
    _create_readme(
        readme_path,
        model_name,
        include_dictionary=include_dict and bool(dictionary_path and dictionary_path.exists()),
        include_filler=include_dict and bool(filler_dict_path and filler_dict_path.exists()),
    )
    result["readme"] = package_dir / readme_path.name

    return result


def _replace_paths(paths: list[tuple[Path, Path]]) -> None:
    """Install staged paths as a transaction, retaining old paths for rollback."""
    backups: dict[Path, Path | None] = {}
    try:
        for _, destination in paths:
            backup_path: Path | None = None
            if destination.exists():
                backup_path = Path(
                    tempfile.mkdtemp(prefix=f".{destination.name}-old-", dir=destination.parent)
                )
                backup_path.rmdir()
                os.replace(destination, backup_path)  # noqa: PTH105
            backups[destination] = backup_path
    except BaseException as install_error:
        _rollback_paths(paths, backups, set(), install_error)
        raise

    installed: set[Path] = set()
    try:
        for staging_path, destination in paths:
            os.replace(staging_path, destination)  # noqa: PTH105
            installed.add(destination)
    except BaseException as install_error:
        _rollback_paths(paths, backups, installed, install_error)
        raise

    for backup_path in backups.values():
        if backup_path is not None:
            _remove_backup(backup_path)


def _rollback_paths(
    paths: list[tuple[Path, Path]],
    backups: dict[Path, Path | None],
    installed: set[Path],
    install_error: BaseException,
) -> None:
    """Restore every retained path, reporting any backup that remains."""
    restore_failure: tuple[Path, BaseException] | None = None
    for _, destination in reversed(paths):
        if destination not in backups:
            continue
        backup_path = backups[destination]
        try:
            if destination in installed:
                _remove_path(destination)
            if backup_path is not None:
                os.replace(backup_path, destination)  # noqa: PTH105
        except BaseException as error:
            if restore_failure is None:
                restore_failure = (backup_path or destination, error)

    if restore_failure is not None:
        recovery_path, restore_error = restore_failure
        raise RuntimeError(
            f"could not restore previous package from {recovery_path}: {restore_error}"
        ) from install_error


def _remove_path(path: Path) -> None:
    """Remove a generated file or directory during rollback."""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _remove_backup(backup_path: Path) -> None:
    """Remove an obsolete backup without failing a completed install."""
    try:
        _remove_path(backup_path)
    except OSError as error:
        logger.warning("Could not remove old package at %s: %s", backup_path, error)


def _create_readme(
    output_path: Path,
    model_name: str | None,
    *,
    include_dictionary: bool,
    include_filler: bool,
) -> None:
    """Create README file for the model package."""
    model_path = model_name or "model"
    dictionary_structure = ""
    dictionary_path = "/path/to/dictionary.dict"
    if include_dictionary or include_filler:
        dictionary_structure = "\ndict/           - Dictionary files included in this package\n"
        if include_dictionary:
            dictionary_structure += "  cmudict.dict  - Pronunciation dictionary\n"
            dictionary_path = f"{model_path}/dict/cmudict.dict"
        if include_filler:
            dictionary_structure += "  filler.dict   - Filler word dictionary\n"
    dictionary_note = ""
    if not include_dictionary:
        dictionary_note = (
            "\nA pronunciation dictionary is not included; supply one when decoding.\n"
        )
    content = f"""pstrain Acoustic Model Package
==========================

Model: {model_name or "unnamed"}
Generator: pstrain (SphinxTrain 2)

Directory Structure
-------------------
pstrain-package.json - pstrain package marker and format version
acoustic/       - Acoustic model files for Sphinx decoders
  feat.params   - Feature extraction parameters
  mdef          - Model definition (phones, states, triphones)
  means         - Gaussian means
  variances     - Gaussian variances
  mixture_weights - Raw mixture occupancy accumulators (normalized on load)
  transition_matrices - Raw HMM transition accumulators (normalized on load)
  noisedict     - Filler/noise dictionary for decoding

{dictionary_structure}{dictionary_note}

Usage with PocketSphinx
-----------------------
Python:
    from pocketsphinx import Decoder

    config = Decoder.default_config()
    config.set_string('-hmm', '/path/to/{model_path}/acoustic')
    config.set_string('-dict', '{dictionary_path}')
    decoder = Decoder(config)

Command line:
    pocketsphinx -hmm {model_path}/acoustic \\
                 -dict {dictionary_path} \\
                 -infile audio.wav

Feature Parameters
------------------
See acoustic/feat.params (copied verbatim from the trained model).

License
-------
See the project repository for license information.
"""
    with output_path.open("w") as f:
        f.write(content)
