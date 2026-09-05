"""Model packaging for distribution.

Creates distributable model packages compatible with PocketSphinx,
Sphinx3, and other Sphinx-based decoders.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

from pstrain.lib.model import MODEL_FILES_REQUIRED, require_complete_model

logger = logging.getLogger(__name__)

__all__ = ["package_model", "create_noisedict"]


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

    Returns:
        Dict mapping file types to output paths

    Notes:
        A named package is fully built before its destination is changed, so its
        path holds the old package, the new package, or nothing, never a partially
        copied package. Replacing a populated directory requires moving the old one
        aside first. If the process stops between those renames, the old package can
        be recovered from a sibling named ``.<name>-old-*``. With no model name,
        ``acoustic``, ``dict``, and ``README.txt`` are replaced as a transaction:
        success installs all new paths, and a handled failure restores all old paths,
        never a mixed or partially copied result. Individual paths can be absent
        during the sequence of renames. Unrelated entries in ``output_dir`` are
        preserved.

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
    source_feat_params = require_complete_model(model_dir)

    package_dir = output_dir / model_name if model_name else output_dir
    staging_parent = package_dir.parent if model_name else package_dir
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
        if model_name:
            _replace_paths([(staging_dir, package_dir)])
        else:
            generated_names = ["acoustic", "README.txt"]
            if include_dict:
                generated_names.insert(1, "dict")
            _replace_paths([(staging_dir / name, package_dir / name) for name in generated_names])
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    logger.info("Packaged model to: %s", package_dir)
    return result


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

    # Create README
    readme_path = staging_dir / "README.txt"
    _create_readme(readme_path, model_name)
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
) -> None:
    """Create README file for the model package."""
    content = f"""pstrain Acoustic Model Package
==========================

Model: {model_name or "unnamed"}
Generator: pstrain (SphinxTrain 2)

Directory Structure
-------------------
acoustic/       - Acoustic model files for Sphinx decoders
  feat.params   - Feature extraction parameters
  mdef          - Model definition (phones, states, triphones)
  means         - Gaussian means
  variances     - Gaussian variances
  mixture_weights - Raw mixture occupancy accumulators (normalized on load)
  transition_matrices - Raw HMM transition accumulators (normalized on load)
  noisedict     - Filler/noise dictionary for decoding

dict/           - Dictionary files
  cmudict.dict  - Pronunciation dictionary
  filler.dict   - Filler word dictionary

Usage with PocketSphinx
-----------------------
Python:
    from pocketsphinx import Decoder

    config = Decoder.default_config()
    config.set_string('-hmm', '/path/to/{model_name or "model"}/acoustic')
    config.set_string('-dict', '/path/to/{model_name or "model"}/dict/cmudict.dict')
    decoder = Decoder(config)

Command line:
    pocketsphinx -hmm {model_name or "model"}/acoustic \\
                 -dict {model_name or "model"}/dict/cmudict.dict \\
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
