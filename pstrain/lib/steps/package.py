"""Model packaging for distribution.

Creates distributable model packages compatible with PocketSphinx,
Sphinx3, and other Sphinx-based decoders.
"""

from __future__ import annotations

import ctypes
import errno
import json
import logging
import os
import secrets
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from pstrain.lib.model import MODEL_FILES_REQUIRED, require_complete_model

logger = logging.getLogger(__name__)

__all__ = ["package_model", "create_noisedict", "validate_package_destination"]

PACKAGE_MANIFEST_NAME = "pstrain-package.json"
PACKAGE_FORMAT_VERSION = 1
_Identity = tuple[int, int]


@dataclass
class _OpenEntry:
    """One filesystem object held open across every transaction step."""

    fd: int
    identity: _Identity
    is_directory: bool

    def close(self) -> None:
        os.close(self.fd)


@dataclass
class _OwnedDirectory:
    """A private transaction directory with one cleanup owner."""

    parent_fd: int
    name: str
    path: Path
    entry: _OpenEntry
    cleaned: bool = False

    def close(self) -> None:
        self.entry.close()


@dataclass
class _StagingDirectory:
    """A newly created staging path whose descriptor owner is attached after opening."""

    path: Path
    owner: _OwnedDirectory | None = None


@contextmanager
def _staging_directory(parent: Path, prefix: str) -> Iterator[_StagingDirectory]:
    """Own a staging directory from creation through descriptor-relative cleanup."""
    path: Path | None = None
    staging: _StagingDirectory | None = None
    try:
        path = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
        staging = _StagingDirectory(path)
        yield staging
    finally:
        if staging is not None and staging.owner is not None:
            _cleanup_owned_directory(staging.owner, strict=False, missing_ok=True)
        elif path is not None:
            shutil.rmtree(path, ignore_errors=True)


@contextmanager
def _owned_descriptor(open_descriptor: Callable[[], int]) -> Iterator[int]:
    """Put a descriptor under cleanup before its acquiring call can return."""
    descriptor: int | None = None
    try:
        descriptor = open_descriptor()
        yield descriptor
    finally:
        if descriptor is not None:
            os.close(descriptor)


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
        A supported package marker permits replacement by default; a recognizable
        legacy package without a marker requires ``overwrite=True``. Unrecognized
        destinations and invalid or unsupported markers are never replaced.

        On macOS and Linux, replacement opens the source, destination parent,
        staging directory, retained package, and recovery directory without
        following their final names. Identities come from those descriptors;
        validation and traversal remain descriptor-relative; and every rename is
        reconciled from the filesystem even when its call raises. Windows retains
        the path-based transaction and makes no guarantee against an active process
        substituting names during packaging. An asynchronous interruption can also
        leave a mixed unnamed package on Windows because its path transaction cannot
        reconcile a rename that completed before raising. No supported platform
        promises safety against every active same-filesystem race because final
        directory-entry deletion has no portable conditional-by-descriptor
        primitive. See ``docs/package-safety.md`` for the exact guarantee, recovery
        instructions, and remaining seams.

        With no model name, ``acoustic``, ``dict``, ``README.txt``, and
        ``pstrain-package.json`` transition separately. On a handled failure the
        implementation reconciles each open identity and attempts to restore the
        previous public set. Individual paths can be absent during that recovery;
        failures are attached to the initiating exception and recovery directories
        are retained when certainty is lost. Unrelated entries in ``output_dir``
        are preserved.

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
    with (
        ExitStack() as resources,
        _staging_directory(staging_parent, f".{package_dir.name}-") as staging_scope,
    ):
        staging_dir = staging_scope.path
        parent_entry: _OpenEntry | None = None
        model_entry: _OpenEntry | None = None
        if _descriptor_transactions_available():
            parent_entry = _open_directory_path(staging_parent, owner=resources)
            staging_entry = _open_child(parent_entry.fd, staging_dir.name, owner=resources)
            if staging_entry is None or not staging_entry.is_directory:
                raise RuntimeError(f"staging directory {staging_dir} changed during creation")
            staging_scope.owner = _OwnedDirectory(
                parent_entry.fd,
                staging_dir.name,
                staging_dir,
                staging_entry,
            )
            model_entry = _open_directory_path(model_dir.resolve(), owner=resources)

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
        if model_name is not None:
            _replace_named_package(
                staging_dir,
                package_dir,
                model_dir=model_dir,
                overwrite=overwrite,
                parent_entry=parent_entry,
                staging_entry=(staging_scope.owner.entry if staging_scope.owner else None),
                model_entry=model_entry,
            )
        else:
            generated_names = ["acoustic", "README.txt"]
            if include_dict:
                generated_names.insert(1, "dict")
            generated_names.append(PACKAGE_MANIFEST_NAME)
            _replace_unnamed_package(
                staging_dir,
                package_dir,
                generated_names,
                model_dir=model_dir,
                overwrite=overwrite,
                parent_entry=parent_entry,
                staging_entry=(staging_scope.owner.entry if staging_scope.owner else None),
                model_entry=model_entry,
            )

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

    _validate_existing_package(
        package_dir,
        display_path=package_dir,
        model_dir=model_dir,
        overwrite=overwrite,
    )
    return package_dir


def _validate_existing_package(
    path: Path,
    *,
    display_path: Path,
    model_dir: Path,
    overwrite: bool,
) -> None:
    """Validate the ownership marker and structure of one existing package object."""
    if path.is_symlink() or not _has_package_structure(path):
        raise ValueError(
            f"Refusing to replace {display_path}: existing destination "
            "is not a recognizable pstrain package."
        )
    marker_status = _package_marker_status(path)
    if marker_status == "absent" and not overwrite:
        raise ValueError(
            f"Refusing to replace {display_path}: existing package has no pstrain package "
            "marker. Use --overwrite to replace the entire existing directory."
        )
    if marker_status == "invalid":
        raise ValueError(
            f"Refusing to replace {display_path}: existing pstrain package marker is invalid "
            "or uses an unsupported format version. An overwrite replaces the entire "
            "existing directory, so the marker must be understood before replacement."
        )
    if _directory_tree_contains(path, model_dir, display_path=display_path):
        raise ValueError(
            f"Package destination {display_path.resolve()} overlaps source model "
            f"{model_dir.resolve()}; choose a separate output directory and package name."
        )


def _directory_tree_contains(root: Path, target: Path, *, display_path: Path) -> bool:
    """Return whether a directory in *root* has the target's filesystem identity."""
    try:
        target_metadata = target.stat()
    except OSError as error:
        raise ValueError(
            f"Cannot safely inspect {display_path} for source overlap: {error}"
        ) from error
    target_identity = (target_metadata.st_dev, target_metadata.st_ino)
    pending = [root]
    visited: set[tuple[int, int]] = set()

    while pending:
        directory = pending.pop()
        try:
            metadata = directory.stat()
            identity = (metadata.st_dev, metadata.st_ino)
            if identity == target_identity:
                return True
            if identity in visited:
                continue
            visited.add(identity)
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        try:
                            entry_metadata = entry.stat(follow_symlinks=True)
                        except FileNotFoundError:
                            continue
                        except OSError as error:
                            raise ValueError(
                                f"Cannot safely inspect {display_path} for source overlap: {error}"
                            ) from error
                        if (
                            stat.S_ISDIR(entry_metadata.st_mode)
                            and (
                                entry_metadata.st_dev,
                                entry_metadata.st_ino,
                            )
                            == target_identity
                        ):
                            return True
                    elif entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
        except ValueError:
            raise
        except OSError as error:
            raise ValueError(
                f"Cannot safely inspect {display_path} for source overlap: {error}"
            ) from error
    return False


def _validate_existing_package_fd(
    package_fd: int,
    *,
    display_path: Path,
    model_identity: _Identity,
    overwrite: bool,
) -> None:
    """Validate a retained package entirely through its already-open descriptor."""
    if not _has_package_structure_fd(package_fd):
        raise ValueError(
            f"Refusing to replace {display_path}: existing destination "
            "is not a recognizable pstrain package."
        )
    marker_status = _package_marker_status_fd(package_fd)
    if marker_status == "absent" and not overwrite:
        raise ValueError(
            f"Refusing to replace {display_path}: existing package has no pstrain package "
            "marker. Use --overwrite to replace the entire existing directory."
        )
    if marker_status == "invalid":
        raise ValueError(
            f"Refusing to replace {display_path}: existing pstrain package marker is invalid "
            "or uses an unsupported format version. An overwrite replaces the entire "
            "existing directory, so the marker must be understood before replacement."
        )
    if _directory_tree_contains_fd(package_fd, model_identity, display_path=display_path):
        raise ValueError(
            f"Package destination {display_path.resolve()} overlaps source model; "
            "choose a separate output directory and package name."
        )


def _regular_child_exists(parent_fd: int, name: str) -> bool:
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode)


def _has_package_structure_fd(package_fd: int) -> bool:
    """Recognize historical package structure without reopening the package root."""
    if not _regular_child_exists(package_fd, "README.txt"):
        return False
    with ExitStack() as resources:
        acoustic = _open_child(package_fd, "acoustic", owner=resources)
        if acoustic is None:
            return False
        required = (*MODEL_FILES_REQUIRED, "feat.params", "noisedict")
        return acoustic.is_directory and all(
            _regular_child_exists(acoustic.fd, name) for name in required
        )


def _package_marker_status_fd(package_fd: int) -> str:
    """Read a marker relative to an open package without following substitutions."""
    with ExitStack() as resources:
        marker = _open_child(package_fd, PACKAGE_MANIFEST_NAME, owner=resources)
        if marker is None:
            return "absent"
        try:
            metadata = os.fstat(marker.fd)
            if not stat.S_ISREG(metadata.st_mode):
                return "invalid"
            marker_copy = resources.enter_context(_owned_descriptor(lambda: os.dup(marker.fd)))
            with os.fdopen(marker_copy, closefd=False, encoding="utf-8") as marker_file:
                document = json.load(marker_file)
        except (OSError, UnicodeError, json.JSONDecodeError):
            return "invalid"
    supported = (
        isinstance(document, dict)
        and type(document.get("format_version")) is int
        and document["format_version"] == PACKAGE_FORMAT_VERSION
        and document.get("generator") == "pstrain"
    )
    return "supported" if supported else "invalid"


def _directory_tree_contains_fd(
    root_fd: int,
    target_identity: _Identity,
    *,
    display_path: Path,
) -> bool:
    """Walk directory identities through anchored descriptors, failing closed on change."""
    with ExitStack() as resources:
        root_copy = resources.enter_context(_owned_descriptor(lambda: os.dup(root_fd)))
        pending = [root_copy]
        visited: set[_Identity] = set()
        try:
            while pending:
                directory_fd = pending.pop()
                identity = _metadata_identity(os.fstat(directory_fd))
                if identity == target_identity:
                    return True
                if identity in visited:
                    continue
                visited.add(identity)
                with os.scandir(directory_fd) as entries:
                    names = [entry.name for entry in entries]
                for name in names:
                    metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if stat.S_ISLNK(metadata.st_mode):
                        followed = os.stat(name, dir_fd=directory_fd, follow_symlinks=True)
                        if (
                            stat.S_ISDIR(followed.st_mode)
                            and _metadata_identity(followed) == target_identity
                        ):
                            return True
                    elif stat.S_ISDIR(metadata.st_mode):
                        child_fd = resources.enter_context(
                            _owned_descriptor(
                                partial(
                                    os.open,
                                    name,
                                    _directory_open_flags(),
                                    dir_fd=directory_fd,
                                )
                            )
                        )
                        child_identity = _metadata_identity(os.fstat(child_fd))
                        if child_identity != _metadata_identity(metadata):
                            raise RuntimeError(
                                f"directory entry {name!r} changed during overlap scan"
                            )
                        pending.append(child_fd)
        except BaseException as error:
            if isinstance(error, ValueError):
                raise
            raise ValueError(
                f"Cannot safely inspect {display_path} for source overlap: {error}"
            ) from error
    return False


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


def _descriptor_transactions_available() -> bool:
    """Return whether this platform exposes the required anchored primitives."""
    return (
        os.name == "posix"
        and (sys.platform == "darwin" or sys.platform.startswith("linux"))
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.scandir in os.supports_fd
        and os.unlink in os.supports_dir_fd
        and os.rmdir in os.supports_dir_fd
    )


def _directory_open_flags() -> int:
    """Flags for opening a directory endpoint without following its final name."""
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _entry_open_flags() -> int:
    """Flags for opening a package entry without blocking on special files."""
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _metadata_identity(metadata: os.stat_result) -> _Identity:
    return metadata.st_dev, metadata.st_ino


def _open_directory_path(path: Path, *, owner: ExitStack | None = None) -> _OpenEntry:
    """Open one directory path and derive its identity only from that descriptor."""
    if owner is None:
        fd = os.open(path, _directory_open_flags())
    else:
        fd = owner.enter_context(_owned_descriptor(lambda: os.open(path, _directory_open_flags())))
    try:
        metadata = os.fstat(fd)
        return _OpenEntry(fd, _metadata_identity(metadata), True)
    except BaseException:
        if owner is None:
            os.close(fd)
        raise


def _open_child(
    parent_fd: int,
    name: str,
    *,
    owner: ExitStack | None = None,
) -> _OpenEntry | None:
    """Open one anchored child, rejecting symlinks and substitutions during open."""
    try:
        if owner is None:
            fd = os.open(name, _entry_open_flags(), dir_fd=parent_fd)
        else:
            fd = owner.enter_context(
                _owned_descriptor(lambda: os.open(name, _entry_open_flags(), dir_fd=parent_fd))
            )
    except FileNotFoundError:
        return None
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError(f"Refusing to act on symbolic link {name!r}") from error
        raise

    try:
        descriptor_metadata = os.fstat(fd)
        name_metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor_identity = _metadata_identity(descriptor_metadata)
        if _metadata_identity(name_metadata) != descriptor_identity:
            raise RuntimeError(f"filesystem entry {name!r} changed while it was opened")
        return _OpenEntry(fd, descriptor_identity, stat.S_ISDIR(descriptor_metadata.st_mode))
    except BaseException:
        if owner is None:
            os.close(fd)
        raise


def _entry_matches(parent_fd: int, name: str, identity: _Identity) -> bool:
    """Return whether an anchored name currently carries an expected identity."""
    return _entry_identity_at(parent_fd, name) == identity


def _entry_identity_at(parent_fd: int, name: str) -> _Identity | None:
    """Return the current identity at an anchored name, or None when absent."""
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return _metadata_identity(metadata)


def _create_owned_directory(
    parent_fd: int,
    parent_path: Path,
    prefix: str,
    *,
    owner: ExitStack | None = None,
) -> _OwnedDirectory:
    """Create and open a private directory, rejecting a creation/open substitution."""
    for _attempt in range(100):
        name = f"{prefix}{secrets.token_hex(8)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        try:
            entry = _open_child(parent_fd, name, owner=owner)
        except BaseException as error:
            try:
                os.rmdir(name, dir_fd=parent_fd)
            except BaseException as cleanup_error:
                _add_recovery_note(
                    error,
                    f"could not remove unowned transaction directory {parent_path / name}",
                    cleanup_error,
                )
            raise
        if entry is None or not entry.is_directory:
            if entry is not None and owner is None:
                entry.close()
            raise RuntimeError(f"private transaction directory {parent_path / name} changed")
        return _OwnedDirectory(parent_fd, name, parent_path / name, entry)
    raise FileExistsError("could not reserve a private package transaction directory")


@dataclass(frozen=True)
class _MoveResult:
    moved: bool
    error: BaseException | None


def _move_reconciled(
    entry: _OpenEntry,
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
) -> _MoveResult:
    """Rename an open object and derive the outcome from the filesystem on every return."""
    operation_error: BaseException | None = None
    try:
        _rename_noreplace_at(
            source_parent_fd,
            source_name,
            destination_parent_fd,
            destination_name,
        )
    except BaseException as error:
        operation_error = error

    try:
        at_source = _entry_matches(source_parent_fd, source_name, entry.identity)
        at_destination = _entry_matches(destination_parent_fd, destination_name, entry.identity)
    except BaseException as reconciliation_error:
        if operation_error is not None:
            operation_error.add_note(f"could not reconcile rename outcome: {reconciliation_error}")
            return _MoveResult(False, operation_error)
        raise RuntimeError("could not reconcile package rename outcome") from reconciliation_error

    if at_destination and not at_source:
        moved = True
    elif at_source and not at_destination:
        moved = False
    else:
        detail = (
            f"expected identity {entry.identity} is at both rename endpoints"
            if at_source
            else f"expected identity {entry.identity} is at neither rename endpoint"
        )
        outcome_error = RuntimeError(f"could not safely reconcile package rename: {detail}")
        if operation_error is not None:
            operation_error.add_note(str(outcome_error))
            return _MoveResult(False, operation_error)
        raise outcome_error

    if operation_error is None and not moved:
        raise RuntimeError("exclusive package rename returned success without moving its source")
    return _MoveResult(moved, operation_error)


def _move_or_raise(
    entry: _OpenEntry,
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
) -> None:
    result = _move_reconciled(
        entry,
        source_parent_fd,
        source_name,
        destination_parent_fd,
        destination_name,
    )
    if result.error is not None:
        raise result.error


def _add_recovery_note(error: BaseException, message: str, failure: BaseException) -> None:
    """Keep the initiating failure primary while reporting a recovery failure."""
    error.add_note(f"{message}: {type(failure).__name__}: {failure}")


def _restore_open_entry(
    entry: _OpenEntry,
    backup_fd: int,
    backup_name: str,
    public_fd: int,
    public_name: str,
    error: BaseException,
) -> bool:
    """Restore an entry if reconciliation finds it retained, without masking *error*."""
    try:
        at_backup = _entry_matches(backup_fd, backup_name, entry.identity)
        at_public = _entry_matches(public_fd, public_name, entry.identity)
        if at_public and not at_backup:
            return True
        if at_backup and not at_public:
            result = _move_reconciled(entry, backup_fd, backup_name, public_fd, public_name)
            if result.error is not None:
                raise result.error
            return True
        location = "both endpoints" if at_backup else "neither endpoint"
        raise RuntimeError(f"retained identity is at {location}")
    except BaseException as recovery_error:
        _add_recovery_note(
            error, f"could not restore retained package entry {public_name!r}", recovery_error
        )
        return False


def _replace_named_package(
    staging_dir: Path,
    package_dir: Path,
    *,
    model_dir: Path,
    overwrite: bool,
    parent_entry: _OpenEntry | None,
    staging_entry: _OpenEntry | None,
    model_entry: _OpenEntry | None,
) -> None:
    """Replace a named package with descriptor anchoring where the platform supports it."""
    if not _descriptor_transactions_available():
        _replace_named_package_by_path(
            staging_dir,
            package_dir,
            model_dir=model_dir,
            overwrite=overwrite,
        )
        return
    if parent_entry is None or staging_entry is None or model_entry is None:
        raise RuntimeError("descriptor-relative package transaction was not anchored")
    _replace_named_package_by_descriptor(
        staging_dir,
        package_dir,
        overwrite=overwrite,
        parent=parent_entry,
        staging=staging_entry,
        model=model_entry,
    )


def _replace_named_package_by_descriptor(
    staging_dir: Path,
    package_dir: Path,
    *,
    overwrite: bool,
    parent: _OpenEntry,
    staging: _OpenEntry,
    model: _OpenEntry,
) -> None:
    """Retain, validate, install, and clean a named package through open directories."""
    with ExitStack() as transaction:
        backup: _OwnedDirectory | None = None
        retained: _OpenEntry | None = None
        success = False
        try:
            if not _entry_matches(parent.fd, staging_dir.name, staging.identity):
                raise RuntimeError(f"staging directory {staging_dir} changed before publication")
            backup = _create_owned_directory(
                parent.fd,
                package_dir.parent,
                f".{package_dir.name}-old-",
                owner=transaction,
            )
            retained = _open_child(parent.fd, package_dir.name, owner=transaction)
            if retained is not None:
                try:
                    _move_or_raise(
                        retained, parent.fd, package_dir.name, backup.entry.fd, "retained"
                    )
                    _validate_existing_package_fd(
                        retained.fd,
                        display_path=package_dir,
                        model_identity=model.identity,
                        overwrite=overwrite,
                    )
                except BaseException as error:
                    _restore_open_entry(
                        retained,
                        backup.entry.fd,
                        "retained",
                        parent.fd,
                        package_dir.name,
                        error,
                    )
                    _cleanup_empty_owned_directory(backup, error)
                    raise

            try:
                _move_or_raise(staging, parent.fd, staging_dir.name, parent.fd, package_dir.name)
            except BaseException as error:
                _rollback_named_descriptor(
                    staging,
                    retained,
                    parent,
                    backup,
                    staging_dir.name,
                    package_dir.name,
                    error,
                )
                _cleanup_empty_owned_directory(backup, error)
                raise

            if retained is not None:
                _remove_open_entry(backup.entry.fd, "retained", retained, backup.path / "retained")
            _remove_empty_owned_directory(backup)
            success = True
        finally:
            if backup is not None and not success and not backup.cleaned:
                logger.warning("Preserved package transaction directory at %s", backup.path)


def _rollback_named_descriptor(
    staging: _OpenEntry,
    retained: _OpenEntry | None,
    parent: _OpenEntry,
    backup: _OwnedDirectory,
    staging_name: str,
    package_name: str,
    error: BaseException,
) -> None:
    """Reconcile a named install before restoring its retained identity."""
    quarantined = False
    try:
        at_public = _entry_matches(parent.fd, package_name, staging.identity)
        at_staging = _entry_matches(parent.fd, staging_name, staging.identity)
        if at_public and not at_staging:
            result = _move_reconciled(
                staging,
                parent.fd,
                package_name,
                backup.entry.fd,
                ".new",
            )
            if result.error is not None:
                raise result.error
            quarantined = True
        elif not at_staging:
            raise RuntimeError("staged package identity is at neither expected endpoint")
    except BaseException as recovery_error:
        _add_recovery_note(error, "could not quarantine installed named package", recovery_error)

    if retained is not None:
        _restore_open_entry(
            retained,
            backup.entry.fd,
            "retained",
            parent.fd,
            package_name,
            error,
        )

    if quarantined:
        try:
            _remove_open_entry(backup.entry.fd, ".new", staging, backup.path / ".new")
        except BaseException as recovery_error:
            _add_recovery_note(error, "could not clean installed named package", recovery_error)


def _replace_named_package_by_path(
    staging_dir: Path,
    package_dir: Path,
    *,
    model_dir: Path,
    overwrite: bool,
) -> None:
    """Path-based Windows fallback; active pathname racing is outside its guarantee."""
    backup_path = _reserve_absent_path(package_dir)
    retained_identity: tuple[int, int] | None = None
    try:
        _rename_noreplace(package_dir, backup_path)
    except FileNotFoundError:
        pass
    else:
        retained_identity = _path_identity(backup_path)
        try:
            _validate_existing_package(
                backup_path,
                display_path=package_dir,
                model_dir=model_dir,
                overwrite=overwrite,
            )
        except BaseException as validation_error:
            _restore_retained_path(
                backup_path,
                package_dir,
                retained_identity,
                validation_error,
            )
            raise

    try:
        _rename_noreplace(staging_dir, package_dir)
    except BaseException as install_error:
        if retained_identity is not None:
            _restore_retained_path(
                backup_path,
                package_dir,
                retained_identity,
                install_error,
            )
        raise

    if retained_identity is not None:
        _remove_backup(backup_path, retained_identity)


def _replace_unnamed_package(
    staging_dir: Path,
    package_dir: Path,
    generated_names: list[str],
    *,
    model_dir: Path,
    overwrite: bool,
    parent_entry: _OpenEntry | None,
    staging_entry: _OpenEntry | None,
    model_entry: _OpenEntry | None,
) -> None:
    """Replace unnamed package entries with descriptor anchoring where available."""
    if not _descriptor_transactions_available():
        _replace_unnamed_package_by_path(
            staging_dir,
            package_dir,
            generated_names,
            model_dir=model_dir,
            overwrite=overwrite,
        )
        return
    if parent_entry is None or staging_entry is None or model_entry is None:
        raise RuntimeError("descriptor-relative package transaction was not anchored")
    _replace_unnamed_package_by_descriptor(
        staging_dir,
        package_dir,
        generated_names,
        overwrite=overwrite,
        package=parent_entry,
        staging=staging_entry,
        model=model_entry,
    )


def _replace_unnamed_package_by_descriptor(
    staging_dir: Path,
    package_dir: Path,
    generated_names: list[str],
    *,
    overwrite: bool,
    package: _OpenEntry,
    staging: _OpenEntry,
    model: _OpenEntry,
) -> None:
    """Reconcile every unnamed-package move from its open identity on all returns."""
    with ExitStack() as transaction:
        backup: _OwnedDirectory | None = None
        retained: dict[str, _OpenEntry] = {}
        installed: dict[str, _OpenEntry] = {}
        success = False
        try:
            if not _entry_matches(package.fd, staging_dir.name, staging.identity):
                raise RuntimeError(f"staging directory {staging_dir} changed before publication")
            backup = _create_owned_directory(
                package.fd,
                package_dir,
                ".pstrain-package-old-",
                owner=transaction,
            )

            try:
                for name in generated_names:
                    entry = _open_child(package.fd, name, owner=transaction)
                    if entry is None:
                        continue
                    retained[name] = entry
                    _move_or_raise(entry, package.fd, name, backup.entry.fd, name)

                if retained:
                    _validate_existing_package_fd(
                        backup.entry.fd,
                        display_path=package_dir,
                        model_identity=model.identity,
                        overwrite=overwrite,
                    )

                for name in generated_names:
                    entry = _open_child(staging.fd, name, owner=transaction)
                    if entry is None:
                        raise RuntimeError(f"staged package entry {name!r} disappeared")
                    installed[name] = entry
                    _move_or_raise(entry, staging.fd, name, package.fd, name)
            except BaseException as error:
                _rollback_unnamed_descriptor(
                    package,
                    staging,
                    backup,
                    retained,
                    installed,
                    error,
                )
                _cleanup_empty_owned_directory(backup, error)
                raise

            for name, entry in retained.items():
                _remove_open_entry(backup.entry.fd, name, entry, backup.path / name)
            _remove_empty_owned_directory(backup)
            success = True
        finally:
            if backup is not None and not success and not backup.cleaned:
                logger.warning("Preserved package transaction directory at %s", backup.path)


def _rollback_unnamed_descriptor(
    package: _OpenEntry,
    staging: _OpenEntry,
    backup: _OwnedDirectory,
    retained: dict[str, _OpenEntry],
    installed: dict[str, _OpenEntry],
    error: BaseException,
) -> None:
    """Recover unnamed entries by observing their identities, never their call history."""
    quarantine: list[tuple[str, _OpenEntry]] = []
    for name, entry in reversed(installed.items()):
        try:
            at_public = _entry_matches(package.fd, name, entry.identity)
            at_staging = _entry_matches(staging.fd, name, entry.identity)
            if at_public and not at_staging:
                quarantine_name = f".new-{name}"
                result = _move_reconciled(
                    entry,
                    package.fd,
                    name,
                    backup.entry.fd,
                    quarantine_name,
                )
                if result.error is not None:
                    raise result.error
                quarantine.append((quarantine_name, entry))
            elif not at_staging:
                raise RuntimeError("installed identity is at neither expected endpoint")
        except BaseException as recovery_error:
            _add_recovery_note(
                error, f"could not quarantine installed entry {name!r}", recovery_error
            )

    for name, entry in reversed(retained.items()):
        _restore_open_entry(
            entry,
            backup.entry.fd,
            name,
            package.fd,
            name,
            error,
        )

    for name, entry in quarantine:
        try:
            _remove_open_entry(backup.entry.fd, name, entry, backup.path / name)
        except BaseException as recovery_error:
            _add_recovery_note(error, f"could not clean installed entry {name!r}", recovery_error)


def _replace_unnamed_package_by_path(
    staging_dir: Path,
    package_dir: Path,
    generated_names: list[str],
    *,
    model_dir: Path,
    overwrite: bool,
) -> None:
    """Path-based Windows fallback; active pathname racing is outside its guarantee."""
    backup_root = Path(tempfile.mkdtemp(prefix=f".{package_dir.name}-old-", dir=package_dir.parent))
    backup_root_identity = _path_identity(backup_root)
    retained: dict[str, tuple[int, int]] = {}
    installed: dict[str, tuple[int, int]] = {}
    try:
        try:
            for name in generated_names:
                destination = package_dir / name
                backup_path = backup_root / name
                try:
                    _rename_noreplace(destination, backup_path)
                except FileNotFoundError:
                    continue
                retained[name] = _path_identity(backup_path)

            if retained:
                _validate_existing_package(
                    backup_root,
                    display_path=package_dir,
                    model_dir=model_dir,
                    overwrite=overwrite,
                )
        except BaseException as retention_error:
            _rollback_unnamed_paths(
                package_dir,
                backup_root,
                retained,
                installed,
                retention_error,
            )
            raise

        try:
            for name in generated_names:
                staging_path = staging_dir / name
                identity = _path_identity(staging_path)
                _rename_noreplace(staging_path, package_dir / name)
                installed[name] = identity
        except BaseException as install_error:
            _rollback_unnamed_paths(
                package_dir,
                backup_root,
                retained,
                installed,
                install_error,
            )
            raise
    finally:
        if not retained and not installed:
            backup_root.rmdir()

    _remove_backup(backup_root, backup_root_identity)


def _reserve_absent_path(destination: Path) -> Path:
    """Reserve an unpredictable sibling name, then make it available for a rename."""
    path = Path(tempfile.mkdtemp(prefix=f".{destination.name}-old-", dir=destination.parent))
    path.rmdir()
    return path


def _path_identity(path: Path) -> tuple[int, int]:
    """Return the filesystem identity of a path without following a final symlink."""
    metadata = path.lstat()
    return metadata.st_dev, metadata.st_ino


def _require_identity(path: Path, expected: tuple[int, int]) -> None:
    """Fail if a private retained path no longer names the object that was moved there."""
    try:
        actual = _path_identity(path)
    except OSError as error:
        raise RuntimeError(f"retained package object at {path} is no longer accessible") from error
    if actual != expected:
        raise RuntimeError(f"retained package object at {path} changed during installation")


def _restore_retained_path(
    backup_path: Path,
    destination: Path,
    retained_identity: tuple[int, int],
    install_error: BaseException,
) -> None:
    """Restore one retained object without replacing a destination that appeared meanwhile."""
    try:
        _require_identity(backup_path, retained_identity)
        _rename_noreplace(backup_path, destination)
    except BaseException as restore_error:
        raise RuntimeError(
            f"could not restore previous package from {backup_path}: {restore_error}"
        ) from install_error


def _rollback_unnamed_paths(
    package_dir: Path,
    backup_root: Path,
    retained: dict[str, tuple[int, int]],
    installed: dict[str, tuple[int, int]],
    install_error: BaseException,
) -> None:
    """Remove only installed objects and restore retained paths without clobbering races."""
    restore_failure: tuple[Path, BaseException] | None = None
    for name, identity in reversed(installed.items()):
        destination = package_dir / name
        quarantine = backup_root / f".new-{name}"
        try:
            _rename_noreplace(destination, quarantine)
            _require_identity(quarantine, identity)
            _remove_path(quarantine)
        except BaseException as error:
            if restore_failure is None:
                restore_failure = (quarantine, error)

    for name, identity in reversed(retained.items()):
        backup_path = backup_root / name
        destination = package_dir / name
        try:
            _require_identity(backup_path, identity)
            _rename_noreplace(backup_path, destination)
        except BaseException as error:
            if restore_failure is None:
                restore_failure = (backup_path, error)

    if restore_failure is not None:
        recovery_path, restore_error = restore_failure
        raise RuntimeError(
            f"could not restore previous package from {recovery_path}: {restore_error}"
        ) from install_error

    backup_root.rmdir()


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename without replacing an existing destination, or fail closed."""
    if sys.platform == "darwin":
        library = ctypes.CDLL(None, use_errno=True)
        try:
            rename_exclusive = library.renamex_np
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace rename is unavailable on this platform",
            ) from error
        result = rename_exclusive(os.fsencode(source), os.fsencode(destination), 0x00000004)
    elif sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        try:
            rename_exclusive = library.renameat2
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace rename is unavailable on this platform",
            ) from error
        result = rename_exclusive(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            0x00000001,
        )
    elif os.name == "nt":
        source.rename(destination)
        return
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace rename is unavailable on this platform",
        )

    if result != 0:
        error_number = ctypes.get_errno() or errno.EIO
        raise OSError(
            error_number,
            os.strerror(error_number),
            f"{source} -> {destination}",
        )


def _rename_noreplace_at(
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
) -> None:
    """Atomically rename anchored entries without replacing an existing destination."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        try:
            rename_exclusive = library.renameatx_np
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "descriptor-relative atomic no-replace rename is unavailable",
            ) from error
        result = rename_exclusive(
            source_parent_fd,
            os.fsencode(source_name),
            destination_parent_fd,
            os.fsencode(destination_name),
            0x00000004,
        )
    elif sys.platform.startswith("linux"):
        try:
            rename_exclusive = library.renameat2
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "descriptor-relative atomic no-replace rename is unavailable",
            ) from error
        result = rename_exclusive(
            source_parent_fd,
            os.fsencode(source_name),
            destination_parent_fd,
            os.fsencode(destination_name),
            0x00000001,
        )
    else:
        raise OSError(
            errno.ENOTSUP,
            "descriptor-relative atomic no-replace rename is unavailable",
        )

    if result != 0:
        error_number = ctypes.get_errno() or errno.EIO
        if error_number == errno.EEXIST:
            raise FileExistsError(
                error_number,
                os.strerror(error_number),
                f"{source_name} -> {destination_name}",
            )
        if error_number == errno.ENOENT:
            raise FileNotFoundError(
                error_number,
                os.strerror(error_number),
                f"{source_name} -> {destination_name}",
            )
        raise OSError(
            error_number,
            os.strerror(error_number),
            f"{source_name} -> {destination_name}",
        )


def _remove_open_entry(
    parent_fd: int,
    name: str,
    entry: _OpenEntry,
    display_path: Path,
) -> None:
    """Remove contents through an open object and its final anchored directory entry."""
    # POSIX unlinkat and directory-relative rmdir still select the final entry by name. Keeping its
    # parent and object open prevents an outer-path substitution from redirecting
    # recursive cleanup, but no portable primitive conditionally unlinks the open
    # identity itself. The remaining last-component race is documented explicitly.
    if not _entry_matches(parent_fd, name, entry.identity):
        raise RuntimeError(f"refusing cleanup because {display_path} changed identity")
    if entry.is_directory:
        _empty_open_directory(entry.fd, display_path)
        if not _entry_matches(parent_fd, name, entry.identity):
            raise RuntimeError(f"refusing cleanup because {display_path} changed identity")
        os.rmdir(name, dir_fd=parent_fd)
    else:
        os.unlink(name, dir_fd=parent_fd)


def _empty_open_directory(directory_fd: int, display_path: Path) -> None:
    """Remove package-owned children relative to an open directory descriptor."""
    with os.scandir(directory_fd) as entries:
        names = [entry.name for entry in entries]
    with ExitStack() as resources:
        for name in names:
            child = _open_child(directory_fd, name, owner=resources)
            if child is None:
                continue
            _remove_open_entry(directory_fd, name, child, display_path / name)


def _cleanup_owned_directory(
    directory: _OwnedDirectory,
    *,
    strict: bool,
    missing_ok: bool = False,
) -> None:
    """Let the directory's sole owner clean through its descriptor exactly once."""
    if directory.cleaned:
        return
    try:
        current_identity = _entry_identity_at(directory.parent_fd, directory.name)
        if current_identity != directory.entry.identity:
            if missing_ok and current_identity is None:
                return
            raise RuntimeError(f"refusing cleanup because {directory.path} changed identity")
        _empty_open_directory(directory.entry.fd, directory.path)
        if not _entry_matches(directory.parent_fd, directory.name, directory.entry.identity):
            raise RuntimeError(f"refusing cleanup because {directory.path} changed identity")
        os.rmdir(directory.name, dir_fd=directory.parent_fd)
        directory.cleaned = True
    except BaseException:
        if strict:
            raise
        logger.warning(
            "Could not remove transaction directory at %s", directory.path, exc_info=True
        )


def _cleanup_empty_owned_directory(directory: _OwnedDirectory, error: BaseException) -> None:
    """Remove an empty recovery root without touching an unexpected child."""
    if directory.cleaned:
        return
    try:
        _remove_empty_owned_directory(directory)
    except BaseException as cleanup_error:
        _add_recovery_note(
            error,
            "could not remove empty package recovery directory",
            cleanup_error,
        )


def _remove_empty_owned_directory(directory: _OwnedDirectory) -> None:
    """Remove a private root only when its open descriptor proves it has no entries."""
    if directory.cleaned:
        return
    with os.scandir(directory.entry.fd) as entries:
        if next(entries, None) is not None:
            raise RuntimeError(
                f"refusing to remove non-empty transaction directory {directory.path}"
            )
    if not _entry_matches(directory.parent_fd, directory.name, directory.entry.identity):
        raise RuntimeError(f"refusing cleanup because {directory.path} changed identity")
    os.rmdir(directory.name, dir_fd=directory.parent_fd)
    directory.cleaned = True


def _remove_path(path: Path) -> None:
    """Remove a generated file or directory during rollback."""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _remove_backup(backup_path: Path, retained_identity: tuple[int, int]) -> None:
    """Remove an obsolete backup without failing a completed install."""
    try:
        _require_identity(backup_path, retained_identity)
        _remove_path(backup_path)
    except (OSError, RuntimeError) as error:
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
