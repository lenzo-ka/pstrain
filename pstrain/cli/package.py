"""CLI command for packaging a trained acoustic model."""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path

from pstrain.cli.base import Command, CommandContext, CommandResult


class PackageCommand(Command):
    """Create a distributable package from a trained model."""

    name = "package"
    help = "Package a trained model for distribution"
    description = "Create a self-contained package from a trained acoustic model"
    needs_project_dir = True

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "target",
            help=(
                "Trained model name (e.g., cd-1g), or an explicit model path "
                "containing a separator (relative paths resolve from the current directory)"
            ),
        )
        parser.add_argument(
            "--config",
            metavar="NAME",
            help="Model profile for a bare target (default: the only profile, or default)",
        )
        parser.add_argument(
            "--out",
            type=Path,
            metavar="DIR",
            help="Output directory (default: <project>/packages)",
        )
        parser.add_argument(
            "--name",
            help="Package name as one path component (default: target name)",
        )
        parser.add_argument(
            "--dict",
            type=Path,
            help="Dictionary file (default: <project>/shared/dictionary.dict)",
        )
        parser.add_argument(
            "--filler-dict",
            type=Path,
            help="Filler dictionary (default: <project>/shared/filler.dict when present)",
        )
        parser.add_argument(
            "--no-dict",
            action="store_true",
            help="Do not include dictionaries in the package",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Replace an unmarked destination that has legacy package structure",
        )

    def execute(self, ctx: CommandContext) -> CommandResult:
        """Package the requested trained model."""
        from pstrain.api import package_model

        project_dir = ctx.project_dir
        target = ctx.args.target
        target_path = Path(target)
        explicit_path = target_path.is_absolute() or os.sep in target
        if os.altsep is not None:
            explicit_path = explicit_path or os.altsep in target
        if explicit_path:
            model_dir = target_path.resolve()
            default_name = target_path.name
        else:
            target_dir = project_dir / "shared" / "models" / target
            config_name = ctx.args.config
            if config_name is None and target_dir.is_dir():
                profiles = sorted(path.name for path in target_dir.iterdir() if path.is_dir())
                if len(profiles) > 1:
                    choices = ", ".join(profiles)
                    return CommandResult.fail(
                        f"Model target {target!r} has multiple profiles: {choices}. "
                        "Choose one with --config NAME."
                    )
                if profiles:
                    config_name = profiles[0]
            model_dir = target_dir / (config_name or "default")
            default_name = target

        output_dir = (ctx.args.out if ctx.args.out else project_dir / "packages").resolve()
        package_name = default_name if ctx.args.name is None else ctx.args.name
        requested_destination = output_dir / package_name
        destination = requested_destination.resolve()
        if (
            not package_name
            or package_name in {".", ".."}
            or len(Path(package_name).parts) != 1
            or not destination.is_relative_to(output_dir)
        ):
            return CommandResult.fail(
                f"Invalid package destination {destination}: package name {package_name!r} "
                "must be exactly one ordinary path component."
            )

        model_dir = model_dir.resolve()
        if _paths_overlap(model_dir, destination):
            return CommandResult.fail(
                f"Package destination {destination} overlaps source model {model_dir}; "
                "choose a separate output directory and package name."
            )

        if requested_destination.exists() or requested_destination.is_symlink():
            if requested_destination.is_symlink() or not _has_package_structure(
                requested_destination
            ):
                return CommandResult.fail(
                    f"Refusing to replace {requested_destination}: existing destination "
                    "is not a recognizable pstrain package."
                )
            marker_status = _package_marker_status(requested_destination)
            if marker_status == "absent" and not ctx.args.overwrite:
                return CommandResult.fail(
                    f"Refusing to replace {requested_destination}: existing package has no "
                    "pstrain package marker. Use --overwrite to replace the entire existing "
                    "directory."
                )
            if marker_status == "invalid":
                return CommandResult.fail(
                    f"Refusing to replace {requested_destination}: existing pstrain package "
                    "marker is invalid or uses an unsupported format version. An overwrite "
                    "replaces the entire existing directory, so the marker must be understood "
                    "before replacement."
                )

        dictionary = ctx.args.dict or project_dir / "shared" / "dictionary.dict"
        filler = ctx.args.filler_dict or project_dir / "shared" / "filler.dict"
        filler_path = filler if filler.exists() else None
        include_dict = not ctx.args.no_dict
        if include_dict and not dictionary.is_file():
            return CommandResult.fail(
                f"Dictionary not found: {dictionary}. Provide --dict PATH or use --no-dict."
            )

        package_dir = requested_destination
        artifact_paths = _artifact_paths(
            package_dir,
            include_dict=include_dict,
            include_filler=include_dict and filler_path is not None,
        )
        if ctx.dry_run:
            print(f"source\t{model_dir}")
            print(f"destination\t{package_dir}")
            print(f"dictionary\t{dictionary if include_dict else 'excluded'}")
            print(f"filler_dictionary\t{filler_path or 'generated noisedict'}")
            for key, path in artifact_paths.items():
                print(f"{key}\t{path}")
            return CommandResult.ok()

        replacing = package_dir.exists()

        artifacts = package_model(
            model_dir=model_dir,
            output_dir=output_dir,
            model_name=package_name,
            dictionary_path=dictionary,
            filler_dict_path=filler_path,
            include_dict=include_dict,
        )
        if replacing:
            print(f"replaced\t{package_dir}")
        for key, path in artifacts.items():
            print(f"{key}\t{path}")
        return CommandResult.ok()


def _artifact_paths(
    package_dir: Path, *, include_dict: bool, include_filler: bool
) -> dict[str, Path]:
    """Return the paths produced by the public packager without touching disk."""
    acoustic = package_dir / "acoustic"
    paths = {
        name: acoustic / name
        for name in ("mdef", "means", "variances", "mixture_weights", "transition_matrices")
    }
    paths["feat_params"] = acoustic / "feat.params"
    paths["noisedict"] = acoustic / "noisedict"
    if include_dict:
        paths["dictionary"] = package_dir / "dict" / "cmudict.dict"
        if include_filler:
            paths["filler_dict"] = package_dir / "dict" / "filler.dict"
    paths["manifest"] = package_dir / "pstrain-package.json"
    paths["readme"] = package_dir / "README.txt"
    return paths


package_command = PackageCommand()


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
    required = (
        "mdef",
        "means",
        "variances",
        "mixture_weights",
        "transition_matrices",
        "feat.params",
        "noisedict",
    )
    return acoustic.is_dir() and all((acoustic / name).is_file() for name in required)


def _package_marker_status(path: Path) -> str:
    """Classify a package marker as supported, absent, or invalid."""
    marker_path = path / "pstrain-package.json"
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
        and marker["format_version"] == 1
        and marker.get("generator") == "pstrain"
    )
    return "supported" if supported else "invalid"
