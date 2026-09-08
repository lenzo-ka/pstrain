"""CLI command for packaging a trained acoustic model."""

from __future__ import annotations

import argparse
import os
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
        from pstrain.api import package_model, validate_package_destination

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
        model_dir = model_dir.resolve()
        include_dict = not ctx.args.no_dict
        package_dir = validate_package_destination(
            model_dir,
            output_dir,
            package_name,
            include_dict=include_dict,
            overwrite=ctx.args.overwrite,
        )

        dictionary = ctx.args.dict or project_dir / "shared" / "dictionary.dict"
        filler = ctx.args.filler_dict or project_dir / "shared" / "filler.dict"
        filler_path = filler if filler.exists() else None
        if include_dict and not dictionary.is_file():
            return CommandResult.fail(
                f"Dictionary not found: {dictionary}. Provide --dict PATH or use --no-dict."
            )

        artifact_paths = _artifact_paths(
            package_dir,
            include_dict=include_dict,
            include_filler=include_dict and filler_path is not None,
        )
        if ctx.dry_run:
            _print_row("source", model_dir)
            _print_row("destination", package_dir)
            _print_row("dictionary", dictionary if include_dict else "excluded")
            _print_row("filler_dictionary", filler_path or "generated noisedict")
            for key, path in artifact_paths.items():
                _print_row(key, path)
            return CommandResult.ok()

        replacing = package_dir.exists()

        artifacts = package_model(
            model_dir=model_dir,
            output_dir=output_dir,
            model_name=package_name,
            dictionary_path=dictionary,
            filler_dict_path=filler_path,
            include_dict=include_dict,
            overwrite=ctx.args.overwrite,
        )
        if replacing:
            _print_row("replaced", package_dir)
        for key, path in artifacts.items():
            _print_row(key, path)
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


def _print_row(key: str, value: object) -> None:
    """Print one tabular row without allowing a field to create extra rows or columns."""
    print(f"{key}\t{_line_field(value)}")


def _line_field(value: object) -> str:
    """Backslash-escape controls only when a field would disrupt line-oriented output."""
    text = str(value)
    if not any(character in text for character in "\t\r\n"):
        return text
    return text.replace("\\", "\\\\").replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n")
