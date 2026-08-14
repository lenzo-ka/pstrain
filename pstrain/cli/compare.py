"""Compare command for debugging and verification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from pstrain.cli.base import Command, CommandResult

if TYPE_CHECKING:
    from pstrain.cli.base import CommandContext

__all__ = ["CompareCommand", "compare_command"]


class CompareCommand(Command):
    """Compare files for debugging and verification.

    Examples:
        pstrain compare file1.mfc file2.mfc          # Auto-detects type
        pstrain compare model_dir1 model_dir2        # Compares full models
        pstrain compare --stats file.mfc             # Show stats for single file
        pstrain compare --type means file1 file2     # Force specific type
    """

    name = "compare"
    help = "Compare feature files, models, or parameters"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "file_a",
            type=Path,
            help="First file or directory",
        )
        parser.add_argument(
            "file_b",
            type=Path,
            nargs="?",
            help="Second file or directory (omit for --stats)",
        )
        parser.add_argument(
            "-t",
            "--type",
            choices=["features", "means", "variances", "mixw", "tmat", "model", "auto"],
            default="auto",
            help="Type of comparison (default: auto-detect)",
        )
        parser.add_argument(
            "--rtol",
            type=float,
            default=1e-5,
            help="Relative tolerance (default: 1e-5)",
        )
        parser.add_argument(
            "--atol",
            type=float,
            default=1e-8,
            help="Absolute tolerance (default: 1e-8)",
        )
        parser.add_argument(
            "--stats",
            action="store_true",
            help="Print statistics for a single file instead of comparing",
        )
        parser.add_argument(
            "--deep",
            action="store_true",
            help="Deep validation: try loading files to verify type (slower)",
        )

    @classmethod
    def execute(cls, ctx: CommandContext) -> CommandResult:
        from pstrain.lib.compare import (
            ModelCompareResult,
            compare_auto,
            compare_features,
            compare_gaussians,
            compare_mixw,
            compare_models,
            compare_tmat,
            print_stats,
        )
        from pstrain.lib.filetypes import (
            FileType,
            describe_file,
            detect_file_type,
            validate_file_type,
        )

        args = ctx.args
        file_a = args.file_a

        # Stats mode - single file
        if args.stats:
            file_type = detect_file_type(file_a)
            print(f"Type: {describe_file(file_a)}")
            print_stats(file_a)
            return CommandResult.ok()

        # Compare mode - two files required
        if args.file_b is None:
            return CommandResult.fail("Second file required for comparison (or use --stats)")

        file_b = args.file_b
        rtol = args.rtol
        atol = args.atol

        # Map CLI type names to FileType enum values
        type_mapping = {
            "features": FileType.FEATURES,
            "means": FileType.MEANS,
            "variances": FileType.VARIANCES,
            "mixw": FileType.MIXTURE_WEIGHTS,
            "tmat": FileType.TRANSITION_MATRICES,
            "model": FileType.MODEL,
        }

        # Auto-detect or use specified type
        if args.type == "auto":
            try:
                file_type, result = compare_auto(file_a, file_b, rtol, atol)
                print(f"Comparing {file_type.value}:")
                print(f"  {file_a}")
                print(f"  {file_b}")
                if isinstance(result, ModelCompareResult):
                    print(result.summary())
                    return (
                        CommandResult.ok()
                        if result.all_compared_components_match
                        else CommandResult.fail("")
                    )
                print(f"Result: {result.summary()}")
                return CommandResult.ok() if result.match else CommandResult.fail("")
            except ValueError as e:
                return CommandResult.fail(str(e))
        else:
            # Explicit type specified - validate that files match expected type
            compare_type = args.type
            expected_type = type_mapping[compare_type]

            # Validate both files
            valid_a, msg_a = validate_file_type(file_a, expected_type, deep=args.deep)
            if not valid_a:
                return CommandResult.fail(
                    f"{file_a}: {msg_a}\n"
                    f"  Use --type auto to auto-detect, or verify the file is correct"
                )

            valid_b, msg_b = validate_file_type(file_b, expected_type, deep=args.deep)
            if not valid_b:
                return CommandResult.fail(
                    f"{file_b}: {msg_b}\n"
                    f"  Use --type auto to auto-detect, or verify the file is correct"
                )

            print(f"Comparing {compare_type}:")
            print(f"  {file_a}")
            print(f"  {file_b}")

            if compare_type == "features":
                result = compare_features(file_a, file_b, rtol, atol)
            elif compare_type in ("means", "variances"):
                result = compare_gaussians(file_a, file_b, rtol, atol)
            elif compare_type == "mixw":
                result = compare_mixw(file_a, file_b, rtol, atol)
            elif compare_type == "tmat":
                result = compare_tmat(file_a, file_b, rtol, atol)
            elif compare_type == "model":
                model_result = compare_models(file_a, file_b, rtol, atol)
                print(model_result.summary())
                return (
                    CommandResult.ok()
                    if model_result.all_compared_components_match
                    else CommandResult.fail("")
                )
            else:
                return CommandResult.fail(f"Unknown type: {compare_type}")

            print(f"Result: {result.summary()}")
            return CommandResult.ok() if result.match else CommandResult.fail("")


# Module-level instance for CLI registration
compare_command = CompareCommand()
