"""CLI command for pstrain system information.

Shows installation paths, versions, and system configuration.
Similar to `python-config --prefix` or `pkg-config --variable`.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from typing import Any

from pstrain import __version__
from pstrain.cli.base import Command, CommandContext, CommandResult
from pstrain.lib.paths import PstrainPaths, get_paths


class InfoCommand(Command):
    """Show pstrain installation and system information."""

    name = "info"
    help = "Show pstrain paths, version, and system information"
    needs_project_dir = False  # This command doesn't need a project
    supports_json_output = True

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--bin-dir",
            action="store_true",
            help="Print only the binary directory path",
        )
        parser.add_argument(
            "--lib-path",
            action="store_true",
            help="Print only the library path",
        )
        parser.add_argument(
            "--include-dir",
            action="store_true",
            help="Print only the C header include directory",
        )
        parser.add_argument(
            "--cflags",
            action="store_true",
            help="Print compiler flags for C development (-I...)",
        )
        parser.add_argument(
            "--ldflags",
            action="store_true",
            help="Print linker flags for C development (-L... -l...)",
        )
        parser.add_argument(
            "--version",
            action="store_true",
            dest="show_version",
            help="Print only the version",
        )

    def execute(self, ctx: CommandContext) -> CommandResult:
        args = ctx.args
        paths = get_paths()

        # Single-value outputs (for scripting)
        if getattr(args, "bin_dir", False):
            if paths.bin_dir:
                print(paths.bin_dir)
                return CommandResult.ok()
            print("not found", file=sys.stderr)
            return CommandResult.fail("Binary directory not found")

        if getattr(args, "lib_path", False):
            if paths.lib_path:
                print(paths.lib_path)
                return CommandResult.ok()
            print("not found", file=sys.stderr)
            return CommandResult.fail("Library not found")

        if getattr(args, "include_dir", False):
            if paths.include_dir:
                print(paths.include_dir)
                return CommandResult.ok()
            print("not found", file=sys.stderr)
            return CommandResult.fail("Include directory not found")

        if getattr(args, "cflags", False):
            flags = []
            if paths.include_dir:
                flags.append(f"-I{paths.include_dir}")
            print(" ".join(flags) if flags else "")
            return CommandResult.ok()

        if getattr(args, "ldflags", False):
            flags = []
            if paths.lib_path:
                lib_dir = paths.lib_path.parent
                flags.append(f"-L{lib_dir}")
                flags.append("-lpstrainc")
            print(" ".join(flags) if flags else "")
            return CommandResult.ok()

        if getattr(args, "show_version", False):
            print(__version__)
            return CommandResult.ok()

        # Full info output
        info = self._gather_info(paths)

        if ctx.json_output:
            print(ctx.format_json(info))
        else:
            self._print_info(info)

        return CommandResult.ok()

    def _gather_info(self, paths: PstrainPaths) -> dict[str, Any]:
        """Gather all system information."""
        lib_available = self._check_lib_available()

        fp_contract = None
        if lib_available:
            from pstrain.lib._cffi.core import get_ffi, get_lib

            fp_contract = get_ffi().string(get_lib().pstrain_fp_contract_policy()).decode("ascii")

        from pstrain.lib.commands import PSTRAIN_BINARIES, resolve_binary

        native_programs: dict[str, dict[str, str]] = {}
        for name in PSTRAIN_BINARIES.values():
            program = resolve_binary(name)
            if program is None:
                native_programs[name] = {"state": "absent"}
            else:
                with program.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                native_programs[name] = {"path": str(program), "sha256": digest}

        return {
            "pstrain": {
                "version": __version__,
                "lib_available": lib_available,
                "fp_contract_declared": fp_contract,
                "native_programs": native_programs,
            },
            "paths": {
                "bin_dir": str(paths.bin_dir) if paths.bin_dir else None,
                "lib_path": str(paths.lib_path) if paths.lib_path else None,
                "include_dir": str(paths.include_dir) if paths.include_dir else None,
                "data_dir": str(paths.data_dir),
                "project_root": str(paths.project_root) if paths.project_root else None,
            },
            "python": {
                "version": platform.python_version(),
                "executable": sys.executable,
                "prefix": sys.prefix,
            },
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "release": platform.release(),
            },
        }

    def _check_lib_available(self) -> bool:
        """Check if the C library can be loaded."""
        try:
            from pstrain.lib._cffi.core import _find_library

            _find_library()
            return True
        except RuntimeError:
            return False

    def _print_info(self, info: dict[str, Any]) -> None:
        """Print info in human-readable format."""
        print("pstrain Information")
        print("=" * 40)
        print()

        print("Version:")
        print(f"  pstrain:           {info['pstrain']['version']}")
        print(
            f"  C library:     {'available' if info['pstrain']['lib_available'] else 'not found'}"
        )
        if info["pstrain"]["fp_contract_declared"] is not None:
            print(
                f"  C library FP contraction (declared): {info['pstrain']['fp_contract_declared']}"
            )
        print()

        print("Paths:")
        paths = info["paths"]
        print(f"  bin_dir:       {paths['bin_dir'] or '(not found)'}")
        print(f"  lib_path:      {paths['lib_path'] or '(not found)'}")
        print(f"  include_dir:   {paths['include_dir'] or '(not found)'}")
        print(f"  data_dir:      {paths['data_dir']}")
        if paths["project_root"]:
            print(f"  project_root:  {paths['project_root']} (development)")
        print()

        print("Resolved native programs (exact artifacts):")
        for name, identity in info["pstrain"]["native_programs"].items():
            if identity.get("state") == "absent":
                print(f"  {name}: absent")
            else:
                print(f"  {name}: {identity['path']} (sha256 {identity['sha256']})")
        print()

        print("Python:")
        py = info["python"]
        print(f"  version:       {py['version']}")
        print(f"  executable:    {py['executable']}")
        print(f"  prefix:        {py['prefix']}")
        print()

        print("Platform:")
        plat = info["platform"]
        print(f"  system:        {plat['system']}")
        print(f"  machine:       {plat['machine']}")
        print(f"  release:       {plat['release']}")

        # Show helpful hints if things are missing
        if not info["pstrain"]["lib_available"]:
            print()
            print("Note: C library not found. Build with:")
            print("  cmake -S . -B build && cmake --build build")

        if not paths["bin_dir"]:
            print()
            print("Note: Binary directory not found. Options:")
            print("  - Build: cmake -S . -B build && cmake --build build")
            print("  - Install: cmake --install build")
            print("  - Set PSTRAIN_BIN_DIR environment variable")


# Module-level command instance
info_command = InfoCommand()
