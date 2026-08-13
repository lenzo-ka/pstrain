#!/usr/bin/env python3
"""Generate platform linker export lists from the CFFI declarations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cffi import FFI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pstrain.lib._cffi.cdef import CDEF  # noqa: E402


def cffi_functions() -> list[str]:
    """Return the complete function surface declared to ABI-mode CFFI."""
    ffi = FFI()
    ffi.cdef(CDEF)
    return sorted(
        key.removeprefix("function ")
        for key in ffi._parser._declarations  # type: ignore[attr-defined]
        if key.startswith("function ")
    )


def rendered_exports() -> dict[Path, str]:
    """Render ELF and Mach-O allowlists from the same symbol set."""
    names = cffi_functions()
    elf = "{\n  global:\n" + "".join(f"    {name};\n" for name in names)
    elf += "  local:\n    *;\n};\n"
    macos = "".join(f"_{name}\n" for name in names)
    return {
        ROOT / "csrc" / "pstrain.exports.elf": elf,
        ROOT / "csrc" / "pstrain.exports.macos": macos,
    }


def main() -> int:
    """Write export lists, or verify checked-in lists with ``--check``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    mismatches = []
    for path, expected in rendered_exports().items():
        if args.check:
            if not path.is_file() or path.read_text() != expected:
                mismatches.append(path.relative_to(ROOT))
        else:
            path.write_text(expected)
    if mismatches:
        print("stale CFFI export lists: " + ", ".join(map(str, mismatches)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
