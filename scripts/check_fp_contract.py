#!/usr/bin/env python3
"""Reject fused multiply-add instructions in every built native artifact."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

FMA = re.compile(r"\b(?:v?f(?:n?madd|n?msub|mla|mls)[a-z0-9.]*)\b", re.IGNORECASE)
REQUIRED = ("bw", "norm", "sphinx_fe")


def artifacts(build_dir: Path) -> list[Path]:
    """Return all libraries and executables emitted into canonical directories."""
    found: set[Path] = set()
    for directory in (build_dir / "bin", build_dir / "lib"):
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.is_file() and (
                path.suffix in {".a", ".so", ".dylib"} or path.stat().st_mode & 0o111
            ):
                found.add(path.resolve())
    missing = [name for name in REQUIRED if not (build_dir / "bin" / name).is_file()]
    if missing:
        raise RuntimeError(f"required standalone artifacts absent: {', '.join(missing)}")
    if not any(path.name.startswith("libpstrainc") for path in found):
        raise RuntimeError("required libpstrainc artifact absent")
    return sorted(found)


def disassemble(path: Path) -> str:
    """Disassemble an ELF/Mach-O object or archive with the platform tool."""
    if sys.platform == "darwin":
        command = ["otool", "-tvV", str(path)]
    else:
        tool = shutil.which("objdump")
        if tool is None:
            raise RuntimeError("objdump is required for the FP-contraction gate")
        command = [tool, "-d", str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"could not disassemble {path}: {result.stderr.strip()}")
    return result.stdout


def main() -> int:
    build_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "build")
    built = artifacts(build_dir)
    failures: list[str] = []
    for path in built:
        matches = sorted({match.group(0) for match in FMA.finditer(disassemble(path))})
        if matches:
            failures.append(f"{path}: {', '.join(matches)}")
    if failures:
        print("FP contraction gate failed; fused instructions found:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"FP contraction gate passed: scanned {len(built)} native artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
