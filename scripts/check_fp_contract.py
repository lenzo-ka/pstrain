#!/usr/bin/env python3
"""Reject fused multiply-add instructions in build trees or built wheels."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

FMA = re.compile(r"\b(?:v?f(?:n?madd|n?msub|mla|mls)[a-z0-9.]*)\b", re.IGNORECASE)
REQUIRED = ("bw", "norm", "sphinx_fe")
NATIVE_MAGICS = {
    b"\x7fELF",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
}
COFF_MACHINES = {
    b"\x4c\x01",  # i386
    b"\x64\x86",  # amd64
    b"\x64\xaa",  # arm64
}


def build_artifacts(build_dir: Path) -> list[Path]:
    """Return libraries and executables emitted into canonical directories."""
    found: set[Path] = set()
    for directory in (build_dir / "bin", build_dir / "lib"):
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.is_file() and (
                path.suffix in {".a", ".so", ".dylib"} or path.stat().st_mode & 0o111
            ):
                found.add(path.resolve())
    _require_training_artifacts(found, build_dir)
    return sorted(found)


def _is_native(path: Path) -> bool:
    try:
        start = path.read_bytes()[:8]
    except OSError:
        return False
    return start[:4] in NATIVE_MAGICS or start == b"!<arch>\n" or start[:2] in COFF_MACHINES


def object_artifacts(object_dir: Path) -> list[Path]:
    """Return COFF object files recursively from an extracted CI artifact."""
    found = sorted(path.resolve() for path in object_dir.rglob("*.obj") if path.is_file())
    if not found:
        raise RuntimeError(f"unchecked object set: no .obj files found in {object_dir}")
    return found


def _require_training_artifacts(found: set[Path], source: Path) -> None:
    names = {path.name for path in found}
    missing = [name for name in REQUIRED if name not in names]
    if missing:
        raise RuntimeError(
            f"required standalone artifacts absent from {source}: {', '.join(missing)}"
        )
    if not any(name.startswith("libpstrainc") for name in names):
        raise RuntimeError(f"required libpstrainc artifact absent from {source}")


def wheel_artifacts(wheel: Path, destination: Path) -> list[Path]:
    """Extract a wheel and return every native object it contains."""
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(destination)
    found = {
        path.resolve() for path in destination.rglob("*") if path.is_file() and _is_native(path)
    }
    _require_training_artifacts(found, wheel)
    return sorted(found)


def _run(command: list[str], path: Path) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        reason = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise RuntimeError(f"unchecked {path}: {' '.join(command[:2])} failed: {reason}")
    return result.stdout


def disassemblies(path: Path) -> list[tuple[str, str]]:
    """Return an explicit architecture label and disassembly for each slice."""
    if sys.platform == "darwin" and path.read_bytes()[:2] not in COFF_MACHINES:
        lipo = shutil.which("lipo")
        otool = shutil.which("otool")
        if lipo is None or otool is None:
            raise RuntimeError(f"unchecked {path}: lipo and otool are required")
        architectures = _run([lipo, "-archs", str(path)], path).split()
        if not architectures:
            raise RuntimeError(f"unchecked {path}: lipo reported no architectures")
        return [
            (architecture, _run([otool, "-arch", architecture, "-tvV", str(path)], path))
            for architecture in architectures
        ]

    objdump = shutil.which("objdump")
    if objdump is None:
        raise RuntimeError(f"unchecked {path}: objdump is required")
    headers = _run([objdump, "-f", str(path)], path)
    architectures = sorted(set(re.findall(r"architecture:\s*([^,\n]+)", headers)))
    if not architectures:
        raise RuntimeError(f"unchecked {path}: objdump reported no architecture")
    return [("+".join(architectures), _run([objdump, "-d", str(path)], path))]


def check(paths: list[tuple[str, Path]]) -> int:
    """Check labelled artifacts and report every inspected architecture."""
    failures: list[str] = []
    checked: list[str] = []
    for source, path in paths:
        for architecture, assembly in disassemblies(path):
            label = f"{source}:{path.name}[{architecture}]"
            checked.append(label)
            matches = sorted({match.group(0) for match in FMA.finditer(assembly)})
            if matches:
                failures.append(f"{label}: {', '.join(matches)}")
    if failures:
        print("FP contraction gate failed; fused instructions found:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"FP contraction gate passed: scanned {len(checked)} native artifact architectures")
    for label in checked:
        print(f"checked {label}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir", nargs="?", type=Path, default=Path("build"))
    parser.add_argument("--wheels", type=Path)
    parser.add_argument("--objects", type=Path)
    args = parser.parse_args()

    if args.objects is not None:
        if args.wheels is not None:
            parser.error("--objects and --wheels are mutually exclusive")
        return check([(str(args.objects), path) for path in object_artifacts(args.objects)])

    if args.wheels is None:
        return check([(str(args.build_dir), path) for path in build_artifacts(args.build_dir)])

    wheels = sorted(args.wheels.glob("*.whl"))
    if not wheels:
        raise RuntimeError(f"unchecked wheel set: no wheels found in {args.wheels}")
    with tempfile.TemporaryDirectory(prefix="pstrain-wheel-fp-") as temporary:
        root = Path(temporary)
        labelled: list[tuple[str, Path]] = []
        for index, wheel in enumerate(wheels):
            labelled.extend(
                (wheel.name, path) for path in wheel_artifacts(wheel, root / f"wheel-{index}")
            )
        return check(labelled)


if __name__ == "__main__":
    raise SystemExit(main())
