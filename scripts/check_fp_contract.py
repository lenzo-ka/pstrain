#!/usr/bin/env python3
"""Reject FP instructions that multiply and accumulate without intermediate rounding.

That criterion follows the fused-operation semantics in the Arm Architecture
Reference Manual.  The enumerated boundary covers x86 FMA3/FMA4 and Arm
scalar, AdvSIMD, SVE, and SVE2 families: ordinary and negated multiply-add or
subtract; destructive multiplicand forms; widening FP16/BF16 long forms;
complex FCMLA; and floating-point FMMLA/BFMMLA matrix operations.  Operand and
element-width forms share mnemonic roots in disassembly.  It also covers the
x86 VDPBF16PS and Arm BFDOT families, whose dot products accumulate unrounded
products.

Integer matrix operations such as SMMLA and UMMLA are outside the boundary
because they do not compute floating-point results.  X86 DPPS/DPPD are outside
the boundary because each multiplication is rounded before its products are
summed.  Arm SDOT, UDOT, USDOT, and SUDOT are integer operations.  Arm BFDOT,
BFMMLALB/BFMMLALT, and BFMMLA are retained because their products are not
rounded before accumulation.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

FMA = re.compile(
    r"\b(?:"
    r"v?f(?:madd|msub|nmadd|nmsub)[a-z0-9.]*"
    r"|f(?:mad|msb|nmad|nmsb|nmla|nmls)[a-z0-9.]*"
    r"|f(?:mla|mls)(?:l2?|lb|lt)?[a-z0-9.]*"
    r"|fcmla[a-z0-9.]*|fmmla[a-z0-9.]*"
    r"|bf(?:dot|mla|mmla)[a-z0-9.]*"
    r"|vdpbf16ps[a-z0-9.]*"
    r")\b",
    re.IGNORECASE,
)
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


def inspect(paths: list[tuple[str, Path]]) -> tuple[list[str], dict[str, list[str]]]:
    """Return inspected labels and fused instructions grouped by architecture."""
    checked: list[str] = []
    fused: dict[str, list[str]] = {}
    for source, path in paths:
        for architecture, assembly in disassemblies(path):
            label = f"{source}:{path.name}[{architecture}]"
            checked.append(label)
            matches = sorted({match.group(0) for match in FMA.finditer(assembly)})
            if matches:
                fused.setdefault(architecture, []).append(f"{label}: {', '.join(matches)}")
    return checked, fused


def check(paths: list[tuple[str, Path]]) -> int:
    """Check labelled artifacts and report every inspected architecture."""
    checked, fused = inspect(paths)
    failures = [failure for architecture in sorted(fused) for failure in fused[architecture]]
    if failures:
        print("FP contraction gate failed; fused instructions found:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"FP contraction gate passed: scanned {len(checked)} native artifact architectures")
    for label in checked:
        print(f"checked {label}")
    return 0


def check_population(production_dir: Path, canary_dir: Path) -> int:
    """Require clean production objects and a rejecting canary for every architecture."""
    production = [(str(production_dir), path) for path in object_artifacts(production_dir)]
    canaries = [(str(canary_dir), path) for path in object_artifacts(canary_dir)]
    checked, production_fused = inspect(production)
    canary_checked, canary_fused = inspect(canaries)
    production_architectures = {label.rsplit("[", 1)[1][:-1] for label in checked}
    canary_architectures = {label.rsplit("[", 1)[1][:-1] for label in canary_checked}

    failures: list[str] = []
    for architecture in sorted(production_fused):
        failures.extend(production_fused[architecture])
    missing = sorted(production_architectures - canary_architectures)
    nondiscriminating = sorted(production_architectures - set(canary_fused))
    if missing:
        failures.append("missing same-architecture canary: " + ", ".join(missing))
    if nondiscriminating:
        failures.append("canary did not trigger scanner: " + ", ".join(nondiscriminating))
    if failures:
        print("FP contraction population gate failed:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1

    architectures = ", ".join(sorted(production_architectures))
    print(
        "FP contraction population gate passed: "
        f"{len(checked)} production object architectures; canary evidence for {architectures}"
    )
    for architecture in sorted(production_architectures):
        print(f"canary rejected [{architecture}]: {'; '.join(canary_fused[architecture])}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir", nargs="?", type=Path, default=Path("build"))
    parser.add_argument("--wheels", type=Path)
    parser.add_argument("--objects", type=Path)
    parser.add_argument("--canaries", type=Path)
    args = parser.parse_args()

    if args.objects is not None:
        if args.wheels is not None:
            parser.error("--objects and --wheels are mutually exclusive")
        if args.canaries is not None:
            return check_population(args.objects, args.canaries)
        return check([(str(args.objects), path) for path in object_artifacts(args.objects)])

    if args.canaries is not None:
        parser.error("--canaries requires --objects")

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
