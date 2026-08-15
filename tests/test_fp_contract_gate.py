"""Re-runnable negative control for the emitted-code contraction gate."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def test_contraction_enabled_build_makes_gate_red(tmp_path: Path) -> None:
    """A contraction-enabled native artifact must make the shipped gate fail."""
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64", "arm64", "aarch64"}:
        pytest.skip(f"negative control has no compiler flags for {machine}")
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("negative control requires a C compiler")

    source = tmp_path / "fused.c"
    source.write_text(
        "#include <stdio.h>\n"
        "__attribute__((noinline)) double fused(double a, double b, double c) "
        "{ return a * b + c; }\n"
        "int main(void) { volatile double a=2, b=3, c=4; "
        'printf("%f\\n", fused(a,b,c)); return 0; }\n'
    )
    artifact = tmp_path / "fused"
    flags = ["-O3", "-ffp-contract=fast"]
    if machine in {"x86_64", "amd64"}:
        flags.append("-mfma")
    subprocess.run([compiler, *flags, str(source), "-o", str(artifact)], check=True)

    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    (build / "lib").mkdir()
    for name in ("bw", "norm", "sphinx_fe"):
        shutil.copy2(artifact, build / "bin" / name)
    shutil.copy2(artifact, build / "lib" / "libpstrainc.so")

    result = subprocess.run(
        [sys.executable, "scripts/check_fp_contract.py", str(build)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "FP contraction gate failed; fused instructions found" in result.stderr


def test_contraction_enabled_wheel_makes_gate_red(tmp_path: Path) -> None:
    """The wheel extraction path must present fused native artifacts to the detector."""
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64", "arm64", "aarch64"}:
        pytest.skip(f"negative control has no compiler flags for {machine}")
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("negative control requires a C compiler")

    source = tmp_path / "fused.c"
    source.write_text("double fused(double a,double b,double c){return a*b+c;}\n")
    artifact = tmp_path / "fused.so"
    flags = ["-O3", "-ffp-contract=fast", "-dynamiclib" if sys.platform == "darwin" else "-shared"]
    if machine in {"x86_64", "amd64"}:
        flags.append("-mfma")
    subprocess.run([compiler, *flags, str(source), "-o", str(artifact)], check=True)

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel = wheelhouse / "negative_control-0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in ("bw", "norm", "sphinx_fe", "libpstrainc.so"):
            archive.write(artifact, f"pstrain/_lib/{name}")

    result = subprocess.run(
        [sys.executable, "scripts/check_fp_contract.py", "--wheels", str(wheelhouse)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "FP contraction gate failed; fused instructions found" in result.stderr


def test_contraction_enabled_coff_object_makes_gate_red(tmp_path: Path) -> None:
    """A contraction-enabled COFF object must make the object scan fail."""
    compiler = shutil.which("clang")
    objdump = shutil.which("objdump")
    if compiler is None or objdump is None:
        pytest.skip("COFF negative control requires clang and objdump")

    objects = tmp_path / "objects"
    objects.mkdir()
    source = objects / "canary.c"
    source.write_text("double fused(double a,double b,double c){return a*b+c;}\n")
    artifact = objects / "fp-contract-canary.obj"
    subprocess.run(
        [
            compiler,
            "--target=x86_64-pc-windows-msvc",
            "-O3",
            "-ffp-contract=fast",
            "-mfma",
            "-c",
            str(source),
            "-o",
            str(artifact),
        ],
        check=True,
    )

    result = subprocess.run(
        [sys.executable, "scripts/check_fp_contract.py", "--objects", str(objects)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "fp-contract-canary.obj" in result.stderr
    assert "FP contraction gate failed; fused instructions found" in result.stderr
