"""Re-runnable negative control for the emitted-code contraction gate."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.check_fp_contract import FMA


def test_fused_mnemonic_token_stream_coverage() -> None:
    """Regex covers the real 4FMAPS family without accepting nearby raw-text tokens."""
    fused = (
        "  v4fmaddps %zmm0, %zmm1\n  v4fmaddss %xmm0, %xmm1\n"
        "  v4fnmaddps %zmm0, %zmm1\n  v4fnmaddss %xmm0, %xmm1\n"
        "  vfmadd213ps %zmm2, %zmm3, %zmm4\n"
    )
    clean = (
        "0000 <v4fmaddps_fallback>:\n  v4fmaddpd %zmm0, %zmm1\n"
        "  v4fnmaddsd %xmm0, %xmm1\n  v4fmaddsubps %zmm0, %zmm1\n"
    )

    assert [match.group(0) for match in FMA.finditer(fused)] == [
        "v4fmaddps",
        "v4fmaddss",
        "v4fnmaddps",
        "v4fnmaddss",
        "vfmadd213ps",
    ]
    assert FMA.search(clean) is None


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


def test_coff_object_without_capable_disassembler_is_unchecked(tmp_path: Path) -> None:
    """An object remains a hard failure when no disassembler can inspect it."""
    compiler = shutil.which("clang")
    if compiler is None:
        pytest.skip("COFF fail-closed control requires clang")

    objects = tmp_path / "objects"
    objects.mkdir()
    source = objects / "clean.c"
    source.write_text("double clean(double value){return value + 1.0;}\n")
    artifact = objects / "clean.obj"
    subprocess.run(
        [
            compiler,
            "--target=x86_64-pc-windows-msvc",
            "-c",
            str(source),
            "-o",
            str(artifact),
        ],
        check=True,
    )

    environment = os.environ.copy()
    environment["PATH"] = ""
    result = subprocess.run(
        [sys.executable, "scripts/check_fp_contract.py", "--objects", str(objects)],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode != 0
    assert "unchecked" in result.stderr
    assert "no available disassembler could read file" in result.stderr


def test_vdpbf16ps_coff_object_makes_gate_red(tmp_path: Path) -> None:
    """The scanner rejects the x86 BF16 fused three-way dot-product construction."""
    compiler = shutil.which("clang")
    objdump = shutil.which("objdump")
    if compiler is None or objdump is None:
        pytest.skip("VDPBF16PS COFF negative control requires clang and objdump")

    objects = tmp_path / "objects"
    objects.mkdir()
    source = objects / "vdpbf16ps.s"
    source.write_text(
        ".text\n.globl fused_dot\nfused_dot:\n  vdpbf16ps %zmm2, %zmm1, %zmm0\n  ret\n"
    )
    artifact = objects / "x86-vdpbf16ps.obj"
    subprocess.run(
        [
            compiler,
            "--target=x86_64-pc-windows-msvc",
            "-march=cooperlake",
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
    assert "x86-vdpbf16ps.obj" in result.stderr
    assert "vdpbf16ps" in result.stderr


def test_object_population_requires_discriminating_same_architecture_canary(
    tmp_path: Path,
) -> None:
    """The staged architecture is green only with a same-ISA rejecting canary."""
    compiler = shutil.which("clang")
    objdump = shutil.which("objdump")
    if compiler is None or objdump is None:
        pytest.skip("COFF population control requires clang and objdump")

    source = tmp_path / "canary.c"
    source.write_text("double fused(double a,double b,double c){return a*b+c;}\n")
    production = tmp_path / "production"
    canaries = tmp_path / "canaries"
    production.mkdir()
    canaries.mkdir()
    common = [
        compiler,
        "--target=x86_64-pc-windows-msvc",
        "-O3",
        "-mfma",
        "-c",
        str(source),
    ]
    subprocess.run(
        [*common, "-ffp-contract=off", "-o", str(production / "production.obj")],
        check=True,
    )
    subprocess.run(
        [*common, "-ffp-contract=fast", "-o", str(canaries / "canary.obj")],
        check=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_fp_contract.py",
            "--objects",
            str(production),
            "--canaries",
            str(canaries),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "canary evidence for" in result.stdout
    assert "canary rejected" in result.stdout

    arm_source = tmp_path / "arm-production.s"
    arm_source.write_text(".text\n.globl clean\nclean:\n  ret\n")
    subprocess.run(
        [
            compiler,
            "--target=aarch64-pc-windows-msvc",
            "-c",
            str(arm_source),
            "-o",
            str(production / "arm-production.obj"),
        ],
        check=True,
    )
    missing = subprocess.run(
        [
            sys.executable,
            "scripts/check_fp_contract.py",
            "--objects",
            str(production),
            "--canaries",
            str(canaries),
        ],
        capture_output=True,
        text=True,
    )
    assert missing.returncode == 1
    assert "missing same-architecture canary" in missing.stderr


@pytest.mark.parametrize("mnemonic", ["fmad", "fmsb", "fnmad", "fnmsb"])
def test_sve_multiplicand_coff_arm64_object_makes_gate_red(tmp_path: Path, mnemonic: str) -> None:
    """The scanner rejects every SVE fused destructive-multiplicand form."""
    compiler = shutil.which("clang")
    objdump = shutil.which("objdump")
    if compiler is None or objdump is None:
        pytest.skip("ARM64 COFF negative control requires clang and objdump")

    objects = tmp_path / "objects"
    objects.mkdir()
    source = objects / f"{mnemonic}.s"
    source.write_text(f".text\n.globl fused\nfused:\n  {mnemonic} z0.d, p0/m, z1.d, z2.d\n  ret\n")
    artifact = objects / f"sve-{mnemonic}.obj"
    subprocess.run(
        [
            compiler,
            "--target=aarch64-pc-windows-msvc",
            "-march=armv8-a+sve",
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
    assert f"sve-{mnemonic}.obj" in result.stderr
    assert mnemonic in result.stderr


@pytest.mark.parametrize(
    ("mnemonic", "instruction", "march"),
    [
        ("fcmla", "fcmla v0.2d, v1.2d, v2.2d, #0", "armv8.3-a"),
        ("fmmla", "fmmla z0.s, z1.s, z2.s", "armv8.6-a+sve+f32mm"),
        ("bfdot", "bfdot v0.4s, v1.8h, v2.8h", "armv8.6-a+bf16"),
        ("bfmlalb", "bfmlalb z0.s, z1.h, z2.h", "armv8.6-a+sve+bf16"),
        ("bfmlalt", "bfmlalt z0.s, z1.h, z2.h", "armv8.6-a+sve+bf16"),
        ("bfmmla", "bfmmla v0.4s, v1.8h, v2.8h", "armv8.6-a+bf16"),
    ],
)
def test_arm_fused_complex_and_matrix_coff_objects_make_gate_red(
    tmp_path: Path, mnemonic: str, instruction: str, march: str
) -> None:
    """The scanner rejects retained ARM64 complex and FP matrix constructions."""
    compiler = shutil.which("clang")
    objdump = shutil.which("objdump")
    if compiler is None or objdump is None:
        pytest.skip("ARM64 COFF negative control requires clang and objdump")

    objects = tmp_path / "objects"
    objects.mkdir()
    source = objects / f"{mnemonic}.s"
    source.write_text(f".text\n.globl fused\nfused:\n  {instruction}\n  ret\n")
    artifact = objects / f"arm64-{mnemonic}.obj"
    subprocess.run(
        [
            compiler,
            "--target=aarch64-pc-windows-msvc",
            f"-march={march}",
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
    assert f"arm64-{mnemonic}.obj" in result.stderr
    assert mnemonic in result.stderr
