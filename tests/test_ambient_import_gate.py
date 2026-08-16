from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts" / "check_ambient_import.py"


def _venv_python(tmp_path: Path) -> Path:
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    binary = "python.exe" if os.name == "nt" else "python"
    return venv / ("Scripts" if os.name == "nt" else "bin") / binary


def _run(python: Path, *, pythonpath: Path | None = None) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    if pythonpath is not None:
        environment["PYTHONPATH"] = str(pythonpath)
    return subprocess.run(
        [str(python), str(CHECK)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_ambient_import_gate_accepts_clean_or_checkout_local_import(
    tmp_path: Path,
) -> None:
    python = _venv_python(tmp_path)
    clean = _run(python)
    assert clean.returncode == 0, clean.stderr
    assert "ambient pstrain import:" in clean.stdout

    checkout_local = _run(python, pythonpath=ROOT)
    assert checkout_local.returncode == 0, checkout_local.stderr
    assert "checkout-local" in checkout_local.stdout


def test_ambient_import_gate_rejects_foreign_import(tmp_path: Path) -> None:
    python = _venv_python(tmp_path)
    package = tmp_path / "foreign" / "pstrain"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("\n", encoding="utf-8")

    result = _run(python, pythonpath=package.parent)

    assert result.returncode != 0
    assert "resolves outside this checkout" in result.stderr
    assert str((package / "__init__.py").resolve()) in result.stderr
