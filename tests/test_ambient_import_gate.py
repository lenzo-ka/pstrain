from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts" / "check_ambient_import.py"


def _run(*, pythonpath: Path | None = None) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    if pythonpath is not None:
        environment["PYTHONPATH"] = str(pythonpath)
    return subprocess.run(
        [sys.executable, str(CHECK)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_ambient_import_gate_accepts_clean_or_checkout_local_import() -> None:
    clean = _run()
    assert clean.returncode == 0, clean.stderr
    assert "ambient pstrain import:" in clean.stdout

    checkout_local = _run(pythonpath=ROOT)
    assert checkout_local.returncode == 0, checkout_local.stderr
    assert "checkout-local" in checkout_local.stdout


def test_ambient_import_gate_rejects_foreign_import(tmp_path: Path) -> None:
    package = tmp_path / "pstrain"
    package.mkdir()
    (package / "__init__.py").write_text("\n", encoding="utf-8")

    result = _run(pythonpath=tmp_path)

    assert result.returncode != 0
    assert "resolves outside this checkout" in result.stderr
    assert str((package / "__init__.py").resolve()) in result.stderr
