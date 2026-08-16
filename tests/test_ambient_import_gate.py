from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts" / "check_ambient_import.py"


def _venv_python(tmp_path: Path) -> Path:
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    binary = "python.exe" if os.name == "nt" else "python"
    return venv / ("Scripts" if os.name == "nt" else "bin") / binary


def _run(
    python: Path, *, pythonpath: Path | list[Path] | None = None
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    if pythonpath is not None:
        paths = pythonpath if isinstance(pythonpath, list) else [pythonpath]
        environment["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    return subprocess.run(
        [str(python), str(CHECK)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _site_packages(python: Path) -> Path:
    result = subprocess.run(
        [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _copy_yaml_dependency(python: Path) -> None:
    spec = find_spec("yaml")
    assert spec is not None and spec.origin is not None
    shutil.copytree(Path(spec.origin).parent, _site_packages(python) / "yaml")


def test_ambient_import_gate_accepts_clean_or_checkout_local_import(
    tmp_path: Path,
) -> None:
    python = _venv_python(tmp_path)
    clean = _run(python)
    assert clean.returncode == 0, clean.stderr
    assert "ambient pstrain import:" in clean.stdout

    _copy_yaml_dependency(python)
    checkout_local = _run(python, pythonpath=ROOT)
    assert checkout_local.returncode == 0, checkout_local.stderr
    assert "checkout-local" in checkout_local.stdout


def test_ambient_import_gate_rejects_foreign_import(tmp_path: Path) -> None:
    python = _venv_python(tmp_path)
    package = tmp_path / "foreign" / "pstrain"
    benchmarks = package / "benchmarks"
    benchmarks.mkdir(parents=True)
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    (benchmarks / "__init__.py").write_text("\n", encoding="utf-8")
    (benchmarks / "arctic.py").write_text("\n", encoding="utf-8")

    result = _run(python, pythonpath=package.parent)

    assert result.returncode != 0
    assert "resolves outside this checkout" in result.stderr
    assert str((package / "__init__.py").resolve()) in result.stderr


def test_ambient_import_gate_uses_script_startup(tmp_path: Path) -> None:
    python = _venv_python(tmp_path)
    foreign = tmp_path / "foreign"
    package = foreign / "pstrain"
    benchmarks = package / "benchmarks"
    benchmarks.mkdir(parents=True)
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    (benchmarks / "__init__.py").write_text("\n", encoding="utf-8")
    (benchmarks / "arctic.py").write_text("\n", encoding="utf-8")
    site_packages = _site_packages(python)
    (site_packages / "argv_conditioned.pth").write_text(
        f"import sys; sys.path.insert(0, {str(foreign)!r}) if sys.argv[0] != '-c' else None\n",
        encoding="utf-8",
    )

    result = _run(python)

    assert result.returncode != 0
    assert str((package / "__init__.py").resolve()) in result.stderr


def test_ambient_import_gate_rejects_foreign_package_path(tmp_path: Path) -> None:
    python = _venv_python(tmp_path)
    foreign = tmp_path / "foreign" / "pstrain"
    benchmarks = foreign / "benchmarks"
    benchmarks.mkdir(parents=True)
    (benchmarks / "__init__.py").write_text("\n", encoding="utf-8")
    (benchmarks / "arctic.py").write_text("\n", encoding="utf-8")
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(
        f"import pstrain\npstrain.__path__.insert(0, {str(foreign)!r})\n",
        encoding="utf-8",
    )

    result = _run(python, pythonpath=[hook, ROOT])

    assert result.returncode != 0
    assert "offending pstrain.__path__" in result.stderr
    assert str(foreign.resolve()) in result.stderr
