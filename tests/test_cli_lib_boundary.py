"""Tests for the CLI-to-lib boundary ratchet."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_cli_lib_boundary", ROOT / "scripts" / "check_cli_lib_boundary.py"
)
assert SPEC is not None and SPEC.loader is not None
boundary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(boundary)


@pytest.mark.parametrize(
    ("source", "package", "expected"),
    [
        ("import pstrain.lib.x", "pstrain.cli", {"pstrain.lib.x"}),
        ("from pstrain.lib.x import a", "pstrain.cli", {"pstrain.lib.x"}),
        (
            "from pstrain.lib import a, b",
            "pstrain.cli",
            {"pstrain.lib.a", "pstrain.lib.b"},
        ),
        ("from ..lib.x import a", "pstrain.cli", {"pstrain.lib.x"}),
        ("from ..lib import a", "pstrain.cli", {"pstrain.lib.a"}),
        ("from .. import lib", "pstrain.cli", {"pstrain.lib"}),
        (
            'import importlib\nimportlib.import_module("pstrain.lib.paths")',
            "pstrain.cli",
            {"pstrain.lib.paths"},
        ),
        (
            'import importlib.util\nimportlib.import_module("pstrain.lib.paths")',
            "pstrain.cli",
            {"pstrain.lib.paths"},
        ),
        (
            'import importlib as il\nil.import_module("pstrain.lib.paths")',
            "pstrain.cli",
            {"pstrain.lib.paths"},
        ),
        (
            "import importlib as first\n"
            "import importlib as second\n"
            'first.import_module("pstrain.lib.paths")\n'
            'second.import_module("pstrain.lib.config")',
            "pstrain.cli",
            {"pstrain.lib.paths", "pstrain.lib.config"},
        ),
        (
            'il.import_module("pstrain.lib.paths")\nimport importlib as il',
            "pstrain.cli",
            {"pstrain.lib.paths"},
        ),
        (
            'from importlib import import_module\nimport_module("pstrain.lib.paths")',
            "pstrain.cli",
            {"pstrain.lib.paths"},
        ),
        (
            'from importlib import import_module as im\nim("pstrain.lib.paths")',
            "pstrain.cli",
            {"pstrain.lib.paths"},
        ),
        ('__import__("pstrain.lib.paths")', "pstrain.cli", {"pstrain.lib.paths"}),
        ('client.import_module("pstrain.lib.paths")', "pstrain.cli", set()),
        ('import_module("pstrain.lib.paths")', "pstrain.cli", set()),
        (
            'def import_module(name):\n    return name\nimport_module("pstrain.lib.paths")',
            "pstrain.cli",
            set(),
        ),
        ('obj.__import__("pstrain.lib.paths")', "pstrain.cli", set()),
        ('__import__ = print\n__import__("pstrain.lib.paths")', "pstrain.cli", set()),
        (
            'import importlib\nimportlib.import_module("pstrain." + name)',
            "pstrain.cli",
            set(),
        ),
    ],
)
def test_discover_edges(source: str, package: str, expected: set[str]) -> None:
    assert boundary.discover_edges(source, package) == expected


def test_find_imports_scans_nested_cli_packages(tmp_path: Path) -> None:
    nested = tmp_path / "pstrain" / "cli" / "admin"
    nested.mkdir(parents=True)
    (nested / "__init__.py").write_text("from ...lib import config\n", encoding="utf-8")
    (nested / "command.py").write_text("from ...lib.paths import get_paths\n", encoding="utf-8")

    assert boundary.find_imports(tmp_path / "pstrain" / "cli", tmp_path) == {
        "pstrain/cli/admin/__init__.py::pstrain.lib.config",
        "pstrain/cli/admin/command.py::pstrain.lib.paths",
    }


def test_dynamic_import_bindings_do_not_leak_across_files(tmp_path: Path) -> None:
    cli_dir = tmp_path / "pstrain" / "cli"
    cli_dir.mkdir(parents=True)
    (cli_dir / "bound.py").write_text("import importlib as il\n", encoding="utf-8")
    (cli_dir / "unbound.py").write_text('il.import_module("pstrain.lib.paths")\n', encoding="utf-8")

    assert boundary.find_imports(cli_dir, tmp_path) == set()


@pytest.mark.parametrize(
    ("source", "allowlist_entry", "message"),
    [
        ("import pstrain.lib.new\n", "", "NEW:"),
        ("", "pstrain/cli/example.py::pstrain.lib.old\n", "STALE:"),
    ],
)
def test_ratchet_rejects_new_and_stale_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    source: str,
    allowlist_entry: str,
    message: str,
) -> None:
    cli_dir = tmp_path / "pstrain" / "cli"
    cli_dir.mkdir(parents=True)
    (cli_dir / "example.py").write_text(source, encoding="utf-8")
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text(allowlist_entry, encoding="utf-8")
    monkeypatch.setattr(boundary, "ROOT", tmp_path)
    monkeypatch.setattr(boundary, "CLI_DIR", cli_dir)
    monkeypatch.setattr(boundary, "ALLOWLIST", allowlist)

    assert boundary.main() == 1
    assert message in capsys.readouterr().err


def test_cli_lib_boundary_gate_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_cli_lib_boundary.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
