"""Tests for the CLI-to-lib boundary ratchet."""

import builtins
import concurrent.futures
import importlib.util
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from tests import cli_lib_boundary_guard

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
        ("from pstrain import lib", "pstrain.cli", {"pstrain.lib"}),
        ("import pstrain\npstrain.lib", "pstrain.cli", {"pstrain.lib"}),
        ("import pstrain as package\npackage.lib.bw", "pstrain.cli", {"pstrain.lib"}),
        ("pstrain.lib", "pstrain.cli", set()),
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


def test_static_scan_catches_cached_library_attribute_after_root_import() -> None:
    library = importlib.import_module("pstrain.lib")
    source = "import pstrain as package\nassert package.lib is expected\n"
    assert boundary.discover_edges(source, "pstrain.cli") == {"pstrain.lib"}

    filename = ROOT / "pstrain" / "cli" / "runtime_import_order_probe.py"
    namespace = {
        "__name__": "pstrain.cli.runtime_import_order_probe",
        "__package__": "pstrain.cli",
        "expected": library,
    }
    # There is no lib import request here for the runtime wrappers to observe.
    exec(compile(source, filename.as_posix(), "exec"), namespace)


@pytest.mark.parametrize(
    ("source", "transitive_source", "target", "import_location"),
    [
        (
            "from pstrain import lib\nlib.bw\n",
            None,
            "pstrain.lib",
            "pstrain/cli/runtime_boundary_probe.py:1",
        ),
        (
            'import importlib\nname = "bw"\nimportlib.import_module("pstrain.lib." + name)\n',
            None,
            "pstrain.lib.bw",
            "pstrain/cli/runtime_boundary_probe.py:3",
        ),
        (
            "import boundary_transitive_probe\n",
            "import pstrain.lib.bw\n",
            "pstrain.lib.bw",
            "boundary_transitive_probe.py:1",
        ),
    ],
    ids=["package-attribute", "computed-name", "transitive-import"],
)
def test_runtime_guard_rejects_cli_lib_import_constructions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    transitive_source: str | None,
    target: str,
    import_location: str,
) -> None:
    # Preload the target to prove sys.modules caching does not bypass the guard.
    importlib.import_module(target)
    if transitive_source is not None:
        (tmp_path / "boundary_transitive_probe.py").write_text(transitive_source, encoding="utf-8")
        monkeypatch.syspath_prepend(tmp_path)

    filename = ROOT / "pstrain" / "cli" / "runtime_boundary_probe.py"
    namespace = {"__name__": "pstrain.cli.runtime_boundary_probe", "__package__": "pstrain.cli"}
    with pytest.raises(
        cli_lib_boundary_guard.CliLibBoundaryViolation,
        match=(
            rf"CLI-to-library boundary violation: {re.escape(target)} imported at "
            rf".*{re.escape(import_location)}; "
        ),
    ):
        exec(compile(source, filename.as_posix(), "exec"), namespace)


def test_runtime_guard_accepts_continuous_api_route_from_cli() -> None:
    api_filename = ROOT / "pstrain" / "api" / "runtime_boundary_probe.py"
    api_namespace = {
        "__name__": "pstrain.api.runtime_boundary_probe",
        "__package__": "pstrain.api",
    }
    exec(
        compile(
            "def import_through_api():\n    import pstrain.lib.bw\n",
            api_filename.as_posix(),
            "exec",
        ),
        api_namespace,
    )
    filename = ROOT / "pstrain" / "cli" / "runtime_boundary_probe.py"
    namespace = {
        "__name__": "pstrain.cli.runtime_boundary_probe",
        "__package__": "pstrain.cli",
        "import_through_api": api_namespace["import_through_api"],
    }
    exec(compile("import_through_api()\n", filename.as_posix(), "exec"), namespace)


def test_runtime_guard_rejects_api_callback_laundering() -> None:
    callback_filename = ROOT / "boundary_callback_probe.py"
    callback_namespace = {"__name__": "boundary_callback_probe", "importlib": importlib}
    exec(
        compile(
            'def callback():\n    importlib.import_module("pstrain.lib.bw")\n',
            callback_filename.as_posix(),
            "exec",
        ),
        callback_namespace,
    )
    api_filename = ROOT / "pstrain" / "api" / "runtime_callback_probe.py"
    api_namespace = {"__name__": "pstrain.api.runtime_callback_probe"}
    exec(
        compile("def invoke(callback):\n    callback()\n", api_filename.as_posix(), "exec"),
        api_namespace,
    )
    cli_filename = ROOT / "pstrain" / "cli" / "runtime_callback_probe.py"
    cli_namespace = {
        "__name__": "pstrain.cli.runtime_callback_probe",
        "callback": callback_namespace["callback"],
        "invoke": api_namespace["invoke"],
    }

    with pytest.raises(
        cli_lib_boundary_guard.CliLibBoundaryViolation,
        match="non-boundary frame boundary_callback_probe",
    ):
        exec(compile("invoke(callback)\n", cli_filename.as_posix(), "exec"), cli_namespace)


def test_runtime_guard_rejects_generator_resumed_by_api() -> None:
    generator_filename = ROOT / "boundary_generator_probe.py"
    generator_namespace = {"__name__": "boundary_generator_probe", "importlib": importlib}
    exec(
        compile(
            'def values():\n    importlib.import_module("pstrain.lib.bw")\n    yield 1\n',
            generator_filename.as_posix(),
            "exec",
        ),
        generator_namespace,
    )
    api_filename = ROOT / "pstrain" / "api" / "runtime_generator_probe.py"
    api_namespace = {"__name__": "pstrain.api.runtime_generator_probe"}
    exec(
        compile("def consume(values):\n    return list(values)\n", api_filename.as_posix(), "exec"),
        api_namespace,
    )
    cli_filename = ROOT / "pstrain" / "cli" / "runtime_generator_probe.py"
    cli_namespace = {
        "__name__": "pstrain.cli.runtime_generator_probe",
        "consume": api_namespace["consume"],
        "values": generator_namespace["values"],
    }

    with pytest.raises(
        cli_lib_boundary_guard.CliLibBoundaryViolation,
        match="non-boundary frame boundary_generator_probe",
    ):
        exec(compile("consume(values())\n", cli_filename.as_posix(), "exec"), cli_namespace)


def test_runtime_guard_rejects_executor_dispatch_from_neutral_helper() -> None:
    helper_filename = ROOT / "boundary_executor_probe.py"
    helper_namespace = {"__name__": "boundary_executor_probe", "importlib": importlib}
    exec(
        compile(
            "def import_lib():\n"
            '    return importlib.import_module("pstrain.lib.bw")\n'
            "def submit(executor):\n"
            "    return executor.submit(import_lib).result()\n",
            helper_filename.as_posix(),
            "exec",
        ),
        helper_namespace,
    )
    cli_filename = ROOT / "pstrain" / "cli" / "runtime_executor_probe.py"
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(lambda: None).result()  # Ensure the worker predates CLI provenance.
        cli_namespace = {
            "__name__": "pstrain.cli.runtime_executor_probe",
            "executor": executor,
            "submit": helper_namespace["submit"],
        }
        with pytest.raises(
            cli_lib_boundary_guard.CliLibBoundaryViolation,
            match="non-boundary frame boundary_executor_probe",
        ):
            exec(compile("submit(executor)\n", cli_filename.as_posix(), "exec"), cli_namespace)


def test_runtime_guard_rejects_first_import_from_preloaded_library_function(
    tmp_path: Path,
) -> None:
    importlib.import_module("pstrain.lib.filetypes")
    filetypes = sys.modules["pstrain.lib.filetypes"]
    validate_file_type = filetypes.validate_file_type
    feature_type = filetypes.FileType.FEATURES
    feature_path = tmp_path / "preloaded.mfc"
    feature_path.write_bytes((1).to_bytes(4, "little", signed=True) + b"\0" * 4)
    cli_filename = ROOT / "pstrain" / "cli" / "runtime_preloaded_probe.py"
    cli_namespace = {
        "__name__": "pstrain.cli.runtime_preloaded_probe",
        "feature_path": feature_path,
        "feature_type": feature_type,
        "validate_file_type": validate_file_type,
    }

    with pytest.raises(
        cli_lib_boundary_guard.CliLibBoundaryViolation,
        match=r"pstrain\.lib\.features imported at pstrain/lib/filetypes\.py:299; "
        r"no pstrain\.api frame",
    ):
        exec(
            compile(
                "validate_file_type(feature_path, feature_type, deep=True)\n",
                cli_filename.as_posix(),
                "exec",
            ),
            cli_namespace,
        )


def test_runtime_guard_accepts_lazy_import_through_public_api(tmp_path: Path) -> None:
    api = importlib.import_module("pstrain.api")
    feature_path = tmp_path / "public.mfc"
    feature_path.write_bytes((1).to_bytes(4, "little", signed=True) + b"\0" * 4)
    cli_filename = ROOT / "pstrain" / "cli" / "runtime_public_probe.py"
    cli_namespace = {
        "__name__": "pstrain.cli.runtime_public_probe",
        "feature_path": feature_path,
        "feature_type": api.FileType.FEATURES,
        "validate_file_type": api.validate_file_type,
    }

    exec(
        compile(
            "result = validate_file_type(feature_path, feature_type, deep=True)\n",
            cli_filename.as_posix(),
            "exec",
        ),
        cli_namespace,
    )
    assert cli_namespace["result"] == (
        False,
        "Failed to load: Cannot determine veclen for 1 floats",
    )


@pytest.mark.parametrize(
    ("owner", "attribute", "label"),
    [
        (builtins, "__import__", "builtins.__import__"),
        (importlib, "import_module", "importlib.import_module"),
        (threading.Thread, "start", "threading.Thread.start"),
        (
            concurrent.futures.ThreadPoolExecutor,
            "submit",
            "ThreadPoolExecutor.submit",
        ),
    ],
)
def test_runtime_guard_detects_lost_wrapper_ownership(
    owner: object, attribute: str, label: str
) -> None:
    wrapper = getattr(owner, attribute)
    setattr(owner, attribute, lambda *args, **kwargs: None)
    try:
        with pytest.raises(AssertionError, match=re.escape(label)):
            cli_lib_boundary_guard.assert_installed()
    finally:
        setattr(owner, attribute, wrapper)


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


def test_cli_lib_boundary_allowlist_is_empty() -> None:
    assert boundary.read_allowlist() == set()
