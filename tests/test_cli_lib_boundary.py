"""Tests for the CLI-to-lib boundary ratchet."""

import builtins
import concurrent.futures
import contextlib
import importlib.util
import inspect
import multiprocessing
import pickle
import pkgutil
import re
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

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


@contextlib.contextmanager
def _boundary_module(
    name: str, filename: Path, source: str, **names: object
) -> Iterator[ModuleType]:
    """Register a module the guard can authenticate as boundary code.

    The guard resolves ownership through ``sys.modules`` and the module's source
    location, so a probe that only sets ``__name__`` no longer counts. Tests that
    assert an accepted route must therefore build a real module object.
    """
    package = ".".join(name.split(".")[:2])
    # A real submodule frame always has its parent package loaded, and the guard
    # resolves the package directory through it.
    importlib.import_module(package)
    module = ModuleType(name)
    module.__file__ = filename.as_posix()
    if "." in name:
        module.__package__ = name.rsplit(".", 1)[0]
    module.__dict__.update(names)
    exec(compile(source, filename.as_posix(), "exec"), module.__dict__)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield module
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def _run_from_cli(source: str, filename: Path, **names: object) -> dict[str, object]:
    namespace: dict[str, object] = {
        "__name__": f"pstrain.cli.{filename.stem}",
        "__package__": "pstrain.cli",
        **names,
    }
    exec(compile(source, filename.as_posix(), "exec"), namespace)
    return namespace


def test_runtime_guard_accepts_continuous_api_route_from_cli() -> None:
    with _boundary_module(
        "pstrain.api.runtime_boundary_probe",
        ROOT / "pstrain" / "api" / "runtime_boundary_probe.py",
        "def import_through_api():\n    import pstrain.lib.bw\n",
    ) as api:
        _run_from_cli(
            "import_through_api()\n",
            ROOT / "pstrain" / "cli" / "runtime_boundary_probe.py",
            import_through_api=api.import_through_api,
        )


def test_runtime_guard_rejects_a_module_that_only_claims_an_api_name() -> None:
    """A neutral module setting ``__name__`` is not the module it names."""
    importlib.import_module("pstrain.lib.bw")
    spoofed = "pstrain.api.not_a_real_module"
    assert spoofed not in sys.modules
    namespace = {"__name__": spoofed, "importlib": importlib}
    exec(
        compile(
            'def reach_lib():\n    return importlib.import_module("pstrain.lib.bw")\n',
            (ROOT / "boundary_spoof_probe.py").as_posix(),
            "exec",
        ),
        namespace,
    )

    with pytest.raises(
        cli_lib_boundary_guard.CliLibBoundaryViolation,
        match=rf"non-boundary frame {re.escape(spoofed)} at boundary_spoof_probe\.py:2",
    ):
        _run_from_cli(
            "reach_lib()\n",
            ROOT / "pstrain" / "cli" / "runtime_spoof_probe.py",
            reach_lib=namespace["reach_lib"],
        )


def test_runtime_guard_rejects_an_api_named_module_defined_outside_the_package() -> None:
    """Registration is not enough; the frame's source must be in the package."""
    importlib.import_module("pstrain.lib.bw")
    with (
        _boundary_module(
            "pstrain.api.runtime_outside_probe",
            ROOT / "boundary_outside_probe.py",
            'def reach_lib():\n    return importlib.import_module("pstrain.lib.bw")\n',
            importlib=importlib,
        ) as impostor,
        pytest.raises(
            cli_lib_boundary_guard.CliLibBoundaryViolation,
            match=r"non-boundary frame pstrain\.api\.runtime_outside_probe at "
            r"boundary_outside_probe\.py:2",
        ),
    ):
        _run_from_cli(
            "reach_lib()\n",
            ROOT / "pstrain" / "cli" / "runtime_outside_probe.py",
            reach_lib=impostor.reach_lib,
        )


_RECV_SOURCE = "def receive(connection):\n    return connection.recv()\n"
_PICKLE_SOURCE = "def serialize(value):\n    return ForkingPickler.dumps(value)\n"


def _pipe_naming_a_library_class() -> object:
    """Return a read end holding a protocol-0 pickle that names a library class."""
    reader, writer = multiprocessing.Pipe(duplex=False)
    writer.send_bytes(b"cpstrain.lib.bw\nBWConfig\n.")
    return reader


def test_runtime_guard_rejects_deserialization_choosing_the_import_below_the_api() -> None:
    """Incoming bytes, not the API, choose this target, so no route is established."""
    importlib.import_module("pstrain.lib.bw")
    with (
        _boundary_module(
            "pstrain.api.runtime_recv_probe",
            ROOT / "pstrain" / "api" / "runtime_recv_probe.py",
            _RECV_SOURCE,
        ) as api,
        pytest.raises(
            cli_lib_boundary_guard.CliLibBoundaryViolation,
            match=r"non-boundary frame multiprocessing\.connection",
        ),
    ):
        _run_from_cli(
            "receive(connection)\n",
            ROOT / "pstrain" / "cli" / "runtime_recv_probe.py",
            receive=api.receive,
            connection=_pipe_naming_a_library_class(),
        )


def test_runtime_guard_accepts_deserialization_inside_the_library() -> None:
    """A library frame unpickling a reply imports library code and crosses nothing."""
    importlib.import_module("pstrain.lib.bw")
    with (
        _boundary_module(
            "pstrain.lib.runtime_recv_probe",
            ROOT / "pstrain" / "lib" / "runtime_recv_probe.py",
            _RECV_SOURCE,
        ) as library,
        _boundary_module(
            "pstrain.api.runtime_recv_route_probe",
            ROOT / "pstrain" / "api" / "runtime_recv_route_probe.py",
            "def receive_through_api(receive, connection):\n    return receive(connection)\n",
        ) as api,
    ):
        namespace = _run_from_cli(
            "obtained = receive_through_api(receive, connection)\n",
            ROOT / "pstrain" / "cli" / "runtime_recv_route_probe.py",
            receive_through_api=api.receive_through_api,
            receive=library.receive,
            connection=_pipe_naming_a_library_class(),
        )
    assert namespace["obtained"].__module__ == "pstrain.lib.bw"


def test_runtime_guard_accepts_target_pickling_from_the_library() -> None:
    """Pickling re-imports the module of an object the library frame chose."""
    library_bw = importlib.import_module("pstrain.lib.bw")
    with (
        _boundary_module(
            "pstrain.lib.runtime_pickle_probe",
            ROOT / "pstrain" / "lib" / "runtime_pickle_probe.py",
            _PICKLE_SOURCE,
            ForkingPickler=multiprocessing.reduction.ForkingPickler,
        ) as library,
        _boundary_module(
            "pstrain.api.runtime_pickle_probe",
            ROOT / "pstrain" / "api" / "runtime_pickle_probe.py",
            "def serialize_through_api(serialize, value):\n    return serialize(value)\n",
        ) as api,
    ):
        namespace = _run_from_cli(
            "payload = serialize_through_api(serialize, value)\n",
            ROOT / "pstrain" / "cli" / "runtime_pickle_probe.py",
            serialize_through_api=api.serialize_through_api,
            serialize=library.serialize,
            value=library_bw.BWConfig,
        )
    assert pickle.loads(bytes(namespace["payload"])) is library_bw.BWConfig


def test_runtime_guard_rejects_target_pickling_from_a_neutral_module() -> None:
    """Serialization is trusted by identity of its caller, not of the pickler.

    The refused import surfaces as ``PicklingError``: the C pickler discards the
    failure from ``__import__`` and raises its own. The boundary crossing is
    still refused, but the violation type does not survive that conversion.
    """
    library_bw = importlib.import_module("pstrain.lib.bw")
    namespace = {
        "__name__": "boundary_pickle_probe",
        "ForkingPickler": multiprocessing.reduction.ForkingPickler,
    }
    exec(
        compile(_PICKLE_SOURCE, (ROOT / "boundary_pickle_probe.py").as_posix(), "exec"),
        namespace,
    )
    with (
        _boundary_module(
            "pstrain.api.runtime_neutral_pickle_probe",
            ROOT / "pstrain" / "api" / "runtime_neutral_pickle_probe.py",
            "def serialize_through_api(serialize, value):\n    return serialize(value)\n",
        ) as api,
        pytest.raises(
            pickle.PicklingError,
            match=r"import of module 'pstrain\.lib\.bw' failed",
        ),
    ):
        _run_from_cli(
            "serialize_through_api(serialize, value)\n",
            ROOT / "pstrain" / "cli" / "runtime_neutral_pickle_probe.py",
            serialize_through_api=api.serialize_through_api,
            serialize=namespace["serialize"],
            value=library_bw.BWConfig,
        )


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
    # The future re-raised it here, and the worker-thread record is now spent.
    assert cli_lib_boundary_guard.drain_escaped_violations()


_THREAD_PROBE_SOURCE = (
    "import threading\n"
    "def work():\n"
    '    importlib.import_module("pstrain.lib.bw")\n'
    "class Worker(threading.Thread):\n"
    "    def run(self):\n"
    '        importlib.import_module("pstrain.lib.bw")\n'
)


def _thread_probe_namespace(module_name: str, filename: Path) -> dict[str, object]:
    """Build a module that imports the library from a thread target and from ``run``."""
    namespace: dict[str, object] = {"__name__": module_name, "importlib": importlib}
    if "." in module_name:
        namespace["__package__"] = module_name.rsplit(".", 1)[0]
    exec(compile(_THREAD_PROBE_SOURCE, filename.as_posix(), "exec"), namespace)
    return namespace


def _run_thread_from_cli(source: str, **names: object) -> None:
    cli_filename = ROOT / "pstrain" / "cli" / "runtime_thread_probe.py"
    namespace: dict[str, object] = {
        "__name__": "pstrain.cli.runtime_thread_probe",
        "__package__": "pstrain.cli",
        "threading": threading,
        **names,
    }
    exec(compile(source, cli_filename.as_posix(), "exec"), namespace)


@pytest.mark.parametrize(
    ("source", "argument"),
    [
        ("thread = threading.Thread(target=work)\nthread.start()\nthread.join()\n", "work"),
        ("thread = Worker()\nthread.start()\nthread.join()\n", "Worker"),
    ],
    ids=["thread-target", "run-override"],
)
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_runtime_guard_rejects_thread_dispatch_from_a_neutral_module(
    source: str, argument: str
) -> None:
    probe = _thread_probe_namespace("boundary_thread_probe", ROOT / "boundary_thread_probe.py")

    # A bare thread prints and discards the violation, so the caller sees nothing.
    _run_thread_from_cli(source, **{argument: probe[argument]})

    with pytest.raises(AssertionError, match="non-boundary frame boundary_thread_probe"):
        cli_lib_boundary_guard.assert_no_escaped_violations()


@pytest.mark.parametrize(
    ("source", "argument"),
    [
        ("thread = threading.Thread(target=work)\nthread.start()\nthread.join()\n", "work"),
        ("thread = Worker()\nthread.start()\nthread.join()\n", "Worker"),
    ],
    ids=["thread-target", "run-override"],
)
def test_runtime_guard_accepts_thread_dispatch_through_the_public_api(
    source: str, argument: str
) -> None:
    with _boundary_module(
        "pstrain.api.runtime_thread_probe",
        ROOT / "pstrain" / "api" / "runtime_thread_probe.py",
        _THREAD_PROBE_SOURCE,
        importlib=importlib,
    ) as probe:
        _run_thread_from_cli(source, **{argument: getattr(probe, argument)})

    cli_lib_boundary_guard.assert_no_escaped_violations()


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


def _public_api_modules() -> list[ModuleType]:
    package = importlib.import_module("pstrain.api")
    modules = [package]
    for info in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}."):
        if any(part.startswith("_") for part in info.name.split(".")):
            continue
        modules.append(importlib.import_module(info.name))
    return modules


def _parameter_shape(function: Callable[..., object]) -> list[tuple[str, object, object]]:
    """Return the parameters that callers depend on, ignoring annotations."""
    return [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(function).parameters.values()
    ]


def _api_forwarders() -> list[tuple[str, Callable[..., object], Callable[..., object]]]:
    """Find public API functions written to put an API frame before a library call.

    The runtime rule needs an API frame, not merely an API name, so the public
    API cannot reach the command line's lazily importing work through a bare
    re-export. Each such name is a function that forwards to a private alias of
    the library callable it fronts.
    """
    forwarders: list[tuple[str, Callable[..., object], Callable[..., object]]] = []
    for module in _public_api_modules():
        for name in getattr(module, "__all__", ()):
            wrapper = getattr(module, name, None)
            target = getattr(module, f"_{name}", None)
            if (
                inspect.isfunction(wrapper)
                and wrapper.__module__ == module.__name__
                and callable(target)
            ):
                forwarders.append((f"{module.__name__}.{name}", wrapper, target))
    return forwarders


def test_public_api_forwarders_keep_their_library_signatures() -> None:
    forwarders = _api_forwarders()
    assert forwarders, "no public API forwarders were discovered"

    drifted = {
        name: (_parameter_shape(wrapper), _parameter_shape(target))
        for name, wrapper, target in forwarders
        if _parameter_shape(wrapper) != _parameter_shape(target)
    }

    assert not drifted, "\n".join(
        f"{name} forwards {wrapper} but its target takes {target}"
        for name, (wrapper, target) in sorted(drifted.items())
    )


def test_public_pipeline_forwarders_keep_their_library_signatures() -> None:
    api_pipeline = importlib.import_module("pstrain.api.pipeline")
    lib_pipeline = importlib.import_module("pstrain.lib.pipeline")

    run_pipeline = _parameter_shape(api_pipeline.run_pipeline)
    pipeline_run = _parameter_shape(lib_pipeline.Pipeline.run)
    assert run_pipeline[0][0] == "pipeline"
    assert pipeline_run[0][0] == "self"
    assert run_pipeline[1:] == pipeline_run[1:]

    assert _parameter_shape(api_pipeline.PipelineContext.from_config) == _parameter_shape(
        lib_pipeline.PipelineContext.from_config
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
