"""Tests for the CLI-to-lib boundary ratchet."""

import builtins
import concurrent.futures
import contextlib
import dataclasses
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

    with pytest.raises(
        cli_lib_boundary_guard.CliLibBoundaryViolation,
        match=(
            rf"CLI-to-library boundary violation: {re.escape(target)} imported at "
            rf".*{re.escape(import_location)}; "
        ),
    ):
        _run_from_cli(source, ROOT / "pstrain" / "cli" / "runtime_boundary_probe.py")


@contextlib.contextmanager
def _anchored_module(
    name: str, filename: Path, source: str, **names: object
) -> Iterator[ModuleType]:
    """Register a module the guard can authenticate as code of its package.

    The guard establishes every role it reasons about -- command line, public
    API, library -- the same way: the live ``sys.modules`` entry for the frame's
    ``__name__`` must own that frame's globals, and the frame's code must come
    from a file under the directory frozen for that package at installation. A
    probe that only sets ``__name__`` is therefore an ordinary neutral frame, so
    a test that needs a genuine frame of any of the three roles must build a
    real module object under the real directory.
    """
    package = ".".join(name.split(".")[:2])
    assert package in {"pstrain.api", "pstrain.cli", "pstrain.lib"}
    module = ModuleType(name)
    module.__file__ = filename.as_posix()
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


@contextlib.contextmanager
def _cli_module(filename: Path, **names: object) -> Iterator[ModuleType]:
    """Register a genuine command-line module for ``filename``."""
    with _anchored_module(f"pstrain.cli.{filename.stem}", filename, "", **names) as module:
        yield module


def _neutral_namespace(
    filename: Path, source: str, *, module_name: str | None = None, **names: object
) -> dict[str, object]:
    """Build a namespace the guard cannot authenticate as any package's code.

    ``module_name`` sets the ``__name__`` the namespace claims. Nothing is
    registered under it, so whatever it claims, the namespace is neutral.
    """
    namespace: dict[str, object] = {"__name__": module_name or filename.stem, **names}
    exec(compile(source, filename.as_posix(), "exec"), namespace)
    return namespace


def _run_from_cli(source: str, filename: Path, **names: object) -> dict[str, object]:
    """Execute ``source`` in a genuine command-line frame and return its namespace."""
    with _cli_module(filename, **names) as module:
        exec(compile(source, filename.as_posix(), "exec"), module.__dict__)
        return dict(module.__dict__)


def test_runtime_guard_accepts_continuous_api_route_from_cli() -> None:
    with _anchored_module(
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
        _anchored_module(
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


def test_runtime_guard_anchors_every_role_to_the_checkout() -> None:
    """The roles are anchored to the checkout, not to a live ``__path__``."""
    assert cli_lib_boundary_guard.anchored_directories() == {
        "pstrain.cli": ROOT / "pstrain" / "cli",
        "pstrain.api": ROOT / "pstrain" / "api",
        "pstrain.lib": ROOT / "pstrain" / "lib",
    }


_REACH_LIB_SOURCE = (
    'import importlib\n\n\ndef reach_lib():\n    return importlib.import_module("pstrain.lib.bw")\n'
)


def test_runtime_guard_ignores_a_rewritten_package_path(tmp_path: Path) -> None:
    """A package's ``__path__`` is mutable, so the anchor cannot be read from it.

    Pointing ``pstrain.api.__path__`` at a directory of the caller's own and
    loading a real module from there once made that module authenticate as
    public API code: the module owned its own globals and its source lay under
    the anchor it had just moved. The anchors now come from the checkout root at
    installation, so the rewrite changes nothing in either direction -- the
    outside module is refused, and genuine API code is still accepted while the
    rewritten ``__path__`` is in place.
    """
    importlib.import_module("pstrain.lib.bw")
    api = importlib.import_module("pstrain.api")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    shadow_file = outside / "shadow.py"
    shadow_file.write_text(_REACH_LIB_SOURCE, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("pstrain.api.shadow", shadow_file)
    assert spec is not None and spec.loader is not None
    shadow = importlib.util.module_from_spec(spec)

    original_path = list(api.__path__)
    api.__path__ = [str(outside)]
    sys.modules["pstrain.api.shadow"] = shadow
    try:
        spec.loader.exec_module(shadow)
        with pytest.raises(
            cli_lib_boundary_guard.CliLibBoundaryViolation,
            match=r"non-boundary frame pstrain\.api\.shadow at .*shadow\.py:5",
        ):
            _run_from_cli(
                "reach_lib()\n",
                ROOT / "pstrain" / "cli" / "runtime_shadow_probe.py",
                reach_lib=shadow.reach_lib,
            )

        with _anchored_module(
            "pstrain.api.runtime_shadow_control_probe",
            ROOT / "pstrain" / "api" / "runtime_shadow_control_probe.py",
            "def import_through_api():\n    import pstrain.lib.bw\n",
        ) as control:
            _run_from_cli(
                "import_through_api()\n",
                ROOT / "pstrain" / "cli" / "runtime_shadow_control_probe.py",
                import_through_api=control.import_through_api,
            )
    finally:
        sys.modules.pop("pstrain.api.shadow", None)
        api.__path__ = original_path


_BRIDGE_SOURCE = "def bridge(importer):\n    return importer()\n"


@pytest.mark.parametrize(
    "claimed_name",
    ["boundary_neutral", "pstrain.cli.impostor"],
    ids=["neutral-name", "claimed-cli-name"],
)
def test_runtime_guard_treats_a_claimed_cli_name_as_a_neutral_frame(claimed_name: str) -> None:
    """Claiming a command-line name must not make the guard more permissive.

    Genuine command-line code calls this bridge, which calls an authenticated
    API importer. The bridge belongs to no package, so the segment is
    interrupted and the import is refused. A merely-claimed ``pstrain.cli`` name
    once ended the stack walk instead, which hid both the bridge and the real
    command-line frame beyond it and turned that refusal into an acceptance.
    Only ``__name__`` differs between these two cases, so both must be refused
    and both must name the real command-line frame as the origin.
    """
    importlib.import_module("pstrain.lib.bw")
    assert claimed_name not in sys.modules
    bridge = _neutral_namespace(
        ROOT / "boundary_bridge_probe.py", _BRIDGE_SOURCE, module_name=claimed_name
    )
    with (
        _anchored_module(
            "pstrain.api.runtime_bridge_probe",
            ROOT / "pstrain" / "api" / "runtime_bridge_probe.py",
            _REACH_LIB_SOURCE,
        ) as api,
        pytest.raises(
            cli_lib_boundary_guard.CliLibBoundaryViolation,
            match=(
                rf"non-boundary frame {re.escape(claimed_name)} at boundary_bridge_probe\.py:2 "
                r"interrupts the route "
                r"\(CLI origin pstrain/cli/runtime_bridge_probe\.py:1\)"
            ),
        ),
    ):
        _run_from_cli(
            "bridge(reach_lib)\n",
            ROOT / "pstrain" / "cli" / "runtime_bridge_probe.py",
            bridge=bridge["bridge"],
            reach_lib=api.reach_lib,
        )


_COMPUTED_IMPORT_SOURCE = (
    "import importlib\n\n\ndef reach_lib(target):\n    return importlib.import_module(target)\n"
)
_NO_API_FRAME = "no pstrain.api frame establishes the route"


def _assert_refused_without_api(reach_lib: Callable[[str], object], origin: str) -> None:
    with pytest.raises(
        cli_lib_boundary_guard.CliLibBoundaryViolation,
        match=rf"{re.escape(_NO_API_FRAME)} \(CLI origin {re.escape(origin)}\)",
    ):
        reach_lib("pstrain.lib.bw")


@pytest.mark.parametrize(
    "lose_identity",
    [
        lambda name: sys.modules.pop(name),
        lambda name: sys.modules.__setitem__(name, ModuleType(name)),
    ],
    ids=["entry-removed", "entry-replaced"],
)
def test_runtime_guard_enforces_command_line_code_that_lost_its_registration(
    lose_identity: Callable[[str], object],
) -> None:
    """Losing a live module identity must not switch enforcement off.

    A frame is authenticated from the live ``sys.modules`` entry for its
    ``__name__``. That entry is ordinary mutable process state, so genuine
    command-line code can lose it -- deleted, or replaced by another object --
    while the very same function object keeps running. Authentication then
    fails, and a stack with no recognized command-line frame once yielded no
    origin at all. With no origin there was no rule to apply, so the import was
    allowed: absence of provenance switched enforcement off rather than on. A
    computed target evades the static scan too, so nothing else caught it.

    The source location under the frozen command-line directory is now kept as
    a fallback origin, so the same call is refused either way.
    """
    importlib.import_module("pstrain.lib.bw")
    name = "pstrain.cli.runtime_identity_probe"
    filename = ROOT / "pstrain" / "cli" / "runtime_identity_probe.py"
    origin = "pstrain/cli/runtime_identity_probe.py:5"
    with _anchored_module(name, filename, _COMPUTED_IMPORT_SOURCE) as cli:
        reach_lib = cli.reach_lib
        _assert_refused_without_api(reach_lib, origin)
        lose_identity(name)
        assert getattr(sys.modules.get(name), "reach_lib", None) is not reach_lib
        _assert_refused_without_api(reach_lib, origin)


def test_runtime_guard_enforces_a_command_line_file_executed_without_registration() -> None:
    """A real command-line file run through a loader is never registered at all.

    ``exec_module`` on a module that was never placed in ``sys.modules`` gives
    genuine command-line code with no live module identity to authenticate
    against, which is the same loss of provenance as a removed or replaced
    entry.
    """
    importlib.import_module("pstrain.lib.bw")
    name = "pstrain.cli.runtime_loader_probe"
    filename = ROOT / "pstrain" / "cli" / "runtime_loader_probe.py"
    filename.write_text(_COMPUTED_IMPORT_SOURCE, encoding="utf-8")
    try:
        spec = importlib.util.spec_from_file_location(name, filename)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        assert name not in sys.modules
        spec.loader.exec_module(module)
        _assert_refused_without_api(module.reach_lib, "pstrain/cli/runtime_loader_probe.py:5")
    finally:
        filename.unlink(missing_ok=True)
        for cached in (filename.parent / "__pycache__").glob(f"{filename.stem}.*.pyc"):
            cached.unlink(missing_ok=True)


def test_runtime_guard_accepts_an_api_route_from_unregistered_command_line_code() -> None:
    """The fallback origin restores the rule; it does not refuse everything.

    Command-line code that has lost its registration is still accepted when it
    reaches the library through a genuine public API frame, so the fallback
    discriminates by route rather than by the loss of identity.
    """
    name = "pstrain.cli.runtime_identity_control_probe"
    filename = ROOT / "pstrain" / "cli" / "runtime_identity_control_probe.py"
    with _anchored_module(
        "pstrain.api.runtime_identity_control_probe",
        ROOT / "pstrain" / "api" / "runtime_identity_control_probe.py",
        _COMPUTED_IMPORT_SOURCE,
    ) as api:
        with _anchored_module(
            name, filename, "def call(reach_lib):\n    return reach_lib('pstrain.lib.bw')\n"
        ) as cli:
            call = cli.call
        assert name not in sys.modules
        call(api.reach_lib)


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
        _anchored_module(
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
        _anchored_module(
            "pstrain.lib.runtime_recv_probe",
            ROOT / "pstrain" / "lib" / "runtime_recv_probe.py",
            _RECV_SOURCE,
        ) as library,
        _anchored_module(
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
        _anchored_module(
            "pstrain.lib.runtime_pickle_probe",
            ROOT / "pstrain" / "lib" / "runtime_pickle_probe.py",
            _PICKLE_SOURCE,
            ForkingPickler=multiprocessing.reduction.ForkingPickler,
        ) as library,
        _anchored_module(
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
        _anchored_module(
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
    neutral = _neutral_namespace(
        ROOT / "boundary_callback_probe.py",
        'def callback():\n    importlib.import_module("pstrain.lib.bw")\n',
        importlib=importlib,
    )
    with (
        _anchored_module(
            "pstrain.api.runtime_callback_probe",
            ROOT / "pstrain" / "api" / "runtime_callback_probe.py",
            "def invoke(callback):\n    callback()\n",
        ) as api,
        pytest.raises(
            cli_lib_boundary_guard.CliLibBoundaryViolation,
            match="non-boundary frame boundary_callback_probe",
        ),
    ):
        _run_from_cli(
            "invoke(callback)\n",
            ROOT / "pstrain" / "cli" / "runtime_callback_probe.py",
            invoke=api.invoke,
            callback=neutral["callback"],
        )


def test_runtime_guard_rejects_generator_resumed_by_api() -> None:
    neutral = _neutral_namespace(
        ROOT / "boundary_generator_probe.py",
        'def values():\n    importlib.import_module("pstrain.lib.bw")\n    yield 1\n',
        importlib=importlib,
    )
    with (
        _anchored_module(
            "pstrain.api.runtime_generator_probe",
            ROOT / "pstrain" / "api" / "runtime_generator_probe.py",
            "def consume(values):\n    return list(values)\n",
        ) as api,
        pytest.raises(
            cli_lib_boundary_guard.CliLibBoundaryViolation,
            match="non-boundary frame boundary_generator_probe",
        ),
    ):
        _run_from_cli(
            "consume(values())\n",
            ROOT / "pstrain" / "cli" / "runtime_generator_probe.py",
            consume=api.consume,
            values=neutral["values"],
        )


def test_runtime_guard_rejects_executor_dispatch_from_neutral_helper() -> None:
    neutral = _neutral_namespace(
        ROOT / "boundary_executor_probe.py",
        "def import_lib():\n"
        '    return importlib.import_module("pstrain.lib.bw")\n'
        "def submit(executor):\n"
        "    return executor.submit(import_lib).result()\n",
        importlib=importlib,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(lambda: None).result()  # Ensure the worker predates CLI provenance.
        with pytest.raises(
            cli_lib_boundary_guard.CliLibBoundaryViolation,
            match="non-boundary frame boundary_executor_probe",
        ):
            _run_from_cli(
                "submit(executor)\n",
                ROOT / "pstrain" / "cli" / "runtime_executor_probe.py",
                submit=neutral["submit"],
                executor=executor,
            )
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


def _run_thread_from_cli(source: str, **names: object) -> None:
    _run_from_cli(
        source,
        ROOT / "pstrain" / "cli" / "runtime_thread_probe.py",
        threading=threading,
        **names,
    )


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
    probe = _neutral_namespace(
        ROOT / "boundary_thread_probe.py", _THREAD_PROBE_SOURCE, importlib=importlib
    )

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
    with _anchored_module(
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

    with pytest.raises(
        cli_lib_boundary_guard.CliLibBoundaryViolation,
        match=r"pstrain\.lib\.features imported at pstrain/lib/filetypes\.py:299; "
        r"no pstrain\.api frame",
    ):
        _run_from_cli(
            "validate_file_type(feature_path, feature_type, deep=True)\n",
            ROOT / "pstrain" / "cli" / "runtime_preloaded_probe.py",
            feature_path=feature_path,
            feature_type=feature_type,
            validate_file_type=validate_file_type,
        )


def test_runtime_guard_accepts_lazy_import_through_public_api(tmp_path: Path) -> None:
    api = importlib.import_module("pstrain.api")
    feature_path = tmp_path / "public.mfc"
    feature_path.write_bytes((1).to_bytes(4, "little", signed=True) + b"\0" * 4)

    namespace = _run_from_cli(
        "result = validate_file_type(feature_path, feature_type, deep=True)\n",
        ROOT / "pstrain" / "cli" / "runtime_public_probe.py",
        feature_path=feature_path,
        feature_type=api.FileType.FEATURES,
        validate_file_type=api.validate_file_type,
    )
    assert namespace["result"] == (
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


def test_public_api_forwarders_publish_their_library_documentation() -> None:
    """A forwarder exists for the guard; the published reference must not pay for it.

    ``automodule`` renders every name in a module's ``__all__``, so replacing a
    re-export with a forwarder would otherwise replace the library's parameter
    documentation with a one-line note about call frames.
    """
    forwarders = _api_forwarders()
    assert forwarders, "no public API forwarders were discovered"

    undocumented = {
        name: (wrapper.__doc__, target.__doc__)
        for name, wrapper, target in forwarders
        if target.__doc__ and wrapper.__doc__ != target.__doc__
    }

    assert not undocumented, "\n".join(
        f"{name} publishes {wrapper!r} instead of its library documentation {target!r}"
        for name, (wrapper, target) in sorted(undocumented.items())
    )


def test_public_pipeline_forwarders_keep_their_library_signatures() -> None:
    api_pipeline = importlib.import_module("pstrain.api.pipeline")
    lib_pipeline = importlib.import_module("pstrain.lib.pipeline")

    run_pipeline = _parameter_shape(api_pipeline.run_pipeline)
    pipeline_run = _parameter_shape(lib_pipeline.Pipeline.run)
    assert run_pipeline[0][0] == "pipeline"
    assert pipeline_run[0][0] == "self"
    assert run_pipeline[1:] == pipeline_run[1:]

    # ``from_config`` is a classmethod, so its bound signature drops ``cls``.
    assert _parameter_shape(api_pipeline.create_pipeline_context) == _parameter_shape(
        lib_pipeline.PipelineContext.from_config
    )


def test_public_pipeline_context_is_the_library_class() -> None:
    """The public name must stay the same object the library exports.

    A subclass would give the command line an API construction frame at the cost
    of the identity, equality, ``isinstance`` and pickle relationships callers
    already rely on. The API factory supplies that frame instead.
    """
    api_pipeline = importlib.import_module("pstrain.api.pipeline")
    lib_pipeline = importlib.import_module("pstrain.lib.pipeline")
    package = importlib.import_module("pstrain.api")

    assert api_pipeline.PipelineContext is lib_pipeline.PipelineContext
    assert api_pipeline.PipelineContext is lib_pipeline.context.PipelineContext
    assert getattr(package, "PipelineContext", api_pipeline.PipelineContext) is (
        lib_pipeline.PipelineContext
    )


def test_public_pipeline_context_keeps_its_value_relationships(tmp_path: Path) -> None:
    api_pipeline = importlib.import_module("pstrain.api.pipeline")
    lib_pipeline = importlib.import_module("pstrain.lib.pipeline")

    public = api_pipeline.PipelineContext(project_dir=tmp_path)
    library = lib_pipeline.PipelineContext(project_dir=tmp_path)

    assert public == library
    assert isinstance(library, api_pipeline.PipelineContext)
    assert isinstance(public, lib_pipeline.PipelineContext)
    assert dataclasses.fields(public) == dataclasses.fields(library)

    restored = pickle.loads(pickle.dumps(public))
    assert type(restored) is lib_pipeline.PipelineContext
    assert restored == library


def test_create_pipeline_context_returns_the_library_class(tmp_path: Path) -> None:
    api_pipeline = importlib.import_module("pstrain.api.pipeline")
    lib_pipeline = importlib.import_module("pstrain.lib.pipeline")

    context = api_pipeline.create_pipeline_context(tmp_path)

    assert type(context) is lib_pipeline.PipelineContext
    assert context == lib_pipeline.PipelineContext.from_config(tmp_path)


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
