"""Runtime observations for the CLI-to-library import boundary.

The guard wraps Python's two supported name-based import entry points for the
pytest session. Unlike a meta-path finder or audit hook, these entry points run
even when ``sys.modules`` already contains the target.

For an executed ``pstrain.lib`` import with CLI provenance, the application
stack segment from import to CLI must stay inside ``pstrain.api`` and
``pstrain.lib`` and must contain an API frame. Membership is established, not
inferred: a frame belongs to a boundary package only when the live
``sys.modules`` entry for its ``__name__`` owns that frame's globals and its
code was compiled from a file inside the package's real directory. A writable
``__name__`` alone proves nothing.

Only two kinds of infrastructure are transparent, both held by identity. The
import machinery is recognized by module-dictionary identity. A short list of
exact multiprocessing code objects covers serialization: pickling re-imports
the defining module of an object its caller already chose, so it is transparent
when an authenticated boundary frame invoked it, while unpickling takes its
target from the incoming bytes and is transparent only inside ``pstrain.lib``,
where the import it triggers crosses no boundary. Every other frame, the rest
of ``multiprocessing`` included, stays in the segment. This rejects callbacks
resumed below an API frame and library functions reached without one.

CLI provenance is copied into direct ``threading.Thread`` targets and
``ThreadPoolExecutor`` submissions so a worker cannot lose the origin merely by
losing the submitting stack. A violation raised on a worker thread is also
recorded, because a bare ``threading.Thread`` prints and discards it instead of
failing the session.

This is not a complete architectural proof. The runtime observation covers
only supported name-based imports executed in the pytest process after
installation. The companion AST scan covers ordinary CLI imports and access
through a directly imported ``pstrain`` alias, including paths tests do not
execute. Their remaining limits are documented in
``docs/design/cli-lib-boundary.md``.
"""

from __future__ import annotations

import builtins
import concurrent.futures
import contextlib
import functools
import importlib
import sys
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, FrameType
from typing import Any

_Callable = Callable[..., Any]

_ORIGINAL_IMPORT: _Callable | None = None
_ORIGINAL_IMPORT_MODULE: _Callable | None = None
_ORIGINAL_THREAD_START: _Callable | None = None
_ORIGINAL_THREAD_SUBMIT: _Callable | None = None
_PROJECT_ROOT: Path | None = None
_OBSERVED_ROUTES: Counter[str] = Counter()
_ESCAPED_VIOLATIONS: list[str] = []

_BOUNDARY_PACKAGES = ("pstrain.api", "pstrain.lib")
_PACKAGE_DIRECTORIES: dict[str, Path] = {}

# Import machinery implements the wrapped operation itself. Membership is by
# module-dictionary identity, not by name, so a module cannot join by writing
# its own ``__name__``.
_IMPORT_MACHINERY: tuple[Mapping[str, object], ...] = ()

# Multiprocessing frames that hand an object the caller already chose to a
# pickler, and the one frame that unpickles a reply. Both are held as exact
# code objects resolved at installation; every other multiprocessing frame is
# an ordinary segment blocker.
_PICKLING_SPECS: tuple[tuple[str, str | None, str], ...] = (
    ("multiprocessing.reduction", "ForkingPickler", "dumps"),
    ("multiprocessing.reduction", None, "dump"),
    ("multiprocessing.connection", "_ConnectionBase", "send"),
    ("multiprocessing.queues", "Queue", "_feed"),
)
_UNPICKLING_SPECS: tuple[tuple[str, str | None, str], ...] = (
    ("multiprocessing.connection", "_ConnectionBase", "recv"),
)
_PICKLING_CODE: frozenset[CodeType] = frozenset()
_UNPICKLING_CODE: frozenset[CodeType] = frozenset()


class CliLibBoundaryViolation(BaseException):
    """Escape ordinary application exception handlers and fail the test session."""


@dataclass(frozen=True)
class _DispatchProvenance:
    cli_origin: str
    api_reached_continuously: bool


def _is_package(module: str, package: str) -> bool:
    return module == package or module.startswith(f"{package}.")


def _absolute_name(name: str, globals_: Mapping[str, object] | None, level: int) -> str:
    if level == 0:
        return name
    package = globals_.get("__package__") if globals_ is not None else None
    if not isinstance(package, str) or not package:
        return name
    return importlib.util.resolve_name(f"{'.' * level}{name}", package)


def _builtin_targets(
    name: str,
    globals_: Mapping[str, object] | None,
    fromlist: Iterable[str] | None,
    level: int,
) -> set[str]:
    base = _absolute_name(name, globals_, level)
    targets = {base}
    targets.update(f"{base}.{item}" for item in (fromlist or ()) if item != "*")
    return {target for target in targets if _is_package(target, "pstrain.lib")}


def _frame_module(frame: FrameType) -> str:
    module = frame.f_globals.get("__name__", "")
    return module if isinstance(module, str) else ""


def _location(frame: FrameType) -> str:
    path = Path(frame.f_code.co_filename)
    if _PROJECT_ROOT is not None:
        with contextlib.suppress(OSError, ValueError):
            path = path.resolve().relative_to(_PROJECT_ROOT)
    return f"{path}:{frame.f_lineno}"


def _package_directory(package: str) -> Path | None:
    """Return the real on-disk directory of an imported boundary package."""
    cached = _PACKAGE_DIRECTORIES.get(package)
    if cached is not None:
        return cached
    locations = getattr(sys.modules.get(package), "__path__", None)
    if not locations:
        return None
    try:
        directory = Path(next(iter(locations))).resolve()
    except OSError:
        return None
    _PACKAGE_DIRECTORIES[package] = directory
    return directory


def _boundary_package(frame: FrameType | None) -> str | None:
    """Return the boundary package that provably owns ``frame``, if any.

    Ownership is not a name. ``__name__`` is an ordinary writable string, so a
    module that merely calls itself ``pstrain.api.something`` proves nothing. A
    frame counts as boundary code only when the live ``sys.modules`` entry for
    that name owns this exact globals mapping and the frame's code was compiled
    from a file inside the package's real directory.
    """
    if frame is None:
        return None
    name = frame.f_globals.get("__name__")
    if not isinstance(name, str):
        return None
    package = next((known for known in _BOUNDARY_PACKAGES if _is_package(name, known)), None)
    if package is None:
        return None
    module = sys.modules.get(name)
    if module is None or getattr(module, "__dict__", None) is not frame.f_globals:
        return None
    directory = _package_directory(package)
    if directory is None:
        return None
    try:
        Path(frame.f_code.co_filename).resolve().relative_to(directory)
    except (OSError, ValueError):
        return None
    return package


def _serialization_caller(frame: FrameType) -> FrameType | None:
    """Return the first frame outside a run of trusted serialization frames."""
    caller = frame.f_back
    while caller is not None and caller.f_code in _PICKLING_CODE:
        caller = caller.f_back
    return caller


def _is_transparent_frame(frame: FrameType) -> bool:
    """Identify infrastructure that cannot itself choose the imported target.

    This guard's own frames and the import machinery merely implement the
    wrapped operation; both are recognized by module-dictionary identity.

    Serialization is different in each direction. Pickling re-imports the
    defining module of an object the caller already chose, so it is transparent
    when an authenticated boundary frame invoked it. Unpickling takes its target
    from the incoming bytes, so it can never establish a boundary crossing; it
    is transparent only inside ``pstrain.lib``, where the import it triggers is
    library-to-library and crosses nothing. Every other frame, multiprocessing
    included, stays in the segment so callbacks cannot launder CLI provenance.
    """
    if any(frame.f_globals is machinery for machinery in _IMPORT_MACHINERY):
        return True
    if frame.f_globals is globals():
        return True
    code = frame.f_code
    if code in _PICKLING_CODE:
        return _boundary_package(_serialization_caller(frame)) is not None
    if code in _UNPICKLING_CODE:
        return _boundary_package(_serialization_caller(frame)) == "pstrain.lib"
    return False


def _run_dispatched(
    provenance: _DispatchProvenance,
    function: _Callable,
    *args: object,
    **kwargs: object,
) -> Any:
    # _stack_route recognizes this frame as the boundary between the submitted
    # callable and executor/thread infrastructure that did not run on the
    # submitting stack.
    return function(*args, **kwargs)


def _stack_route() -> tuple[tuple[FrameType, ...], _DispatchProvenance] | None:
    """Return frames between this operation and its physical or carried CLI origin."""
    frames: list[FrameType] = []
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code is _run_dispatched.__code__:
            provenance = frame.f_locals.get("provenance")
            assert isinstance(provenance, _DispatchProvenance)
            return tuple(frames), provenance
        if _is_transparent_frame(frame):
            frame = frame.f_back
            continue
        module = _frame_module(frame)
        if _is_package(module, "pstrain.cli"):
            return tuple(frames), _DispatchProvenance(_location(frame), False)
        frames.append(frame)
        frame = frame.f_back
    return None


def _segment_state(
    frames: Iterable[FrameType], inherited_api: bool
) -> tuple[bool, FrameType | None]:
    api_reached = inherited_api
    blocker: FrameType | None = None
    for frame in frames:
        package = _boundary_package(frame)
        if package == "pstrain.api":
            api_reached = True
        elif package is None:
            blocker = frame
            break
    return api_reached and blocker is None, blocker


def _capture_dispatch_provenance() -> _DispatchProvenance | None:
    route = _stack_route()
    if route is None:
        return None
    frames, origin = route
    continuous, _ = _segment_state(frames, origin.api_reached_continuously)
    return _DispatchProvenance(origin.cli_origin, continuous)


def _check_targets(targets: Iterable[str]) -> None:
    targets = sorted(set(targets))
    if not targets:
        return
    route = _stack_route()
    if route is None:
        return
    frames, provenance = route
    continuous, blocker = _segment_state(frames, provenance.api_reached_continuously)
    target = targets[0]
    importer = _location(frames[0]) if frames else provenance.cli_origin
    if not continuous:
        if blocker is None:
            reason = "no pstrain.api frame establishes the route"
        else:
            module = _frame_module(blocker) or "<unknown>"
            reason = f"non-boundary frame {module} at {_location(blocker)} interrupts the route"
        message = (
            "CLI-to-library boundary violation: "
            f"{target} imported at {importer}; {reason} "
            f"(CLI origin {provenance.cli_origin})"
        )
        if threading.current_thread() is not threading.main_thread():
            # Nothing re-raises an exception that escapes a worker thread, so
            # record it for the lifecycle check that runs on the main thread.
            _ESCAPED_VIOLATIONS.append(message)
        raise CliLibBoundaryViolation(message)
    _OBSERVED_ROUTES[f"{target} via continuous pstrain.api route at {importer}"] += 1


def _guarded_import(
    name: str,
    globals: Mapping[str, object] | None = None,
    locals: Mapping[str, object] | None = None,
    fromlist: Iterable[str] | None = (),
    level: int = 0,
) -> Any:
    _check_targets(_builtin_targets(name, globals, fromlist, level))
    assert _ORIGINAL_IMPORT is not None
    return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)


def _guarded_import_module(name: str, package: str | None = None) -> Any:
    target = importlib.util.resolve_name(name, package) if name.startswith(".") else name
    _check_targets({target} if _is_package(target, "pstrain.lib") else set())
    assert _ORIGINAL_IMPORT_MODULE is not None
    return _ORIGINAL_IMPORT_MODULE(name, package)


def _guarded_thread_submit(
    executor: concurrent.futures.ThreadPoolExecutor,
    function: _Callable,
    /,
    *args: object,
    **kwargs: object,
) -> concurrent.futures.Future[Any]:
    assert _ORIGINAL_THREAD_SUBMIT is not None
    provenance = _capture_dispatch_provenance()
    if provenance is None:
        return _ORIGINAL_THREAD_SUBMIT(executor, function, *args, **kwargs)
    dispatched = functools.partial(_run_dispatched, provenance, function)
    return _ORIGINAL_THREAD_SUBMIT(executor, dispatched, *args, **kwargs)


def _guarded_thread_start(thread: threading.Thread, *args: object, **kwargs: object) -> Any:
    assert _ORIGINAL_THREAD_START is not None
    provenance = _capture_dispatch_provenance()
    target = getattr(thread, "_target", None)
    # ThreadPoolExecutor has a persistent worker target. Per-submission wrapping
    # above avoids leaking the provenance of the first submit into later work.
    if provenance is None or getattr(target, "__module__", "") == "concurrent.futures.thread":
        return _ORIGINAL_THREAD_START(thread, *args, **kwargs)

    run = thread.run
    run_function = getattr(run, "__func__", None)
    if target is not None and run_function is threading.Thread.run:
        thread._target = functools.partial(_run_dispatched, provenance, target)  # type: ignore[attr-defined]
        try:
            return _ORIGINAL_THREAD_START(thread, *args, **kwargs)
        except BaseException:
            thread._target = target  # type: ignore[attr-defined]
            raise

    had_instance_run = "run" in thread.__dict__

    def run_with_provenance() -> Any:
        try:
            return _run_dispatched(provenance, run)
        finally:
            if had_instance_run:
                thread.run = run  # type: ignore[method-assign]
            else:
                thread.__dict__.pop("run", None)

    thread.run = run_with_provenance  # type: ignore[method-assign]
    try:
        return _ORIGINAL_THREAD_START(thread, *args, **kwargs)
    except BaseException:
        if had_instance_run:
            thread.run = run  # type: ignore[method-assign]
        else:
            thread.__dict__.pop("run", None)
        raise


def assert_installed() -> None:
    """Fail if test code has displaced any process-wide guard wrapper."""
    missing: list[str] = []
    if builtins.__import__ is not _guarded_import:
        missing.append("builtins.__import__")
    if importlib.import_module is not _guarded_import_module:
        missing.append("importlib.import_module")
    if threading.Thread.start is not _guarded_thread_start:
        missing.append("threading.Thread.start")
    if concurrent.futures.ThreadPoolExecutor.submit is not _guarded_thread_submit:
        missing.append("ThreadPoolExecutor.submit")
    if missing:
        raise AssertionError(f"CLI-to-library guard lost ownership of: {', '.join(missing)}")


def drain_escaped_violations() -> list[str]:
    """Return and forget violations raised outside the main thread."""
    escaped = list(_ESCAPED_VIOLATIONS)
    _ESCAPED_VIOLATIONS.clear()
    return escaped


def assert_no_escaped_violations() -> None:
    """Fail for a violation that only reached a worker thread's exception hook.

    A ``ThreadPoolExecutor`` submission re-raises through its future, but a bare
    ``threading.Thread`` discards the exception after printing it. Without this
    check the runtime rule would detect such a route and still report success.
    """
    escaped = drain_escaped_violations()
    if escaped:
        raise AssertionError(
            "CLI-to-library boundary violations reached only a worker thread:\n  "
            + "\n  ".join(escaped)
        )


def _resolve_code(module_name: str, owner: str | None, attribute: str) -> CodeType:
    """Resolve one trusted serialization entry point to its exact code object."""
    module = importlib.import_module(module_name)
    holder: Any = module if owner is None else getattr(module, owner)
    function = getattr(holder, attribute)
    # Unwrap classmethod and bound-method descriptors to reach the raw function.
    function = getattr(function, "__func__", function)
    code = function.__code__
    assert isinstance(code, CodeType)
    return code


def _resolve_trusted_code() -> None:
    """Resolve trusted infrastructure identities before the wrappers go in.

    A rename in a future CPython makes this fail loudly at installation rather
    than silently widening or narrowing what the guard trusts.
    """
    global _IMPORT_MACHINERY, _PICKLING_CODE, _UNPICKLING_CODE
    _IMPORT_MACHINERY = tuple(
        module.__dict__
        for module in (
            sys.modules.get("importlib._bootstrap"),
            sys.modules.get("importlib._bootstrap_external"),
        )
        if module is not None
    )
    assert len(_IMPORT_MACHINERY) == 2, "import bootstrap modules are not loaded"
    _PICKLING_CODE = frozenset(_resolve_code(*spec) for spec in _PICKLING_SPECS)
    _UNPICKLING_CODE = frozenset(_resolve_code(*spec) for spec in _UNPICKLING_SPECS)
    assert len(_PICKLING_CODE) == len(_PICKLING_SPECS)
    assert len(_UNPICKLING_CODE) == len(_UNPICKLING_SPECS)
    assert not (_PICKLING_CODE & _UNPICKLING_CODE)


def install(project_root: Path) -> None:
    """Install the process-wide import and dispatch wrappers once."""
    global _ORIGINAL_IMPORT, _ORIGINAL_IMPORT_MODULE
    global _ORIGINAL_THREAD_START, _ORIGINAL_THREAD_SUBMIT, _PROJECT_ROOT
    if _ORIGINAL_IMPORT is not None:
        assert_installed()
        return
    _PROJECT_ROOT = project_root.resolve()
    _OBSERVED_ROUTES.clear()
    _ESCAPED_VIOLATIONS.clear()
    _PACKAGE_DIRECTORIES.clear()
    _resolve_trusted_code()
    _ORIGINAL_IMPORT = builtins.__import__
    _ORIGINAL_IMPORT_MODULE = importlib.import_module
    _ORIGINAL_THREAD_START = threading.Thread.start
    _ORIGINAL_THREAD_SUBMIT = concurrent.futures.ThreadPoolExecutor.submit
    builtins.__import__ = _guarded_import  # type: ignore[assignment]
    importlib.import_module = _guarded_import_module  # type: ignore[assignment]
    threading.Thread.start = _guarded_thread_start  # type: ignore[method-assign]
    concurrent.futures.ThreadPoolExecutor.submit = _guarded_thread_submit  # type: ignore[method-assign]


def restore() -> None:
    """Verify ownership, then restore every entry point wrapped by :func:`install`."""
    global _ORIGINAL_IMPORT, _ORIGINAL_IMPORT_MODULE
    global _ORIGINAL_THREAD_START, _ORIGINAL_THREAD_SUBMIT, _PROJECT_ROOT
    if _ORIGINAL_IMPORT is None:
        return
    assert _ORIGINAL_IMPORT_MODULE is not None
    assert _ORIGINAL_THREAD_START is not None
    assert _ORIGINAL_THREAD_SUBMIT is not None
    try:
        assert_installed()
    finally:
        builtins.__import__ = _ORIGINAL_IMPORT  # type: ignore[assignment]
        importlib.import_module = _ORIGINAL_IMPORT_MODULE  # type: ignore[assignment]
        threading.Thread.start = _ORIGINAL_THREAD_START  # type: ignore[method-assign]
        concurrent.futures.ThreadPoolExecutor.submit = _ORIGINAL_THREAD_SUBMIT  # type: ignore[method-assign]
        _ORIGINAL_IMPORT = None
        _ORIGINAL_IMPORT_MODULE = None
        _ORIGINAL_THREAD_START = None
        _ORIGINAL_THREAD_SUBMIT = None
        _PROJECT_ROOT = None
        _PACKAGE_DIRECTORIES.clear()


def observed_routes() -> Counter[str]:
    """Return a copy of continuous API routes observed from CLI origins."""
    return _OBSERVED_ROUTES.copy()
