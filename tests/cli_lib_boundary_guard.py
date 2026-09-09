"""Runtime observations for the CLI-to-library import boundary.

The guard wraps Python's two supported name-based import entry points for the
pytest session. Unlike a meta-path finder or audit hook, these entry points run
even when ``sys.modules`` already contains the target.

For an executed ``pstrain.lib`` import with CLI provenance, the complete stack
segment from import to CLI must stay inside ``pstrain.api`` and ``pstrain.lib``
and must contain an API frame. This rejects callbacks resumed below an API
frame and library functions reached without one. CLI provenance is copied into
direct ``threading.Thread`` targets and ``ThreadPoolExecutor`` submissions so a
worker cannot lose the origin merely by losing the submitting stack.

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
from types import FrameType
from typing import Any

_Callable = Callable[..., Any]

_ORIGINAL_IMPORT: _Callable | None = None
_ORIGINAL_IMPORT_MODULE: _Callable | None = None
_ORIGINAL_THREAD_START: _Callable | None = None
_ORIGINAL_THREAD_SUBMIT: _Callable | None = None
_PROJECT_ROOT: Path | None = None
_OBSERVED_ROUTES: Counter[str] = Counter()


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


def _is_transparent_frame(module: str) -> bool:
    """Identify infrastructure that does not choose the imported target.

    Import bootstrap frames merely implement the wrapped operation. The
    multiprocessing package also re-imports a function's defining module while
    pickling a process target chosen by an already validated library frame.
    All other neutral frames remain part of the segment so callbacks cannot
    launder CLI provenance.
    """
    return (
        module == __name__
        or module.startswith("importlib._bootstrap")
        or _is_package(module, "multiprocessing")
    )


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
        module = _frame_module(frame)
        if _is_transparent_frame(module):
            frame = frame.f_back
            continue
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
        module = _frame_module(frame)
        if _is_package(module, "pstrain.api"):
            api_reached = True
        elif not _is_package(module, "pstrain.lib"):
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
        raise CliLibBoundaryViolation(
            "CLI-to-library boundary violation: "
            f"{target} imported at {importer}; {reason} "
            f"(CLI origin {provenance.cli_origin})"
        )
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


def install(project_root: Path) -> None:
    """Install the process-wide import and dispatch wrappers once."""
    global _ORIGINAL_IMPORT, _ORIGINAL_IMPORT_MODULE
    global _ORIGINAL_THREAD_START, _ORIGINAL_THREAD_SUBMIT, _PROJECT_ROOT
    if _ORIGINAL_IMPORT is not None:
        assert_installed()
        return
    _PROJECT_ROOT = project_root.resolve()
    _OBSERVED_ROUTES.clear()
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


def observed_routes() -> Counter[str]:
    """Return a copy of continuous API routes observed from CLI origins."""
    return _OBSERVED_ROUTES.copy()
