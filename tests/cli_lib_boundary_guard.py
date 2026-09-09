"""Runtime observations for the CLI-to-library import boundary.

The guard wraps Python's two supported name-based import entry points for the
pytest session. Unlike a meta-path finder or audit hook, these entry points run
even when ``sys.modules`` already contains the target.

For an executed ``pstrain.lib`` import with CLI provenance, the application
stack segment from import to CLI must stay inside ``pstrain.api`` and
``pstrain.lib`` and must contain an API frame.

Every role the guard reasons about -- command line, public API, library -- is
established the same way, and from something the code being judged cannot
rewrite. The three package directories are derived eagerly at installation from
the verified checkout root and are then fixed for the session; the guard never
consults a package's ``__path__`` afterwards, because ``__path__`` is an
ordinary mutable list. A frame belongs to a package only when the live
``sys.modules`` entry for its ``__name__`` owns that frame's globals and its
code was compiled from a file under that frozen directory. A writable
``__name__`` alone proves nothing, and it proves nothing for a command-line
frame either: a frame that merely claims a ``pstrain.cli`` name is an ordinary
neutral frame, so it does not end the stack walk and cannot hide the frames
beyond it.

Because that live entry is mutable, genuine command-line code can run with it
missing or replaced, and then nothing on the stack authenticates as the command
line. Having no origin would switch enforcement off rather than on, so the walk
also keeps a frame compiled under the frozen command-line directory as a
fallback origin. An authenticated origin always wins; the fallback is consulted
only where the walk would otherwise have ended with no origin at all. Where
several frames carry such a source location the outermost one is kept, so the
segment always spans every frame between the import and the furthest command-
line source location on the stack. A forged ``co_filename`` can therefore only
lengthen that segment, never cut it short.

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
import importlib.util
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

_CLI_PACKAGE = "pstrain.cli"
_BOUNDARY_PACKAGES = ("pstrain.api", "pstrain.lib")
# Every role is anchored the same way, the command line included.
_ANCHORED_PACKAGES = (_CLI_PACKAGE, *_BOUNDARY_PACKAGES)
# Frozen at installation from the checkout root. Never refreshed from a live
# ``__path__``, which the code under judgement can rewrite.
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


def _resolve_package_directories(project_root: Path) -> None:
    """Anchor every role to a directory derived from the verified checkout root.

    The directories come from the project root, never from a package's
    ``__path__``. ``__path__`` is an ordinary mutable list, so resolving the
    anchor lazily let the code being judged point it at a directory of its own
    before the first check and then authenticate from there. Deriving the
    anchors eagerly, once, removes that move.

    Each package is also resolved through the import system here and asserted to
    land on the derived directory. That runs before any wrapper is installed and
    before any test module is imported, so it reads a ``__path__`` nothing under
    judgement has had a chance to touch. An interpreter that would import a
    different ``pstrain`` therefore fails loudly at installation instead of
    quietly authenticating the wrong tree.
    """
    _PACKAGE_DIRECTORIES.clear()
    for package in _ANCHORED_PACKAGES:
        directory = project_root.joinpath(*package.split(".")).resolve()
        assert (directory / "__init__.py").is_file(), (
            f"{package} is not a package of the checkout at {project_root}"
        )
        spec = importlib.util.find_spec(package)
        assert spec is not None, f"{package} does not resolve to an importable package"
        locations = [Path(entry).resolve() for entry in (spec.submodule_search_locations or ())]
        assert locations == [directory], (
            f"{package} resolves to {locations}, not to the checkout directory {directory}"
        )
        _PACKAGE_DIRECTORIES[package] = directory


def _compiled_under(frame: FrameType, package: str) -> bool:
    """Report whether ``frame``'s code was compiled from ``package``'s frozen directory."""
    directory = _PACKAGE_DIRECTORIES.get(package)
    if directory is None:
        return False
    try:
        Path(frame.f_code.co_filename).resolve().relative_to(directory)
    except (OSError, ValueError):
        return False
    return True


def _anchored_package(frame: FrameType | None) -> str | None:
    """Return the package that provably owns ``frame``, if any.

    Ownership is not a name. ``__name__`` is an ordinary writable string, so a
    module that merely calls itself ``pstrain.api.something`` -- or
    ``pstrain.cli.something`` -- proves nothing. A frame counts as owned only
    when the live ``sys.modules`` entry for that name owns this exact globals
    mapping and the frame's code was compiled from a file under the directory
    frozen for that package at installation.
    """
    if frame is None:
        return None
    name = frame.f_globals.get("__name__")
    if not isinstance(name, str):
        return None
    package = next((known for known in _ANCHORED_PACKAGES if _is_package(name, known)), None)
    if package is None:
        return None
    module = sys.modules.get(name)
    if module is None or getattr(module, "__dict__", None) is not frame.f_globals:
        return None
    return package if _compiled_under(frame, package) else None


def _boundary_package(frame: FrameType | None) -> str | None:
    """Return the ``pstrain.api`` or ``pstrain.lib`` package that owns ``frame``."""
    package = _anchored_package(frame)
    return package if package in _BOUNDARY_PACKAGES else None


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
    """Return frames between this operation and its physical or carried CLI origin.

    A command-line frame is authenticated from its live ``sys.modules`` entry,
    and that entry is ordinary mutable process state. Genuine command-line code
    can therefore be running with no entry under its name, or with the name
    bound to some other object, and then no frame on the stack authenticates as
    the command line. Yielding no origin in that case switched enforcement off
    rather than on: with no origin there is no rule to apply and the import was
    simply allowed.

    So the walk also remembers a frame whose code was compiled from a file under
    the frozen command-line directory, together with the segment collected up to
    it, and falls back to that source location when the walk finds no
    authenticated origin. A source location is not authentication --
    ``co_filename`` is chosen by whoever compiled the code -- and it is not used
    as such. It is consulted only after the walk has run to the end without an
    authenticated origin, which is exactly the set of stacks that previously
    returned ``None`` and were allowed unconditionally.

    Where several frames carry such a source location, the walk keeps the
    outermost one. Keeping the innermost was a way to make enforcement *more*
    permissive by forging a ``co_filename``: an inner candidate's segment stops
    short of every frame beyond it, so a forged frame placed just below the API
    could discard the neutral frames that interrupt the route and the genuine
    command-line frame that follows them, turning a refusal into an acceptance.
    Keeping the outermost candidate leaves those intervening frames in the
    segment, where an inner candidate is judged as the ordinary neutral frame it
    is -- exactly as it already is when an authenticated command-line frame lies
    further out. A forged source location can then only lengthen the segment, so
    it can make the check stricter or leave it unchanged, never weaker.
    """
    frames: list[FrameType] = []
    fallback: tuple[tuple[FrameType, ...], FrameType] | None = None
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code is _run_dispatched.__code__:
            provenance = frame.f_locals.get("provenance")
            assert isinstance(provenance, _DispatchProvenance)
            return tuple(frames), provenance
        if _is_transparent_frame(frame):
            frame = frame.f_back
            continue
        # A command-line frame ends the walk, so it is authenticated exactly as
        # an API or library frame is. A frame that merely claims the name is an
        # ordinary neutral frame: it stays in the segment and the walk continues
        # past it to whatever really dispatched the call.
        if _anchored_package(frame) == _CLI_PACKAGE:
            return tuple(frames), _DispatchProvenance(_location(frame), False)
        if _compiled_under(frame, _CLI_PACKAGE):
            # Overwrite rather than keep the first: the walk runs inwards to
            # outwards, so the last candidate seen is the outermost one.
            fallback = (tuple(frames), frame)
        frames.append(frame)
        frame = frame.f_back
    if fallback is not None:
        segment, origin = fallback
        return segment, _DispatchProvenance(_location(origin), False)
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
    """Install the process-wide import and dispatch wrappers once.

    Every role the rule reasons about is anchored to this one checkout root, so
    a second installation naming a different root is a configuration error. It
    used to be ignored in favor of the root already frozen, which would have
    enforced the boundary of one checkout while the session ran another.
    """
    global _ORIGINAL_IMPORT, _ORIGINAL_IMPORT_MODULE
    global _ORIGINAL_THREAD_START, _ORIGINAL_THREAD_SUBMIT, _PROJECT_ROOT
    resolved_root = project_root.resolve()
    if _ORIGINAL_IMPORT is not None:
        assert resolved_root == _PROJECT_ROOT, (
            f"CLI-to-library guard is already anchored to {_PROJECT_ROOT}, not to {resolved_root}"
        )
        assert_installed()
        return
    _PROJECT_ROOT = resolved_root
    _OBSERVED_ROUTES.clear()
    _ESCAPED_VIOLATIONS.clear()
    _resolve_package_directories(_PROJECT_ROOT)
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


def anchored_directories() -> dict[str, Path]:
    """Return a copy of the directory frozen for each role at installation."""
    return dict(_PACKAGE_DIRECTORIES)
