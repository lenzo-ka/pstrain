"""Runtime enforcement for the CLI-to-library import boundary.

The guard wraps Python's two supported name-based import entry points for the
pytest session.  Unlike a meta-path finder or an audit hook, these entry points
are invoked even when the requested module is already present in
``sys.modules``.  Looking at the runtime target also makes aliases and computed
``importlib.import_module`` names irrelevant.

An import of ``pstrain.lib`` or a child is rejected when the nearest pstrain
boundary frame is in ``pstrain.cli``.  It is allowed when a ``pstrain.api`` or
``pstrain.lib`` frame comes first: the former is the intended public route and
the latter is an internal library import whose original entry was checked
separately.

This is an exact runtime guarantee, not whole-program proof.  It covers imports
executed in the pytest process after this module installs the wrappers.  It
cannot see unexecuted paths, imports in spawned children or subprocesses,
module objects obtained without importing (for example through
``sys.modules``), custom loader execution, private importlib entry points, or
an import callable captured before installation.  The AST scanner remains a
cheap early warning for ordinary imports in unexecuted CLI paths.
"""

from __future__ import annotations

import builtins
import contextlib
import importlib
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from types import FrameType
from typing import Any

_ImportCallable = Callable[..., Any]

_ORIGINAL_IMPORT: _ImportCallable | None = None
_ORIGINAL_IMPORT_MODULE: _ImportCallable | None = None
_PROJECT_ROOT: Path | None = None
_OBSERVED_ROUTES: Counter[str] = Counter()


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


def _import_route() -> tuple[str, FrameType, FrameType] | None:
    """Return (nearest boundary, importer, CLI frame) for a CLI-routed import."""
    frame = sys._getframe(1)
    importer: FrameType | None = None
    nearest_boundary = ""
    while frame is not None:
        module = _frame_module(frame)
        if module != __name__ and importer is None:
            importer = frame
        if not nearest_boundary:
            if _is_package(module, "pstrain.api"):
                nearest_boundary = "pstrain.api"
            elif _is_package(module, "pstrain.lib"):
                nearest_boundary = "pstrain.lib"
            elif _is_package(module, "pstrain.cli"):
                assert importer is not None
                return "pstrain.cli", importer, frame
        elif _is_package(module, "pstrain.cli"):
            assert importer is not None
            return nearest_boundary, importer, frame
        frame = frame.f_back
    return None


def _check_targets(targets: Iterable[str]) -> None:
    targets = sorted(set(targets))
    if not targets:
        return
    route = _import_route()
    if route is None:
        return
    boundary, importer, cli_frame = route
    target = targets[0]
    if boundary == "pstrain.cli":
        raise AssertionError(
            "CLI-to-library boundary violation: "
            f"{target} imported at {_location(importer)} without passing through pstrain.api "
            f"(CLI frame {_location(cli_frame)})"
        )
    _OBSERVED_ROUTES[f"{target} via {boundary} at {_location(importer)}"] += 1


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


def install(project_root: Path) -> None:
    """Install the process-wide import wrappers once."""
    global _ORIGINAL_IMPORT, _ORIGINAL_IMPORT_MODULE, _PROJECT_ROOT
    if _ORIGINAL_IMPORT is not None:
        return
    _PROJECT_ROOT = project_root.resolve()
    _OBSERVED_ROUTES.clear()
    _ORIGINAL_IMPORT = builtins.__import__
    _ORIGINAL_IMPORT_MODULE = importlib.import_module
    builtins.__import__ = _guarded_import  # type: ignore[assignment]
    importlib.import_module = _guarded_import_module  # type: ignore[assignment]


def restore() -> None:
    """Restore the import entry points wrapped by :func:`install`."""
    global _ORIGINAL_IMPORT, _ORIGINAL_IMPORT_MODULE, _PROJECT_ROOT
    if _ORIGINAL_IMPORT is None or _ORIGINAL_IMPORT_MODULE is None:
        return
    builtins.__import__ = _ORIGINAL_IMPORT  # type: ignore[assignment]
    importlib.import_module = _ORIGINAL_IMPORT_MODULE  # type: ignore[assignment]
    _ORIGINAL_IMPORT = None
    _ORIGINAL_IMPORT_MODULE = None
    _PROJECT_ROOT = None


def observed_routes() -> Counter[str]:
    """Return a copy of the public/internal routes observed from CLI frames."""
    return _OBSERVED_ROUTES.copy()
