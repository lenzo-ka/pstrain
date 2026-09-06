"""Pytest configuration and shared fixtures.

This module centralizes test configuration, including:
- C library availability detection
- Common fixtures for test data
- Skip markers for tests requiring the C library
"""

from __future__ import annotations

import concurrent.futures
import functools
import inspect
import multiprocessing.pool
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import pstrain

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PRODUCTION_ROOT = _PROJECT_ROOT / "pstrain"
# Authoritative pool invariant: every process pool constructed by production
# code must close native helpers before CPython joins child processes at exit.
# Observing constructors here covers aliases, subclasses, and dynamic factories.
# The static inventory is only a non-exhaustive early check. This runtime guard
# is authoritative for constructions the suite executes, but cannot observe a
# pool built in a spawned child/native helper, a subprocess, or during the
# pstrain import above before pytest_configure installs these wrappers.
_POOL_INITIALIZERS: set[object] = set()
_ORIGINAL_POOL_CONSTRUCTORS: list[tuple[type[Any], Callable[..., Any]]] = []
_OBSERVED_PRODUCTION_POOLS: Counter[str] = Counter()


def _production_caller() -> tuple[Path, int] | None:
    frame = sys._getframe(1)
    while frame is not None:
        path = Path(frame.f_code.co_filename).resolve()
        try:
            path.relative_to(_PRODUCTION_ROOT)
        except ValueError:
            frame = frame.f_back
            continue
        return path, frame.f_lineno
    return None


def _guard_pool_constructor(constructor: Callable[..., Any]) -> Callable[..., Any]:
    signature = inspect.signature(constructor)

    @functools.wraps(constructor)
    def guarded(self: object, *args: object, **kwargs: object) -> Any:
        if isinstance(self, multiprocessing.pool.ThreadPool):
            return constructor(self, *args, **kwargs)
        caller = _production_caller()
        if caller is not None:
            path, line = caller
            relative = path.relative_to(_PROJECT_ROOT)
            if path.exists():
                _OBSERVED_PRODUCTION_POOLS[f"{relative}:{line}"] += 1
            arguments = signature.bind_partial(self, *args, **kwargs).arguments
            initializer = _normalized_initializer(arguments.get("initializer"))
            try:
                approved = initializer in _POOL_INITIALIZERS
            except TypeError:
                approved = False
            if not approved:
                if isinstance(self, multiprocessing.pool.Pool):
                    self._state = multiprocessing.pool.INIT
                raise AssertionError(
                    "production process pool initializer must install the native-helper shutdown "
                    f"finalizer; constructed at {relative}:{line}"
                )
        return constructor(self, *args, **kwargs)

    return guarded


def _normalized_initializer(initializer: object) -> object:
    seen: set[int] = set()
    while id(initializer) not in seen:
        seen.add(id(initializer))
        if isinstance(initializer, functools.partial):
            initializer = initializer.func
            continue
        wrapped = getattr(initializer, "__wrapped__", None)
        if wrapped is not None:
            initializer = wrapped
            continue
        break
    return initializer


def _install_pool_constructor_guards() -> None:
    from pstrain.lib.native_worker import close_helper_before_children_are_joined
    from pstrain.lib.pipeline.runner import _initialize_pool_worker
    from pstrain.lib.steps.train import _initialize_bw_pool_worker

    _OBSERVED_PRODUCTION_POOLS.clear()
    _POOL_INITIALIZERS.update(
        {
            close_helper_before_children_are_joined,
            _initialize_pool_worker,
            _initialize_bw_pool_worker,
        }
    )
    for pool_class in (concurrent.futures.ProcessPoolExecutor, multiprocessing.pool.Pool):
        constructor = pool_class.__init__
        _ORIGINAL_POOL_CONSTRUCTORS.append((pool_class, constructor))
        pool_class.__init__ = _guard_pool_constructor(constructor)  # type: ignore[method-assign]


def _restore_pool_constructor_guards() -> None:
    while _ORIGINAL_POOL_CONSTRUCTORS:
        pool_class, constructor = _ORIGINAL_POOL_CONSTRUCTORS.pop()
        pool_class.__init__ = constructor  # type: ignore[method-assign]
    _POOL_INITIALIZERS.clear()


def _assert_path_in_checkout(*, subject: str, actual: Path) -> None:
    """Fail collection when a tested artifact is outside this checkout."""
    expected = _PROJECT_ROOT.resolve()
    resolved = actual.resolve()
    try:
        resolved.relative_to(expected)
    except ValueError:
        raise pytest.UsageError(
            f"SUBJECT IDENTITY GATE FAILED ({subject}):\n"
            f"  expected repository root: {expected}\n"
            f"  actual resolved path:     {resolved}"
        ) from None


def _assert_test_subject_identity() -> None:
    """Prove that Python and resolved native artifacts belong to this checkout."""
    package_file = getattr(pstrain, "__file__", None)
    if package_file is None:
        raise pytest.UsageError(
            "SUBJECT IDENTITY GATE FAILED (Python package 'pstrain'):\n"
            f"  expected repository root: {_PROJECT_ROOT.resolve()}\n"
            "  actual resolved path:     <pstrain.__file__ is None>"
        )
    _assert_path_in_checkout(subject="Python package 'pstrain'", actual=Path(package_file))

    # _find_library is the exact path subsequently passed to cffi.FFI.dlopen.
    # If no library is selectable, the existing PSTRAIN_REQUIRE_CLIB gate below
    # decides whether that is an error or tests may be skipped.
    from pstrain.lib._cffi.core import _find_library

    try:
        library_path = _find_library()
    except RuntimeError:
        return
    _assert_path_in_checkout(subject="native library 'libpstrainc'", actual=library_path)

    # Resolve commands through the same default route used by CommandBuilder.
    # PSTRAIN_BIN_DIR remains a supported override, but test runs may only select
    # an override (or PATH entry) whose binary belongs to this checkout.
    from pstrain.lib.commands import PSTRAIN_BINARIES, resolve_binary

    for binary_name in sorted(set(PSTRAIN_BINARIES.values())):
        binary_path = resolve_binary(binary_name)
        if binary_path is None:
            raise pytest.UsageError(
                f"SUBJECT IDENTITY GATE FAILED (core command '{binary_name}'):\n"
                f"  expected repository root: {_PROJECT_ROOT.resolve()}\n"
                "  actual resolved path:     <not found>"
            )
        _assert_path_in_checkout(subject=f"core command '{binary_name}'", actual=binary_path)


_assert_test_subject_identity()

# Single source of truth for libpstrainc availability. This import must follow the
# identity gate: importing tests.clib can load a sibling checkout's native library.
# Re-exported here so tests and other conftests can rely on the marker.
from tests.clib import (  # noqa: E402, F401
    C_LIBRARY_AVAILABLE,
    c_library_available,
    require_clib_env,
    requires_c_library,
)

# =============================================================================
# CI gate: fail loudly when the C library is required but missing
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Abort collection if PSTRAIN_REQUIRE_CLIB is set but libpstrainc can't load.

    Without this, a misconfigured CI job (lib not built, wrong path) would
    silently skip the entire CFFI/parity tier and still report green.
    """
    if require_clib_env() and not c_library_available():
        raise pytest.UsageError(
            "PSTRAIN_REQUIRE_CLIB is set but libpstrainc could not be loaded. "
            "Build it first (e.g. 'make build-c') or unset PSTRAIN_REQUIRE_CLIB."
        )
    _install_pool_constructor_guards()


def pytest_unconfigure(config: pytest.Config) -> None:
    _restore_pool_constructor_guards()


def pytest_terminal_summary(terminalreporter: Any) -> None:
    terminalreporter.write_sep("=", "observed production process pools")
    for location, count in sorted(_OBSERVED_PRODUCTION_POOLS.items()):
        terminalreporter.write_line(f"{location}: {count}")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return _PROJECT_ROOT


@pytest.fixture
def sample_data_dir() -> Path:
    """Return the sample data directory."""
    return _PROJECT_ROOT / "pstrain" / "data" / "sample"


@pytest.fixture
def sample_audio(sample_data_dir: Path) -> Path:
    """Return path to sample audio file."""
    return sample_data_dir / "kevin-alice-16k.wav"


@pytest.fixture
def sample_transcript(sample_data_dir: Path) -> Path:
    """Return path to sample transcript file."""
    return sample_data_dir / "kevin-alice-16k.txt"


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Create and return a temporary directory for model files."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def basic_phones() -> list[str]:
    """Return a basic phone set for testing."""
    return ["SIL", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D"]
