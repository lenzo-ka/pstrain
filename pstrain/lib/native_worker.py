"""Crash containment for the ``contained-all-operations`` native surface.

One-shot wrappers execute as complete Python/CFFI operations in the helper;
BW, alignment, feature extraction, and logmath use opaque remote objects.
See ``docs/design/native-boundary.md`` for the phase contract.

Each Python process owns at most one lazily spawned helper process, created
through an explicit ``spawn`` context and reused across calls.  The helper
services one request at a time; the owning process classifies every outcome
(clean result, native error return, signal death, nonzero exit, clean exit
mid-request, transport EOF) into the exception hierarchy below.

Diagnostics are captured worker-side: the helper redirects its own stderr --
where every ``E_INFO``/``E_ERROR``/``E_FATAL`` line lands -- into a private
temporary file, and the owner attaches the tail of that file to the raised
exception.
"""

from __future__ import annotations

import atexit
import contextlib
import multiprocessing
import os
import select
import struct
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from typing import Any, NoReturn, ParamSpec, TypeVar, cast

_START_TIMEOUT = 30.0
_REQUEST_TIMEOUT = 300.0
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_BULK_REQUEST_BYTES = 256 * 1024 * 1024
_DIAGNOSTIC_TAIL_BYTES = 64 * 1024
_inside_worker = False
P = ParamSpec("P")
R = TypeVar("R")

#: Worker protocol operations accepted from containment wrappers and proxies.
GUARDED_OPERATIONS = frozenset(
    {
        "prune_tree",
        "make_quests",
        "mdef_gen_ci",
        "python_call",
        "object_create",
        "object_call",
        "object_close",
    }
)

#: Fault-injection operations used only by the reliability tests.  They exist
#: in the helper because a clean ``exit(0)`` mid-request and a signal death
#: cannot be induced from outside the child.
_FAULT_OPERATIONS = frozenset(
    {"_fault_exit_zero", "_fault_signal", "_session_probe_set", "_session_probe_is_set"}
)
_STATUS_OPERATIONS = frozenset({"prune_tree", "make_quests", "mdef_gen_ci"})

_OPERATIONS = GUARDED_OPERATIONS | _FAULT_OPERATIONS


class PstrainError(RuntimeError):
    """Base class for public pstrain failures."""


class PstrainNativeError(PstrainError):
    """A contained native operation failed."""

    def __init__(
        self,
        operation: str,
        input_paths: tuple[str, ...],
        diagnostic: str = "",
        returncode: int | None = None,
    ) -> None:
        self.operation = operation
        self.input_paths = input_paths
        self.input_path = input_paths[0] if input_paths else None
        self.diagnostic = diagnostic
        self.returncode = returncode
        detail = diagnostic.strip() or "native operation failed"
        super().__init__(f"{operation} failed for {', '.join(input_paths)}: {detail}")


class PstrainNativeFatalError(PstrainNativeError):
    """The native worker exited nonzero or returned a diagnosed failure."""


class PstrainNativeCrashError(PstrainNativeError):
    """The native worker died from a signal."""

    def __init__(self, *args: Any, signal: int, **kwargs: Any) -> None:
        self.signal = signal
        super().__init__(*args, **kwargs)


class PstrainWorkerProtocolError(PstrainNativeError):
    """The worker violated the request/response protocol.

    Raised in particular when the helper exits cleanly (status 0) with a
    request still outstanding: a successful-looking exit is not a result.
    """


class PstrainInvalidInputError(PstrainNativeError):
    """Python-side validation rejected the request before it reached a worker."""


class PstrainWorkerError(PstrainError):
    """The contained worker or process-pool infrastructure is unavailable."""


@dataclass(frozen=True)
class _WorkerState:
    """Picklable construction state handed to the spawned helper."""

    diagnostic_path: str


def _diagnostic_tail(path: str) -> str:
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - _DIAGNOSTIC_TAIL_BYTES))
            return stream.read().decode(errors="replace")
    except OSError:
        return ""


def _resolve_python_target(module_name: str, target_name: str) -> Any:
    """Resolve a module-level function or class without importing CFFI in the owner."""
    import importlib

    target: Any = importlib.import_module(module_name)
    for part in target_name.split("."):
        target = getattr(target, part)
    return target


def _dispatch(
    lib: Any,
    ffi: Any,
    operation: str,
    arguments: tuple[Any, ...],
    objects: dict[str, Any],
) -> Any:
    """Run one native operation inside the helper process."""
    if operation == "prune_tree":
        return int(
            lib.pstrain_prune_tree(
                arguments[0].encode(),
                arguments[1].encode(),
                arguments[2].encode(),
                arguments[3].encode(),
                arguments[4],
                arguments[5],
                arguments[6],
            )
        )
    if operation == "make_quests":

        def optional(value: str | None) -> Any:
            return ffi.NULL if value is None else value.encode()

        return int(
            lib.pstrain_make_quests(
                arguments[0].encode(),
                arguments[1].encode(),
                optional(arguments[2]),
                optional(arguments[3]),
                arguments[4].encode(),
                arguments[5],
                arguments[6],
                arguments[7],
                arguments[8],
                arguments[9],
            )
        )
    if operation == "mdef_gen_ci":
        return int(
            lib.pstrain_mdef_gen_ci(arguments[0].encode(), arguments[1].encode(), arguments[2])
        )
    if operation == "python_call":
        module_name, target_name, args, kwargs = arguments
        return _resolve_python_target(module_name, target_name)(*args, **kwargs)
    if operation == "object_create":
        object_id, module_name, target_name, args, kwargs = arguments
        objects[object_id] = _resolve_python_target(module_name, target_name)(*args, **kwargs)
        return None
    if operation == "object_call":
        object_id, method_name, args, kwargs = arguments
        return getattr(objects[object_id], method_name)(*args, **kwargs)
    if operation == "object_close":
        object_id = arguments[0]
        instance = objects.pop(object_id, None)
        if instance is not None:
            close = getattr(instance, "close", None)
            if close is not None:
                close()
        return None
    if operation == "_fault_exit_zero":
        os._exit(0)
    if operation == "_fault_signal":
        os.kill(os.getpid(), arguments[0])
        return 0
    if operation == "_session_probe_set":
        return int(lib.pstrain_session_probe_set())
    if operation == "_session_probe_is_set":
        return int(lib.pstrain_session_probe_is_set())
    raise ValueError(f"unknown native operation: {operation}")


def _worker_main(connection: Connection, state: _WorkerState) -> None:
    """Entry point of the spawned helper process."""
    global _inside_worker
    _inside_worker = True
    # Own stderr permanently: sphinxbase writes every diagnostic level there,
    # and the parent cannot install an err callback (err_cb_f is variadic).
    diagnostic_fd = os.open(state.diagnostic_path, os.O_WRONLY | os.O_APPEND)
    os.dup2(diagnostic_fd, 2)
    os.close(diagnostic_fd)

    # Importing CFFI and dlopening libpstrainc is deliberately child-only for
    # guarded operations, and happens before "ready" so that a load failure is
    # classified as an unstartable worker rather than as a failed request.
    from pstrain.lib import _pstrainc

    lib = _pstrainc.get_lib()
    ffi = _pstrainc.get_ffi()
    objects: dict[str, Any] = {}

    connection.send(("ready", os.getpid()))

    while True:
        try:
            request = connection.recv()
        except EOFError:
            return
        if request is None:
            return
        request_id, operation, arguments = request
        try:
            result = _dispatch(lib, ffi, operation, arguments, objects)
            # Stateful BW/alignment/logmath objects span multiple RPCs. Their
            # native contexts may retain cmd_ln-backed configuration, so reset
            # at the coarse operation boundary (when the final object closes),
            # not between methods of a live object.
            if not objects:
                lib.pstrain_session_reset()
            connection.send(("result", request_id, result))
        except BaseException as exc:
            # Python-level faults are reportable; native process termination
            # bypasses this arm entirely and is classified by the owner.
            try:
                if not objects:
                    lib.pstrain_session_reset()
            finally:
                if isinstance(exc, (FileNotFoundError, ValueError)):
                    connection.send(("validation_error", request_id, exc))
                else:
                    connection.send(("error", request_id, repr(exc)))


class _NativeWorker:
    """One helper process, lazily started and reused, one request at a time."""

    def __init__(self) -> None:
        self._process: BaseProcess | None = None
        self._connection: Connection | None = None
        self._diagnostic_path: str | None = None
        self._request_id = 0

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def close(self) -> None:
        """Ask the helper to exit, then reclaim it and its diagnostic file."""
        if self._connection is not None:
            with contextlib.suppress(BrokenPipeError, EOFError, OSError):
                self._connection.send(None)
        self._discard()

    def _discard(self) -> None:
        """Drop the current helper unconditionally; the next call respawns."""
        connection, process = self._connection, self._process
        self._connection = None
        self._process = None
        if connection is not None:
            with contextlib.suppress(OSError):
                connection.close()
        if process is not None:
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join()
        if self._diagnostic_path is not None:
            Path(self._diagnostic_path).unlink(missing_ok=True)
            self._diagnostic_path = None

    def _start(self) -> None:
        self._discard()
        fd, path = tempfile.mkstemp(prefix="pstrain-native-", suffix=".log")
        os.close(fd)
        self._diagnostic_path = path
        parent: Connection | None = None
        child: Connection | None = None
        started = False
        try:
            context = multiprocessing.get_context("spawn")
            parent, child = context.Pipe()
            process = context.Process(target=_worker_main, args=(child, _WorkerState(path)))
            process.start()
            started = True
        except Exception as exc:
            raise PstrainWorkerError(f"cannot start native worker: {exc}") from exc
        finally:
            if not started:
                if parent is not None:
                    parent.close()
                if child is not None:
                    child.close()
                self._diagnostic_path = None
                Path(path).unlink(missing_ok=True)
        assert parent is not None and child is not None
        child.close()
        self._process, self._connection = process, parent
        ready = wait([parent, process.sentinel], timeout=_START_TIMEOUT)
        if parent in ready:
            message = None
            with contextlib.suppress(EOFError, OSError):
                message = parent.recv()
            if message and message[0] == "ready":
                return
        diagnostic = self._tail().strip()
        self._discard()
        raise PstrainWorkerError(f"native worker failed to start: {diagnostic or 'no diagnostic'}")

    def call(
        self,
        operation: str,
        arguments: tuple[Any, ...],
        inputs: tuple[str, ...],
        reported_operation: str | None = None,
    ) -> Any:
        label = reported_operation or operation
        request_id = self._request_id + 1
        request = bytes(ForkingPickler.dumps((request_id, operation, arguments)))
        maximum = (
            _MAX_BULK_REQUEST_BYTES
            if operation in {"python_call", "object_call", "object_create"}
            else _MAX_REQUEST_BYTES
        )
        if len(request) > maximum:
            raise PstrainInvalidInputError(
                label,
                inputs,
                f"serialized native-worker request is {len(request)} bytes; "
                f"maximum is {maximum} bytes",
            )
        deadline = time.monotonic() + _REQUEST_TIMEOUT
        if self._process is None or not self._process.is_alive():
            self._start()
        assert self._connection is not None and self._process is not None
        self._request_id = request_id
        self._truncate_diagnostic()
        try:
            self._send_request(request, deadline)
        except (BrokenPipeError, EOFError, OSError):
            # The helper died between requests; respawn and send once more.
            self._discard()
            self._start()
            assert self._connection is not None and self._process is not None
            try:
                self._send_request(request, deadline)
            except TimeoutError:
                self._raise_timeout(label)
            except (BrokenPipeError, EOFError, OSError):
                # The fresh helper died during the retried send; give up loudly
                # rather than leaving a connection holding a partial frame.
                self._discard()
                self._raise_death(label, inputs, eof=True)
        except TimeoutError:
            self._raise_timeout(label)

        remaining = max(0.0, deadline - time.monotonic())
        ready = wait([self._connection, self._process.sentinel], timeout=remaining)
        if not ready:
            self._raise_timeout(label)
        if self._connection in ready:
            try:
                message = self._connection.recv()
            except EOFError:
                return self._raise_death(label, inputs, eof=True)
            kind, response_id, payload = message
            if response_id != request_id or kind not in {"result", "error", "validation_error"}:
                diagnostic = self._tail()
                self._discard()
                raise PstrainWorkerProtocolError(label, inputs, diagnostic)
            if kind == "error":
                raise PstrainNativeError(label, inputs, str(payload))
            if kind == "validation_error":
                raise payload
            if operation in _STATUS_OPERATIONS and payload != 0:
                # The helper is alive and has run pstrain_session_reset; the
                # operation itself reported failure.
                diagnostic = self._tail()
                if diagnostic.strip():
                    raise PstrainNativeFatalError(label, inputs, diagnostic, int(payload))
                raise PstrainNativeError(label, inputs, diagnostic, int(payload))
            return payload
        return self._raise_death(label, inputs, eof=False)

    def _send_request(self, payload: bytes, deadline: float) -> None:
        """Send one bounded request without allowing a full pipe to hang."""
        assert self._connection is not None
        framed = struct.pack("!i", len(payload)) + payload
        descriptor = self._connection.fileno()
        os.set_blocking(descriptor, False)
        sent = 0
        try:
            while sent < len(framed):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                _, writable, _ = select.select([], [descriptor], [], remaining)
                if not writable:
                    raise TimeoutError
                try:
                    sent += os.write(descriptor, framed[sent:])
                except BlockingIOError:
                    continue
        finally:
            with contextlib.suppress(OSError):
                os.set_blocking(descriptor, True)

    def _raise_timeout(self, operation: str) -> NoReturn:
        assert self._process is not None
        self._process.kill()
        self._process.join()
        diagnostic = self._tail()
        self._discard()
        raise PstrainWorkerError(
            f"native worker timed out during {operation}: {diagnostic or 'no diagnostic'}"
        )

    def _truncate_diagnostic(self) -> None:
        """Start each request with an empty diagnostic file."""
        if self._diagnostic_path is None:
            return
        with contextlib.suppress(OSError):
            fd = os.open(self._diagnostic_path, os.O_WRONLY | os.O_TRUNC)
            os.close(fd)

    def _tail(self) -> str:
        return _diagnostic_tail(self._diagnostic_path or "")

    def _raise_death(self, operation: str, inputs: tuple[str, ...], *, eof: bool) -> NoReturn:
        assert self._process is not None
        self._process.join(timeout=1)
        returncode = self._process.exitcode
        diagnostic = self._tail()
        self._discard()
        if returncode is not None and returncode < 0:
            raise PstrainNativeCrashError(
                operation, inputs, diagnostic, returncode, signal=-returncode
            )
        if returncode == 0:
            raise PstrainWorkerProtocolError(operation, inputs, diagnostic, returncode)
        if returncode is not None:
            raise PstrainNativeFatalError(operation, inputs, diagnostic, returncode)
        suffix = "transport EOF" if eof else "transport closed"
        raise PstrainWorkerProtocolError(operation, inputs, diagnostic or suffix)


_owner_pid = os.getpid()
_worker: _NativeWorker | None = None
_lock = threading.Lock()


def _owned_worker() -> _NativeWorker:
    """Return this process's worker, creating a fresh one after a fork."""
    global _owner_pid, _worker
    pid = os.getpid()
    if _worker is None or _owner_pid != pid:
        _owner_pid = pid
        _worker = _NativeWorker()
    return _worker


def call(operation: str, arguments: tuple[Any, ...], inputs: tuple[Path | str, ...]) -> Any:
    """Execute one guarded native operation in this process's helper.

    Args:
        operation: One of :data:`GUARDED_OPERATIONS`.
        arguments: Picklable positional arguments for the native entry point.
        inputs: Input paths, recorded on any raised exception.

    Raises:
        PstrainInvalidInputError: The operation is not routed through the helper.
            Requests whose serialized representation exceeds 64 KiB are also
            rejected before anything is sent.
        PstrainWorkerError: The helper could not be started, or timed out.
        PstrainNativeCrashError: The helper died on a signal.
        PstrainWorkerProtocolError: The helper exited cleanly mid-request.
        PstrainNativeFatalError: The operation failed with a diagnostic.
        PstrainNativeError: The operation failed without one.
    """
    paths = tuple(str(item) for item in inputs)
    if operation not in _OPERATIONS:
        raise PstrainInvalidInputError(
            operation, paths, f"operation is not routed through the native worker: {operation}"
        )
    with _lock:
        return _owned_worker().call(operation, arguments, paths)


def in_worker() -> bool:
    """Return whether code is executing inside the contained native helper."""
    return _inside_worker


def call_python(
    module_name: str,
    target_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    inputs: tuple[Path | str, ...] = (),
) -> Any:
    """Run a complete Python/CFFI operation in the contained helper."""
    paths = tuple(str(item) for item in inputs)
    with _lock:
        return _owned_worker().call(
            "python_call", (module_name, target_name, args, kwargs), paths, target_name
        )


def contained(function: Callable[P, R]) -> Callable[P, R]:
    """Route a complete stateless public CFFI operation through the helper."""
    module_name = function.__module__
    target_name = function.__name__

    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if in_worker():
            return function(*args, **kwargs)
        inputs = tuple(item for item in (*args, *kwargs.values()) if isinstance(item, (Path, str)))
        return cast("R", call_python(module_name, target_name, args, kwargs, inputs))

    return wrapper


class NativeObjectProxy:
    """Proxy a stateful CFFI-backed Python object living in the helper."""

    def __init__(
        self,
        module_name: str,
        target_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        inputs: tuple[Path | str, ...] = (),
    ) -> None:
        self._object_id: str | None = uuid.uuid4().hex
        self._inputs = inputs
        call(
            "object_create",
            (self._object_id, module_name, target_name, args, kwargs),
            inputs,
        )

    def call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        if self._object_id is None:
            raise PstrainWorkerError("native object is closed")
        return call(
            "object_call",
            (self._object_id, method_name, args, kwargs),
            self._inputs,
        )

    def close(self) -> None:
        if self._object_id is not None:
            try:
                call("object_close", (self._object_id,), self._inputs)
            finally:
                self._object_id = None


def _shutdown() -> None:
    if _worker is not None:
        _worker.close()


atexit.register(_shutdown)
