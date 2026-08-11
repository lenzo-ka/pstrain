"""Persistent native-worker lifecycle and failure classification tests.

These are the reliability tests for the ``contained-3`` phase: malformed user
input reaching ``prune_tree``, ``make_quests`` or ``mdef_gen_ci`` must surface
as a typed exception with this interpreter still standing.
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest

from pstrain.lib import dtree, mdef, native_worker
from tests.clib import requires_c_library


@pytest.fixture(autouse=True)
def fresh_worker() -> Iterator[None]:
    """Isolate each test from any helper left over by another test."""
    native_worker._shutdown()
    yield
    native_worker._shutdown()


def _mdef_body(path: Path) -> str:
    """mdef text without the generation-timestamp comment."""
    return "\n".join(line for line in path.read_text().splitlines() if not line.startswith("#"))


def _phone_list(tmp_path: Path) -> Path:
    phones = tmp_path / "phones"
    phones.write_text("AA\nSIL\n")
    return phones


def test_unrouted_operation_is_rejected_before_any_worker_starts(tmp_path: Path) -> None:
    """Only the contained-3 operations may be asked of the helper."""
    with pytest.raises(native_worker.PstrainInvalidInputError) as raised:
        native_worker.call("tie_states", (), (tmp_path / "input",))
    assert raised.value.operation == "tie_states"
    assert native_worker._owned_worker().pid is None


def test_guarded_set_includes_the_full_surface_routes() -> None:
    assert set(native_worker.GUARDED_OPERATIONS) == {
        "prune_tree",
        "make_quests",
        "mdef_gen_ci",
        "python_call",
        "object_create",
        "object_call",
        "object_close",
    }


@requires_c_library
def test_session_reset_clears_cmd_ln_state_between_requests() -> None:
    """A worker request must observe reset state, not the prior probe key."""
    assert native_worker.call("_session_probe_set", (), ()) == 0
    assert native_worker.call("_session_probe_is_set", (), ()) == 0


def test_startup_pipe_failure_cleans_diagnostic_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    diagnostic = tmp_path / "diagnostic.log"

    def make_diagnostic(*args: object, **kwargs: object) -> tuple[int, str]:
        return os.open(diagnostic, os.O_RDWR | os.O_CREAT), str(diagnostic)

    context = native_worker.multiprocessing.get_context("spawn")
    monkeypatch.setattr(native_worker.tempfile, "mkstemp", make_diagnostic)
    monkeypatch.setattr(context, "Pipe", lambda: (_ for _ in ()).throw(OSError("no pipe")))
    monkeypatch.setattr(native_worker.multiprocessing, "get_context", lambda method: context)

    with pytest.raises(native_worker.PstrainWorkerError, match="cannot start native worker"):
        native_worker._owned_worker()._start()
    assert not diagnostic.exists()
    assert native_worker._owned_worker()._diagnostic_path is None


def test_oversized_request_is_rejected_before_worker_starts(tmp_path: Path) -> None:
    oversized = "x" * native_worker._MAX_REQUEST_BYTES
    with pytest.raises(native_worker.PstrainInvalidInputError, match="maximum is 65536 bytes"):
        native_worker.call("_fault_exit_zero", (oversized,), (tmp_path / "input",))
    assert native_worker._owned_worker().pid is None


@requires_c_library
def test_repeated_calls_reuse_one_worker(tmp_path: Path) -> None:
    """Consecutive guarded calls are serviced by the same helper process."""
    phones = _phone_list(tmp_path)
    first = tmp_path / "first.mdef"
    second = tmp_path / "second.mdef"

    assert native_worker._owned_worker().pid is None
    mdef.generate_ci_mdef(phones, first)
    pid = native_worker._owned_worker().pid
    assert pid is not None

    for index in range(3):
        mdef.generate_ci_mdef(phones, tmp_path / f"extra-{index}.mdef")
        assert native_worker._owned_worker().pid == pid

    mdef.generate_ci_mdef(phones, second)
    assert native_worker._owned_worker().pid == pid
    assert _mdef_body(second) == _mdef_body(first)


@requires_c_library
def test_succeed_fail_succeed_in_one_worker_is_unaffected_by_the_failure(
    tmp_path: Path,
) -> None:
    """A diagnosed native failure leaves the reused helper's state clean.

    ``pstrain_session_reset`` frees ``global_cmdln`` after every request; this
    is the test that catches key accumulation across the eight different
    argument tables, since the third result must match the first exactly.
    """
    phones = _phone_list(tmp_path)
    first = tmp_path / "first.mdef"
    mdef.generate_ci_mdef(phones, first)
    pid = native_worker._owned_worker().pid
    assert pid is not None

    # A partially-written tree makes prune_tree return -1 without dying, so
    # the same helper services the failure and the call that follows it.
    pset = tmp_path / "pset"
    pset.write_text("")
    trees = tmp_path / "trees"
    trees.mkdir()
    for state in range(3):
        (trees / f"AA-{state}.dtree").write_text("n_node 5\n0 1 2 0.0 0.0\n")

    with pytest.raises(native_worker.PstrainNativeError):
        dtree.prune_tree(first, pset, trees, tmp_path / "out", 1)
    assert native_worker._owned_worker().pid == pid

    third = tmp_path / "third.mdef"
    mdef.generate_ci_mdef(phones, third)
    assert native_worker._owned_worker().pid == pid
    assert _mdef_body(third) == _mdef_body(first)


@requires_c_library
def test_nonzero_exit_is_fatal_and_the_worker_is_replaced(tmp_path: Path) -> None:
    """E_FATAL in the helper is a fatal error, not a dead toolkit."""
    phones = _phone_list(tmp_path)
    first = tmp_path / "first.mdef"
    mdef.generate_ci_mdef(phones, first)
    pid = native_worker._owned_worker().pid

    with pytest.raises(native_worker.PstrainNativeFatalError) as raised:
        mdef.generate_ci_mdef(tmp_path / "missing", tmp_path / "bad.mdef")
    assert raised.value.operation == "mdef_gen_ci"
    assert raised.value.input_path == str(tmp_path / "missing")
    assert raised.value.diagnostic

    recovered = tmp_path / "recovered.mdef"
    mdef.generate_ci_mdef(phones, recovered)
    assert native_worker._owned_worker().pid != pid
    assert _mdef_body(recovered) == _mdef_body(first)


@requires_c_library
def test_exit_zero_during_request_is_protocol_error(tmp_path: Path) -> None:
    """A clean child exit with a request outstanding is never a success."""
    with pytest.raises(native_worker.PstrainWorkerProtocolError) as raised:
        native_worker.call("_fault_exit_zero", (), (tmp_path / "input",))
    assert raised.value.returncode == 0

    phones = _phone_list(tmp_path)
    mdef.generate_ci_mdef(phones, tmp_path / "recovered.mdef")


@requires_c_library
def test_signal_death_is_crash_and_next_call_respawns(tmp_path: Path) -> None:
    with pytest.raises(native_worker.PstrainNativeCrashError) as raised:
        native_worker.call("_fault_signal", (signal.SIGSEGV,), (tmp_path / "input",))
    assert raised.value.signal == signal.SIGSEGV

    phones = _phone_list(tmp_path)
    mdef.generate_ci_mdef(phones, tmp_path / "recovered.mdef")


@requires_c_library
def test_worker_killed_while_idle_respawns_transparently(tmp_path: Path) -> None:
    """Killing the helper between requests is invisible to the next caller."""
    phones = _phone_list(tmp_path)
    mdef.generate_ci_mdef(phones, tmp_path / "first.mdef")
    worker = native_worker._owned_worker()
    old_pid = worker.pid
    assert old_pid is not None

    os.kill(old_pid, signal.SIGKILL)
    assert worker._process is not None
    worker._process.join()

    second = tmp_path / "second.mdef"
    mdef.generate_ci_mdef(phones, second)
    assert native_worker._owned_worker().pid != old_pid
    assert second.exists()


@requires_c_library
def test_wedged_worker_send_times_out_and_next_call_respawns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phones = _phone_list(tmp_path)
    mdef.generate_ci_mdef(phones, tmp_path / "first.mdef")
    worker = native_worker._owned_worker()
    old_pid = worker.pid
    assert old_pid is not None and worker._connection is not None

    duplicate = socket.socket(fileno=os.dup(worker._connection.fileno()))
    try:
        duplicate.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
    finally:
        duplicate.close()
    original_timeout = native_worker._REQUEST_TIMEOUT
    os.kill(old_pid, signal.SIGSTOP)
    try:
        monkeypatch.setattr(native_worker, "_REQUEST_TIMEOUT", 0.2)

        payload = "x" * (native_worker._MAX_REQUEST_BYTES - 1024)
        with pytest.raises(native_worker.PstrainWorkerError, match="timed out during"):
            native_worker.call("_fault_exit_zero", (payload,), (tmp_path / "input",))

        # The short timeout is only for the wedged call; recovery includes a
        # fresh spawn, which slower CI machines cannot finish inside 200 ms.
        monkeypatch.setattr(native_worker, "_REQUEST_TIMEOUT", original_timeout)

        recovered = tmp_path / "recovered.mdef"
        mdef.generate_ci_mdef(phones, recovered)
        assert recovered.exists()
        assert native_worker._owned_worker().pid != old_pid
    finally:
        # Never leave a stopped helper behind: a failed assertion above would
        # otherwise hang fixture teardown on a blocking send to a full pipe.
        with contextlib.suppress(ProcessLookupError):
            os.kill(old_pid, signal.SIGCONT)


@requires_c_library
def test_shutdown_reaps_the_worker_and_its_diagnostic_file(tmp_path: Path) -> None:
    phones = _phone_list(tmp_path)
    mdef.generate_ci_mdef(phones, tmp_path / "first.mdef")
    worker = native_worker._owned_worker()
    diagnostic_path = worker._diagnostic_path
    assert diagnostic_path is not None and Path(diagnostic_path).exists()

    native_worker._shutdown()
    assert worker.pid is None
    assert not Path(diagnostic_path).exists()
