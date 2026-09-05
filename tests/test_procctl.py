from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import Mock, call

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "procctl.py"
SPEC = importlib.util.spec_from_file_location("procctl", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
procctl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(procctl)

# Bound on waits for something the operating system will do shortly. Generous,
# because a loaded machine can take seconds to start an interpreter, and a test
# that reaches this bound should be reporting a hang rather than a slow machine.
LIVENESS_TIMEOUT = 30.0
# How long teardown lets a stop wait for each signal before escalating.
STOP_TIMEOUT = 5.0
# The harness must outlast the tool it runs, or a slow but valid launch is
# killed here before it prints a fingerprint and leaves a detached child no
# fixture ever registered. Launch costs at most LAUNCH_SETTLE_DEADLINE plus the
# four LAUNCH_CLEANUP_TIMEOUT periods a refused launch spends disposing of its
# child, which is eight seconds and so already inside the margin below; stop
# costs at most one live reading, which is two readers each capped at
# READER_SUBPROCESS_TIMEOUT, plus two STOP_TIMEOUT signal waits. This covers the
# larger of the two with room for interpreter start-up, so it fires only on a
# genuine hang.
RUN_TIMEOUT = (
    procctl.LAUNCH_SETTLE_DEADLINE + 2 * procctl.READER_SUBPROCESS_TIMEOUT + 2 * STOP_TIMEOUT + 30.0
)

Launcher = Callable[..., tuple[Path, int]]


class _Clock:
    """A monotonic clock the test advances, so settling never really sleeps.

    A reader charges its own cost with `advance`, which is how a slow `ps` or
    working-directory reader is modeled without depending on machine speed.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, duration: float) -> None:
        self.now += duration

    def sleep(self, duration: float) -> None:
        self.advance(duration)


def _inject_clock(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> None:
    monkeypatch.setattr(procctl.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(procctl.time, "sleep", clock.sleep)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    """Run procctl, bounded, so a hung launch or stop fails instead of hanging."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=RUN_TIMEOUT,
    )


def _wait_until(predicate: Callable[[], bool], message: str) -> None:
    deadline = time.monotonic() + LIVENESS_TIMEOUT
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    if not predicate():
        pytest.fail(f"{message} within {LIVENESS_TIMEOUT:.0f} seconds")


def _pid_is_running(pid: int) -> bool:
    result = subprocess.run(
        ["ps", "-p", str(pid)],
        capture_output=True,
        timeout=procctl.READER_SUBPROCESS_TIMEOUT,
    )
    return result.returncode == 0


@pytest.fixture
def launcher(tmp_path: Path) -> Iterator[Launcher]:
    """Launch detached processes and stop them through the verified stop path.

    Teardown signals no PID or process-group number on its own authority: it
    hands a pristine copy of each fingerprint back to `procctl stop`, which
    refuses without signaling unless the live process is still the one that was
    launched. A PID retired during the test and reused by something else can
    therefore never be signaled, including when it has been reused by a
    concurrent run of this file in another checkout.

    A refusal that leaves the recorded PID running is a leak, not a success,
    so teardown checks the result and fails the test with the PID and the
    refusal text rather than letting a survivor go unnoticed.
    """
    cleanup: list[tuple[Path, int]] = []

    def register(fingerprint: Path) -> int:
        recorded = fingerprint.read_text()
        # Tests mutate the fingerprint to provoke refusals, so teardown keeps
        # the identity as it was actually recorded.
        pristine = fingerprint.with_suffix(".cleanup.json")
        pristine.write_text(recorded)
        pid = int(json.loads(recorded)["pid"])
        cleanup.append((pristine, pid))
        return pid

    def launch(*command: str) -> tuple[Path, int]:
        fingerprint = tmp_path / "process.json"
        if not command:
            command = (sys.executable, "-c", "import time; time.sleep(60)")
        arguments = (
            "launch",
            "--attempt",
            str(tmp_path),
            "--fingerprint",
            str(fingerprint),
            "--log",
            str(tmp_path / "process.log"),
            "--",
            *command,
        )
        try:
            result = _run(*arguments)
        except subprocess.TimeoutExpired:
            # The launcher outlived the harness. It may still have written a
            # fingerprint, so register whatever it recorded before failing;
            # otherwise the detached child would be left with no owner.
            if fingerprint.exists():
                register(fingerprint)
            raise
        assert result.returncode == 0, result.stderr
        return fingerprint, register(fingerprint)

    yield launch

    leaked = []
    for pristine, pid in cleanup:
        try:
            result = _run("stop", str(pristine), "--timeout", str(STOP_TIMEOUT))
        except subprocess.TimeoutExpired:
            leaked.append(f"PID {pid}: stop did not finish within {RUN_TIMEOUT:.0f} seconds")
            continue
        if result.returncode == 0:
            continue
        if not _pid_is_running(pid):
            # A refusal on a PID that has already exited is the tool declining
            # to signal a process it can no longer identify, which is correct.
            continue
        leaked.append(
            f"PID {pid} still running after stop exited {result.returncode}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    if leaked:
        pytest.fail("procctl left launched processes running:\n" + "\n".join(leaked))


def test_stop_verified_process(launcher: Launcher) -> None:
    fingerprint, _pid = launcher()
    result = _run("stop", str(fingerprint), "--timeout", "2")
    assert result.returncode == 0, result.stderr
    assert "sent SIGTERM to verified process group" in result.stdout
    assert "stopped" in result.stdout


def test_launch_records_the_identity_left_by_exec_indirection(
    launcher: Launcher, tmp_path: Path
) -> None:
    """A wrapper that replaces itself with its target must still launch.

    The indirection is a real `os.execv` from a Python wrapper, not a shell
    exec, so the command `ps` reports changes underneath a PID that never does.
    """
    wrapper_path = tmp_path / "wrapper.py"
    wrapper_path.write_text("""
import os
import sys
import time

time.sleep(0.5)
os.execv(sys.executable, [sys.executable, '-c', 'import time; time.sleep(60)'])
""")
    fingerprint, pid = launcher(sys.executable, str(wrapper_path))

    recorded = json.loads(fingerprint.read_text())
    assert recorded["pid"] == pid
    assert recorded["requested_command"].endswith("wrapper.py")
    # The settled identity is the target the wrapper exec'd, not the wrapper.
    assert "wrapper.py" not in recorded["observed_command"]
    assert "time.sleep(60)" in recorded["observed_command"]


def test_settlement_requires_sustained_identity_after_exec_indirection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = {"command": "wrapper", "pid": 42}
    target = {"command": "target", "pid": 42}
    observations = iter((wrapper, target, target))
    clock = _Clock()
    monkeypatch.setattr(
        procctl, "_live_fingerprint", lambda _pid, _deadline=None: next(observations)
    )
    _inject_clock(monkeypatch, clock)

    assert procctl._settled_live_fingerprint(42) == target
    # The change opens a fresh window, so the target is confirmed a full settle
    # duration after it was first seen rather than after the launch began.
    assert clock.now == pytest.approx(2 * procctl.FINGERPRINT_SETTLE_DURATION)


def test_settlement_survives_a_reader_slower_than_the_settle_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loaded machine settles late; it does not refuse a stable identity."""
    target = {"command": "target", "pid": 42}
    clock = _Clock()
    reads = 0

    def observe(_pid: int, _deadline: float | None = None) -> dict[str, object]:
        nonlocal reads
        reads += 1
        clock.advance(5 * procctl.FINGERPRINT_SETTLE_DURATION)
        return target

    monkeypatch.setattr(procctl, "_live_fingerprint", observe)
    _inject_clock(monkeypatch, clock)

    assert procctl._settled_live_fingerprint(42) == target
    assert reads == 2


def test_settlement_retries_transient_reader_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = {"command": "target", "pid": 42}
    observations = iter((procctl.Refusal("ps hiccup"), target, target))
    clock = _Clock()

    def observe(_pid: int, _deadline: float | None = None) -> dict[str, object]:
        value = next(observations)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(procctl, "_live_fingerprint", observe)
    _inject_clock(monkeypatch, clock)

    assert procctl._settled_live_fingerprint(42) == target


def test_settlement_treats_a_reader_failure_as_an_interruption_not_a_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failing to read an identity is not evidence that the identity differs.

    One identity, read successfully four times but interrupted between each
    read, must settle as soon as two reads agree across a full window. Counting
    those interruptions as changes would refuse it as four distinct identities.
    """
    target = {"command": "target", "pid": 42}
    hiccup = procctl.Refusal("ps hiccup")
    observations = iter((target, hiccup, target, hiccup, target, hiccup, target, target))
    clock = _Clock()

    def observe(_pid: int, _deadline: float | None = None) -> dict[str, object]:
        value = next(observations)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(procctl, "_live_fingerprint", observe)
    _inject_clock(monkeypatch, clock)

    assert procctl._settled_live_fingerprint(42) == target


def test_settlement_refuses_when_the_outer_deadline_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller-visible bound holds even when reads keep being interrupted."""
    target = {"command": "target", "pid": 42}
    clock = _Clock()
    reads = 0

    def observe(_pid: int, _deadline: float | None = None) -> dict[str, object]:
        nonlocal reads
        reads += 1
        # Every window is interrupted, so the identity is never confirmed and
        # only the outer deadline can end this.
        if reads % 2 == 0:
            raise procctl.Refusal("ps hiccup")
        return target

    monkeypatch.setattr(procctl, "_live_fingerprint", observe)
    _inject_clock(monkeypatch, clock)

    with pytest.raises(procctl.Refusal) as refusal:
        procctl._settled_live_fingerprint(42)

    assert clock.now <= procctl.LAUNCH_SETTLE_DEADLINE
    message = str(refusal.value)
    assert f"of {procctl.LAUNCH_SETTLE_DEADLINE:.0f}s" in message
    # The one identity seen is reported as one, not as one per interruption.
    assert "distinct identities: 1" in message
    assert "windows interrupted by reader failures: " in message


def test_ps_reads_the_command_untruncated(monkeypatch: pytest.MonkeyPatch) -> None:
    """ps must be asked for unlimited width, with the command read last.

    Without `-ww` both BSD and procps ps cut the command to the terminal width,
    and settling compares whole identities, so two commands sharing a surviving
    prefix would compare equal. Only the Linux legs of CI cut it short enough to
    notice, so this guards the invocation itself on every platform.
    """
    recorded: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        recorded.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout="user Sun Aug 16 12:00:00 2026 the command line\n", stderr=""
        )

    monkeypatch.setattr(procctl.subprocess, "run", fake_run)

    details = procctl._process_details(42)

    assert recorded[0][0] == "ps"
    assert "-ww" in recorded[0]
    # The command is the last field, so no fixed-width column can cut it.
    assert recorded[0][-1] == "command="
    assert details["command"] == "the command line"


def test_process_details_keeps_a_command_containing_spaces_whole() -> None:
    """A real ps reading must return the arguments, not just the executable."""
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    with subprocess.Popen(command, stdout=subprocess.DEVNULL) as child:
        try:
            details = procctl._process_details(child.pid)
            deadline = time.monotonic() + STOP_TIMEOUT
            # macOS reports a temporary `(name)` placeholder until arguments
            # are readable.
            while "time.sleep(60)" not in details["command"] and time.monotonic() < deadline:
                time.sleep(0.05)
                details = procctl._process_details(child.pid)
            assert "time.sleep(60)" in details["command"], details["command"]
        finally:
            child.kill()
            child.wait(timeout=LIVENESS_TIMEOUT)


def test_reader_bounds_each_inspection_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hung ps or lsof becomes a refusal instead of consuming the deadline."""
    # A PID that cannot exist, so the Linux /proc fast path in _process_cwd is
    # absent and both readers reach their subprocess on every platform.
    unusable_pid = 2**31 - 1
    calls: list[object] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        calls.append(kwargs.get("timeout"))
        raise subprocess.TimeoutExpired(str(command), procctl.READER_SUBPROCESS_TIMEOUT)

    monkeypatch.setattr(procctl.subprocess, "run", fake_run)

    with pytest.raises(procctl.Refusal, match="could not run ps"):
        procctl._process_details(unusable_pid)
    with pytest.raises(procctl.Refusal, match="could not inspect working directory"):
        procctl._process_cwd(unusable_pid)

    assert calls == [procctl.READER_SUBPROCESS_TIMEOUT, procctl.READER_SUBPROCESS_TIMEOUT]


def test_reader_timeout_is_clamped_to_the_launch_deadline() -> None:
    """Readers may not run past the deadline, which is what bounds the launch."""
    now = time.monotonic()
    # Plenty of budget left: the hang cap governs.
    assert procctl._reader_timeout(42, now + 10_000) == procctl.READER_SUBPROCESS_TIMEOUT
    # Little budget left: the deadline governs.
    assert procctl._reader_timeout(42, now + 1.0) < procctl.READER_SUBPROCESS_TIMEOUT
    # No budget left: the reader is not started at all.
    with pytest.raises(procctl.Refusal, match="no time left to inspect PID 42"):
        procctl._reader_timeout(42, now - 100.0)
    # No deadline, as when stop reads a live identity outside a launch.
    assert procctl._reader_timeout(42, None) == procctl.READER_SUBPROCESS_TIMEOUT


def test_settlement_refuses_a_reading_that_finished_after_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late reading is not evidence, even when it matches the one before it."""
    target = {"command": "target", "pid": 42}
    clock = _Clock()
    reads = 0

    def observe(_pid: int, _deadline: float | None = None) -> dict[str, object]:
        nonlocal reads
        reads += 1
        if reads == 2:
            clock.advance(2 * procctl.LAUNCH_SETTLE_DEADLINE)
        return target

    monkeypatch.setattr(procctl, "_live_fingerprint", observe)
    _inject_clock(monkeypatch, clock)

    # Without the post-read check this pair would confirm the identity, because
    # the two readings agree and span far more than the settle duration.
    with pytest.raises(procctl.Refusal, match="finished after the launch deadline"):
        procctl._settled_live_fingerprint(42)
    assert reads == 2


def test_settlement_refuses_an_endlessly_changing_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    reads = 0

    def observe(_pid: int, _deadline: float | None = None) -> dict[str, object]:
        nonlocal reads
        reads += 1
        return {"command": f"stage-{reads}", "pid": 42}

    monkeypatch.setattr(procctl, "_live_fingerprint", observe)
    _inject_clock(monkeypatch, clock)

    with pytest.raises(procctl.Refusal) as refusal:
        procctl._settled_live_fingerprint(42)

    assert reads == procctl.FINGERPRINT_SETTLE_CHANGES + 1
    message = str(refusal.value)
    assert f"distinct identities: {reads}" in message
    assert f"stage-{procctl.FINGERPRINT_SETTLE_CHANGES}" in message


def test_settlement_refuses_an_unreadable_process_and_names_the_reader_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()

    def observe(_pid: int, _deadline: float | None = None) -> dict[str, object]:
        raise procctl.Refusal("PID 42 is not running")

    monkeypatch.setattr(procctl, "_live_fingerprint", observe)
    _inject_clock(monkeypatch, clock)

    with pytest.raises(procctl.Refusal) as refusal:
        procctl._settled_live_fingerprint(42)

    message = str(refusal.value)
    assert "last reader error: 'PID 42 is not running'" in message
    assert "last identity: None" in message


def test_refuses_stale_fingerprint_without_signalling(launcher: Launcher) -> None:
    fingerprint, pid = launcher()
    stale = json.loads(fingerprint.read_text())
    stale["start_time"] = "Mon Jan  1 00:00:00 1990"
    fingerprint.write_text(json.dumps(stale))

    result = _run("stop", str(fingerprint))

    assert result.returncode == 2
    assert "REFUSED: fingerprint mismatch; no signal sent" in result.stderr
    assert "start_time" in result.stderr
    assert _pid_is_running(pid)


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ({"start_time": "Mon Jan  1 00:00:00 1990"}, "start_time"),
        ({"host": "different-host"}, "host"),
        ({"user": "different-user"}, "user"),
        ({"attempt_path": "/different/attempt"}, "attempt_path"),
    ],
)
def test_stop_refuses_identity_mismatch_without_signalling(
    launcher: Launcher,
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, str],
    expected_message: str,
) -> None:
    fingerprint, _pid = launcher()
    stale = json.loads(fingerprint.read_text())
    stale.update(mutation)
    fingerprint.write_text(json.dumps(stale))
    killpg = Mock(side_effect=AssertionError("stop must not signal a mismatched process"))
    monkeypatch.setattr(os, "killpg", killpg)
    args = argparse.Namespace(fingerprint=fingerprint, timeout=0.01)

    with pytest.raises(procctl.Refusal, match=expected_message):
        procctl.stop(args)
    killpg.assert_not_called()


def test_stop_refuses_dead_pid_without_signalling(
    launcher: Launcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    fingerprint, pid = launcher()
    # Retire the PID through the verified path rather than with an unverified
    # kill this test sends itself.
    assert _run("stop", str(fingerprint), "--timeout", "5").returncode == 0
    _wait_until(lambda: not _pid_is_running(pid), f"PID {pid} did not exit")
    killpg = Mock(side_effect=AssertionError("stop must not signal a dead PID"))
    monkeypatch.setattr(os, "killpg", killpg)

    with pytest.raises(procctl.Refusal, match="is not running"):
        procctl.stop(argparse.Namespace(fingerprint=fingerprint, timeout=0.01))
    killpg.assert_not_called()


@pytest.mark.parametrize(
    ("requested_command", "live_command"),
    (
        ("/resolved/wrapper launched-target", "launched-target"),
        ("python3 wrapper.py target", "target"),
    ),
    ids=("resolved-wrapper-exec", "python-wrapper-exec"),
)
def test_stop_accepts_arbitrary_post_exec_command(
    monkeypatch: pytest.MonkeyPatch, requested_command: str, live_command: str
) -> None:
    fingerprint = {
        "host": "host",
        "user": "user",
        "pid": 42,
        "pgid": 42,
        "start_time": "start",
        "requested_command": requested_command,
        "observed_command": requested_command,
        "attempt_path": "/attempt",
    }
    live = {
        "host": "host",
        "user": "user",
        "pid": 42,
        "start_time": "start",
        "command": live_command,
        "attempt_path": "/attempt",
    }
    killpg = Mock()
    monkeypatch.setattr(procctl, "_read_fingerprint", lambda _path: fingerprint)
    monkeypatch.setattr(procctl, "_live_fingerprint", lambda _pid: live)
    monkeypatch.setattr(procctl, "_wait_for_process_group_exit", lambda _pgid, _timeout: True)
    monkeypatch.setattr(procctl.os, "killpg", killpg)

    result = procctl.stop(argparse.Namespace(fingerprint=Path("unused"), timeout=0.01))

    assert result == 0
    killpg.assert_called_once_with(42, procctl.signal.SIGTERM)


def test_stop_refuses_same_second_pid_reuse_by_precise_start_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fingerprint = {
        "host": "host",
        "user": "user",
        "pid": 42,
        "pgid": 42,
        "start_time": "darwin-timeval:1786852800:100000",
        "requested_command": "wrapper launched-target",
        "observed_command": "wrapper launched-target",
        "attempt_path": "/attempt",
    }
    start_time = Mock(return_value="darwin-timeval:1786852800:900000")
    killpg = Mock(side_effect=AssertionError("stop must not signal an imposter"))
    monkeypatch.setattr(procctl, "_read_fingerprint", lambda _path: fingerprint)
    monkeypatch.setattr(
        procctl,
        "_process_details",
        lambda _pid, _deadline=None: {
            "user": "user",
            "start_time": "Sun Aug 16 12:00:00 2026",
            "command": "launched-target",
        },
    )
    monkeypatch.setattr(procctl, "_process_start_time", start_time)
    monkeypatch.setattr(procctl, "_process_cwd", lambda _pid, _deadline=None: "/attempt")
    monkeypatch.setattr(procctl.socket, "gethostname", lambda: "host")
    monkeypatch.setattr(procctl.os, "killpg", killpg)

    with pytest.raises(procctl.Refusal, match="start_time"):
        procctl.stop(argparse.Namespace(fingerprint=Path("unused"), timeout=0.01))
    start_time.assert_called_once_with(42, "Sun Aug 16 12:00:00 2026")
    killpg.assert_not_called()


def test_stop_terminates_sigterm_ignoring_process_group(launcher: Launcher, tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    wrapper_path = tmp_path / "wrapper.py"
    wrapper_path.write_text("""
import os
import signal
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = os.fork()
if child == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(1)
else:
    with open('child.pid', 'w', encoding='utf-8') as stream:
        stream.write(str(child))
    while True:
        time.sleep(1)
""")
    fingerprint, pid = launcher(sys.executable, str(wrapper_path))
    pgid = os.getpgid(pid)
    _wait_until(child_pid_path.exists, "the wrapper did not fork a child")

    result = _run("stop", str(fingerprint), "--timeout", "2")

    assert result.returncode == 0, result.stderr
    _wait_until(
        lambda: not procctl._process_group_exists(pgid),
        f"process group {pgid} survived stop",
    )


def test_launch_failure_escalates_and_confirms_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = Mock(pid=42)
    process.wait.side_effect = (subprocess.TimeoutExpired("procctl", 2), 0)
    killpg = Mock()
    monkeypatch.setattr(procctl, "_wait_for_process_group_exit", Mock(side_effect=(False, True)))
    monkeypatch.setattr(procctl.os, "killpg", killpg)

    message = procctl._terminate_failed_launch(process, 42)

    assert killpg.call_args_list == [
        call(42, procctl.signal.SIGTERM),
        call(42, procctl.signal.SIGKILL),
    ]
    assert process.wait.call_count == 2
    assert message == "sent SIGTERM then SIGKILL to process group 42; exit confirmed"
