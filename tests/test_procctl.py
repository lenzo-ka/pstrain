from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "procctl.py"
SPEC = importlib.util.spec_from_file_location("procctl", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
procctl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(procctl)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


def _launch(tmp_path: Path, *command: str) -> tuple[Path, int]:
    fingerprint = tmp_path / "process.json"
    if not command:
        command = (sys.executable, "-c", "import time; time.sleep(60)")
    result = _run(
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
    assert result.returncode == 0, result.stderr
    return fingerprint, json.loads(fingerprint.read_text())["pid"]


def test_stop_verified_process(tmp_path: Path) -> None:
    fingerprint, _pid = _launch(tmp_path)
    result = _run("stop", str(fingerprint), "--timeout", "2")
    assert result.returncode == 0, result.stderr
    assert "sent SIGTERM to verified PID" in result.stdout
    assert "stopped" in result.stdout


def test_settlement_requires_sustained_identity_after_exec_indirection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = {"command": "wrapper", "pid": 42}
    target = {"command": "target", "pid": 42}
    observations = iter((wrapper, target, target))
    clock = Mock(side_effect=(0.0, 0.0, 0.0, 0.5, 0.5, 1.5))
    monkeypatch.setattr(procctl, "_live_fingerprint", lambda _pid: next(observations))
    monkeypatch.setattr(procctl.time, "monotonic", clock)
    monkeypatch.setattr(procctl.time, "sleep", lambda _seconds: None)

    assert procctl._settled_live_fingerprint(42) == target
    assert clock.call_count == 6


def test_settlement_retries_transient_reader_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = {"command": "target", "pid": 42}
    observations = iter((target, procctl.Refusal("ps hiccup"), target, target))
    clock = iter((0.0, 0.0, 0.0, 0.2, 0.2, 0.4, 0.4, 1.5))

    def observe(_pid: int) -> dict[str, object]:
        value = next(observations)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(procctl, "_live_fingerprint", observe)
    monkeypatch.setattr(procctl.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(procctl.time, "sleep", lambda _seconds: None)

    assert procctl._settled_live_fingerprint(42) == target


def test_refuses_stale_fingerprint_without_signalling(tmp_path: Path) -> None:
    fingerprint, pid = _launch(tmp_path)
    stale = json.loads(fingerprint.read_text())
    stale["start_time"] = "Mon Jan  1 00:00:00 1990"
    fingerprint.write_text(json.dumps(stale))
    try:
        result = _run("stop", str(fingerprint))
        assert result.returncode == 2
        assert "REFUSED: fingerprint mismatch; no signal sent" in result.stderr
        assert "start_time" in result.stderr
        subprocess.run(["ps", "-p", str(pid)], check=True, capture_output=True)
    finally:
        subprocess.run(["kill", str(pid)], check=False)
        time.sleep(0.1)


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ({"start_time": "Mon Jan  1 00:00:00 1990"}, "start_time"),
        (
            {
                "requested_command": "unrelated --one",
                "observed_command": "unrelated --two",
            },
            "command",
        ),
    ],
)
def test_stop_refuses_identity_mismatch_without_signalling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, str],
    expected_message: str,
) -> None:
    fingerprint, pid = _launch(tmp_path)
    stale = json.loads(fingerprint.read_text())
    stale.update(mutation)
    fingerprint.write_text(json.dumps(stale))
    kill = Mock(side_effect=AssertionError("stop must not signal a mismatched process"))
    monkeypatch.setattr(os, "kill", kill)
    args = argparse.Namespace(fingerprint=fingerprint, timeout=0.01)
    try:
        with pytest.raises(procctl.Refusal, match=expected_message):
            procctl.stop(args)
        kill.assert_not_called()
    finally:
        subprocess.run(["kill", str(pid)], check=False, capture_output=True)


def test_stop_refuses_dead_pid_without_signalling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fingerprint, pid = _launch(tmp_path)
    subprocess.run(["kill", str(pid)], check=True, capture_output=True)
    time.sleep(0.1)
    kill = Mock(side_effect=AssertionError("stop must not signal a dead PID"))
    monkeypatch.setattr(os, "kill", kill)

    with pytest.raises(procctl.Refusal, match="is not running"):
        procctl.stop(argparse.Namespace(fingerprint=fingerprint, timeout=0.01))
    kill.assert_not_called()


@pytest.mark.parametrize("matching_field", ("requested_command", "observed_command"))
def test_stop_accepts_either_recorded_command(
    monkeypatch: pytest.MonkeyPatch, matching_field: str
) -> None:
    fingerprint = {
        "host": "host",
        "user": "user",
        "pid": 42,
        "start_time": "start",
        "requested_command": "requested",
        "observed_command": "observed",
        "attempt_path": "/attempt",
    }
    live = {
        "host": "host",
        "user": "user",
        "pid": 42,
        "start_time": "start",
        "command": fingerprint[matching_field],
        "attempt_path": "/attempt",
    }
    kill = Mock()
    monkeypatch.setattr(procctl, "_read_fingerprint", lambda _path: fingerprint)
    monkeypatch.setattr(procctl, "_live_fingerprint", lambda _pid: live)
    monkeypatch.setattr(procctl, "_process_details", Mock(side_effect=procctl.Refusal("stopped")))
    monkeypatch.setattr(procctl.os, "kill", kill)

    result = procctl.stop(argparse.Namespace(fingerprint=Path("unused"), timeout=0.01))

    assert result == 0
    kill.assert_called_once_with(42, procctl.signal.SIGTERM)


def test_launch_failure_escalates_and_confirms_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = Mock(pid=42)
    process.wait.side_effect = (subprocess.TimeoutExpired("procctl", 2), 0)

    message = procctl._terminate_failed_launch(process)

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.wait.call_count == 2
    assert message == "sent SIGTERM then SIGKILL to PID 42; exit confirmed"
