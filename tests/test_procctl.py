from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "procctl.py"


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


def test_fingerprints_command_after_exec_indirection(tmp_path: Path) -> None:
    shim = tmp_path / "exec-shim"
    shim.write_text('#!/bin/sh\nsleep 0.05\nexec "$@"\n')
    shim.chmod(0o755)
    fingerprint, pid = _launch(
        tmp_path,
        str(shim),
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
    )
    recorded = json.loads(fingerprint.read_text())
    try:
        assert str(shim) not in recorded["command"]
        result = _run("stop", str(fingerprint), "--timeout", "2")
        assert result.returncode == 0, result.stderr
    finally:
        subprocess.run(["kill", str(pid)], check=False, capture_output=True)


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
