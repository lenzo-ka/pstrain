#!/usr/bin/env python3
"""Launch and safely stop one detached process using a JSON fingerprint."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

FIELDS = (
    "host",
    "user",
    "pid",
    "pgid",
    "start_time",
    "requested_command",
    "observed_command",
    "attempt_path",
)
FINGERPRINT_SETTLE_INTERVAL = 0.1
FINGERPRINT_SETTLE_DURATION = 1.0
FINGERPRINT_SETTLE_TIMEOUT = 2.0
LAUNCH_CLEANUP_TIMEOUT = 2.0


class Refusal(Exception):
    """The fingerprint does not identify the current process."""


def _process_details(pid: int) -> dict[str, str]:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "user=", "-o", "lstart=", "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise Refusal(f"could not run ps to inspect PID {pid}: {error}") from error
    line = result.stdout.strip()
    if result.returncode != 0 or not line:
        raise Refusal(f"PID {pid} is not running")
    parts = line.split(maxsplit=6)
    if len(parts) != 7:
        raise Refusal(f"could not inspect PID {pid} with ps")
    user, *started, command = parts
    return {"user": user, "start_time": " ".join(started), "command": command}


def _process_cwd(pid: int) -> str:
    proc_cwd = Path(f"/proc/{pid}/cwd")
    if proc_cwd.exists():
        return str(proc_cwd.resolve(strict=True))
    try:
        result = subprocess.run(
            ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise Refusal(f"could not inspect working directory for PID {pid}: {error}") from error
    paths = [line[1:] for line in result.stdout.splitlines() if line.startswith("n")]
    if result.returncode != 0 or len(paths) != 1:
        raise Refusal(f"could not inspect working directory for PID {pid}")
    return str(Path(paths[0]).resolve(strict=True))


def _live_fingerprint(pid: int) -> dict[str, Any]:
    details = _process_details(pid)
    return {
        "host": socket.gethostname(),
        "user": details["user"],
        "pid": pid,
        "start_time": details["start_time"],
        "command": details["command"],
        "attempt_path": _process_cwd(pid),
    }


def _settled_live_fingerprint(pid: int) -> dict[str, Any]:
    """Read an identity unchanged for a full second within the launch budget."""
    started = time.monotonic()
    deadline = started + FINGERPRINT_SETTLE_TIMEOUT
    previous: dict[str, Any] | None = None
    stable_since = started
    while True:
        now = time.monotonic()
        if now > deadline:
            break
        try:
            current = _live_fingerprint(pid)
        except Refusal:
            previous = None
            stable_since = now
        else:
            if current != previous:
                previous = current
                stable_since = now
            elif now - stable_since >= FINGERPRINT_SETTLE_DURATION:
                return current
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(FINGERPRINT_SETTLE_INTERVAL, remaining))
    raise Refusal(f"process identity for PID {pid} did not settle after launch")


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_group_exists(pgid):
            return True
        time.sleep(0.05)
    return not _process_group_exists(pgid)


def _terminate_failed_launch(process: subprocess.Popen[bytes], pgid: int) -> str:
    """Terminate a child after launch bookkeeping fails, escalating if needed."""
    os.killpg(pgid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=LAUNCH_CLEANUP_TIMEOUT)
    if _wait_for_process_group_exit(pgid, LAUNCH_CLEANUP_TIMEOUT):
        return f"sent SIGTERM to process group {pgid}; exit confirmed"
    os.killpg(pgid, signal.SIGKILL)
    try:
        process.wait(timeout=LAUNCH_CLEANUP_TIMEOUT)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"process group {pgid} survived SIGTERM and SIGKILL after launch failure"
        ) from error
    if not _wait_for_process_group_exit(pgid, LAUNCH_CLEANUP_TIMEOUT):
        raise RuntimeError(
            f"process group {pgid} survived SIGTERM and SIGKILL after launch failure"
        )
    return f"sent SIGTERM then SIGKILL to process group {pgid}; exit confirmed"


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def launch(args: argparse.Namespace) -> int:
    attempt = args.attempt.resolve(strict=True)
    fingerprint_path = args.fingerprint.resolve()
    log_path = args.log.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            args.command,
            cwd=attempt,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pgid = os.getpgid(process.pid)
    if pgid != process.pid:
        cleanup = _terminate_failed_launch(process, pgid)
        raise RuntimeError(f"detached process did not lead its process group; {cleanup}")
    try:
        fingerprint = _settled_live_fingerprint(process.pid)
        if fingerprint["attempt_path"] != str(attempt):
            raise RuntimeError("detached process did not start in the attempt path")
        fingerprint["observed_command"] = fingerprint.pop("command")
        fingerprint["requested_command"] = " ".join(args.command)
        fingerprint["pgid"] = pgid
        _write_json_atomic(fingerprint_path, fingerprint)
    except BaseException:
        cleanup = _terminate_failed_launch(process, pgid)
        print(f"launch failed; {cleanup}", file=sys.stderr)
        raise
    print(f"launched PID {process.pid}; fingerprint: {fingerprint_path}")
    return 0


def _read_fingerprint(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refusal(f"cannot read fingerprint {path}: {error}") from error
    if not isinstance(value, dict):
        raise Refusal("fingerprint must be a JSON object")
    missing = [field for field in FIELDS if field not in value]
    extra = [field for field in value if field not in FIELDS]
    integer_fields = ("pid", "pgid")
    invalid_strings = [
        field
        for field in FIELDS
        if field not in integer_fields and not isinstance(value.get(field), str)
    ]
    invalid_integers = [field for field in integer_fields if type(value.get(field)) is not int]
    if (
        missing
        or extra
        or invalid_integers
        or invalid_strings
        or value.get("pgid") != value.get("pid")
    ):
        raise Refusal(
            "invalid fingerprint fields "
            f"(missing={missing}, extra={extra}, non_strings={invalid_strings}, "
            f"non_integers={invalid_integers}, pgid_must_equal_pid={value.get('pgid') != value.get('pid')})"
        )
    return value


def stop(args: argparse.Namespace) -> int:
    expected = _read_fingerprint(args.fingerprint)
    live = _live_fingerprint(expected["pid"])
    identity_fields = ("host", "user", "pid", "start_time", "attempt_path")
    mismatches = [
        f"{field}: fingerprint={expected[field]!r}, live={live[field]!r}"
        for field in identity_fields
        if expected[field] != live[field]
    ]
    if mismatches:
        raise Refusal("fingerprint mismatch; no signal sent:\n  " + "\n  ".join(mismatches))
    pgid = expected["pgid"]
    os.killpg(pgid, signal.SIGTERM)
    print(f"sent SIGTERM to verified process group {pgid}")
    if _wait_for_process_group_exit(pgid, args.timeout):
        print(f"process group {pgid} stopped")
        return 0
    os.killpg(pgid, signal.SIGKILL)
    print(f"sent SIGKILL to verified process group {pgid}", file=sys.stderr)
    if _wait_for_process_group_exit(pgid, args.timeout):
        print(f"process group {pgid} stopped")
        return 0
    print(f"process group {pgid} survived SIGKILL", file=sys.stderr)
    return 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="action", required=True)
    launch_parser = commands.add_parser("launch", help="launch a detached process")
    launch_parser.add_argument("--attempt", type=Path, required=True)
    launch_parser.add_argument("--fingerprint", type=Path, required=True)
    launch_parser.add_argument("--log", type=Path, required=True)
    launch_parser.add_argument("command", nargs=argparse.REMAINDER)
    launch_parser.set_defaults(run=launch)
    stop_parser = commands.add_parser("stop", help="verify a fingerprint and send SIGTERM")
    stop_parser.add_argument("fingerprint", type=Path)
    stop_parser.add_argument("--timeout", type=float, default=10.0)
    stop_parser.set_defaults(run=stop)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.action == "launch" and (not args.command or args.command[0] == "--"):
        if args.command and args.command[0] == "--":
            args.command = args.command[1:]
        if not args.command:
            parser().error("launch requires a command after --")
    try:
        return args.run(args)
    except Refusal as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
