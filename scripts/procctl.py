#!/usr/bin/env python3
"""Launch and safely stop one detached process using a JSON fingerprint."""

from __future__ import annotations

import argparse
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

FIELDS = ("host", "user", "pid", "start_time", "command", "attempt_path")
FINGERPRINT_SETTLE_INTERVAL = 0.1
FINGERPRINT_SETTLE_TIMEOUT = 2.0


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
    """Read a live identity after any immediate exec indirection has settled."""
    previous = _live_fingerprint(pid)
    deadline = time.monotonic() + FINGERPRINT_SETTLE_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(FINGERPRINT_SETTLE_INTERVAL)
        current = _live_fingerprint(pid)
        if current == previous:
            return current
        previous = current
    raise Refusal(f"process identity for PID {pid} did not settle after launch")


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
    try:
        fingerprint = _settled_live_fingerprint(process.pid)
        if fingerprint["attempt_path"] != str(attempt):
            raise RuntimeError("detached process did not start in the attempt path")
        _write_json_atomic(fingerprint_path, fingerprint)
    except BaseException:
        process.terminate()
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
    invalid_strings = [
        field for field in FIELDS if field != "pid" and not isinstance(value.get(field), str)
    ]
    if missing or extra or type(value.get("pid")) is not int or invalid_strings:
        raise Refusal(
            "invalid fingerprint fields "
            f"(missing={missing}, extra={extra}, non_strings={invalid_strings})"
        )
    return value


def stop(args: argparse.Namespace) -> int:
    expected = _read_fingerprint(args.fingerprint)
    live = _live_fingerprint(expected["pid"])
    mismatches = [
        f"{field}: fingerprint={expected[field]!r}, live={live[field]!r}"
        for field in FIELDS
        if expected[field] != live[field]
    ]
    if mismatches:
        raise Refusal("fingerprint mismatch; no signal sent:\n  " + "\n  ".join(mismatches))
    os.kill(expected["pid"], signal.SIGTERM)
    print(f"sent SIGTERM to verified PID {expected['pid']}")
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            _process_details(expected["pid"])
        except Refusal:
            print(f"PID {expected['pid']} stopped")
            return 0
        time.sleep(0.05)
    print(f"PID {expected['pid']} still running after {args.timeout:g}s", file=sys.stderr)
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
