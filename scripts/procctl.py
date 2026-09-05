#!/usr/bin/env python3
"""Launch and safely stop one detached process using a JSON fingerprint."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import signal
import socket
import struct
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
FINGERPRINT_SETTLE_CHANGES = 3
FINGERPRINT_READER_TIMEOUT = 2.0
# The outer bound on identity settling, which a caller can rely on: settling
# produces an identity or refuses within this many seconds, plus the cost of
# reaping the last reader subprocess. Readers are clamped to the time left
# inside it and none is started once it has passed. A refused launch also pays
# for cleanup below, so it is not the bound on launch as a whole.
LAUNCH_SETTLE_DEADLINE = 60.0
# A hang cap for one reader, not a latency threshold. Measured worst case for
# `lsof` on a loaded twelve-core machine at nice 19 is under seven seconds, so
# this fires only on a reader that has genuinely stopped, never on a slow one.
READER_SUBPROCESS_TIMEOUT = 30.0
# Each of the four waits a failed launch makes while disposing of its child:
# for the leader and then its group after SIGTERM, and both again after
# SIGKILL. A refused launch therefore returns within the settling bound plus
# four of these.
LAUNCH_CLEANUP_TIMEOUT = 2.0


class Refusal(Exception):
    """The fingerprint does not identify the current process."""


def _reader_timeout(pid: int, deadline: float | None) -> float:
    """Bound one reader by its hang cap and by the time left before `deadline`.

    A reader is never started once the deadline has passed, because a reader
    granted even a token timeout could return a sample the launcher has no
    budget left to trust.
    """
    if deadline is None:
        return READER_SUBPROCESS_TIMEOUT
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise Refusal(f"no time left to inspect PID {pid} before the launch deadline")
    return min(READER_SUBPROCESS_TIMEOUT, remaining)


def _process_details(pid: int, deadline: float | None = None) -> dict[str, str]:
    try:
        result = subprocess.run(
            # -ww lifts the width limit both BSD and procps ps otherwise impose,
            # which truncates the command to the terminal width and, with no
            # terminal, to eighty columns less the fields printed before it. The
            # loss is not cosmetic: settling compares whole identities, so a
            # wrapper and the target it execs can share a surviving prefix and
            # compare equal, which would settle an identity that never stopped
            # changing. The command stays last so no fixed-width column cuts it.
            ["ps", "-ww", "-p", str(pid), "-o", "user=", "-o", "lstart=", "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=_reader_timeout(pid, deadline),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Refusal(f"could not run ps to inspect PID {pid}: {error}") from error
    line = result.stdout.strip()
    if result.returncode != 0 or not line:
        raise Refusal(f"PID {pid} is not running")
    parts = line.split(maxsplit=6)
    if len(parts) != 7:
        raise Refusal(f"could not inspect PID {pid} with ps")
    user, *started, command = parts
    return {"user": user, "start_time": " ".join(started), "command": command}


def _linux_process_start_time(pid: int) -> str:
    """Return Linux's process start tick, which is stable across exec."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise Refusal(f"could not inspect start time for PID {pid}: {error}") from error
    closing_parenthesis = stat.rfind(")")
    fields_after_command = stat[closing_parenthesis + 2 :].split()
    if closing_parenthesis < 0 or len(fields_after_command) < 20:
        raise Refusal(f"could not inspect start time for PID {pid} from /proc")
    return f"linux-proc-ticks:{fields_after_command[19]}"


def _darwin_process_start_time(pid: int) -> str:
    """Return kinfo_proc.p_starttime as seconds and microseconds on macOS."""
    libc = ctypes.CDLL(None, use_errno=True)
    mib = (ctypes.c_int * 4)(1, 14, 1, pid)  # CTL_KERN, KERN_PROC, KERN_PROC_PID
    size = ctypes.c_size_t()
    if libc.sysctl(mib, 4, None, ctypes.byref(size), None, 0) != 0:
        error = ctypes.get_errno()
        raise Refusal(f"could not inspect start time for PID {pid}: {os.strerror(error)}")
    if size.value < 16:
        raise Refusal(f"PID {pid} is not running")
    buffer = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 4, buffer, ctypes.byref(size), None, 0) != 0:
        error = ctypes.get_errno()
        raise Refusal(f"could not inspect start time for PID {pid}: {os.strerror(error)}")
    if size.value < 16:
        raise Refusal(f"PID {pid} is not running")
    seconds, microseconds = struct.unpack_from("@qi", buffer.raw)
    return f"darwin-timeval:{seconds}:{microseconds}"


def _process_start_time(pid: int, ps_start_time: str) -> str:
    if sys.platform.startswith("linux"):
        return _linux_process_start_time(pid)
    if sys.platform == "darwin":
        return _darwin_process_start_time(pid)
    # Unsupported platforms retain the second-granularity ps value. This has a
    # documented same-second PID-reuse residual and is never supplemented by a
    # command match.
    return f"ps-lstart:{ps_start_time}"


def _process_cwd(pid: int, deadline: float | None = None) -> str:
    proc_cwd = Path(f"/proc/{pid}/cwd")
    if proc_cwd.exists():
        return str(proc_cwd.resolve(strict=True))
    try:
        result = subprocess.run(
            ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            check=False,
            capture_output=True,
            text=True,
            timeout=_reader_timeout(pid, deadline),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Refusal(f"could not inspect working directory for PID {pid}: {error}") from error
    paths = [line[1:] for line in result.stdout.splitlines() if line.startswith("n")]
    if result.returncode != 0 or len(paths) != 1:
        raise Refusal(f"could not inspect working directory for PID {pid}")
    return str(Path(paths[0]).resolve(strict=True))


def _live_fingerprint(pid: int, deadline: float | None = None) -> dict[str, Any]:
    details = _process_details(pid, deadline)
    return {
        "host": socket.gethostname(),
        "user": details["user"],
        "pid": pid,
        "start_time": _process_start_time(pid, details["start_time"]),
        "command": details["command"],
        "attempt_path": _process_cwd(pid, deadline),
    }


def _settled_live_fingerprint(pid: int) -> dict[str, Any]:
    """Read an identity unchanged for a full second, confirming it by observation.

    Two rules apply. The inner rule counts confirmation windows rather than
    wall-clock seconds: a sample that differs from the one before it opens a
    new window, which is what a wrapper that execs its target needs, and
    `FINGERPRINT_SETTLE_CHANGES` such changes are tolerated. A reader failure
    interrupts the window and the settle wait restarts, but it is not a change,
    because failing to read an identity is not evidence that it differs. So a
    machine slow enough to make `ps` and the working-directory reader take
    longer than a window settles later instead of refusing.

    The outer rule is `LAUNCH_SETTLE_DEADLINE`, the bound a caller can rely on
    for settling: this returns an identity or refuses within that many seconds
    plus the cost of reaping the last reader subprocess. No reader is started
    once the deadline has passed, each is capped by `READER_SUBPROCESS_TIMEOUT`
    and clamped again to the time left, and a reading that still finishes late
    is refused rather than accepted, so no identity is ever confirmed on
    evidence gathered after the deadline. A refusal reports the last identity
    read, the last reader error, and counts of the distinct identities,
    interrupted windows, and reader failures behind them.

    That bound covers settling only. A caller whose launch is refused waits for
    it and then for `_terminate_failed_launch`, which spends up to four
    `LAUNCH_CLEANUP_TIMEOUT` periods disposing of the detached child, so a
    refused launch returns in about sixty-eight seconds rather than sixty.
    """
    started = time.monotonic()
    outer_deadline = started + LAUNCH_SETTLE_DEADLINE
    previous: dict[str, Any] | None = None
    stable_since: float | None = None
    changes = 0
    interruptions = 0
    failures = 0
    last_error: str | None = None
    reader_deadline = started + FINGERPRINT_READER_TIMEOUT
    while True:
        try:
            current = _live_fingerprint(pid, outer_deadline)
        except Refusal as error:
            last_error = str(error)
            failures += 1
            if stable_since is not None:
                # The window is interrupted, not invalidated: the identity is
                # unconfirmed, but nothing says it changed.
                interruptions += 1
                stable_since = None
            now = time.monotonic()
            # Two attempts before giving up, so one slow failing read cannot
            # decide the launch on its own.
            if now >= outer_deadline or (failures >= 2 and now >= reader_deadline):
                break
            time.sleep(min(FINGERPRINT_SETTLE_INTERVAL, outer_deadline - now))
            continue
        observed_at = time.monotonic()
        if observed_at >= outer_deadline:
            # The read overran the budget it was clamped to. Accepting it would
            # confirm an identity on evidence gathered after the deadline.
            last_error = "the last reading finished after the launch deadline"
            break
        failures = 0
        reader_deadline = observed_at + FINGERPRINT_READER_TIMEOUT
        if current == previous:
            if stable_since is None:
                stable_since = observed_at
            elif observed_at - stable_since >= FINGERPRINT_SETTLE_DURATION:
                return current
        else:
            changes += 1
            if changes > FINGERPRINT_SETTLE_CHANGES:
                break
            previous = current
            stable_since = observed_at
        now = time.monotonic()
        if now >= outer_deadline:
            break
        delay = min(stable_since + FINGERPRINT_SETTLE_DURATION - now, outer_deadline - now)
        if delay > 0:
            time.sleep(delay)
    raise Refusal(
        f"process identity for PID {pid} did not settle after launch "
        f"({time.monotonic() - started:.1f}s of {LAUNCH_SETTLE_DEADLINE:.0f}s; "
        f"distinct identities: {changes}, windows interrupted by reader failures: "
        f"{interruptions}, consecutive reader failures: {failures}, "
        f"last identity: {previous!r}, last reader error: {last_error!r})"
    )


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
