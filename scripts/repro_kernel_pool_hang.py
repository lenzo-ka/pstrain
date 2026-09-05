#!/usr/bin/env python3
"""Reproduce the Jupyter ``Pipeline.run(..., jobs>1)`` process-pool stall.

Run from the repository root after building the native library::

    PYTHONPATH=. nice -n 19 python scripts/repro_kernel_pool_hang.py \\
        --runs 20 --jobs 2 --timeout 120 [--server] [--reuse-kernel]

Each attempt trains the mini-Arctic fixture to a target, driving the pipeline
exactly the way the tutorial notebook does. This is a diagnostic stress
harness, not a fix. An attempt that overruns its timeout is recorded as
stalled, together with the process tree and a stack sample of every
descendant; a kernel that dies is recorded as a failure rather than a stall.
Any stall or failure makes the program exit nonzero.

There are two ways to get a kernel, and they differ in the plumbing under it,
which is the axis this harness exists to vary:

* by default the harness starts an ipykernel itself through
  ``jupyter_client.KernelManager``, so the kernel is a child of the harness;
* ``--server`` starts a real ``jupyter server`` on a random free port with a
  generated token and asks it for a kernel over ``POST /api/kernels``, so the
  kernel is a child of the server and its stdio and ZMQ plumbing are the
  server's, as they are under JupyterLab. Everything downstream of getting a
  client -- the executed code, the completion marker, the classification and
  the stall capture -- is shared with the default mode.

``--reuse-kernel`` runs every attempt in one kernel rather than a fresh one per
run, which is the shape of a notebook session; the run stops at the first
attempt that does not complete, since a kernel that stalled cannot serve the
rest.

Cleanup is a process group. The default mode kills the kernel's own group,
which jupyter_client creates with ``start_new_session=True``, so spawned pool
workers go with it. ``--server`` first asks the server to take the kernel down
over ``DELETE /api/kernels/<id>`` -- the kernel gets its own session, so the
server's group does not contain it -- and then kills the server's group. In
both modes the census is rooted at the process the harness launched, so under
``--server`` the kernel itself is one of the descendants that must be gone
afterwards. Nothing else is signalled. After the kill the harness counts any
captured descendant still present and reports it as a survivor, because a
nonzero count means the group kill missed something and is a finding about the
harness. It does not try to kill survivors: identifying a process by pid and
start time is not sound enough to authorize a signal, since a start time is
only accurate to the second and a pid recycled inside that second would pass
the check. That same-second window can only turn a dead process into a
reported survivor, which is the safe direction for a count nobody acts on.

The one kill the harness does issue itself, the server's group under
``--server``, is gated on the server still being the process launched here and
still leading its own group; a group whose identity cannot be confirmed is left
alone and the teardown is reported untrusted. A provisioning attempt that fails
after the kernel exists deletes and censuses it on the way out for the same
reason: the kernel is in its own session, so abandoning it would leave it
running with nothing to say so.

The census fails loud, not quiet. A ``ps`` that exits nonzero, produces nothing
parseable, or yields rows the parser has to drop is a distinct state from a
census that ran and found nobody, and it is reported as a harness failure with
its reason rather than as zero survivors.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, TextIO

from jupyter_client import BlockingKernelClient, KernelManager

COMPLETION_MARKER = "pstrain-kernel-repro: completed"
PS_HEADER = f"{'PID':>7} {'PPID':>7} {'STAT':>5} {'%CPU':>5} {'STARTED':<24} COMMAND"

#: ``ps`` reports process identity portably on macOS and Linux only.
POSIX = os.name == "posix"
#: One listing, so a row's parent and start time always describe the same
#: process. Merging two listings could pair one process's parent with another's
#: start time if a pid were recycled between them. ``lstart`` is an absolute
#: start time, stable for a process's whole life; it has one-second resolution,
#: which is enough to notice a recycled pid but not enough to act on one.
_PS_FORMAT = "pid=,ppid=,stat=,%cpu=,lstart=,command="
#: ``lstart`` renders as five whitespace-separated tokens: weekday, month, day,
#: time, year. Everything after them is the command.
_LSTART_TOKENS = 5
_LEADING_FIELDS = 4

OUTCOME_COMPLETED = "completed"
OUTCOME_FAILED = "failed"
OUTCOME_STALLED = "stalled"

#: The kernelspec ``--server`` asks the server for. ``python3`` is the spec the
#: interpreter running this harness installs, so the kernel is the same Python.
DEFAULT_KERNEL_NAME = "python3"
#: How long a ``jupyter server`` gets to answer its REST API, and how long it
#: then gets to write the kernel's connection file.
SERVER_STARTUP_TIMEOUT = 90.0
#: A kernel's liveness under ``--server`` costs an HTTP round trip, so it is
#: asked for at most this often rather than on every pass of the message loop.
LIVENESS_INTERVAL = 5.0


@dataclass(frozen=True)
class ProcessInfo:
    """One row of a ``ps`` listing: a process, its parent and its start time."""

    pid: int
    ppid: int
    stat: str
    cpu: str
    started: str
    command: str

    @property
    def row(self) -> str:
        """The process rendered under `PS_HEADER`."""
        return (
            f"{self.pid:>7} {self.ppid:>7} {self.stat:>5} "
            f"{self.cpu:>5} {self.started:<24} {self.command}"
        )


@dataclass(frozen=True)
class ProcessSnapshot:
    """A parsed ``ps`` listing, or the fact that there is not a usable one.

    ``available`` false means ``ps`` failed or produced nothing parseable, which
    is not the same as a listing that ran and found nobody. ``dropped`` counts
    rows the parser could not read, so a partial listing cannot masquerade as a
    complete one.
    """

    processes: Mapping[int, ProcessInfo]
    dropped: int
    available: bool

    @property
    def trustworthy(self) -> bool:
        return self.available and self.dropped == 0


@dataclass(frozen=True)
class DescendantCensus:
    """The kernel's descendants, or why they could not be established."""

    descendants: Mapping[int, ProcessInfo]
    trusted: bool
    reason: str = ""


@dataclass(frozen=True)
class SurvivorReport:
    """What was still present after the kernel's process group was killed."""

    survivors: tuple[int, ...]
    trusted: bool
    reason: str = ""


@dataclass(frozen=True)
class TeardownReport:
    """Whether the group kill was issued, and why not when it was not.

    Refusing to signal is a distinct outcome from signalling and finding
    nothing left: a survivor count only means something if the kill actually
    happened, so a refusal makes the census untrusted rather than clean.
    """

    signalled: bool
    reason: str = ""


@dataclass(frozen=True)
class KernelExecution:
    """Outcome and diagnostic streams from one kernel execution."""

    reached_idle: bool
    kernel_alive: bool
    saw_error: bool
    elapsed: float
    stdout: str
    stderr: str


@dataclass(frozen=True)
class KernelSession:
    """A live kernel, its client, and the process the census is rooted at.

    ``pid`` and ``started`` name the process the harness launched: the kernel
    itself in the default mode, and the server under ``--server``, where the
    kernel is the server's child and so one of the descendants. ``started``
    pins that process down: a provisioner keeps its numeric pid after the
    process dies, so the pid alone cannot say whether the process running under
    it now is the one this session launched.

    ``is_alive`` and ``terminate`` are what the two modes disagree about, and
    holding them here is what lets the attempt loop, the classification and the
    stall capture be the same code for both.
    """

    client: BlockingKernelClient
    pid: int | None
    started: str | None
    is_alive: Callable[[], bool]
    terminate: Callable[[], TeardownReport]
    root: str = "kernel"


@dataclass(frozen=True)
class JupyterServer:
    """A running ``jupyter server`` and what is needed to talk to it.

    The token is generated per server and passed in the environment rather than
    on the command line, so it never appears in a ``ps`` listing -- which
    matters here, because this harness prints ``ps`` listings.

    ``pid``, ``started`` and ``pgid`` are recorded at launch and are what
    licenses the group kill later: a pid alone cannot say whether the process
    wearing it now is the server this harness started, nor whether that pid is
    still the leader of the group a kill would land on.
    """

    process: subprocess.Popen[bytes]
    base_url: str
    token: str
    runtime_dir: Path
    root_dir: Path
    pid: int
    started: str | None
    pgid: int | None


def parse_process_snapshot(output: str, *, status: int = 0) -> ProcessSnapshot:
    """Parse one ``ps`` listing, keeping track of what could not be read.

    Every field of a row comes from that row, so a process's parent and its
    start time always describe the same process. A row that does not parse is
    dropped rather than guessed at, and counted so the caller can tell a clean
    census from a partial one.
    """
    if status != 0:
        return ProcessSnapshot({}, dropped=0, available=False)
    wanted = _LEADING_FIELDS + _LSTART_TOKENS
    processes: dict[int, ProcessInfo] = {}
    dropped = 0
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split(maxsplit=wanted)
        if len(fields) <= wanted or not fields[0].isdigit() or not fields[1].isdigit():
            dropped += 1
            continue
        processes[int(fields[0])] = ProcessInfo(
            pid=int(fields[0]),
            ppid=int(fields[1]),
            stat=fields[2],
            cpu=fields[3],
            started=" ".join(fields[_LEADING_FIELDS:wanted]),
            command=fields[wanted],
        )
    return ProcessSnapshot(processes, dropped=dropped, available=bool(processes))


def select_descendants(snapshot: Mapping[int, ProcessInfo], root_pid: int) -> list[int]:
    """Return ``root_pid`` and every transitive child in the snapshot."""
    if root_pid not in snapshot:
        return []
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, info in snapshot.items():
            if info.ppid in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return sorted(selected)


def surviving_pids(
    captured: Mapping[int, ProcessInfo], current: Mapping[int, ProcessInfo]
) -> list[int]:
    """Return the captured pids still present as the same process.

    A pid whose start time has changed is a different process that inherited
    the number, and is not a survivor. This census is reported, never acted on:
    a start time is only accurate to the second, so a pid recycled inside that
    second would match here, which is tolerable for a count and not tolerable
    for a signal.
    """
    return sorted(
        pid
        for pid, info in captured.items()
        if (present := current.get(pid)) is not None and present.started == info.started
    )


def classify(execution: KernelExecution, *, marker: str | None = None) -> str:
    """Name the outcome of an execution.

    A dead kernel is a failure, never a stall: the harness hunts a live kernel
    parked on a pool, and a kernel that died is a different animal that must
    not be counted as the thing being hunted.
    """
    if not execution.kernel_alive:
        return OUTCOME_FAILED
    if not execution.reached_idle:
        return OUTCOME_STALLED
    if execution.saw_error:
        return OUTCOME_FAILED
    if marker is not None and marker not in execution.stdout:
        return OUTCOME_FAILED
    return OUTCOME_COMPLETED


def process_snapshot() -> ProcessSnapshot:
    """Snapshot every visible process in one ``ps`` listing.

    Off POSIX there is no listing to take, which is reported as unavailable
    rather than as an empty machine.
    """
    if not POSIX:
        return ProcessSnapshot({}, dropped=0, available=False)
    result = subprocess.run(["ps", "-axo", _PS_FORMAT], check=False, capture_output=True, text=True)
    return parse_process_snapshot(result.stdout, status=result.returncode)


def assess_census(
    snapshot: ProcessSnapshot, pid: int | None, started: str | None
) -> DescendantCensus:
    """Establish the kernel's descendants from a snapshot taken while it lives.

    Every way of not knowing is a distinct untrusted result with a reason, so a
    census that could not run is never confused with one that found nobody. The
    kernel's own row must be present: this runs before the kill, so its absence
    means the snapshot is wrong, not that the kernel is gone.
    """
    if not snapshot.available:
        return DescendantCensus({}, False, "ps was unavailable or produced no parseable rows")
    if snapshot.dropped:
        return DescendantCensus({}, False, f"{snapshot.dropped} ps row(s) could not be parsed")
    if pid is None:
        return DescendantCensus({}, False, "the kernel's pid was never recorded")
    if started is None:
        return DescendantCensus({}, False, f"pid {pid} had no start time recorded at launch")
    kernel = snapshot.processes.get(pid)
    if kernel is None:
        return DescendantCensus({}, False, f"the kernel's own row for pid {pid} is missing")
    if kernel.started != started:
        return DescendantCensus({}, False, f"pid {pid} now belongs to a different process")
    return DescendantCensus(
        {child: snapshot.processes[child] for child in select_descendants(snapshot.processes, pid)},
        True,
    )


def assess_survivors(captured: Mapping[int, ProcessInfo], after: ProcessSnapshot) -> SurvivorReport:
    """Decide which captured descendants outlived the kill.

    The kernel's own row is expected to be gone by now, so unlike `assess_census`
    this asks only that the listing itself be complete and readable.
    """
    if not after.available:
        return SurvivorReport((), False, "ps was unavailable or produced no parseable rows")
    if after.dropped:
        return SurvivorReport((), False, f"{after.dropped} ps row(s) could not be parsed")
    return SurvivorReport(tuple(surviving_pids(captured, after.processes)), True)


def captured_descendants(
    census: DescendantCensus, root_pid: int | None
) -> Mapping[int, ProcessInfo]:
    """The census minus the root process itself.

    The root is the process the harness launched, and it is meant to be gone
    once teardown has run, so it is never counted as a survivor. Under
    ``--server`` the root is the server, which leaves the kernel in the
    captured set: a kernel that outlives the teardown is exactly the kind of
    finding this census exists to make.
    """
    return {pid: info for pid, info in census.descendants.items() if pid != root_pid}


def kernel_descendants(session: KernelSession) -> DescendantCensus:
    """Census the live launched process and its descendants."""
    return assess_census(process_snapshot(), session.pid, session.started)


def report_teardown(
    teardown: TeardownReport,
    census: DescendantCensus,
    root_pid: int | None,
    after: ProcessSnapshot,
) -> SurvivorReport:
    """Fold a teardown and the two censuses into one survivor verdict.

    Three ways of not knowing, each with its own reason: the kill was never
    issued, the tree was never established, or the listing afterwards could not
    be read. None of them is zero survivors.
    """
    if not teardown.signalled:
        return SurvivorReport((), False, f"the process group was not signalled: {teardown.reason}")
    if not census.trusted:
        return SurvivorReport((), False, f"before the kill, {census.reason}")
    return assess_survivors(captured_descendants(census, root_pid), after)


def shutdown_kernel(session: KernelSession) -> SurvivorReport:
    """Tear the session down and census what outlived it.

    Teardown is the session's own, which ends in a process-group kill in both
    modes. The reported pids are descendants still present afterwards; they are
    never signalled. A census that could not be taken, and a kill that was
    refused, are both reported untrusted, never as zero survivors.
    """
    before = kernel_descendants(session)
    after = ProcessSnapshot({}, dropped=0, available=False)
    teardown = TeardownReport(False, "the teardown raised before it could signal")
    session.client.stop_channels()
    try:
        teardown = session.terminate()
    finally:
        after = process_snapshot()
    return report_teardown(teardown, before, session.pid, after)


def render_tree(snapshot: Mapping[int, ProcessInfo], pids: list[int]) -> str:
    """Format a ``ps``-style block for the given pids."""
    rows = [snapshot[pid].row for pid in pids if pid in snapshot]
    return "\n".join([PS_HEADER, *rows]) if rows else f"{PS_HEADER}\n(no matching processes)"


def stack_sample(pid: int, *, seconds: int = 3) -> str:
    """Dump ``pid``'s stacks with py-spy, falling back to macOS ``sample``."""
    if shutil.which("py-spy"):
        command = ["py-spy", "dump", "--pid", str(pid)]
    elif shutil.which("sample"):
        command = ["sample", str(pid), str(seconds)]
    else:
        return f"pid {pid}: neither py-spy nor sample is available"
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=seconds + 60
        )
    except subprocess.TimeoutExpired:
        return f"pid {pid}: {command[0]} timed out"
    return f"$ {' '.join(command)}\n{result.stdout}{result.stderr}".rstrip()


def stall_report(session: KernelSession) -> str:
    """Process tree plus a stack for the launched process and its descendants."""
    census = kernel_descendants(session)
    if not census.trusted:
        return f"no process tree for {session.root} pid {session.pid}: {census.reason}"
    pids = sorted(census.descendants)
    sections = [
        f"process tree for {session.root} pid {session.pid}:",
        render_tree(census.descendants, pids),
    ]
    sections.extend(stack_sample(pid) for pid in pids)
    return "\n".join(sections)


def process_identity(pid: int) -> tuple[int | None, str | None]:
    """Read a freshly launched process's pid and start time."""
    info = process_snapshot().processes.get(pid)
    return pid, (info.started if info is not None else None)


def _launch_identity(manager: KernelManager) -> tuple[int | None, str | None]:
    """Read the freshly launched kernel's pid and start time."""
    pid = getattr(getattr(manager, "provisioner", None), "pid", None)
    if not isinstance(pid, int):
        return None, None
    return process_identity(pid)


def throttle(check: Callable[[], bool], *, interval: float) -> Callable[[], bool]:
    """Rate-limit a liveness check that costs more than a syscall.

    A ``True`` is only as fresh as ``interval``, which is what makes the check
    cheap enough to sit in the message loop. A ``False`` is remembered: a
    process that has gone does not come back, and re-asking would only give a
    restarted stranger the chance to answer for it.
    """
    state: dict[str, Any] = {"asked": float("-inf"), "alive": True}

    def checked() -> bool:
        now = time.monotonic()
        if state["alive"] and now - state["asked"] >= interval:
            state["asked"] = now
            state["alive"] = check()
        return bool(state["alive"])

    return checked


def start_direct_session(
    *, cwd: Path | None = None, launch_stderr: TextIO | None = None, ready_timeout: float = 60.0
) -> KernelSession:
    """Start an ipykernel as a child of this process and connect to it.

    The kernel inherits this process's environment with ``cwd`` prepended to
    ``PYTHONPATH`` so it imports the checkout under test rather than an
    installed copy.
    """
    launch: dict[str, object] = {}
    if launch_stderr is not None:
        launch["stderr"] = launch_stderr
    if cwd is not None:
        launch["cwd"] = str(cwd)
        launch["env"] = checkout_environment(cwd)

    manager = KernelManager()
    manager.start_kernel(**launch)
    pid, started = _launch_identity(manager)

    def terminate() -> TeardownReport:
        # jupyter_client owns this kill and does its own identity handling, so
        # there is nothing here for the harness to gate.
        manager.shutdown_kernel(now=True)
        return TeardownReport(True)

    session = KernelSession(
        client=manager.client(),
        pid=pid,
        started=started,
        is_alive=manager.is_alive,
        terminate=terminate,
        root="kernel",
    )
    connect(session, ready_timeout=ready_timeout)
    return session


def connect(session: KernelSession, *, ready_timeout: float) -> None:
    """Open the session's channels and wait for the kernel to answer."""
    session.client.start_channels()
    try:
        session.client.wait_for_ready(timeout=ready_timeout)
    except BaseException:
        shutdown_kernel(session)
        raise


def execute_in_kernel(
    code: str,
    *,
    timeout: float,
    cwd: Path | None = None,
    launch_stderr: TextIO | None = None,
) -> tuple[KernelExecution, KernelSession]:
    """Start a kernel as a child of this process and execute ``code`` in it.

    The caller must always call `shutdown_kernel`, including on the timeout
    path, where the live kernel is what the caller inspects.
    """
    session = start_direct_session(
        cwd=cwd, launch_stderr=launch_stderr, ready_timeout=min(timeout, 60.0)
    )
    try:
        return run_in_session(session, code, timeout=timeout), session
    except BaseException:
        shutdown_kernel(session)
        raise


def run_in_session(session: KernelSession, code: str, *, timeout: float) -> KernelExecution:
    """Execute ``code`` in an already connected session.

    The elapsed time is the execution's alone. This is a deliberate change from
    the harness's first form, where the clock started before ``wait_for_ready``
    and a slow readiness ate into the run's own timeout. Two things follow, and
    both are wanted: the timeout now bounds the training run rather than the
    run plus whatever provisioning cost, and a run's number means the same
    thing in both modes, where otherwise ``--server`` would carry a couple of
    seconds of server startup that the default mode does not.
    """
    client = session.client
    begun = time.monotonic()
    stdout: list[str] = []
    stderr: list[str] = []
    saw_error = False

    def outcome(*, reached_idle: bool, kernel_alive: bool) -> KernelExecution:
        return KernelExecution(
            reached_idle=reached_idle,
            kernel_alive=kernel_alive,
            saw_error=saw_error,
            elapsed=time.monotonic() - begun,
            stdout="".join(stdout),
            stderr="".join(stderr),
        )

    message_id = client.execute(code, stop_on_error=False)
    while True:
        if not session.is_alive():
            return outcome(reached_idle=False, kernel_alive=False)
        remaining = timeout - (time.monotonic() - begun)
        if remaining <= 0:
            return outcome(reached_idle=False, kernel_alive=True)
        try:
            message = client.get_iopub_msg(timeout=min(remaining, 1.0))
        except Empty:
            continue
        if message.get("parent_header", {}).get("msg_id") != message_id:
            continue
        message_type = message["header"]["msg_type"]
        content = message["content"]
        if message_type == "stream":
            target = stderr if content["name"] == "stderr" else stdout
            target.append(content["text"])
        elif message_type == "error":
            saw_error = True
            stderr.append("\n".join(content.get("traceback", [])) + "\n")
        elif message_type == "status" and content.get("execution_state") == "idle":
            return outcome(reached_idle=True, kernel_alive=True)


def checkout_environment(cwd: Path, **extra: str) -> dict[str, str]:
    """This process's environment with ``cwd`` first on ``PYTHONPATH``.

    A kernel started from it imports the checkout under test rather than an
    installed copy, whether it is started here or by a server that inherits
    this environment.
    """
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = f"{cwd}{os.pathsep}{existing}" if existing else str(cwd)
    environment.update(extra)
    return environment


def free_port() -> int:
    """Ask the operating system for a free localhost port and hand it back.

    The port is not held, so this races anything else on the machine that binds
    in the meantime; the server is told not to retry, so losing that race is a
    loud startup failure rather than a server quietly listening somewhere else.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def api_url(base_url: str, path: str) -> str:
    """Join a server base URL and an API path over exactly one separator."""
    return f"{base_url.rstrip('/')}/{path.strip('/')}"


def auth_headers(token: str) -> dict[str, str]:
    """Authenticate a REST call with the server's token.

    The token goes in a header, not a query string: out of the server's request
    log, and, because the request is then token-authenticated rather than
    cookie-authenticated, past the XSRF check that would otherwise reject a
    ``POST`` from something that is not a browser.
    """
    return {"Authorization": f"token {token}", "Content-Type": "application/json"}


def api_call(
    server: JupyterServer,
    path: str,
    *,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
    timeout: float = 30.0,
) -> Any:
    """Make one REST call against the server and decode its JSON reply."""
    body = None if payload is None else json.dumps(dict(payload)).encode("utf-8")
    # The URL is one this harness built for its own localhost server, so there
    # is no scheme or host here that came from anywhere else.
    request = urllib.request.Request(
        api_url(server.base_url, path),
        data=body,
        method=method,
        headers=auth_headers(server.token),
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8")
    return json.loads(text) if text.strip() else {}


def connection_file(runtime_dir: Path, kernel_id: str) -> Path:
    """Where a server writes the connection file for one of its kernels.

    The name is the multi-kernel manager's own: it hands each kernel manager a
    connection file named after the kernel id, in its connection directory,
    which is the server's runtime directory.
    """
    return runtime_dir / f"kernel-{kernel_id}.json"


def read_connection_file(path: Path) -> dict[str, Any] | None:
    """Return usable connection info, or ``None`` while there is not any yet.

    The server writes the file some time after the REST call returns, and a
    file that exists is not yet a file that is finished, so an absent file, an
    unparseable one and one still missing the fields a client needs are all the
    same answer: not yet.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        info = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(info, dict) or not {"key", "shell_port", "iopub_port"} <= set(info):
        return None
    return info


def await_connection_file(runtime_dir: Path, kernel_id: str, *, deadline: float) -> Path:
    """Wait for the server to finish writing a kernel's connection file."""
    path = connection_file(runtime_dir, kernel_id)
    while time.monotonic() < deadline:
        if read_connection_file(path) is not None:
            return path
        time.sleep(0.1)
    raise RuntimeError(f"the server wrote no usable connection file at {path}")


def server_command(*, port: int, root_dir: Path) -> list[str]:
    """The argv for a headless ``jupyter server`` this harness can drive.

    Automatic restarts are off: a kernel that dies must be classified as a
    failure, and a server that quietly replaces it would hide exactly that.
    The port is not retried, so the server either listens where the harness
    expects or fails loudly. The token is absent, because it is passed in the
    environment instead.
    """
    return [
        sys.executable,
        "-m",
        "jupyter_server",
        "--no-browser",
        "--ServerApp.ip=127.0.0.1",
        f"--ServerApp.port={port}",
        "--ServerApp.port_retries=0",
        "--ServerApp.open_browser=False",
        f"--ServerApp.root_dir={root_dir}",
        "--KernelManager.autorestart=False",
    ]


def wait_for_server(server: JupyterServer, *, marker: str, deadline: float) -> None:
    """Block until the spawned server answers, or say why it never will.

    An answer on the port is not enough. The harness reserved the port and gave
    it up again before the server bound it, so something else may hold it, and
    the harness is about to signal a process group on the strength of this.
    Three things have to hold together: the reply carries this instance's
    token, the child launched here is still running, and the server is serving
    the root directory this harness made, which the marker file it contains
    proves.
    """
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            raise RuntimeError(
                f"jupyter server exited with code {server.process.returncode} "
                f"before answering {server.base_url}"
            )
        try:
            api_call(server, "api/status", timeout=5.0)
            listed = api_call(server, f"api/contents/{marker}", timeout=5.0)
        except OSError:
            time.sleep(0.2)
            continue
        if server.process.poll() is not None:
            raise RuntimeError(
                f"the jupyter server launched here exited while {server.base_url} "
                f"was answering, so the answer came from something else"
            )
        if listed.get("name") != marker:
            raise RuntimeError(
                f"{server.base_url} answers but is not serving this harness's root directory"
            )
        return
    raise RuntimeError(f"jupyter server did not answer {server.base_url} in time")


def start_jupyter_server(
    *, cwd: Path, scratch: Path, log: TextIO | None = None, deadline: float
) -> JupyterServer:
    """Start a headless ``jupyter server`` in its own process group.

    Its runtime directory is under ``scratch``, so the connection files it
    writes are this server's and nobody else's, and its environment carries the
    checkout on ``PYTHONPATH``, which the kernels it starts inherit. A marker
    file goes into the root directory before launch, so readiness can prove the
    server answering is the one started here.
    """
    runtime_dir = scratch / "runtime"
    root_dir = scratch / "root"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    root_dir.mkdir(parents=True, exist_ok=True)
    marker = f"{secrets.token_hex(8)}.txt"
    (root_dir / marker).write_text("pstrain kernel pool stall repro\n", encoding="utf-8")
    token = secrets.token_hex(24)
    port = free_port()
    environment = checkout_environment(
        cwd, JUPYTER_TOKEN=token, JUPYTER_RUNTIME_DIR=str(runtime_dir)
    )
    stream = log if log is not None else subprocess.DEVNULL
    process = subprocess.Popen(
        server_command(port=port, root_dir=root_dir),
        cwd=str(cwd),
        env=environment,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _, started = process_identity(process.pid)
    server = JupyterServer(
        process=process,
        base_url=f"http://127.0.0.1:{port}",
        token=token,
        runtime_dir=runtime_dir,
        root_dir=root_dir,
        pid=process.pid,
        started=started,
        pgid=current_group(process.pid),
    )
    try:
        wait_for_server(server, marker=marker, deadline=deadline)
    except BaseException:
        report = stop_server(server)
        if not report.signalled:
            print(f"the jupyter server was left running: {report.reason}", flush=True)
        raise
    return server


def group_kill_refusal(
    snapshot: ProcessSnapshot,
    pid: int,
    started: str | None,
    *,
    launched_pgid: int | None,
    current_pgid: int | None,
) -> str:
    """Why the server's process group must not be signalled, or ``""`` if it may.

    A signal is the only thing this harness does that cannot be taken back, so
    it is gated harder than the census that nobody acts on. Everything is asked
    of one listing, so a row's start time and the pid it belongs to always
    describe the same process: the listing must be complete, the pid must still
    be wearing the start time it wore at launch, and the pid must still be the
    leader of its own group -- otherwise the kill would land on a group this
    harness never created.
    """
    if not snapshot.available:
        return "ps was unavailable or produced no parseable rows"
    if snapshot.dropped:
        return f"{snapshot.dropped} ps row(s) could not be parsed"
    if started is None:
        return f"pid {pid} had no start time recorded at launch"
    info = snapshot.processes.get(pid)
    if info is None:
        return f"pid {pid} is already gone"
    if info.started != started:
        return f"pid {pid} now belongs to a different process"
    if launched_pgid != pid:
        return f"pid {pid} did not lead its own group at launch (group {launched_pgid})"
    if current_pgid is None:
        return f"pid {pid} has no readable process group"
    if current_pgid != pid:
        return f"pid {pid} no longer leads its own group (group {current_pgid})"
    return ""


def current_group(pid: int) -> int | None:
    """The process group ``pid`` belongs to now, or ``None`` if unreadable."""
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def issue_group_kill(
    pid: int, pgid: int | None, *, kill: Callable[[int, int], None] = os.killpg
) -> TeardownReport:
    """Signal a process group whose identity has already been established.

    The two ways this fails are opposites and must not share a path.

    ``ProcessLookupError`` means the group is already gone. The identity gate
    ran a moment earlier and found the process alive, so it exited in between:
    there is nothing left to kill and nothing left to doubt, and the teardown
    counts as done.

    ``PermissionError`` means the process is still there and the signal was
    refused, so the server is still running. That cannot be reported as a
    finished teardown: the server is the census root and is excluded from the
    survivor count, so a signalled teardown here would report a clean machine
    while the server this harness started is still on it.
    """
    try:
        kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return TeardownReport(True)
    except PermissionError as error:
        return TeardownReport(
            False, f"SIGKILL to process group {pgid} (server pid {pid}) was denied: {error}"
        )
    return TeardownReport(True)


def stop_server(server: JupyterServer) -> TeardownReport:
    """Kill the server's process group, unless its identity cannot be confirmed.

    A refusal is reported rather than worked around: leaving a server running
    is recoverable, and signalling a stranger's process group is not.
    """
    pgid = current_group(server.pid)
    refusal = group_kill_refusal(
        process_snapshot(),
        server.pid,
        server.started,
        launched_pgid=server.pgid,
        current_pgid=pgid,
    )
    if refusal:
        return TeardownReport(False, refusal)
    report = issue_group_kill(server.pid, pgid)
    if not report.signalled:
        return report
    with contextlib.suppress(subprocess.TimeoutExpired):
        server.process.wait(timeout=30)
    return report


def delete_kernel_and_stop(server: JupyterServer, kernel_id: str | None) -> TeardownReport:
    """Ask the server to end its kernel, then kill the server's group.

    The kernel has its own session, so the server's process group does not
    contain it; the server's own kernel manager is the only thing that reliably
    ends it, and the group kill then finishes the server off.
    """
    if kernel_id is not None:
        try:
            api_call(server, f"api/kernels/{kernel_id}", method="DELETE", timeout=60.0)
        except OSError as error:
            print(f"DELETE /api/kernels/{kernel_id} failed: {error}", flush=True)
    return stop_server(server)


def abandon_provisioning(
    server: JupyterServer, kernel_id: str | None, census: DescendantCensus
) -> SurvivorReport:
    """Tear down a session that failed part way through being provisioned.

    A kernel the server created and this harness then gave up on would outlive
    the group kill -- it is in its own session -- and nothing would say so, so
    the same DELETE, the same gated group kill and the same survivor census run
    here as on the ordinary path. ``census`` is the tree as it stood while the
    kernel certainly existed, which is what makes the kernel one of the
    processes that has to be gone afterwards.
    """
    teardown = delete_kernel_and_stop(server, kernel_id)
    return report_teardown(teardown, census, server.pid, process_snapshot())


def kernel_alive(server: JupyterServer, kernel_id: str) -> bool:
    """Ask the server whether it still has this kernel.

    Anything other than a definite ``404`` counts as alive: a server that is
    slow or briefly unreachable has not told us the kernel died, and calling a
    stall a death would lose the very case being hunted.
    """
    if server.process.poll() is not None:
        return False
    try:
        api_call(server, f"api/kernels/{kernel_id}", timeout=10.0)
    except urllib.error.HTTPError as error:
        return error.code != 404
    except OSError:
        return True
    return True


def start_server_session(
    *,
    cwd: Path,
    scratch: Path,
    kernel_name: str = DEFAULT_KERNEL_NAME,
    log: TextIO | None = None,
    ready_timeout: float = 60.0,
) -> KernelSession:
    """Start a ``jupyter server`` and take a kernel from it over the REST API.

    This is the provisioning path a notebook actually uses: the kernel is the
    server's child, started by the server's own kernel manager, with the
    server's stdio and ZMQ plumbing rather than this harness's. The census is
    rooted at the server for that reason -- the kernel is one of its
    descendants.
    """
    deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
    server = start_jupyter_server(cwd=cwd, scratch=scratch, log=log, deadline=deadline)
    kernel_id: str | None = None
    census = DescendantCensus({}, False, "the kernel was never created")
    try:
        created = api_call(server, "api/kernels", method="POST", payload={"name": kernel_name})
        # Retain the id and the tree the moment the kernel exists. Everything
        # after this point can fail, and if it does the kernel still has to be
        # deleted and still has to be censused: the server's group kill cannot
        # reach into the kernel's own session, so an abandoned kernel would
        # simply keep running with nobody reporting it.
        kernel_id = str(created["id"])
        census = assess_census(process_snapshot(), server.pid, server.started)
        path = await_connection_file(server.runtime_dir, kernel_id, deadline=deadline)
        client = BlockingKernelClient()
        client.load_connection_file(str(path))
    except BaseException:
        message = survivor_message(
            "abandoned provisioning", abandon_provisioning(server, kernel_id, census)
        )
        if message is not None:
            print(message, flush=True)
        raise

    session = KernelSession(
        client=client,
        pid=server.pid,
        started=server.started,
        is_alive=throttle(lambda: kernel_alive(server, kernel_id), interval=LIVENESS_INTERVAL),
        terminate=lambda: delete_kernel_and_stop(server, kernel_id),
        root="server",
    )
    connect(session, ready_timeout=ready_timeout)
    return session


def training_code(fixture: Path, *, target: str, jobs: int, utterances: int) -> str:
    """Build kernel code for one isolated mini-Arctic training attempt.

    Imports go through ``pstrain.api``, the boundary the tutorial notebook
    uses, so the harness exercises the same call path that stalls.
    """
    return f"""
from pathlib import Path
from tempfile import TemporaryDirectory

from pstrain.api import setup_project
from pstrain.api.pipeline import PipelineContext, build_pipeline

fixture = Path({str(fixture)!r})
with TemporaryDirectory(prefix="pstrain-kernel-repro-") as temporary:
    root = Path(temporary)
    audio = root / "audio"
    audio.mkdir()
    lines = (fixture / "transcription.txt").read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if line.strip()][:{utterances}]
    if not lines:
        raise RuntimeError("no utterances selected")
    for line in lines:
        source = fixture / "wav" / (line.split()[0] + ".wav")
        (audio / source.name).symlink_to(source)
    transcription = root / "transcription.txt"
    transcription.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
    project = root / "project"
    setup_project(
        project,
        transcription_path=transcription,
        audio_path=audio,
        dictionary_path=fixture / "dictionary.dict",
        phoneset_path=fixture / "phoneset.txt",
        filler_dict_path=fixture / "filler.dict",
    )
    context = PipelineContext.from_config(
        project,
        experiment="default",
        config_name="default",
        cli_overrides={{"runner": {{"jobs": {jobs}}}}},
    )
    return_code = build_pipeline(context).run({target!r}, jobs={jobs})
    if return_code:
        raise RuntimeError(f"pipeline returned {{return_code}}")
print({COMPLETION_MARKER!r})
"""


def pipeline_run_call(code: str) -> ast.Call | None:
    """Return the ``.run(...)`` call in generated kernel code, if there is one."""
    for node in ast.walk(ast.parse(code)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
        ):
            return node
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20, help="number of training runs")
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="wall timeout per run in seconds"
    )
    parser.add_argument("--jobs", type=int, default=2, help="Pipeline.run worker count")
    parser.add_argument(
        "--utterances", type=int, default=10, help="mini-Arctic utterances to train"
    )
    parser.add_argument(
        "--target",
        default="ci-1g",
        help="pipeline target; 'cd-1g' also fans out the tree-building group",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="provision the kernel through a jupyter server, the way a notebook does",
    )
    parser.add_argument(
        "--kernel-name",
        default=DEFAULT_KERNEL_NAME,
        help=f"kernelspec the server starts (--server only, default {DEFAULT_KERNEL_NAME})",
    )
    parser.add_argument(
        "--reuse-kernel",
        action="store_true",
        help="run every attempt in one long-lived kernel, the shape of a notebook session",
    )
    args = parser.parse_args(argv)
    if args.runs < 1 or args.timeout <= 0 or args.jobs < 2 or args.utterances < 1:
        parser.error("require runs >= 1, timeout > 0, jobs >= 2, and utterances >= 1")
    if args.kernel_name != DEFAULT_KERNEL_NAME and not args.server:
        parser.error("--kernel-name names the kernelspec a server starts, so it needs --server")
    return args


def attempt_groups(runs: int, *, reuse: bool) -> list[list[int]]:
    """Split the run numbers into the kernels that will serve them.

    A fresh kernel per run by default, so no run can be contaminated by the
    one before it; one kernel for every run under ``--reuse-kernel``, which is
    what a notebook does and what the original observation was doing.
    """
    numbers = list(range(1, runs + 1))
    return [numbers] if reuse else [[number] for number in numbers]


def report_attempt(
    label: str, execution: KernelExecution, session: KernelSession, launch_stderr: str
) -> str:
    """Print one attempt's result and diagnostics; return its outcome."""
    result = classify(execution, marker=COMPLETION_MARKER)
    if result == OUTCOME_COMPLETED:
        print(f"{label}: completed in {execution.elapsed:.2f}s", flush=True)
        return result
    if result == OUTCOME_STALLED:
        print(f"{label}: STALLED after {execution.elapsed:.2f}s", flush=True)
        print(stall_report(session), flush=True)
    elif not execution.kernel_alive:
        print(f"{label}: FAILED, kernel died after {execution.elapsed:.2f}s", flush=True)
    else:
        print(f"{label}: FAILED in {execution.elapsed:.2f}s", flush=True)
    print("kernel stdout:", flush=True)
    print(execution.stdout or "(none)", flush=True)
    print("kernel stderr:", flush=True)
    print((execution.stderr + launch_stderr) or "(none)", flush=True)
    return result


@dataclass
class Tally:
    """What the run produced, in the four counts the summary reports."""

    stalls: int = 0
    failures: int = 0
    outlived: int = 0
    untrusted: int = 0
    attempted: int = 0
    kernels: int = 0

    @property
    def clean(self) -> bool:
        return not (self.stalls or self.failures or self.outlived or self.untrusted)


def survivor_message(label: str, report: SurvivorReport) -> str | None:
    """The line a survivor census earns, or ``None`` when it found nobody."""
    if not report.trusted:
        return (
            f"{label}: CENSUS UNTRUSTED, cannot say what outlived the group kill: {report.reason}"
        )
    if report.survivors:
        return (
            f"{label}: {len(report.survivors)} process(es) outlived the group kill and were "
            f"left alone: {', '.join(str(pid) for pid in report.survivors)}"
        )
    return None


def record_shutdown(label: str, session: KernelSession, tally: Tally) -> None:
    """Tear a session down and fold the survivor census into ``tally``."""
    report = shutdown_kernel(session)
    message = survivor_message(label, report)
    if message is not None:
        if report.trusted:
            tally.outlived += 1
        else:
            tally.untrusted += 1
        print(message, flush=True)


def start_session(
    args: argparse.Namespace, *, cwd: Path, scratch: Path, log: TextIO
) -> KernelSession:
    """Get a kernel the way the requested mode gets one."""
    ready_timeout = min(args.timeout, 60.0)
    if args.server:
        return start_server_session(
            cwd=cwd,
            scratch=scratch,
            kernel_name=args.kernel_name,
            log=log,
            ready_timeout=ready_timeout,
        )
    return start_direct_session(cwd=cwd, launch_stderr=log, ready_timeout=ready_timeout)


def run_attempts(args: argparse.Namespace, code: str, *, repository: Path) -> Tally:
    """Drive every attempt and report each one as it finishes."""
    tally = Tally()
    for group in attempt_groups(args.runs, reuse=args.reuse_kernel):
        tally.kernels += 1
        label = f"run {group[0]}/{args.runs}"
        with (
            tempfile.TemporaryDirectory(prefix="pstrain-kernel-repro-") as scratch,
            tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as launch_log,
        ):
            session = start_session(args, cwd=repository, scratch=Path(scratch), log=launch_log)
            consumed = 0
            try:
                for number in group:
                    label = f"run {number}/{args.runs}"
                    tally.attempted += 1
                    execution = run_in_session(session, code, timeout=args.timeout)
                    launch_log.seek(consumed)
                    result = report_attempt(label, execution, session, launch_log.read())
                    consumed = launch_log.tell()
                    tally.stalls += result == OUTCOME_STALLED
                    tally.failures += result == OUTCOME_FAILED
                    if result != OUTCOME_COMPLETED and number != group[-1]:
                        # The rest of this group would run in a kernel that has
                        # already gone wrong, which measures nothing.
                        print(f"{label}: abandoning the remaining runs in this kernel", flush=True)
                        break
            finally:
                record_shutdown(label, session, tally)
    return tally


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    fixture = repository / "tests" / "fixtures" / "mini_arctic"
    code = training_code(fixture, target=args.target, jobs=args.jobs, utterances=args.utterances)
    provisioning = "jupyter server" if args.server else "direct KernelManager"
    print(
        f"kernel pool stall repro: runs={args.runs} jobs={args.jobs} target={args.target} "
        f"utterances={args.utterances} timeout={args.timeout:.1f}s "
        f"provisioning={provisioning} "
        f"kernels={'one, reused' if args.reuse_kernel else 'one per run'}",
        flush=True,
    )
    tally = run_attempts(args, code, repository=repository)
    print(
        f"summary: {tally.stalls}/{tally.attempted} stalled, "
        f"{tally.failures}/{tally.attempted} failed, "
        f"{tally.outlived}/{tally.kernels} left survivors, "
        f"{tally.untrusted}/{tally.kernels} census untrusted",
        flush=True,
    )
    return 0 if tally.clean else 1


if __name__ == "__main__":
    sys.exit(main())
