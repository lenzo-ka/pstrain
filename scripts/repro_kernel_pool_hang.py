#!/usr/bin/env python3
"""Reproduce the Jupyter ``Pipeline.run(..., jobs>1)`` process-pool stall.

Run from the repository root after building the native library::

    PYTHONPATH=. nice -n 19 python scripts/repro_kernel_pool_hang.py \\
        --runs 20 --jobs 2 --timeout 120 [--server] [--reuse-kernel] \\
        [--decode [--test-utterances N]] \\
        [--server-stdout {file,pipe}] [--no-capture-fd-output] \\
        [--corpus DIR --dictionary FILE] [--import-root DIR] [--hold-iopub S]

Each attempt trains the mini-Arctic fixture to a target, driving the pipeline
exactly the way the tutorial notebook does; ``--corpus`` trains a slice of a
full Arctic corpus instead, which is how the volume of output is raised past
what a pipe can absorb. There are two pools a pstrain session creates, and by
default this harness measures the pipeline's; ``--decode`` measures the other
one, the process pool inside ``test_model``, by training the fixture serially
and then decoding the split's holdout with ``--jobs`` workers, which is the
tutorial's decode cell. This is a diagnostic stress harness, not a fix. An
attempt that overruns its timeout is recorded as stalled, together with the
process tree, a stack sample of every descendant and, where ``lsof`` is
available, the descriptors each one holds, since a stall on fd 1 can only be
read off the descriptors; a kernel that dies is recorded as a failure rather
than a stall. Any stall or failure makes the program exit nonzero.

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

Two hops carry a kernel's fd 1, and only the second of them can ever block.
Inside a kernel fd 1 is not a terminal: with ``IPKernelApp.capture_fd_output``
on, which is the default, it is a 64 KB pipe ipykernel made, and every pool
worker and native helper the kernel spawns inherits it. A daemon thread in
ipykernel drains that pipe and copies each read straight through to the
kernel's *original* fd 1 with a blocking, unguarded ``os.write``; that
descriptor is whatever the jupyter server's own stdout was, inherited
unchanged. So if the far end stops draining, the copy blocks, the drain stops,
the capture pipe fills, and every process writing to fd 1 parks in ``write(2)``
-- which presents as a pool that has hung with every worker at 0% CPU.

``--server-stdout`` is the knob that decides whether that far end can block,
and it is the one that reproduces the stall. The default, ``file``, gives the
server the harness's own launch log, which always drains, so no volume of
output can stall anything and the whole class is invisible; ``pipe`` gives the
server a pipe whose read end this harness holds open and never reads, which is
the shape of a server launched by a supervisor, an editor or a tty under flow
control. ``--no-capture-fd-output`` removes the first hop, so writers block on
the server's descriptor directly. ``--hold-iopub`` is deliberately not a
substitute for either: iopub is a ZMQ PUB socket, which drops at its high-water
mark rather than pushing back on the kernel.

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
import dataclasses
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
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
#: The kernelspec ``--no-capture-fd-output`` writes and asks the server for. It
#: is the same interpreter as `DEFAULT_KERNEL_NAME`, with ipykernel's fd 1/2
#: capture turned off.
NO_FD_CAPTURE_KERNEL_NAME = "python3-nofdcap"
#: How long a ``jupyter server`` gets to answer its REST API, and how long it
#: then gets to write the kernel's connection file.
SERVER_STARTUP_TIMEOUT = 90.0
#: A kernel's liveness under ``--server`` costs an HTTP round trip, so it is
#: asked for at most this often rather than on every pass of the message loop.
LIVENESS_INTERVAL = 5.0
#: How long a held stdout pipe gets, after the group kill, to reach end of
#: file. Every writer is meant to be dead by then, so this only has to cover
#: the last of them being reaped, not any amount of buffered output.
HELD_PIPE_TIMEOUT = 5.0


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

    ``unresolved`` is the same kind of fact about the pipe held open as the
    server's stdout: a drain that never reached end of file means something is
    still holding a descriptor this teardown handed out, which no survivor
    count taken by pid can be trusted over.
    """

    signalled: bool
    reason: str = ""
    unresolved: str = ""


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


@dataclass
class HeldPipe:
    """The read end of a pipe, held open by this process and never read from.

    Holding it open unread is the whole point: a full pipe blocks the processes
    writing into it rather than killing them with ``SIGPIPE``, which is the
    backpressure the harness exists to apply.

    `release` ends that at teardown by handing the descriptor to a daemon
    thread that reads and discards until the last writer has gone, and then
    closes it. `resolve` waits for that within a bound and says whether it
    happened, since a drain that never reaches end of file is not a tidy detail
    to leak: it means a writer outlived the teardown. Both steps are
    idempotent, and the descriptor is forgotten as it is closed, so a second
    teardown cannot start a second reader or close a number the operating
    system has since handed to something else.
    """

    read_end: int
    reader: threading.Thread | None = None

    def release(self) -> None:
        """Start discarding whatever is in the pipe; close it at the far end's."""
        if self.read_end < 0 or self.reader is not None:
            return
        self.reader = threading.Thread(target=self._drain, args=(self.read_end,), daemon=True)
        self.reader.start()

    def resolve(self, *, timeout: float) -> bool:
        """Wait out the drain within ``timeout``; say whether it finished."""
        if self.reader is not None:
            self.reader.join(timeout)
        return self.read_end < 0

    def _drain(self, descriptor: int) -> None:
        with contextlib.suppress(OSError):
            while os.read(descriptor, 65536):
                pass
        self.close()

    def close(self) -> None:
        """Release the read end, if it has not already been released."""
        if self.read_end < 0:
            return
        descriptor, self.read_end = self.read_end, -1
        with contextlib.suppress(OSError):
            os.close(descriptor)


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

    ``held_stdout`` is set only under ``--server-stdout pipe``: it is the read
    end of the pipe the server's stdout is, kept open here so the pipe fills
    and blocks instead of collapsing, and released once the group kill lands.
    """

    process: subprocess.Popen[bytes]
    base_url: str
    token: str
    runtime_dir: Path
    root_dir: Path
    pid: int
    started: str | None
    pgid: int | None
    held_stdout: HeldPipe | None = None


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

    Four ways of not knowing, each with its own reason: the kill was never
    issued, a descriptor this teardown handed out is still held by somebody,
    the tree was never established, or the listing afterwards could not be
    read. None of them is zero survivors.
    """
    if not teardown.signalled:
        return SurvivorReport((), False, f"the process group was not signalled: {teardown.reason}")
    if teardown.unresolved:
        return SurvivorReport((), False, teardown.unresolved)
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


def open_files(pid: int, *, seconds: int = 5) -> str:
    """Dump the descriptors ``pid`` holds with ``lsof``.

    A stack says a process is parked in ``write(2)``; only the descriptor table
    says what it is writing to, which is what separates fd 1 backpressure from
    every other reason a worker might be idle.

    The timeout is short because this runs once per captured process, and a
    stall report over a whole tree that waits a minute on each is a report
    nobody sees until the run is over.
    """
    if not shutil.which("lsof"):
        return f"pid {pid}: lsof is not available"
    command = ["lsof", "-n", "-P", "-p", str(pid)]
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=seconds
        )
    except subprocess.TimeoutExpired:
        return f"pid {pid}: lsof timed out"
    return f"$ {' '.join(command)}\n{result.stdout}{result.stderr}".rstrip()


def stall_report(session: KernelSession) -> str:
    """Process tree, a stack and a descriptor table for every captured process."""
    census = kernel_descendants(session)
    if not census.trusted:
        return f"no process tree for {session.root} pid {session.pid}: {census.reason}"
    pids = sorted(census.descendants)
    sections = [
        f"process tree for {session.root} pid {session.pid}:",
        render_tree(census.descendants, pids),
    ]
    sections.extend(stack_sample(pid) for pid in pids)
    sections.extend(open_files(pid) for pid in pids)
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


def run_in_session(
    session: KernelSession, code: str, *, timeout: float, hold_iopub: float = 0.0
) -> KernelExecution:
    """Execute ``code`` in an already connected session.

    ``hold_iopub`` stops reading iopub for that many seconds after the execute
    request goes out, the shape of a busy or throttled frontend. It is a
    control, not a stall knob: iopub is a ZMQ PUB socket, which discards at its
    high-water mark, so a client that stops reading it never pushes back on the
    kernel. The hold is bounded by the run's own timeout, so a held run still
    classifies rather than running past the clock.

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
    if hold_iopub > 0:
        held = min(hold_iopub, timeout)
        print(f"  not reading iopub for {held:.0f}s", flush=True)
        time.sleep(held)
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


def server_stdout(mode: str, log: TextIO | None) -> tuple[Any, HeldPipe | None]:
    """What the server's stdout will be, and the read end held open for it.

    The kernel inherits this descriptor and ipykernel's watcher thread copies
    every captured byte to it, so whether it can block is the whole experiment.
    ``file`` hands over the harness's launch log, or ``/dev/null`` when there
    is none; both always drain. ``pipe`` makes a pipe and returns its write end
    for the child together with the read end the caller must keep open, since a
    pipe with no reader kills its writers instead of blocking them.
    """
    if mode != "pipe":
        return (log if log is not None else subprocess.DEVNULL), None
    read_end, write_end = os.pipe()
    return write_end, HeldPipe(read_end)


def write_no_capture_kernelspec(root: Path) -> str:
    """Write a kernelspec with ipykernel's fd capture off; return its name.

    The spec runs the same interpreter as the default one, so the only
    difference between the two runs is the hop under test: with capture off the
    kernel's fd 1 stays the descriptor the server gave it, and writers block on
    that directly rather than on ipykernel's pipe. A server finds the spec
    through ``JUPYTER_PATH``, which is why ``root`` is handed back to the
    caller's environment rather than installed anywhere shared.
    """
    directory = root / "kernels" / NO_FD_CAPTURE_KERNEL_NAME
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                    "--IPKernelApp.capture_fd_output=False",
                ],
                "display_name": "Python 3 (no fd capture)",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    return NO_FD_CAPTURE_KERNEL_NAME


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
    *,
    cwd: Path,
    scratch: Path,
    log: TextIO | None = None,
    deadline: float,
    stdout_mode: str = "file",
    capture_fd_output: bool = True,
) -> JupyterServer:
    """Start a headless ``jupyter server`` in its own process group.

    Its runtime directory is under ``scratch``, so the connection files it
    writes are this server's and nobody else's, and its environment carries the
    checkout on ``PYTHONPATH``, which the kernels it starts inherit. A marker
    file goes into the root directory before launch, so readiness can prove the
    server answering is the one started here.

    ``stdout_mode`` sets the descriptor the kernel will inherit as its original
    fd 1, and ``capture_fd_output`` false points the server at a kernelspec,
    written under ``scratch`` and found through ``JUPYTER_PATH``, that turns
    ipykernel's own fd capture off. Both are scoped to this server: nothing
    outside ``scratch`` is written and no ambient environment is mutated.
    """
    runtime_dir = scratch / "runtime"
    root_dir = scratch / "root"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    root_dir.mkdir(parents=True, exist_ok=True)
    marker = f"{secrets.token_hex(8)}.txt"
    (root_dir / marker).write_text("pstrain kernel pool stall repro\n", encoding="utf-8")
    token = secrets.token_hex(24)
    port = free_port()
    extra: dict[str, str] = {}
    if not capture_fd_output:
        specs = scratch / "kernelspecs"
        write_no_capture_kernelspec(specs)
        # First on the search path, not instead of it: a checkout or a virtual
        # environment may already put kernelspecs on JUPYTER_PATH, and dropping
        # those would change what else the server can start.
        inherited = os.environ.get("JUPYTER_PATH", "")
        extra["JUPYTER_PATH"] = f"{specs}{os.pathsep}{inherited}" if inherited else str(specs)
    environment = checkout_environment(
        cwd, JUPYTER_TOKEN=token, JUPYTER_RUNTIME_DIR=str(runtime_dir), **extra
    )
    stream, held_stdout = server_stdout(stdout_mode, log)
    process = subprocess.Popen(
        server_command(port=port, root_dir=root_dir),
        cwd=str(cwd),
        env=environment,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    if held_stdout is not None:
        # Only the server and the kernels under it hold the write end now, so
        # filling the pipe blocks them and never this process.
        os.close(stream)
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
        held_stdout=held_stdout,
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

    Any held stdout pipe is released first and waited out last; see
    `release_held_stdout` and `unresolved_stdout`.
    """
    release_held_stdout(server)
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
    return dataclasses.replace(report, unresolved=unresolved_stdout(server))


def release_held_stdout(server: JupyterServer) -> None:
    """Drain and close the pipe this harness held open as the server's stdout.

    This is the first act of teardown, before the kernel is even asked to shut
    down, because everything downstream of it needs the pipe to move. A server
    whose stdout has filled is itself blocked the moment it logs anything, so
    it cannot serve the DELETE; and a kernel parked on that pipe cannot answer
    a shutdown request. Leaving the descriptor open until after the kill spends
    the DELETE's whole timeout and then reports the kernel and its children as
    survivors, which they are.

    Draining first and closing after is what makes the release clean: a reader
    that simply closes turns the backpressure into ``SIGPIPE`` for everyone
    parked on the pipe, so processes die where they stood instead of finishing
    their writes and shutting down in order.

    It is not gated the way the group kill is, and does not need to be. A group
    kill is aimed at a number, and a number can come to be worn by a stranger;
    this reaches exactly the processes holding the write end of a pipe this
    harness created and handed to the server it launched, which can only ever
    be that server and the kernels under it.
    """
    if server.held_stdout is not None:
        server.held_stdout.release()


def unresolved_stdout(server: JupyterServer, *, timeout: float = HELD_PIPE_TIMEOUT) -> str:
    """Why the held stdout pipe is still open after the kill, or ``""``.

    End of file arrives when the last write end closes, so once the group kill
    has landed the drain should finish at once. If it has not within
    ``timeout``, something is still holding a descriptor this teardown handed
    out -- and the reader thread and its fd are still alive here to prove it.
    That is a finding, not a leak to be swallowed: it is reported so a survivor
    count taken by pid cannot claim a clean machine over the top of it.
    """
    if server.held_stdout is None or server.held_stdout.resolve(timeout=timeout):
        return ""
    return (
        f"the pipe held open as the server's stdout had not reached end of file "
        f"{timeout:.0f}s after the group kill, so something still holds the write end"
    )


def delete_kernel_and_stop(server: JupyterServer, kernel_id: str | None) -> TeardownReport:
    """Ask the server to end its kernel, then kill the server's group.

    The kernel has its own session, so the server's process group does not
    contain it; the server's own kernel manager is the only thing that reliably
    ends it, and the group kill then finishes the server off. Any held stdout
    pipe is released before the DELETE; see `release_held_stdout`.
    """
    release_held_stdout(server)
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
    stdout_mode: str = "file",
    capture_fd_output: bool = True,
    log: TextIO | None = None,
    ready_timeout: float = 60.0,
) -> KernelSession:
    """Start a ``jupyter server`` and take a kernel from it over the REST API.

    This is the provisioning path a notebook actually uses: the kernel is the
    server's child, started by the server's own kernel manager, with the
    server's stdio and ZMQ plumbing rather than this harness's. The census is
    rooted at the server for that reason -- the kernel is one of its
    descendants.

    ``stdout_mode`` and ``capture_fd_output`` set the two hops the kernel's
    fd 1 travels; turning capture off replaces the kernelspec, since that is
    the only place ipykernel's own option can be set.
    """
    deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
    server = start_jupyter_server(
        cwd=cwd,
        scratch=scratch,
        log=log,
        deadline=deadline,
        stdout_mode=stdout_mode,
        capture_fd_output=capture_fd_output,
    )
    requested = kernel_name if capture_fd_output else NO_FD_CAPTURE_KERNEL_NAME
    kernel_id: str | None = None
    census = DescendantCensus({}, False, "the kernel was never created")
    try:
        created = api_call(server, "api/kernels", method="POST", payload={"name": requested})
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


#: The imports every generated cell opens with. Shared because both corpora
#: need exactly these and nothing else; anything a corpus needs on top of them
#: it imports for itself.
KERNEL_IMPORTS = """
from pathlib import Path
from tempfile import TemporaryDirectory

from pstrain.api import setup_project
from pstrain.api.pipeline import PipelineContext, build_pipeline
"""


def pipeline_run_code(*, target: str, jobs: int) -> str:
    """The tail of every generated cell: run the pipeline, then say so.

    This much is identical whichever corpus was selected from, and it goes
    through ``pstrain.api``, the boundary the tutorial notebook uses, so the
    harness exercises the same call path that stalls. It is the only part the
    two generators share below the imports: the selection and the project they
    build differ, and were made to look alike once, at the cost of changing the
    cell the default path had been measured with. They are kept apart now.
    """
    return f"""    context = PipelineContext.from_config(
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


def training_code(fixture: Path, *, target: str, jobs: int, utterances: int) -> str:
    """Build kernel code for one isolated mini-Arctic training attempt.

    Imports go through ``pstrain.api``, the boundary the tutorial notebook
    uses, so the harness exercises the same call path that stalls.

    This cell is byte-for-byte what every recorded measurement was taken with,
    and `test_the_default_cell_is_the_one_every_measurement_was_taken_with`
    pins it there. The fixture's transcription is written through unchanged
    rather than reconstructed, because a default that quietly drifts makes the
    hundreds of clean runs behind it unusable as a baseline.
    """
    return f"""{KERNEL_IMPORTS}
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
{pipeline_run_code(target=target, jobs=jobs)}"""


def corpus_training_code(
    corpus: Path, dictionary: Path, *, target: str, jobs: int, utterances: int
) -> str:
    """Build kernel code for one training attempt over a full Arctic corpus.

    The bundled fixture is ten utterances, whose per-utterance output is far
    too small to fill a 64 KB pipe, so the volume needed to see fd 1
    backpressure has to come from a real corpus. Selection is the tutorial
    notebook's own: parse ``etc/txt.done.data``, normalize each prompt, reject
    any carrying a word the dictionary does not have, and take the first N by
    utterance id so a rerun selects the same utterances.

    The cell also reports which ``pstrain`` it imported, since ``--import-root``
    exists to point it somewhere other than this checkout and a run against the
    wrong tree measures nothing. That reporting lives here and not on the
    default path, which stays as it was measured.
    """
    return f"""{KERNEL_IMPORTS}
import re

import pstrain

corpus = Path({str(corpus)!r})
dictionary = Path({str(dictionary)!r})
lexicon = set()
for line in dictionary.read_text(encoding="utf-8").splitlines():
    if line.strip() and not line.startswith("#"):
        lexicon.add(line.split()[0])
prompt = re.compile(r'\\(\\s*(\\S+)\\s+"(.*)"\\s*\\)')
wav_dir = corpus / "wav"
in_vocabulary = {{}}
for line in (corpus / "etc" / "txt.done.data").read_text(encoding="utf-8").splitlines():
    match = prompt.match(line.strip())
    if match is None:
        continue
    utterance, text = match.group(1), match.group(2)
    words = [
        word.strip("'")
        for word in re.sub(r"[^\\w\\s']", " ", text.lower()).split()
        if word.strip("'")
    ]
    if not words or not all(word in lexicon for word in words):
        continue
    if not (wav_dir / (utterance + ".wav")).exists():
        continue
    in_vocabulary[utterance] = " ".join(words)
selected = dict(sorted(in_vocabulary.items())[:{utterances}])
if len(selected) < {utterances}:
    raise RuntimeError(f"only {{len(selected)}} in-vocabulary utterance(s) available")
print("pstrain imported from", pstrain.__file__)
print("selected", len(selected), "utterance(s) from", corpus)
with TemporaryDirectory(prefix="pstrain-kernel-repro-") as temporary:
    root = Path(temporary)
    audio = root / "audio"
    audio.mkdir()
    for utterance in selected:
        (audio / (utterance + ".wav")).symlink_to(wav_dir / (utterance + ".wav"))
    transcription = root / "transcription.txt"
    transcription.write_text(
        "".join(f"{{utterance}} {{text}}\\n" for utterance, text in selected.items()),
        encoding="utf-8",
    )
    project = root / "project"
    setup_project(
        project,
        transcription_path=transcription,
        audio_path=audio,
        dictionary_path=dictionary,
    )
{pipeline_run_code(target=target, jobs=jobs)}"""


def decode_code(fixture: Path, *, jobs: int, utterances: int, test_utterances: int) -> str:
    """Build kernel code for one mini-Arctic decode attempt.

    The subject here is ``test_model``'s own process pool
    (``pstrain/lib/testing/test.py``), not the pipeline's, so the training that
    has to happen first is run serially: a stall then belongs to the decode
    pool and to nothing else. The call is the tutorial's own decode cell --
    ``pstrain.api.testing.test_model`` with an explicit ``jobs`` -- which is
    where the defect was reported from.

    ``test_utterances`` is the split's holdout, and it is what decides how many
    pool workers the decode actually gets: ``_decode_files`` clamps its worker
    count to the number of files, so a holdout smaller than ``jobs`` measures a
    smaller pool than was asked for.
    """
    return f"""{KERNEL_IMPORTS}
from pstrain.api import parse_transcription_file
from pstrain.api.testing import test_model

fixture = Path({str(fixture)!r})
with TemporaryDirectory(prefix="pstrain-kernel-repro-") as temporary:
    root = Path(temporary)
    audio = root / "audio"
    audio.mkdir()
    lines = (fixture / "transcription.txt").read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if line.strip()][:{utterances}]
    if len(lines) <= {test_utterances}:
        raise RuntimeError("no utterances left to train on")
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
        cli_overrides={{
            "runner": {{"jobs": 1}},
            "split": {{"test_count": {test_utterances}}},
        }},
    )
    pipeline = build_pipeline(context)
    for stage in ("ci-1g", "lm"):
        return_code = pipeline.run(stage, jobs=1)
        if return_code:
            raise RuntimeError(f"pipeline stage {{stage}} returned {{return_code}}")
    etc = project / "experiments" / "default" / "etc"
    transcripts = parse_transcription_file(etc / "test.transcription")
    print("decoding", len(transcripts), "utterance(s) with jobs={jobs}")
    decoded = test_model(
        project / "shared" / "models" / "ci-1g" / "default",
        project / "audio",
        transcripts,
        project / "shared" / "dictionary.dict",
        project / "shared" / "filler.dict",
        lm=project / "experiments" / "default" / "lm" / "train.arpa",
        verbose=True,
        jobs={jobs},
    )
    if decoded.n_decoded != len(transcripts):
        raise RuntimeError(f"decoded {{decoded.n_decoded}} of {{len(transcripts)}}")
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
        "--decode",
        action="store_true",
        help=(
            "measure test_model's decode pool instead of the pipeline's: train the fixture "
            "serially, then decode the split's holdout with --jobs workers"
        ),
    )
    parser.add_argument(
        "--test-utterances",
        type=int,
        default=8,
        help="holdout the --decode run decodes, and so its worker ceiling (default 8)",
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
    parser.add_argument(
        "--server-stdout",
        choices=("file", "pipe"),
        default="file",
        help=(
            "what the jupyter server's own stdout is, and so what the kernel inherits as its "
            "original fd 1: 'file', the harness's launch log, which always drains, or 'pipe', "
            "a pipe nobody reads, which fills and blocks its writers (--server only)"
        ),
    )
    parser.add_argument(
        "--no-capture-fd-output",
        action="store_true",
        help=(
            "start the kernel with IPKernelApp.capture_fd_output=False, so fd 1 is the "
            "server's descriptor directly rather than ipykernel's capture pipe (--server only)"
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="train in-vocabulary utterances from this Arctic corpus instead of the fixture",
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=None,
        help="lexicon the --corpus prompts are filtered against and trained with",
    )
    parser.add_argument(
        "--import-root",
        type=Path,
        default=None,
        help="checkout the kernel imports pstrain from (default: this harness's own)",
    )
    parser.add_argument(
        "--hold-iopub",
        type=float,
        default=0.0,
        help=(
            "stop reading iopub for this many seconds after each execute request; a control, "
            "not a stall knob, since ZMQ PUB drops at its high-water mark and never blocks"
        ),
    )
    args = parser.parse_args(argv)
    # A serial pipeline run has no pool to stall, so the training modes need at
    # least two jobs to measure anything. A decode run is different: jobs=1 is
    # the documented workaround, and measuring it alongside jobs=4 is the point.
    minimum_jobs = 1 if args.decode else 2
    if args.runs < 1 or args.timeout <= 0 or args.jobs < minimum_jobs or args.utterances < 1:
        parser.error(f"require runs >= 1, timeout > 0, jobs >= {minimum_jobs}, and utterances >= 1")
    if args.test_utterances < 1:
        parser.error("--test-utterances must be at least 1")
    if args.decode and args.corpus is not None:
        parser.error("--decode trains the bundled fixture, so it cannot take --corpus")
    if args.hold_iopub < 0:
        parser.error("--hold-iopub cannot be negative")
    if args.kernel_name != DEFAULT_KERNEL_NAME and not args.server:
        parser.error("--kernel-name names the kernelspec a server starts, so it needs --server")
    if args.server_stdout != "file" and not args.server:
        parser.error("--server-stdout is the stdout of a server, so it needs --server")
    if args.no_capture_fd_output and not args.server:
        parser.error(
            "--no-capture-fd-output configures a kernelspec a server starts, so it needs --server"
        )
    if args.no_capture_fd_output and args.kernel_name != DEFAULT_KERNEL_NAME:
        parser.error(
            "--no-capture-fd-output writes and requests its own kernelspec, so the one named "
            "by --kernel-name would be ignored"
        )
    if (args.corpus is None) != (args.dictionary is None):
        parser.error("--corpus and --dictionary are only meaningful together")
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
            stdout_mode=args.server_stdout,
            capture_fd_output=not args.no_capture_fd_output,
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
                    execution = run_in_session(
                        session, code, timeout=args.timeout, hold_iopub=args.hold_iopub
                    )
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


def attempt_code(args: argparse.Namespace, *, repository: Path) -> str:
    """The kernel code one attempt runs, for the pool and corpus asked for."""
    if args.decode:
        return decode_code(
            repository / "tests" / "fixtures" / "mini_arctic",
            jobs=args.jobs,
            utterances=args.utterances,
            test_utterances=args.test_utterances,
        )
    if args.corpus is not None:
        return corpus_training_code(
            args.corpus.resolve(),
            args.dictionary.resolve(),
            target=args.target,
            jobs=args.jobs,
            utterances=args.utterances,
        )
    return training_code(
        repository / "tests" / "fixtures" / "mini_arctic",
        target=args.target,
        jobs=args.jobs,
        utterances=args.utterances,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    code = attempt_code(args, repository=repository)
    # The fixture and the kernel's import path are separate choices: the
    # fixture belongs to this checkout, while --import-root points the kernel
    # at the tree whose behavior is being measured, which may be another one.
    import_root = (args.import_root or repository).resolve()
    provisioning = "jupyter server" if args.server else "direct KernelManager"
    subject = (
        f"test_model decode pool, holdout={args.test_utterances}"
        if args.decode
        else f"pipeline pool, target={args.target}"
    )
    print(
        f"kernel pool stall repro: runs={args.runs} jobs={args.jobs} subject={subject} "
        f"utterances={args.utterances} timeout={args.timeout:.1f}s "
        f"provisioning={provisioning} "
        f"kernels={'one, reused' if args.reuse_kernel else 'one per run'} "
        f"corpus={args.corpus or 'mini fixture'} import_root={import_root} "
        f"server_stdout={args.server_stdout} "
        f"capture_fd_output={not args.no_capture_fd_output} "
        f"hold_iopub={args.hold_iopub:.0f}s",
        flush=True,
    )
    tally = run_attempts(args, code, repository=import_root)
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
