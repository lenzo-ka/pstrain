#!/usr/bin/env python3
"""Reproduce the Jupyter ``Pipeline.run(..., jobs>1)`` process-pool stall.

Run from the repository root after building the native library::

    PYTHONPATH=. nice -n 19 python scripts/repro_kernel_pool_hang.py \\
        --runs 20 --jobs 2 --timeout 120

Each attempt starts a fresh ipykernel and trains the mini-Arctic fixture to a
target, driving the pipeline exactly the way the tutorial notebook does. This
is a diagnostic stress harness, not a fix. An attempt that overruns its timeout
is recorded as stalled, together with the kernel's process tree and a stack
sample of every descendant; a kernel that dies is recorded as a failure rather
than a stall. Any stall or failure makes the program exit nonzero.

Cleanup is the kernel's process group, which jupyter_client creates with
``start_new_session=True`` and kills outright, so spawned pool workers go with
it. Nothing else is signalled. After the kill the harness counts any captured
descendant still present and reports it as a survivor, because a nonzero count
means the group kill missed something and is a finding about the harness. It
does not try to kill survivors: identifying a process by pid and start time is
not sound enough to authorize a signal, since a start time is only accurate to
the second and a pid recycled inside that second would pass the check. That
same-second window can only turn a dead process into a reported survivor, which
is the safe direction for a count nobody acts on.

The census fails loud, not quiet. A ``ps`` that exits nonzero, produces nothing
parseable, or yields rows the parser has to drop is a distinct state from a
census that ran and found nobody, and it is reported as a harness failure with
its reason rather than as zero survivors.
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import TextIO

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
    """A live kernel plus the identity it had at launch.

    ``started`` pins the kernel down: a provisioner keeps its numeric pid after
    the process dies, so the pid alone cannot say whether the process running
    under it now is the kernel this session launched.
    """

    manager: KernelManager
    client: BlockingKernelClient
    pid: int | None
    started: str | None


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


def kernel_descendants(session: KernelSession) -> DescendantCensus:
    """Census the live kernel and its descendants."""
    return assess_census(process_snapshot(), session.pid, session.started)


def shutdown_kernel(session: KernelSession) -> SurvivorReport:
    """Kill the kernel's process group and census what outlived it.

    The process group is the whole of the cleanup. The reported pids are
    descendants still present afterwards; they are never signalled. A census
    that could not be taken is reported untrusted, never as zero survivors.
    """
    before = kernel_descendants(session)
    captured = {pid: info for pid, info in before.descendants.items() if pid != session.pid}
    after = ProcessSnapshot({}, dropped=0, available=False)
    session.client.stop_channels()
    try:
        session.manager.shutdown_kernel(now=True)
    finally:
        after = process_snapshot()
    if not before.trusted:
        return SurvivorReport((), False, f"before the kill, {before.reason}")
    return assess_survivors(captured, after)


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
    """Process tree plus a stack for the kernel and each of its descendants."""
    census = kernel_descendants(session)
    if not census.trusted:
        return f"no process tree for kernel pid {session.pid}: {census.reason}"
    pids = sorted(census.descendants)
    sections = [
        f"process tree for kernel pid {session.pid}:",
        render_tree(census.descendants, pids),
    ]
    sections.extend(stack_sample(pid) for pid in pids)
    return "\n".join(sections)


def _launch_identity(manager: KernelManager) -> tuple[int | None, str | None]:
    """Read the freshly launched kernel's pid and start time."""
    pid = getattr(getattr(manager, "provisioner", None), "pid", None)
    if not isinstance(pid, int):
        return None, None
    info = process_snapshot().processes.get(pid)
    return pid, (info.started if info is not None else None)


def execute_in_kernel(
    code: str,
    *,
    timeout: float,
    cwd: Path | None = None,
    launch_stderr: TextIO | None = None,
) -> tuple[KernelExecution, KernelSession]:
    """Start a kernel and execute ``code``, leaving shutdown to the caller.

    The kernel inherits this process's environment with ``cwd`` prepended to
    ``PYTHONPATH`` so it imports the checkout under test rather than an
    installed copy. The caller must always call `shutdown_kernel`, including on
    the timeout path, where the live kernel is what the caller inspects.
    """
    launch: dict[str, object] = {}
    if launch_stderr is not None:
        launch["stderr"] = launch_stderr
    if cwd is not None:
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = f"{cwd}{os.pathsep}{existing}" if existing else str(cwd)
        launch["cwd"] = str(cwd)
        launch["env"] = environment

    manager = KernelManager()
    manager.start_kernel(**launch)
    pid, started = _launch_identity(manager)
    session = KernelSession(manager=manager, client=manager.client(), pid=pid, started=started)
    client = session.client
    client.start_channels()

    begun = time.monotonic()
    stdout: list[str] = []
    stderr: list[str] = []
    saw_error = False

    def outcome(*, reached_idle: bool, kernel_alive: bool) -> tuple[KernelExecution, KernelSession]:
        return (
            KernelExecution(
                reached_idle=reached_idle,
                kernel_alive=kernel_alive,
                saw_error=saw_error,
                elapsed=time.monotonic() - begun,
                stdout="".join(stdout),
                stderr="".join(stderr),
            ),
            session,
        )

    try:
        client.wait_for_ready(timeout=min(timeout, 60.0))
        message_id = client.execute(code, stop_on_error=False)
        while True:
            if not manager.is_alive():
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
    except BaseException:
        shutdown_kernel(session)
        raise


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
    parser.add_argument("--runs", type=int, default=20, help="number of fresh-kernel runs")
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
    args = parser.parse_args(argv)
    if args.runs < 1 or args.timeout <= 0 or args.jobs < 2 or args.utterances < 1:
        parser.error("require runs >= 1, timeout > 0, jobs >= 2, and utterances >= 1")
    return args


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    fixture = repository / "tests" / "fixtures" / "mini_arctic"
    code = training_code(fixture, target=args.target, jobs=args.jobs, utterances=args.utterances)
    stalls = 0
    failures = 0
    outlived = 0
    untrusted = 0
    print(
        f"kernel pool stall repro: runs={args.runs} jobs={args.jobs} target={args.target} "
        f"utterances={args.utterances} timeout={args.timeout:.1f}s",
        flush=True,
    )
    for run_number in range(1, args.runs + 1):
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as launch_log:
            execution, session = execute_in_kernel(
                code, timeout=args.timeout, cwd=repository, launch_stderr=launch_log
            )
            try:
                launch_log.seek(0)
                result = report_attempt(
                    f"run {run_number}/{args.runs}", execution, session, launch_log.read()
                )
                stalls += result == OUTCOME_STALLED
                failures += result == OUTCOME_FAILED
            finally:
                report = shutdown_kernel(session)
                label = f"run {run_number}/{args.runs}"
                if not report.trusted:
                    untrusted += 1
                    print(
                        f"{label}: CENSUS UNTRUSTED, cannot say what outlived "
                        f"the group kill: {report.reason}",
                        flush=True,
                    )
                elif report.survivors:
                    outlived += 1
                    print(
                        f"{label}: {len(report.survivors)} process(es) outlived the group "
                        f"kill and were left alone: "
                        f"{', '.join(str(pid) for pid in report.survivors)}",
                        flush=True,
                    )
    print(
        f"summary: {stalls}/{args.runs} stalled, {failures}/{args.runs} failed, "
        f"{outlived}/{args.runs} left survivors, {untrusted}/{args.runs} census untrusted",
        flush=True,
    )
    return 1 if stalls or failures or outlived or untrusted else 0


if __name__ == "__main__":
    sys.exit(main())
