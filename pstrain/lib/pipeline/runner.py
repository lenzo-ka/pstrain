"""Pipeline runner: tasks, dependency resolution, staleness, execution.

This module is intentionally independent of pstrain specifics — it operates on
`Task` objects whose `fn` is an opaque callable. The actual pstrain tasks live in
`pstrain.lib.pipeline.tasks`.

Staleness model
---------------
A task is stale if any of:
  * Any declared output is missing.
  * Its completion marker is missing.
  * The newest input mtime is greater than or equal to the oldest output mtime.

Completion markers make interrupted writes stale; mtimes retain the deliberately
small file-path DAG model for ordinary dependencies.

Execution model
---------------
The planner returns a topologically-sorted list of tasks to run. The executor
runs them in dependency order. Ready tasks marked with the same
`parallel_group` are batched together and dispatched to a
`ProcessPoolExecutor` — this is how feature extraction fans out across fileids.

Dry-run prints the plan with staleness markers and never executes.
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing
import os
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

from pstrain.lib.native_worker import PstrainWorkerError

logger = logging.getLogger(__name__)

# When a fan-out is in flight, emit a progress line every Nth completion
# where N = max(1, total // _PROGRESS_REPORT_BUCKETS). Larger bucket
# count => more frequent updates.
_PROGRESS_REPORT_BUCKETS = 10

# Per-batch upper bound on how many individual failure messages to
# print. A long fan-out can produce hundreds of identical errors; we
# print the first few and a "... and N more" summary.
_MAX_FAILURES_TO_REPORT = 5

# A worker should start and return this module-level no-op quickly. Allow ample
# time for slow or heavily loaded hosts before treating startup as unavailable.
_POOL_STARTUP_TIMEOUT_SECONDS = 30.0


class UnknownTargetError(KeyError):
    """Raised when a build target is not registered with the pipeline."""


class TaskFailure(RuntimeError):
    """Raised when a task's callable raises, or when declared outputs are
    missing after execution."""


@dataclass(frozen=True)
class Task:
    """A single unit of work in the pipeline.

    Tasks are immutable. The `fn` callable should take no arguments — bind
    parameters via `functools.partial` or a closure when constructing the task.
    """

    name: str
    fn: Callable[[], None]
    inputs: tuple[Path, ...] = ()
    outputs: tuple[Path, ...] = ()
    description: str = ""
    # Tasks sharing a `parallel_group` may be run concurrently by the executor.
    # Use this for fan-outs (one task per fileid). Leave empty for the linear
    # training chain where ordering matters.
    parallel_group: str = ""

    @property
    def completion_marker(self) -> Path | None:
        """Private marker written only after all outputs are complete."""
        if not self.outputs:
            return None
        first = Path(self.outputs[0])
        task_id = hashlib.sha256(self.name.encode()).hexdigest()[:12]
        return first.parent / f".{first.name}.{task_id}.complete"


@dataclass
class _PlanEntry:
    task: Task
    stale: bool
    reason: str


class Pipeline:
    """Registers tasks and resolves their dependency graph by file paths.

    Multiple tasks may not produce the same output. Tasks may declare inputs
    that no other task produces — those are treated as required external
    files (e.g. raw audio, hand-written transcripts).
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._producer_by_output: dict[Path, str] = {}
        self._targets: dict[str, Path] = {}

    def add(self, task: Task) -> None:
        if task.name in self._tasks:
            raise ValueError(f"duplicate task name: {task.name!r}")
        for out in task.outputs:
            out = Path(out)
            if out in self._producer_by_output:
                other = self._producer_by_output[out]
                raise ValueError(f"two tasks produce {out}: {other!r} and {task.name!r}")
            self._producer_by_output[out] = task.name
        self._tasks[task.name] = task

    def add_all(self, tasks: Iterable[Task]) -> None:
        for t in tasks:
            self.add(t)

    def register_target(self, name: str, sentinel_output: Path) -> None:
        """Register a human-readable build target name (e.g. "cd-8g") that
        resolves to a representative output path. The pipeline will plan
        everything required to produce that output."""
        self._targets[name] = Path(sentinel_output)

    def targets(self) -> dict[str, Path]:
        return dict(self._targets)

    def tasks(self) -> dict[str, Task]:
        return dict(self._tasks)

    def resolve_target(self, target: str | Path) -> Path:
        """Map a target name to an output path, or pass through if already a
        path."""
        if isinstance(target, Path):
            return target
        if target in self._targets:
            return self._targets[target]
        as_path = Path(target)
        if as_path in self._producer_by_output:
            return as_path
        raise UnknownTargetError(target)

    def plan(
        self,
        target: str | Path,
        *,
        force: bool = False,
    ) -> list[_PlanEntry]:
        """Return a topologically-sorted plan to build `target`.

        Tasks whose outputs are already up to date are included with
        `stale=False` so the caller can show them or skip them.

        Staleness propagates: if any upstream task is going to re-run, its
        new outputs will be newer than this task's outputs, so this task
        must also re-run.
        """
        target_path = self.resolve_target(target)
        if target_path not in self._producer_by_output:
            raise UnknownTargetError(str(target))

        ordered_names = self._toposort_for(target_path)
        plan: list[_PlanEntry] = []

        # First pass: direct staleness from filesystem mtimes.
        entries_by_name: dict[str, _PlanEntry] = {}
        for name in ordered_names:
            task = self._tasks[name]
            stale, reason = self._staleness(task)
            if force:
                stale, reason = True, "forced"
            entry = _PlanEntry(task=task, stale=stale, reason=reason)
            plan.append(entry)
            entries_by_name[name] = entry

        # Second pass: propagate staleness from upstream tasks. If any
        # producer of one of my inputs is stale, I'm stale too — its new
        # outputs will be newer than mine.
        for entry in plan:
            if entry.stale:
                continue
            for dep in entry.task.inputs:
                producer = self._producer_by_output.get(Path(dep))
                if producer and entries_by_name[producer].stale:
                    entry.stale = True
                    entry.reason = f"upstream {producer!r} will run"
                    break

        return plan

    def run(
        self,
        target: str | Path,
        *,
        dry_run: bool = False,
        force: bool = False,
        jobs: int | None = None,
    ) -> int:
        """Build `target`. ``jobs=None`` uses the available CPU count."""
        plan = self.plan(target, force=force)

        if dry_run:
            _print_plan(plan, target=str(target))
            return 0

        to_run = [e for e in plan if e.stale]
        if not to_run:
            print(f"Up to date: {target}")
            return 0

        return _execute(to_run, jobs=_resolve_jobs(jobs))

    def _toposort_for(self, target: Path) -> list[str]:
        """Return task names in dependency order, reachable from `target`."""
        order: list[str] = []
        visited: set[str] = set()
        on_stack: set[str] = set()

        def visit(out: Path) -> None:
            producer = self._producer_by_output.get(out)
            if producer is None:
                return
            if producer in visited:
                return
            if producer in on_stack:
                cycle = " -> ".join([*on_stack, producer])
                raise RuntimeError(f"cycle detected in task graph: {cycle}")
            on_stack.add(producer)
            task = self._tasks[producer]
            for dep in task.inputs:
                visit(Path(dep))
            on_stack.discard(producer)
            visited.add(producer)
            order.append(producer)

        visit(target)
        return order

    def _staleness(self, task: Task) -> tuple[bool, str]:
        if not task.outputs:
            return True, "no outputs (always runs)"
        outputs = [Path(p) for p in task.outputs]
        missing = [p for p in outputs if not p.exists()]
        if missing:
            return True, f"missing output: {missing[0]}"
        marker = task.completion_marker
        if marker is not None and not marker.exists():
            return True, "missing completion marker"
        out_mtimes = [p.stat().st_mtime for p in outputs]
        oldest_out = min(out_mtimes)
        existing_inputs = [Path(p) for p in task.inputs if Path(p).exists()]
        if not existing_inputs:
            return False, "up to date"
        newest_in = max(p.stat().st_mtime for p in existing_inputs)
        if newest_in >= oldest_out:
            return True, "inputs not older than outputs"
        return False, "up to date"


def _print_plan(plan: list[_PlanEntry], *, target: str) -> None:
    """Print the plan in a Make-style format."""
    print(f"# Plan for target: {target}")
    print(f"# {len(plan)} task(s); {sum(1 for e in plan if e.stale)} stale")
    print()
    for i, entry in enumerate(plan, 1):
        marker = "*" if entry.stale else "."
        print(f"{marker} [{i:2d}] {entry.task.name}  ({entry.reason})")
        if entry.task.description:
            print(f"        {entry.task.description}")
    print()
    print("# Legend: * = will run, . = up to date")


def _resolve_jobs(jobs: int | None) -> int:
    """Resolve the API's auto worker setting to at least one worker."""
    if jobs is None:
        return max(1, os.cpu_count() or 1)
    return jobs


def _execute(entries: list[_PlanEntry], *, jobs: int) -> int:
    """Execute entries, batching all dependency-ready members of a group."""
    pending = list(entries)
    producer_by_output = {
        Path(output): entry.task.name for entry in entries for output in entry.task.outputs
    }
    executed: set[str] = set()

    while pending:
        entry = pending[0]
        group = entry.task.parallel_group
        if group and jobs > 1:
            batch = [
                candidate
                for candidate in pending
                if candidate.task.parallel_group == group
                and all(
                    (producer := producer_by_output.get(Path(task_input))) is None
                    or producer in executed
                    for task_input in candidate.task.inputs
                )
            ]
            rc = _run_parallel_batch(batch, jobs=jobs)
            if rc != 0:
                return rc
            batch_names = {candidate.task.name for candidate in batch}
            executed.update(batch_names)
            pending = [candidate for candidate in pending if candidate.task.name not in batch_names]
        else:
            rc = _run_one(entry.task)
            if rc != 0:
                return rc
            executed.add(entry.task.name)
            pending.pop(0)
    return 0


def _run_one(task: Task) -> int:
    """Run a single task in-process. Returns exit code."""
    logger.info("Running %s", task.name)
    print(f"-> {task.name}")
    start = time.monotonic()
    try:
        _execute_task(task)
    except Exception as exc:
        logger.exception("Task %s failed", task.name)
        print(f"!! {task.name} failed: {exc}")
        return 1
    elapsed = time.monotonic() - start
    print(f"   {task.name} done in {elapsed:.1f}s")
    return 0


def _run_parallel_batch(batch: list[_PlanEntry], *, jobs: int) -> int:
    """Run a batch of independent tasks in a process pool.

    A no-op probe proves that the pool can start before any real work is
    submitted. If construction or the probe fails, the batch is aborted: it is
    never quietly rerun in this process. Guarded native
    operations exist precisely so that malformed input cannot end the
    interpreter, and running them here after isolation failed would hand that
    guarantee back. Additional workers may start lazily after submission and
    fail after other tasks complete. The batch then fails; completed tasks keep
    their valid manifests, while unfinished tasks remain stale for the rerun.
    """
    group_name = batch[0].task.parallel_group
    n = len(batch)
    workers = min(jobs, n)
    print(f"-> fan-out [{group_name}]: {n} task(s), {workers} worker(s)")
    start = time.monotonic()
    failures: list[tuple[str, BaseException]] = []
    completed = 0
    try:
        pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
    except (OSError, BrokenProcessPool) as exc:
        return _abort_unstartable_batch(group_name=group_name, exc=exc)

    try:
        probe = pool.submit(_pool_startup_probe)
        probe.result(timeout=_POOL_STARTUP_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        # Deliberately abandon a wedged worker process: leaking it is preferable
        # to hanging the entire run while waiting for shutdown.
        pool.shutdown(wait=False, cancel_futures=True)
        return _abort_unstartable_batch(group_name=group_name, exc=exc)
    except (OSError, BrokenProcessPool) as exc:
        pool.shutdown(wait=True, cancel_futures=True)
        return _abort_unstartable_batch(group_name=group_name, exc=exc)

    try:
        future_to_task: dict[Future[None], Task] = {}
        for entry in batch:
            try:
                future = pool.submit(_worker, entry.task)
            except Exception as exc:
                failures.append((entry.task.name, exc))
                logger.exception("Process pool failed while submitting %s", entry.task.name)
                break
            future_to_task[future] = entry.task

        for fut in as_completed(future_to_task):
            task = future_to_task[fut]
            try:
                fut.result()
                _verify_outputs(task)
                completed += 1
                report_every = max(1, n // _PROGRESS_REPORT_BUCKETS)
                if completed % report_every == 0 or completed == n:
                    print(f"   [{group_name}] {completed}/{n}")
            except BaseException as exc:
                failures.append((task.name, exc))
                logger.exception("Parallel task %s failed", task.name)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    elapsed = time.monotonic() - start
    if failures:
        print(f"!! fan-out [{group_name}]: {len(failures)} failure(s)")
        for failed_name, failed_exc in failures[:_MAX_FAILURES_TO_REPORT]:
            print(f"   {failed_name}: {failed_exc}")
        if len(failures) > _MAX_FAILURES_TO_REPORT:
            print(f"   ... and {len(failures) - _MAX_FAILURES_TO_REPORT} more")
        return 1
    print(f"   fan-out [{group_name}] done in {elapsed:.1f}s")
    return 0


def _abort_unstartable_batch(*, group_name: str, exc: BaseException) -> int:
    """Abort a batch whose process isolation cannot be established."""
    detail = str(exc) or type(exc).__name__
    error = PstrainWorkerError(f"cannot start process pool for {group_name}: {detail}")
    logger.error("%s", error)
    print(f"!! fan-out [{group_name}] aborted: {error}")
    return 1


def _verify_outputs(task: Task) -> None:
    """Raise when a completed task omitted any declared output."""
    missing = [path for path in task.outputs if not Path(path).exists()]
    if missing:
        raise TaskFailure(f"did not produce: {missing}")


def _execute_task(task: Task) -> None:
    """Execute a task and publish its completion marker last."""
    marker = task.completion_marker
    if marker is not None:
        marker.unlink(missing_ok=True)
    task.fn()
    _verify_outputs(task)
    if marker is not None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        temporary = marker.with_name(f"{marker.name}.tmp-{os.getpid()}")
        temporary.write_text(f"task={task.name}\n", encoding="utf-8")
        temporary.replace(marker)


def _pool_startup_probe() -> None:
    """Importable no-op used to prove that a process-pool worker can start."""


def _worker(task: Task) -> None:
    """Entry point for ProcessPoolExecutor workers.

    Runs the task's callable. Must be importable at module top level so
    pickling works.
    """
    _execute_task(task)
