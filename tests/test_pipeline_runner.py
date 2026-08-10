"""Tests for pstrain.lib.pipeline.runner.

Focus on the runner's behavior in isolation: dependency resolution, topo
sort, staleness detection, dry-run output, and parallel fan-out execution.

These tests use trivial file-touching tasks; they do not exercise any
actual pstrain training code.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import Future
from pathlib import Path

import pytest

from pstrain.lib.pipeline import Pipeline, Task, UnknownTargetError, runner
from pstrain.lib.pipeline import timings as pipeline_timings


def _touch(path: Path, contents: str = "") -> None:
    """Module-level worker for parallel-execution tests; must be picklable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


def _raise_worker_error() -> None:
    """Picklable failing callable for parallel-execution tests."""
    raise RuntimeError("worker task failed")


def _produce_nothing() -> None:
    """Picklable callable that deliberately omits its declared output."""


def _burn_cpu_in_child(output: Path) -> None:
    subprocess.run([sys.executable, "-c", "sum(i * i for i in range(2_000_000))"], check=True)
    output.touch()


def _make_touch_task(name: str, out: Path, *, inputs: tuple[Path, ...] = ()) -> Task:
    """A simple task that writes `name` to `out` when executed."""
    return Task(
        name=name,
        fn=functools.partial(_touch, out, name),
        inputs=inputs,
        outputs=(out,),
    )


def _mark_complete(task: Task) -> None:
    marker = task.completion_marker
    assert marker is not None
    marker.write_text("complete")


def test_simple_linear_chain_runs_in_order(tmp_path: Path) -> None:
    """flat -> ci-1g; both outputs missing; both should run, in order."""
    flat = tmp_path / "flat.txt"
    ci = tmp_path / "ci.txt"

    pl = Pipeline()
    pl.add(_make_touch_task("flat", flat))
    pl.add(_make_touch_task("ci-1g", ci, inputs=(flat,)))
    pl.register_target("ci-1g", ci)

    rc = pl.run("ci-1g")
    assert rc == 0
    assert flat.read_text() == "flat"
    assert ci.read_text() == "ci-1g"
    assert flat.stat().st_mtime <= ci.stat().st_mtime


def test_skip_when_up_to_date(tmp_path: Path) -> None:
    """If outputs are newer than inputs, nothing runs and rc=0."""
    flat = tmp_path / "flat.txt"
    ci = tmp_path / "ci.txt"
    flat.write_text("old-flat")
    time.sleep(0.01)
    ci.write_text("old-ci")

    ran: list[str] = []

    def record_and_touch(name: str, out: Path) -> None:
        ran.append(name)
        out.write_text(name)

    pl = Pipeline()
    pl.add(
        Task(
            name="flat",
            fn=functools.partial(record_and_touch, "flat", flat),
            outputs=(flat,),
        )
    )
    pl.add(
        Task(
            name="ci-1g",
            fn=functools.partial(record_and_touch, "ci-1g", ci),
            inputs=(flat,),
            outputs=(ci,),
        )
    )
    pl.register_target("ci-1g", ci)
    for task in pl.tasks().values():
        _mark_complete(task)

    rc = pl.run("ci-1g")
    assert rc == 0
    assert ran == []
    assert ci.read_text() == "old-ci"


def test_force_reruns_everything(tmp_path: Path) -> None:
    """--force runs all reachable tasks even if up to date."""
    flat = tmp_path / "flat.txt"
    ci = tmp_path / "ci.txt"
    flat.write_text("old")
    time.sleep(0.01)
    ci.write_text("old")

    ran: list[str] = []

    def record_and_touch(name: str, out: Path) -> None:
        ran.append(name)
        out.write_text(name)

    pl = Pipeline()
    pl.add(
        Task(
            name="flat",
            fn=functools.partial(record_and_touch, "flat", flat),
            outputs=(flat,),
        )
    )
    pl.add(
        Task(
            name="ci-1g",
            fn=functools.partial(record_and_touch, "ci-1g", ci),
            inputs=(flat,),
            outputs=(ci,),
        )
    )
    pl.register_target("ci-1g", ci)
    rc = pl.run("ci-1g", force=True)
    assert rc == 0
    assert ran == ["flat", "ci-1g"]


def test_stale_input_triggers_rerun(tmp_path: Path) -> None:
    """An input newer than any output marks the consumer task stale."""
    flat = tmp_path / "flat.txt"
    ci = tmp_path / "ci.txt"
    ci.write_text("old-ci")
    time.sleep(0.05)
    flat.write_text("new-flat")  # Newer than ci

    ran: list[str] = []

    def record_and_touch(name: str, out: Path) -> None:
        ran.append(name)
        out.write_text(name)

    pl = Pipeline()
    pl.add(
        Task(
            name="flat",
            fn=functools.partial(record_and_touch, "flat", flat),
            outputs=(flat,),
        )
    )
    pl.add(
        Task(
            name="ci-1g",
            fn=functools.partial(record_and_touch, "ci-1g", ci),
            inputs=(flat,),
            outputs=(ci,),
        )
    )
    pl.register_target("ci-1g", ci)
    _mark_complete(pl.tasks()["flat"])

    rc = pl.run("ci-1g")
    assert rc == 0
    assert ran == ["ci-1g"]  # flat is up to date; only ci-1g reruns


def test_equal_mtime_input_triggers_rerun(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.write_text("source")
    output.write_text("old")
    task = _make_touch_task("build", output, inputs=(source,))
    _mark_complete(task)
    same_mtime = output.stat().st_mtime_ns
    source.write_text("changed")
    source.touch()
    os.utime(source, ns=(same_mtime, same_mtime))

    pl = Pipeline()
    pl.add(task)
    pl.register_target("build", output)

    assert pl.plan("build")[0].stale
    assert pl.run("build") == 0
    assert output.read_text() == "build"


def test_failed_rebuild_removes_completion_marker(tmp_path: Path) -> None:
    output = tmp_path / "artifact"

    def partial_then_fail() -> None:
        output.write_text("partial")
        raise RuntimeError("interrupted")

    task = Task("build", partial_then_fail, outputs=(output,))
    output.write_text("previously complete")
    _mark_complete(task)
    pl = Pipeline()
    pl.add(task)
    pl.register_target("build", output)

    assert pl.run("build", force=True) != 0
    assert task.completion_marker is not None
    assert not task.completion_marker.exists()
    assert pl.plan("build")[0].stale


def test_dry_run_does_not_execute(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Dry-run prints a plan and does not run any task callables."""
    out = tmp_path / "out.txt"
    ran: list[str] = []

    def boom() -> None:
        ran.append("ran")
        raise RuntimeError("should not be called in dry-run")

    pl = Pipeline()
    pl.add(Task(name="t1", fn=boom, outputs=(out,)))
    pl.register_target("t1", out)

    rc = pl.run("t1", dry_run=True)
    assert rc == 0
    assert ran == []
    assert not out.exists()
    captured = capsys.readouterr()
    assert "Plan for target: t1" in captured.out
    assert "t1" in captured.out


def test_unknown_target_raises(tmp_path: Path) -> None:
    pl = Pipeline()
    pl.add(Task(name="t", fn=lambda: None, outputs=(tmp_path / "x",)))
    with pytest.raises(UnknownTargetError):
        pl.run("does-not-exist")


def test_duplicate_outputs_rejected(tmp_path: Path) -> None:
    pl = Pipeline()
    out = tmp_path / "shared.txt"
    pl.add(Task(name="a", fn=lambda: None, outputs=(out,)))
    with pytest.raises(ValueError, match="two tasks produce"):
        pl.add(Task(name="b", fn=lambda: None, outputs=(out,)))


def test_cycle_detected(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    pl = Pipeline()
    pl.add(Task(name="ta", fn=lambda: None, inputs=(b,), outputs=(a,)))
    pl.add(Task(name="tb", fn=lambda: None, inputs=(a,), outputs=(b,)))
    pl.register_target("ta", a)
    with pytest.raises(RuntimeError, match="cycle"):
        pl.plan("ta")


def test_task_failure_returns_nonzero(tmp_path: Path) -> None:
    out = tmp_path / "x.txt"

    def boom() -> None:
        raise RuntimeError("nope")

    pl = Pipeline()
    pl.add(Task(name="t", fn=boom, outputs=(out,)))
    pl.register_target("t", out)
    rc = pl.run("t")
    assert rc != 0
    assert not out.exists()


def test_missing_output_after_run_fails(tmp_path: Path) -> None:
    """A task whose fn returns without producing its declared outputs fails."""
    out = tmp_path / "x.txt"

    def silent() -> None:
        pass  # Don't actually create the file

    pl = Pipeline()
    pl.add(Task(name="t", fn=silent, outputs=(out,)))
    pl.register_target("t", out)
    rc = pl.run("t")
    assert rc != 0


def test_diamond_dependency_visits_each_task_once(tmp_path: Path) -> None:
    """a -> b, a -> c, both -> d. `a` should appear once in the plan."""
    a, b, c, d = (tmp_path / x for x in ["a", "b", "c", "d"])
    ran: list[str] = []

    def record(name: str, out: Path) -> None:
        ran.append(name)
        out.write_text(name)

    pl = Pipeline()
    pl.add(Task("a", functools.partial(record, "a", a), outputs=(a,)))
    pl.add(Task("b", functools.partial(record, "b", b), inputs=(a,), outputs=(b,)))
    pl.add(Task("c", functools.partial(record, "c", c), inputs=(a,), outputs=(c,)))
    pl.add(Task("d", functools.partial(record, "d", d), inputs=(b, c), outputs=(d,)))
    pl.register_target("d", d)

    rc = pl.run("d")
    assert rc == 0
    assert ran.count("a") == 1
    # a must come before b and c; b and c must come before d
    assert ran.index("a") < ran.index("b") < ran.index("d")
    assert ran.index("a") < ran.index("c") < ran.index("d")


def test_external_inputs_are_not_treated_as_tasks(tmp_path: Path) -> None:
    """If a task's input has no producer, it's an external file."""
    external = tmp_path / "external.wav"
    external.write_text("audio")
    out = tmp_path / "out.txt"

    pl = Pipeline()
    pl.add(
        Task(
            name="t",
            fn=functools.partial(_touch, out, "ok"),
            inputs=(external,),
            outputs=(out,),
        )
    )
    pl.register_target("t", out)
    rc = pl.run("t")
    assert rc == 0
    assert out.read_text() == "ok"


def test_parallel_fanout_writes_all_outputs(tmp_path: Path) -> None:
    """All parallel-group tasks should run; outputs must exist after."""
    n = 6
    outputs = [tmp_path / f"f{i}.txt" for i in range(n)]
    pl = Pipeline()
    for i, out in enumerate(outputs):
        pl.add(
            Task(
                name=f"extract:{i}",
                fn=functools.partial(_touch, out, f"data-{i}"),
                outputs=(out,),
                parallel_group="features",
            )
        )

    # The "features" target points at the last output, but tasks have no
    # cross-dependencies, so we use a sentinel that depends on all of them.
    sentinel = tmp_path / "sentinel.txt"
    pl.add(
        Task(
            name="sentinel",
            fn=functools.partial(_touch, sentinel, "done"),
            inputs=tuple(outputs),
            outputs=(sentinel,),
        )
    )
    pl.register_target("features", sentinel)

    rc = pl.run("features", jobs=4)
    assert rc == 0
    for i, out in enumerate(outputs):
        assert out.read_text() == f"data-{i}"
    assert sentinel.read_text() == "done"


def test_parallel_group_batches_ready_non_adjacent_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ready group members form one batch despite an interleaved task."""
    producer = tmp_path / "producer.txt"
    first = tmp_path / "first.txt"
    unrelated = tmp_path / "unrelated.txt"
    second = tmp_path / "second.txt"
    sentinel = tmp_path / "sentinel.txt"
    batches: list[list[str]] = []

    def record_batch(batch: list[runner._PlanEntry], *, jobs: int) -> int:
        assert jobs == 2
        assert producer.exists()
        batches.append([entry.task.name for entry in batch])
        for entry in batch:
            if runner._run_one(entry.task) != 0:
                return 1
        return 0

    monkeypatch.setattr(runner, "_run_parallel_batch", record_batch)

    pl = Pipeline()
    pl.add(_make_touch_task("producer", producer))
    pl.add(
        Task(
            "group:first",
            functools.partial(_touch, first, "first"),
            inputs=(producer,),
            outputs=(first,),
            parallel_group="features",
        )
    )
    pl.add(_make_touch_task("unrelated", unrelated))
    pl.add(
        Task(
            "group:second",
            functools.partial(_touch, second, "second"),
            inputs=(producer,),
            outputs=(second,),
            parallel_group="features",
        )
    )
    pl.add(
        Task(
            "sentinel",
            functools.partial(_touch, sentinel, "done"),
            inputs=(first, unrelated, second),
            outputs=(sentinel,),
        )
    )
    pl.register_target("all", sentinel)

    assert [entry.task.name for entry in pl.plan("all")] == [
        "producer",
        "group:first",
        "unrelated",
        "group:second",
        "sentinel",
    ]
    assert pl.run("all", jobs=2) == 0
    assert batches == [["group:first", "group:second"]]
    assert unrelated.exists()
    assert sentinel.exists()


def test_jobs_none_uses_cpu_count_bounded_by_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = [tmp_path / f"out-{i}" for i in range(3)]
    sentinel = tmp_path / "sentinel"
    observed: list[tuple[int, int]] = []

    def record_batch(batch: list[runner._PlanEntry], *, jobs: int) -> int:
        observed.append((jobs, min(jobs, len(batch))))
        for entry in batch:
            rc = runner._run_one(entry.task)
            if rc != 0:
                return rc
        return 0

    monkeypatch.setattr(runner.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(runner, "_run_parallel_batch", record_batch)
    pl = Pipeline()
    for i, output in enumerate(outputs):
        pl.add(
            Task(
                f"group:{i}",
                functools.partial(_touch, output),
                outputs=(output,),
                parallel_group="group",
            )
        )
    pl.add(_make_touch_task("sentinel", sentinel, inputs=tuple(outputs)))
    pl.register_target("all", sentinel)

    assert pl.run("all") == 0
    assert observed == [(8, 3)]


def test_jobs_one_runs_inline_without_constructing_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"

    def unexpected_pool(*args: object, **kwargs: object) -> None:
        pytest.fail("ProcessPoolExecutor must not be constructed for jobs=1")

    monkeypatch.setattr(runner, "ProcessPoolExecutor", unexpected_pool)
    pl = Pipeline()
    pl.add(
        Task(
            "grouped",
            functools.partial(_touch, output),
            outputs=(output,),
            parallel_group="group",
        )
    )
    pl.register_target("grouped", output)

    assert pl.run("grouped", jobs=1) == 0
    assert output.exists()


def test_pool_probe_failure_aborts_batch_without_inline_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    outputs = [tmp_path / f"out-{i}" for i in range(2)]
    sentinel = tmp_path / "sentinel"

    submitted: list[object] = []
    shutdown_calls: list[tuple[bool, bool]] = []

    class ProbeDeniedPool:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def submit(self, fn: object, *args: object, **kwargs: object) -> None:
            # ProcessPoolExecutor registers work before worker startup can fail.
            submitted.append(fn)
            raise PermissionError("semaphores denied")

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            shutdown_calls.append((wait, cancel_futures))

    monkeypatch.setattr(runner, "ProcessPoolExecutor", ProbeDeniedPool)
    pl = Pipeline()
    for i, output in enumerate(outputs):
        pl.add(
            Task(
                f"group:{i}",
                functools.partial(_touch, output),
                outputs=(output,),
                parallel_group="group",
            )
        )
    pl.add(_make_touch_task("sentinel", sentinel, inputs=tuple(outputs)))
    pl.register_target("all", sentinel)

    with caplog.at_level("ERROR"):
        assert pl.run("all", jobs=2) == 1
    assert all(not output.exists() for output in outputs)
    assert not sentinel.exists()
    assert submitted == [runner._pool_startup_probe]
    assert shutdown_calls == [(True, True)]
    assert "cannot start process pool for group" in caplog.text


def test_pool_probe_timeout_abandons_worker_and_aborts_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs = [tmp_path / f"out-{i}" for i in range(2)]
    submitted: list[object] = []
    shutdown_calls: list[tuple[bool, bool]] = []

    class TimedOutProbeFuture(Future[None]):
        def result(self, timeout: float | None = None) -> None:
            assert timeout == runner._POOL_STARTUP_TIMEOUT_SECONDS
            raise TimeoutError

    class WedgedProbePool:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def submit(self, fn: object, *args: object, **kwargs: object) -> Future[None]:
            submitted.append(fn)
            return TimedOutProbeFuture()

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            shutdown_calls.append((wait, cancel_futures))

    monkeypatch.setattr(runner, "ProcessPoolExecutor", WedgedProbePool)
    batch = [
        runner._PlanEntry(
            Task(
                f"group:{i}",
                functools.partial(_touch, output, str(i)),
                outputs=(output,),
                parallel_group="group",
            ),
            stale=True,
            reason="test",
        )
        for i, output in enumerate(outputs)
    ]

    assert runner._run_parallel_batch(batch, jobs=2) == 1
    assert submitted == [runner._pool_startup_probe]
    assert shutdown_calls == [(False, True)]
    assert all(not output.exists() for output in outputs)


def test_real_submit_failure_after_probe_fails_without_inline_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    outputs = [tmp_path / f"out-{i}" for i in range(2)]
    submitted: list[object] = []
    shutdown_calls: list[tuple[bool, bool]] = []

    class TrackingFuture(Future[None]):
        result_calls = 0

        def result(self, timeout: float | None = None) -> None:
            self.result_calls += 1
            return super().result(timeout)

    first_real_future = TrackingFuture()
    first_real_future.set_result(None)

    class RegisterThenFailPool:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def submit(self, fn: object, *args: object, **kwargs: object) -> Future[None]:
            submitted.append(fn)
            future: Future[None] = Future()
            if fn is runner._pool_startup_probe:
                future.set_result(None)
                return future
            if len(submitted) == 2:
                args[0].fn()
                return first_real_future
            # Mirror executor ordering: register the work, then fail startup.
            raise PermissionError("worker spawn denied")

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            shutdown_calls.append((wait, cancel_futures))

    def unexpected_inline_run(task: Task) -> int:
        pytest.fail("real tasks must not be rerun inline after a successful probe")

    monkeypatch.setattr(runner, "_run_one", unexpected_inline_run)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", RegisterThenFailPool)
    batch = [
        runner._PlanEntry(
            Task(
                f"group:{i}",
                functools.partial(_touch, output, str(i)),
                outputs=(output,),
                parallel_group="group",
            ),
            stale=True,
            reason="test",
        )
        for i, output in enumerate(outputs)
    ]

    assert runner._run_parallel_batch(batch, jobs=2) != 0
    assert submitted == [runner._pool_startup_probe, runner._worker, runner._worker]
    assert first_real_future.result_calls == 1
    assert outputs[0].exists()
    assert not outputs[1].exists()
    assert shutdown_calls == [(True, True)]
    assert "group:1: worker spawn denied" in capsys.readouterr().out


def test_parallel_worker_failure_does_not_fall_back_inline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "output"

    def unexpected_inline_run(task: Task) -> int:
        pytest.fail("worker failures must not be rerun inline")

    class AcceptedFailingPool:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def submit(self, fn: object, *args: object, **kwargs: object) -> Future[None]:
            future: Future[None] = Future()
            if fn is runner._pool_startup_probe:
                future.set_result(None)
            else:
                future.set_exception(RuntimeError("worker task failed"))
            return future

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            assert wait
            assert cancel_futures

    monkeypatch.setattr(runner, "_run_one", unexpected_inline_run)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", AcceptedFailingPool)
    pl = Pipeline()
    pl.add(
        Task(
            "grouped",
            _raise_worker_error,
            outputs=(output,),
            parallel_group="group",
        )
    )
    pl.register_target("grouped", output)

    assert pl.run("grouped", jobs=2) != 0
    assert "grouped: worker task failed" in capsys.readouterr().out
    assert not output.exists()


def test_parallel_mixed_batch_reports_task_with_missing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outputs = [tmp_path / f"output-{i}" for i in range(3)]

    class AcceptedSuccessfulPool:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def submit(self, fn: object, *args: object, **kwargs: object) -> Future[None]:
            future: Future[None] = Future()
            if fn is runner._worker:
                args[0].fn()
            future.set_result(None)
            return future

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            assert wait
            assert cancel_futures

    monkeypatch.setattr(runner, "ProcessPoolExecutor", AcceptedSuccessfulPool)
    batch = [
        runner._PlanEntry(
            Task(
                f"group:{i}",
                _produce_nothing if i == 1 else functools.partial(_touch, output, str(i)),
                outputs=(output,),
                parallel_group="group",
            ),
            stale=True,
            reason="test",
        )
        for i, output in enumerate(outputs)
    ]

    assert runner._run_parallel_batch(batch, jobs=2) != 0
    output = capsys.readouterr().out
    assert f"group:1: did not produce: [{outputs[1]!r}]" in output
    assert "group:0: did not produce" not in output
    assert "group:2: did not produce" not in output
    assert outputs[0].exists()
    assert not outputs[1].exists()
    assert outputs[2].exists()


def test_staleness_propagates_to_downstream(tmp_path: Path) -> None:
    """If A is stale and B depends on A, B is stale even if B's outputs are
    currently newer than B's existing inputs."""
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    c = tmp_path / "c.txt"

    # b.txt exists and is newer than a (a is missing); c.txt is also there.
    b.write_text("old-b")
    c.write_text("old-c")
    # a is missing, so A is stale; B will rerun because A reruns;
    # C will rerun because B reruns.

    ran: list[str] = []

    def record(name: str, out: Path) -> None:
        ran.append(name)
        out.write_text(name)

    pl = Pipeline()
    pl.add(Task("A", functools.partial(record, "A", a), outputs=(a,)))
    pl.add(Task("B", functools.partial(record, "B", b), inputs=(a,), outputs=(b,)))
    pl.add(Task("C", functools.partial(record, "C", c), inputs=(b,), outputs=(c,)))
    pl.register_target("C", c)

    plan = pl.plan("C")
    assert all(e.stale for e in plan), [e.task.name + ":" + e.reason for e in plan]
    rc = pl.run("C")
    assert rc == 0
    assert ran == ["A", "B", "C"]


def test_force_from_path_target_works(tmp_path: Path) -> None:
    """Resolving a target by its output path (not a registered name) works."""
    out = tmp_path / "out.txt"
    pl = Pipeline()
    pl.add(Task(name="t", fn=functools.partial(_touch, out, "ok"), outputs=(out,)))
    rc = pl.run(out)
    assert rc == 0
    assert out.read_text() == "ok"


def test_inline_and_pool_task_timings_are_persisted_and_rolled_up(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    final = tmp_path / "final"
    pipeline = Pipeline(tmp_path)
    pipeline.add(
        Task(
            "features:first",
            functools.partial(_touch, first),
            outputs=(first,),
            parallel_group="features",
        )
    )
    pipeline.add(
        Task(
            "features:second",
            functools.partial(_touch, second),
            outputs=(second,),
            parallel_group="features",
        )
    )
    pipeline.add(_make_touch_task("finish", final, inputs=(first, second)))
    pipeline.register_target("all", final)

    assert pipeline.run("all", jobs=2) == 0
    artifacts = list((tmp_path / ".pstrain" / "timings").glob("*.json"))
    assert len(artifacts) == 1
    document = json.loads(artifacts[0].read_text())
    assert set(document) == {
        "schema_version",
        "run_id",
        "target",
        "start",
        "end",
        "tasks",
        "stages",
    }
    assert document["schema_version"] == 1
    assert {item["task"] for item in document["tasks"]} == {
        "features:first",
        "features:second",
        "finish",
    }
    features = next(item for item in document["stages"] if item["stage"] == "features")
    feature_tasks = [item for item in document["tasks"] if item["stage"] == "features"]
    assert features["wall"] == pytest.approx(sum(item["wall"] for item in feature_tasks))
    assert features["cpu"] == pytest.approx(
        sum(
            item["cpu_user"]
            + item["cpu_sys"]
            + item["cpu_children_user"]
            + item["cpu_children_sys"]
            for item in feature_tasks
        )
    )


def test_cpu_timing_includes_reaped_child(tmp_path: Path) -> None:
    output = tmp_path / "child-output"
    timing = runner._worker(
        Task("child-cpu", functools.partial(_burn_cpu_in_child, output), outputs=(output,))
    )
    assert timing.cpu_children_user + timing.cpu_children_sys > 0


def test_timing_write_failure_does_not_fail_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    output = tmp_path / "output"
    pipeline = Pipeline(tmp_path)
    pipeline.add(
        Task("work", functools.partial(Path.write_text, output, "done"), outputs=(output,))
    )
    pipeline.register_target("work", output)

    blocker = tmp_path / "not-a-directory"
    blocker.write_text("blocked")
    monkeypatch.setattr(pipeline_timings, "timings_dir", lambda project: blocker / "timings")
    assert pipeline.run("work", jobs=1) == 0
    assert output.exists()
    assert "Could not write pipeline timings" in caplog.text


def test_rollup_schema_and_summary_are_value_tolerant() -> None:
    record = pipeline_timings.TaskTiming(
        task="features:a",
        stage="features",
        group="features",
        wall=2.0,
        cpu_user=0.5,
        cpu_sys=0.25,
        cpu_children_user=0.5,
        cpu_children_sys=0.25,
        start="2026-08-10T00:00:00+00:00",
        end="2026-08-10T00:00:02+00:00",
    )
    document = pipeline_timings.build_document(
        [record],
        run_id="named",
        target="features",
        started=record.start,
        ended=record.end,
    )
    assert document["stages"] == [
        {"stage": "features", "wall": 2.0, "cpu": 1.5, "cpu_wall_ratio": 0.75}
    ]
    assert "features" in pipeline_timings.format_summary(document)
    assert "0.75x" in pipeline_timings.format_summary(document)
