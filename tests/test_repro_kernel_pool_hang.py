"""Plumbing checks for the Jupyter kernel stall reproduction harness.

The kernel-starting check carries the ``kernel`` marker, which the default
pytest options deselect, so CI never starts a kernel. Everything else is a pure
check over the harness's classification and process-identity helpers, so the
teardown rules can be tested without a kernel and without a real ``ps``.
"""

import ast
import dataclasses
import os
from pathlib import Path

import pytest

from scripts.repro_kernel_pool_hang import (
    COMPLETION_MARKER,
    OUTCOME_COMPLETED,
    OUTCOME_FAILED,
    OUTCOME_STALLED,
    KernelExecution,
    assess_census,
    assess_survivors,
    classify,
    execute_in_kernel,
    parse_args,
    parse_process_snapshot,
    pipeline_run_call,
    select_descendants,
    shutdown_kernel,
    surviving_pids,
    training_code,
)

# One ``ps`` listing: a kernel at pid 100 with two spawned pool workers, one of
# them nested, plus an unrelated process that is nobody's descendant.
PS_OUTPUT = """\
  100     1 SNs    0.0 Wed Aug 26 21:33:47 2026 python -m ipykernel_launcher -f /tmp/kernel.json
  101   100 SN     0.0 Wed Aug 26 21:33:48 2026 python -c from multiprocessing.spawn import main
  102   101 RN     0.0 Wed Aug 26 21:33:49 2026 python -c from multiprocessing.spawn import main
  200     1 Ss     0.1 Tue Aug 25 09:00:00 2026 /usr/sbin/unrelated --daemon
"""

HEALTHY = KernelExecution(
    reached_idle=True,
    kernel_alive=True,
    saw_error=False,
    elapsed=1.0,
    stdout=f"{COMPLETION_MARKER}\n",
    stderr="",
)


@pytest.mark.kernel
@pytest.mark.skipif(
    os.name != "posix", reason="the harness identifies kernel descendants through POSIX ps"
)
def test_harness_starts_executes_and_shuts_down_a_kernel() -> None:
    """The harness can drive a real kernel end to end and clean it up."""
    execution, session = execute_in_kernel("print('kernel-ready')", timeout=60.0)
    try:
        assert classify(execution) == OUTCOME_COMPLETED, execution.stderr
        assert execution.stdout == "kernel-ready\n"
        assert session.pid is not None
        assert session.started is not None
    finally:
        # The process-group kill is the whole cleanup: the census runs, and it
        # finds nothing left over.
        report = shutdown_kernel(session)
        assert report.trusted, report.reason
        assert report.survivors == ()


def test_an_error_reply_is_a_failure_not_a_completion() -> None:
    errored = dataclasses.replace(HEALTHY, saw_error=True)
    assert classify(errored, marker=COMPLETION_MARKER) == OUTCOME_FAILED


def test_a_missing_completion_marker_is_not_success() -> None:
    silent = dataclasses.replace(HEALTHY, stdout="training finished\n")
    assert classify(silent, marker=COMPLETION_MARKER) == OUTCOME_FAILED
    # With no marker demanded, the same execution counts as a completion.
    assert classify(silent) == OUTCOME_COMPLETED


def test_a_timeout_stalls_but_a_dead_kernel_fails() -> None:
    timed_out = dataclasses.replace(HEALTHY, reached_idle=False)
    assert classify(timed_out, marker=COMPLETION_MARKER) == OUTCOME_STALLED
    died = dataclasses.replace(timed_out, kernel_alive=False)
    assert classify(died, marker=COMPLETION_MARKER) == OUTCOME_FAILED


def test_a_healthy_execution_completes() -> None:
    assert classify(HEALTHY, marker=COMPLETION_MARKER) == OUTCOME_COMPLETED


def test_descendant_selection_is_transitive_and_excludes_strangers() -> None:
    snapshot = parse_process_snapshot(PS_OUTPUT)
    assert snapshot.processes[101].ppid == 100
    assert snapshot.processes[100].started == "Wed Aug 26 21:33:47 2026"
    assert snapshot.processes[100].command.startswith("python -m ipykernel_launcher")
    assert select_descendants(snapshot.processes, 100) == [100, 101, 102]
    assert select_descendants(snapshot.processes, 999) == []


def test_unreadable_rows_are_counted_not_silently_dropped() -> None:
    snapshot = parse_process_snapshot(PS_OUTPUT + "not a process row at all\n")
    assert sorted(snapshot.processes) == [100, 101, 102, 200]
    assert snapshot.dropped == 1
    assert snapshot.available is True
    # Present but incomplete, so it must not be used as a census.
    assert snapshot.trustworthy is False


def test_a_failed_ps_is_census_unavailable_not_an_empty_machine() -> None:
    failed = parse_process_snapshot("", status=1)
    assert failed.available is False
    assert failed.trustworthy is False
    assert failed.processes == {}
    # Output with nothing parseable in it is equally unavailable.
    assert parse_process_snapshot("ps: illegal option\n").available is False


def test_a_pre_kill_census_needs_the_kernels_own_row() -> None:
    snapshot = parse_process_snapshot(PS_OUTPUT)
    good = assess_census(snapshot, 100, "Wed Aug 26 21:33:47 2026")
    assert good.trusted
    assert sorted(good.descendants) == [100, 101, 102]

    # The kernel is still alive here, so a listing without its row describes a
    # machine that cannot be right.
    without_kernel = parse_process_snapshot(
        "\n".join(line for line in PS_OUTPUT.splitlines() if not line.startswith("  100")) + "\n"
    )
    missing = assess_census(without_kernel, 100, "Wed Aug 26 21:33:47 2026")
    assert not missing.trusted
    assert "own row for pid 100 is missing" in missing.reason

    stale = assess_census(snapshot, 100, "Mon Jan 1 00:00:00 2001")
    assert not stale.trusted
    assert "different process" in stale.reason

    unavailable = assess_census(parse_process_snapshot("", status=1), 100, "Wed Aug 26 21:33:47")
    assert not unavailable.trusted
    assert "unavailable" in unavailable.reason


def test_a_census_that_could_not_run_is_never_reported_as_zero_survivors() -> None:
    captured = dict(parse_process_snapshot(PS_OUTPUT).processes)
    unavailable = assess_survivors(captured, parse_process_snapshot("", status=1))
    assert unavailable.survivors == ()
    assert not unavailable.trusted
    assert "unavailable" in unavailable.reason

    partial = assess_survivors(captured, parse_process_snapshot(PS_OUTPUT + "garbage\n"))
    assert partial.survivors == ()
    assert not partial.trusted
    assert "could not be parsed" in partial.reason


def test_the_kernels_own_row_may_be_gone_after_the_kill() -> None:
    """After the kill the kernel is meant to be dead, so its absence is normal."""
    captured = dict(parse_process_snapshot(PS_OUTPUT).processes)
    del captured[100]
    del captured[200]
    after = parse_process_snapshot("  200     1 Ss     0.1 Tue Aug 25 09:00:00 2026 /usr/sbin/x\n")
    report = assess_survivors(captured, after)
    assert report.trusted
    assert report.survivors == ()


def test_a_recycled_pid_is_not_reported_as_a_survivor() -> None:
    captured = dict(parse_process_snapshot(PS_OUTPUT).processes)
    del captured[200]
    # After the group kill, 101 is gone and 102's number has been handed to a
    # new process, so only 100 is still the process that was captured.
    current = parse_process_snapshot(
        PS_OUTPUT.replace("  101   100 SN", "  199   100 SN").replace(
            "  102   101 RN     0.0 Wed Aug 26 21:33:49 2026",
            "  102   101 RN     0.0 Thu Sep  3 08:00:00 2026",
        )
    )
    # The parser normalizes ps's day-of-month padding, so both sides of the
    # comparison are spaced the same way and a match means a real match.
    assert current.processes[102].started == "Thu Sep 3 08:00:00 2026"
    assert surviving_pids(captured, current.processes) == [100]


def test_generated_kernel_code_parses_and_drives_the_requested_run() -> None:
    code = training_code(Path("/fixture"), target="cd-1g", jobs=3, utterances=4)
    call = pipeline_run_call(code)
    assert call is not None
    assert ast.literal_eval(call.args[0]) == "cd-1g"
    keywords = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
    assert keywords == {"jobs": 3}


def test_parse_args_rejects_serial_and_empty_configurations() -> None:
    args = parse_args(["--runs", "5", "--jobs", "4", "--target", "cd-1g"])
    assert (args.runs, args.jobs, args.target) == (5, 4, "cd-1g")
    for bad in (["--jobs", "1"], ["--runs", "0"], ["--utterances", "0"], ["--timeout", "0"]):
        with pytest.raises(SystemExit):
            parse_args(bad)
