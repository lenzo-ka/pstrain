"""Plumbing checks for the Jupyter kernel stall reproduction harness.

The kernel-starting checks carry the ``kernel`` marker, which the default
pytest options deselect, so CI never starts a kernel or a server. Everything
else is a pure check over the harness's classification, process-identity and
server-provisioning helpers, so the teardown rules can be tested without a
kernel and without a real ``ps``.
"""

import ast
import dataclasses
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import repro_kernel_pool_hang as harness
from scripts.repro_kernel_pool_hang import (
    COMPLETION_MARKER,
    DEFAULT_KERNEL_NAME,
    OUTCOME_COMPLETED,
    OUTCOME_FAILED,
    OUTCOME_STALLED,
    KernelExecution,
    TeardownReport,
    api_call,
    api_url,
    assess_census,
    assess_survivors,
    attempt_groups,
    auth_headers,
    captured_descendants,
    classify,
    connection_file,
    execute_in_kernel,
    group_kill_refusal,
    kernel_descendants,
    parse_args,
    parse_process_snapshot,
    pipeline_run_call,
    read_connection_file,
    run_in_session,
    select_descendants,
    server_command,
    shutdown_kernel,
    start_server_session,
    surviving_pids,
    throttle,
    training_code,
)

REPOSITORY = Path(__file__).resolve().parents[1]

# One ``ps`` listing: a kernel at pid 100 with two spawned pool workers, one of
# them nested, plus an unrelated process that is nobody's descendant.
PS_OUTPUT = """\
  100     1 SNs    0.0 Wed Aug 26 21:33:47 2026 python -m ipykernel_launcher -f /tmp/kernel.json
  101   100 SN     0.0 Wed Aug 26 21:33:48 2026 python -c from multiprocessing.spawn import main
  102   101 RN     0.0 Wed Aug 26 21:33:49 2026 python -c from multiprocessing.spawn import main
  200     1 Ss     0.1 Tue Aug 25 09:00:00 2026 /usr/sbin/unrelated --daemon
"""

# The same machine under ``--server``: the kernel at 301 is the server's child,
# so the whole tree hangs off pid 300 rather than off the kernel.
SERVER_PS_OUTPUT = """\
  300     1 SNs    0.2 Sat Sep  5 04:36:23 2026 python -m jupyter_server --no-browser
  301   300 SNs    0.0 Sat Sep  5 04:36:24 2026 python -m ipykernel_launcher -f /run/kernel-a.json
  302   301 SN     0.0 Sat Sep  5 04:36:30 2026 python -c from multiprocessing.spawn import main
  303   302 RN     0.0 Sat Sep  5 04:36:31 2026 python -c from multiprocessing.spawn import main
  400     1 Ss     0.1 Tue Aug 25 09:00:00 2026 /usr/sbin/unrelated --daemon
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


@pytest.mark.kernel
@pytest.mark.skipif(
    os.name != "posix", reason="the harness identifies kernel descendants through POSIX ps"
)
def test_a_server_provisioned_kernel_runs_and_shuts_down_clean(tmp_path: Path) -> None:
    """A kernel taken from a real jupyter server runs and cleans up completely."""
    # jupyter_server is a dev extra, and the harness launches it as `python -m
    # jupyter_server`, so both the import and the module entry point have to be
    # there; without them this is a skip, not a subprocess that exits and a
    # failure fifteen seconds later.
    pytest.importorskip("jupyter_server")
    pytest.importorskip("jupyter_server.__main__")
    session = start_server_session(cwd=REPOSITORY, scratch=tmp_path, ready_timeout=60.0)
    try:
        execution = run_in_session(session, "print('kernel-ready')", timeout=60.0)
        assert classify(execution) == OUTCOME_COMPLETED, execution.stderr
        assert execution.stdout == "kernel-ready\n"
        # The census is rooted at the server, so it sees the server and the
        # kernel the server started under it.
        assert session.root == "server"
        census = kernel_descendants(session)
        assert census.trusted, census.reason
        assert len(census.descendants) >= 2
    finally:
        # DELETE for the kernel and the process group for the server is the
        # whole cleanup, and it leaves nothing behind.
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


def test_parse_args_takes_the_server_mode_and_guards_its_options() -> None:
    args = parse_args(["--server", "--reuse-kernel", "--kernel-name", "python3-alt"])
    assert (args.server, args.reuse_kernel, args.kernel_name) == (True, True, "python3-alt")
    default = parse_args([])
    assert (default.server, default.reuse_kernel, default.kernel_name) == (
        False,
        False,
        DEFAULT_KERNEL_NAME,
    )
    # Only a server starts a kernelspec by name, so naming one without --server
    # would silently do nothing.
    with pytest.raises(SystemExit):
        parse_args(["--kernel-name", "python3-alt"])


def test_attempts_get_a_fresh_kernel_each_unless_one_is_reused() -> None:
    assert attempt_groups(3, reuse=False) == [[1], [2], [3]]
    # One kernel serving every run is the shape of a notebook session.
    assert attempt_groups(3, reuse=True) == [[1, 2, 3]]


def test_api_urls_join_over_exactly_one_separator() -> None:
    assert api_url("http://127.0.0.1:8888", "api/kernels") == "http://127.0.0.1:8888/api/kernels"
    assert api_url("http://127.0.0.1:8888/", "/api/kernels/") == "http://127.0.0.1:8888/api/kernels"


class _Reply:
    """The little of ``urlopen``'s return value that `api_call` touches."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_Reply":
        return self

    def __exit__(self, *exception: object) -> bool:
        return False


def test_the_token_travels_in_a_header_and_never_in_a_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = auth_headers("s3cret")
    assert headers["Authorization"] == "token s3cret"
    assert headers["Content-Type"] == "application/json"

    # Inspect the request api_call actually builds, not just the helper: a
    # regression that appended ?token= to the URL would satisfy the helper and
    # still leak the token into the server's request log and into any ps
    # listing this harness prints.
    sent: list[Any] = []

    def fake_urlopen(request: Any, timeout: float | None = None) -> _Reply:
        sent.append(request)
        return _Reply(b'{"id": "abc-123"}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    server = SimpleNamespace(base_url="http://127.0.0.1:8888", token="s3cret")
    created = api_call(server, "api/kernels", method="POST", payload={"name": "python3"})
    assert created == {"id": "abc-123"}
    (request,) = sent
    assert request.get_header("Authorization") == "token s3cret"
    assert request.full_url == "http://127.0.0.1:8888/api/kernels"
    assert "s3cret" not in request.full_url


def test_a_group_kill_is_refused_unless_the_pid_is_still_the_server_launched() -> None:
    snapshot = parse_process_snapshot(SERVER_PS_OUTPUT)
    allowed = group_kill_refusal(
        snapshot, 300, "Sat Sep 5 04:36:23 2026", launched_pgid=300, current_pgid=300
    )
    assert allowed == ""

    # The pid was recycled while the harness held it, so the group behind it is
    # a stranger's and must not be signalled.
    recycled = group_kill_refusal(
        snapshot, 300, "Mon Jan 1 00:00:00 2001", launched_pgid=300, current_pgid=300
    )
    assert "different process" in recycled

    gone = group_kill_refusal(
        snapshot, 999, "Sat Sep 5 04:36:23 2026", launched_pgid=999, current_pgid=999
    )
    assert "already gone" in gone

    # Leading its own group is what makes the kill reach the server's children
    # and nothing else.
    moved = group_kill_refusal(
        snapshot, 300, "Sat Sep 5 04:36:23 2026", launched_pgid=300, current_pgid=42
    )
    assert "no longer leads its own group" in moved
    never_led = group_kill_refusal(
        snapshot, 300, "Sat Sep 5 04:36:23 2026", launched_pgid=42, current_pgid=300
    )
    assert "did not lead its own group at launch" in never_led
    unreadable = group_kill_refusal(
        snapshot, 300, "Sat Sep 5 04:36:23 2026", launched_pgid=300, current_pgid=None
    )
    assert "no readable process group" in unreadable

    # And a listing that cannot be trusted is never a licence to signal.
    blind = group_kill_refusal(
        parse_process_snapshot("", status=1),
        300,
        "Sat Sep 5 04:36:23 2026",
        launched_pgid=300,
        current_pgid=300,
    )
    assert "unavailable" in blind


def test_a_denied_group_kill_is_never_reported_as_a_finished_teardown() -> None:
    def denied(pid: int, number: int) -> None:
        raise PermissionError(1, "Operation not permitted")

    report = harness.issue_group_kill(300, 300, kill=denied)
    assert not report.signalled
    assert "process group 300" in report.reason
    assert "server pid 300" in report.reason
    assert "Operation not permitted" in report.reason

    # The refusal has to reach the census. The server is the census root and is
    # excluded from the survivor count, so a signalled teardown here would
    # report a clean machine with the server still running on it.
    snapshot = parse_process_snapshot(SERVER_PS_OUTPUT)
    verdict = harness.report_teardown(
        report, assess_census(snapshot, 300, "Sat Sep 5 04:36:23 2026"), 300, snapshot
    )
    assert verdict.survivors == ()
    assert not verdict.trusted
    assert "Operation not permitted" in verdict.reason


def test_a_group_that_exited_before_the_signal_counts_as_torn_down() -> None:
    def gone(pid: int, number: int) -> None:
        raise ProcessLookupError(3, "No such process")

    # The identity gate found the process alive a moment before, so this is it
    # exiting in between: nothing left to kill, and nothing left to doubt.
    report = harness.issue_group_kill(300, 300, kill=gone)
    assert report.signalled
    assert report.reason == ""


def test_a_kill_that_lands_signals_the_verified_group() -> None:
    sent: list[tuple[int, int]] = []
    report = harness.issue_group_kill(300, 300, kill=lambda pid, number: sent.append((pid, number)))
    assert sent == [(300, signal.SIGKILL)]
    assert report.signalled


def test_a_refused_group_kill_makes_the_survivor_census_untrusted() -> None:
    census = assess_census(parse_process_snapshot(SERVER_PS_OUTPUT), 300, "Sat Sep 5 04:36:23 2026")
    refused = harness.report_teardown(
        TeardownReport(False, "pid 300 now belongs to a different process"),
        census,
        300,
        parse_process_snapshot(SERVER_PS_OUTPUT),
    )
    # Nothing was signalled, so "nothing survived" would be a claim about a
    # kill that never happened.
    assert refused.survivors == ()
    assert not refused.trusted
    assert "was not signalled" in refused.reason


def test_provisioning_that_fails_after_the_post_still_deletes_and_censuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A kernel the harness gave up on is deleted and reported, not orphaned."""
    snapshot = parse_process_snapshot(SERVER_PS_OUTPUT)
    server = SimpleNamespace(
        pid=300,
        started="Sat Sep 5 04:36:23 2026",
        runtime_dir=tmp_path,
        base_url="http://127.0.0.1:8888",
        token="s3cret",
    )
    calls: list[tuple[str, str]] = []

    def fake_api_call(target: Any, path: str, **options: Any) -> dict[str, str]:
        method = str(options.get("method", "GET"))
        calls.append((method, path))
        return {"id": "abc-123"} if method == "POST" else {}

    def refuse_connection_file(*args: Any, **options: Any) -> Path:
        raise RuntimeError("the server wrote no usable connection file")

    monkeypatch.setattr(harness, "start_jupyter_server", lambda **options: server)
    monkeypatch.setattr(harness, "api_call", fake_api_call)
    monkeypatch.setattr(harness, "await_connection_file", refuse_connection_file)
    monkeypatch.setattr(harness, "stop_server", lambda target: TeardownReport(True))
    monkeypatch.setattr(harness, "process_snapshot", lambda: snapshot)

    with pytest.raises(RuntimeError, match="no usable connection file"):
        start_server_session(cwd=tmp_path, scratch=tmp_path)

    # The kernel is in its own session, so the server's group kill cannot reach
    # it: the DELETE has to be attempted even though provisioning gave up.
    assert calls == [("POST", "api/kernels"), ("DELETE", "api/kernels/abc-123")]

    # And the census has to run against the tree captured while the kernel
    # existed, so an abandoned kernel is reported rather than silently left.
    reported = capsys.readouterr().out
    assert "abandoned provisioning" in reported
    assert "301, 302, 303" in reported


def test_connection_info_is_read_only_once_the_server_has_finished_writing(tmp_path: Path) -> None:
    path = connection_file(tmp_path, "abc-123")
    assert path == tmp_path / "kernel-abc-123.json"
    # Absent, half-written, and complete-but-not-yet-usable are all "not yet".
    assert read_connection_file(path) is None
    path.write_text('{"key": "k", "shell', encoding="utf-8")
    assert read_connection_file(path) is None
    path.write_text('{"key": "k", "shell_port": 1}', encoding="utf-8")
    assert read_connection_file(path) is None
    path.write_text('{"key": "k", "shell_port": 1, "iopub_port": 2}', encoding="utf-8")
    assert read_connection_file(path) == {"key": "k", "shell_port": 1, "iopub_port": 2}


def test_under_a_server_the_census_is_rooted_at_the_server_not_the_kernel() -> None:
    snapshot = parse_process_snapshot(SERVER_PS_OUTPUT)
    census = assess_census(snapshot, 300, "Sat Sep 5 04:36:23 2026")
    assert census.trusted, census.reason
    assert sorted(census.descendants) == [300, 301, 302, 303]

    # Only the process the harness launched is exempt from the survivor count.
    # The kernel is a descendant here, so a kernel that outlives teardown is
    # reported rather than excused.
    assert sorted(captured_descendants(census, 300)) == [301, 302, 303]

    # Rooting at the kernel, as the direct mode does, would not see the server.
    kernel_rooted = assess_census(snapshot, 301, "Sat Sep 5 04:36:24 2026")
    assert sorted(kernel_rooted.descendants) == [301, 302, 303]
    assert sorted(captured_descendants(kernel_rooted, 301)) == [302, 303]


def test_the_server_is_started_headless_without_restarts_or_port_drift() -> None:
    command = server_command(port=61784, root_dir=Path("/scratch/root"))
    assert command[:3] == [sys.executable, "-m", "jupyter_server"]
    assert "--ServerApp.port=61784" in command
    # A retried port would leave the server listening somewhere the harness is
    # not talking to, and a restarted kernel would hide a death behind a
    # replacement, so both are off.
    assert "--ServerApp.port_retries=0" in command
    assert "--KernelManager.autorestart=False" in command
    assert "--ServerApp.root_dir=/scratch/root" in command
    # The token is passed in the environment, so it is in no argv anywhere.
    assert not any("token" in argument.lower() for argument in command)


def test_a_throttled_liveness_check_is_not_asked_on_every_pass() -> None:
    asked = 0

    def check() -> bool:
        nonlocal asked
        asked += 1
        return True

    checked = throttle(check, interval=60.0)
    assert checked() and checked() and checked()
    assert asked == 1


def test_a_throttled_check_that_says_dead_is_never_asked_again() -> None:
    answers = iter([False, True])
    checked = throttle(lambda: next(answers), interval=0.0)
    assert checked() is False
    # A process that has gone does not come back, so re-asking could only let a
    # restarted stranger answer for it.
    assert checked() is False
