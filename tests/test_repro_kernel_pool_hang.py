"""Plumbing checks for the Jupyter kernel stall reproduction harness.

The kernel-starting checks carry the ``kernel`` marker, which the default
pytest options deselect, so CI never starts a kernel or a server. Everything
else is a pure check over the harness's classification, process-identity and
server-provisioning helpers, so the teardown rules can be tested without a
kernel and without a real ``ps``.

One of the marked checks reproduces the fd 1 write backpressure the harness
exists to catch, with a synthetic writer and no pstrain in it at all, so the
instrument can be shown to see that class of stall rather than only asserted to.
"""

import ast
import dataclasses
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import repro_kernel_pool_hang as harness
from scripts.repro_kernel_pool_hang import (
    COMPLETION_MARKER,
    DEFAULT_KERNEL_NAME,
    NO_FD_CAPTURE_KERNEL_NAME,
    OUTCOME_COMPLETED,
    OUTCOME_FAILED,
    OUTCOME_STALLED,
    HeldPipe,
    KernelExecution,
    ProcessInfo,
    TeardownReport,
    api_call,
    api_url,
    assess_census,
    assess_survivors,
    attempt_code,
    attempt_groups,
    auth_headers,
    captured_descendants,
    classify,
    connection_file,
    corpus_training_code,
    decode_code,
    execute_in_kernel,
    group_kill_refusal,
    kernel_descendants,
    parse_args,
    parse_process_snapshot,
    pipeline_run_call,
    read_connection_file,
    release_held_stdout,
    run_in_session,
    select_descendants,
    server_command,
    server_stdout,
    shutdown_kernel,
    stall_report,
    start_server_session,
    surviving_pids,
    throttle,
    training_code,
    write_no_capture_kernelspec,
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


# The writer lives in its own module file rather than in the executed cell,
# because a spawn child re-imports its target by name and a function defined in
# a kernel's __main__ is not importable anywhere.
FD1_WRITER_MODULE = '''"""A child that fills fd 1, the way a pool worker's native printf does."""

import os


def write_to_fd1(payload_size: int, writes: int) -> None:
    for _ in range(writes):
        os.write(1, b"x" * payload_size)
'''

#: Enough to overrun both hops -- ipykernel's 64 KB capture pipe and the 64 KB
#: pipe the server's stdout is -- so no buffer along the way can absorb it.
FD1_FLOOD_BYTES = 300 * 1024
#: Long enough for a kernel to start a child and for the child to fill both
#: pipes, which takes a fraction of a second, and short enough that a stalled
#: attempt plus its stack sampling stays well inside a minute.
FD1_FLOOD_TIMEOUT = 12.0


def fd1_flood_code(directory: Path) -> str:
    """Kernel code whose spawned child writes `FD1_FLOOD_BYTES` to fd 1.

    The writer is a spawn child and not the kernel's own thread because that is
    the shape of the thing that stalls: a process that inherited the kernel's
    fd 1 and knows nothing about ipykernel. There is no pstrain here -- the
    mechanism is the plumbing, and this reproduces it on its own.
    """
    return f"""
import multiprocessing
import sys
from pathlib import Path

directory = Path({str(directory)!r})
directory.mkdir(parents=True, exist_ok=True)
(directory / "fd1_writer.py").write_text({FD1_WRITER_MODULE!r}, encoding="utf-8")
if str(directory) not in sys.path:
    sys.path.insert(0, str(directory))
import fd1_writer

child = multiprocessing.get_context("spawn").Process(
    target=fd1_writer.write_to_fd1, args=(4096, {FD1_FLOOD_BYTES // 4096})
)
child.start()
child.join()
print({COMPLETION_MARKER!r})
"""


def flood_a_servers_stdout(
    scratch: Path, *, stdout_mode: str
) -> tuple[str, str, dict[int, ProcessInfo], Any]:
    """Run the fd 1 flood under a server with the given stdout, and tear down.

    Returns the outcome, the stall report and process tree when there is one,
    and the teardown report, so every assertion can be made after the server is
    gone rather than while it is holding a pipe open.
    """
    scratch.mkdir(parents=True)
    outcome, report = OUTCOME_FAILED, ""
    descendants: dict[int, ProcessInfo] = {}
    with (scratch / "launch.log").open("w+", encoding="utf-8") as log:
        session = start_server_session(
            cwd=REPOSITORY,
            scratch=scratch,
            stdout_mode=stdout_mode,
            log=log,
            ready_timeout=60.0,
        )
        try:
            execution = run_in_session(
                session, fd1_flood_code(scratch / "writer"), timeout=FD1_FLOOD_TIMEOUT
            )
            outcome = classify(execution, marker=COMPLETION_MARKER)
            if outcome != OUTCOME_COMPLETED:
                census = kernel_descendants(session)
                assert census.trusted, census.reason
                descendants = dict(census.descendants)
                report = stall_report(session)
        finally:
            teardown = shutdown_kernel(session)
    return outcome, report, descendants, teardown


def report_section(report: str, command: str) -> str:
    """The block of a stall report produced by one ``$ command`` line."""
    for block in report.split("\n$ "):
        if block.startswith(command):
            return block
    return ""


@pytest.mark.kernel
@pytest.mark.skipif(
    os.name != "posix", reason="the harness identifies kernel descendants through POSIX ps"
)
def test_a_server_stdout_nobody_reads_stalls_a_child_writing_to_fd_1(tmp_path: Path) -> None:
    """The stall this harness hunts, reproduced on demand without pstrain.

    A child of the kernel writes more to fd 1 than the two pipes between it and
    the server's stdout can hold. With nothing draining the far end the copy
    ipykernel's watcher thread makes blocks, the capture pipe fills behind it,
    and the child parks in ``write(2)`` -- so the run must classify as stalled,
    and the stall report must show the child in that syscall holding a pipe on
    fd 1. Anything less and the instrument cannot see this class of defect.
    """
    pytest.importorskip("jupyter_server")
    pytest.importorskip("jupyter_server.__main__")
    outcome, report, descendants, teardown = flood_a_servers_stdout(
        tmp_path / "unread-pipe", stdout_mode="pipe"
    )
    assert teardown.trusted, teardown.reason
    assert teardown.survivors == ()
    assert outcome == OUTCOME_STALLED

    writers = [pid for pid, info in descendants.items() if "spawn_main" in info.command]
    assert len(writers) == 1, report
    writer = writers[0]

    if shutil.which("py-spy"):
        stack = report_section(report, f"py-spy dump --pid {writer}")
        assert stack, report
        # py-spy walks Python frames, so the flooding function is what names it.
        assert "write_to_fd1" in stack, stack
    elif shutil.which("sample"):
        stack = report_section(report, f"sample {writer} ")
        assert stack, report
        # macOS `sample` walks native frames, so the syscall itself is visible.
        assert "os_write" in stack, stack
        assert "write  (in libsystem_kernel" in stack or "write_nocancel" in stack, stack
    else:
        # Neither sampler is installed, so the harness took no stack and there
        # is none to read. Everything else about the stall still holds and is
        # still asserted; only the syscall goes unwitnessed.
        assert f"pid {writer}: neither py-spy nor sample is available" in report, report

    if shutil.which("lsof"):
        files = report_section(report, f"lsof -n -P -p {writer}")
        assert files, report
        rows = [line.split() for line in files.splitlines()]
        on_fd1 = [row for row in rows if len(row) > 4 and row[3].rstrip("rwu") == "1"]
        assert on_fd1, files
        # A pipe on fd 1, not a terminal and not a file: what the child is
        # parked on is the thing that cannot take another byte.
        assert all(row[4] in {"PIPE", "FIFO"} for row in on_fd1), files


@pytest.mark.kernel
@pytest.mark.skipif(
    os.name != "posix", reason="the harness identifies kernel descendants through POSIX ps"
)
def test_the_same_flood_completes_when_the_servers_stdout_drains(tmp_path: Path) -> None:
    """The control: identical code, a stdout that drains, and no stall.

    This is what makes the stalled run mean something. The only difference is
    the descriptor the server was given, so a completion here says the volume
    of output is not itself the cause -- the far end not draining is.
    """
    pytest.importorskip("jupyter_server")
    pytest.importorskip("jupyter_server.__main__")
    outcome, report, _, teardown = flood_a_servers_stdout(
        tmp_path / "launch-log", stdout_mode="file"
    )
    assert teardown.trusted, teardown.reason
    assert teardown.survivors == ()
    assert outcome == OUTCOME_COMPLETED, report


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


# The mini-fixture cell exactly as every recorded run of this harness executed
# it. Hundreds of clean attempts are only a baseline for the next measurement
# if the next measurement runs the same code, so this is spelled out in full
# rather than described: a change here has to be made deliberately, and any
# result taken before it has to be re-read in the light of it.
DEFAULT_CELL = """
from pathlib import Path
from tempfile import TemporaryDirectory

from pstrain.api import setup_project
from pstrain.api.pipeline import PipelineContext, build_pipeline

fixture = Path('/fixture')
with TemporaryDirectory(prefix="pstrain-kernel-repro-") as temporary:
    root = Path(temporary)
    audio = root / "audio"
    audio.mkdir()
    lines = (fixture / "transcription.txt").read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if line.strip()][:10]
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
        cli_overrides={"runner": {"jobs": 2}},
    )
    return_code = build_pipeline(context).run('ci-1g', jobs=2)
    if return_code:
        raise RuntimeError(f"pipeline returned {return_code}")
print('pstrain-kernel-repro: completed')
"""


def test_the_default_cell_is_the_one_every_measurement_was_taken_with() -> None:
    assert training_code(Path("/fixture"), target="ci-1g", jobs=2, utterances=10) == DEFAULT_CELL
    # The additions made for the corpus path are on the corpus path only. None
    # of them may reach the default cell, which is the measured baseline.
    default = training_code(Path("/fixture"), target="ci-1g", jobs=4, utterances=500)
    for addition in ("import pstrain", "pstrain imported from", 'print("selected"', "wav_dir"):
        assert addition not in default, addition


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


def test_parse_args_takes_the_backpressure_options_and_guards_them() -> None:
    args = parse_args(
        ["--server", "--server-stdout", "pipe", "--no-capture-fd-output", "--hold-iopub", "5"]
    )
    assert (args.server_stdout, args.no_capture_fd_output, args.hold_iopub) == ("pipe", True, 5.0)
    default = parse_args([])
    # The defaults are the ones every measurement so far was taken under: a
    # stdout that drains, ipykernel's capture on, iopub read as it arrives.
    assert (default.server_stdout, default.no_capture_fd_output) == ("file", False)
    assert (default.hold_iopub, default.corpus, default.dictionary, default.import_root) == (
        0.0,
        None,
        None,
        None,
    )
    for bad in (
        # Both options configure a server, or the kernelspec one starts.
        ["--server-stdout", "pipe"],
        ["--no-capture-fd-output"],
        # The no-capture spec is the one requested, so a named one is ignored.
        ["--server", "--no-capture-fd-output", "--kernel-name", "python3-alt"],
        # A corpus is selected against a dictionary; neither alone says which
        # utterances to train.
        ["--corpus", "/corpus"],
        ["--dictionary", "/lexicon.dict"],
        ["--hold-iopub", "-1"],
    ):
        with pytest.raises(SystemExit):
            parse_args(bad)


def test_a_corpus_run_selects_in_vocabulary_utterances_and_drives_the_run() -> None:
    code = corpus_training_code(
        Path("/corpus"), Path("/lexicon.dict"), target="cd-1g", jobs=4, utterances=300
    )
    call = pipeline_run_call(code)
    assert call is not None
    assert ast.literal_eval(call.args[0]) == "cd-1g"
    assert {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords} == {
        "jobs": 4
    }
    # Prompts come from where a full Arctic corpus keeps them, the selection is
    # bounded and ordered so a rerun trains the same utterances, and the
    # fixture's phoneset and filler dictionary play no part in a real corpus.
    assert "txt.done.data" in code
    assert "sorted(in_vocabulary.items())[:300]" in code
    assert "phoneset_path" not in code


def test_a_decode_run_trains_serially_and_decodes_with_the_requested_jobs() -> None:
    """The decode cell's subject is test_model's pool, so nothing else may fan out."""
    code = decode_code(Path("/fixture"), jobs=4, utterances=10, test_utterances=8)
    tree = ast.parse(code)
    runs = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "run"
    ]
    assert len(runs) == 1
    assert {keyword.arg: ast.literal_eval(keyword.value) for keyword in runs[0].keywords} == {
        "jobs": 1
    }
    decodes = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "test_model"
    ]
    assert len(decodes) == 1
    assert {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in decodes[0].keywords
        if keyword.arg in {"jobs", "verbose"}
    } == {"jobs": 4, "verbose": True}
    # The holdout is what the decode pool's worker count is clamped to, so it
    # has to reach the split rather than only the transcript that is read back.
    assert '"split": {"test_count": 8}' in code
    # The one run walks a fixed, literal sequence of stages, and it is that
    # sequence -- not the presence or absence of some substring in the cell --
    # that says the decode is reached without fanning anything out on the way.
    literal_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple)
    ]
    assert len(literal_loops) == 1
    stages = literal_loops[0]
    assert ast.literal_eval(stages.iter) == ("ci-1g", "lm")
    assert isinstance(stages.target, ast.Name)
    assert len(runs[0].args) == 1
    assert isinstance(runs[0].args[0], ast.Name)
    assert runs[0].args[0].id == stages.target.id


def test_a_decode_run_is_the_only_one_that_may_ask_for_a_single_job() -> None:
    args = parse_args(["--decode", "--jobs", "1", "--test-utterances", "3"])
    assert (args.decode, args.jobs, args.test_utterances) == (True, 1, 3)
    assert parse_args([]).decode is False
    assert parse_args([]).test_utterances == 8
    for bad in (
        ["--jobs", "1"],
        ["--decode", "--jobs", "0"],
        ["--decode", "--test-utterances", "0"],
        ["--decode", "--corpus", "/corpus", "--dictionary", "/lexicon.dict"],
    ):
        with pytest.raises(SystemExit):
            parse_args(bad)


def test_the_decode_flag_decides_which_code_an_attempt_runs() -> None:
    training = attempt_code(parse_args([]), repository=REPOSITORY)
    decode = attempt_code(parse_args(["--decode"]), repository=REPOSITORY)
    assert "test_model" not in training
    assert "test_model" in decode
    assert str(REPOSITORY / "tests" / "fixtures" / "mini_arctic") in decode


def test_the_corpus_options_decide_which_code_an_attempt_runs() -> None:
    fixture = str(REPOSITORY / "tests" / "fixtures" / "mini_arctic")
    assert fixture in attempt_code(parse_args([]), repository=REPOSITORY)
    corpus = attempt_code(
        parse_args(["--corpus", "/corpus", "--dictionary", "/lexicon.dict"]),
        repository=REPOSITORY,
    )
    assert "/corpus" in corpus
    assert fixture not in corpus


def test_a_file_stdout_hands_over_a_descriptor_that_cannot_block(tmp_path: Path) -> None:
    with (tmp_path / "launch.log").open("w+", encoding="utf-8") as log:
        stream, held = server_stdout("file", log)
        assert stream is log
        assert held is None
    # Without a launch log there is still nothing that can fill up.
    stream, held = server_stdout("file", None)
    assert stream is subprocess.DEVNULL
    assert held is None


def test_a_pipe_stdout_is_never_read_and_teardown_closes_it() -> None:
    write_end, held = server_stdout("pipe", None)
    assert held is not None
    try:
        assert stat.S_ISFIFO(os.fstat(write_end).st_mode)
        # Nothing drains the read end, so a writer fills the pipe and then has
        # nowhere to put the next byte. On a blocking descriptor, which is what
        # a server and its kernels inherit, that is the stall being reproduced.
        os.set_blocking(write_end, False)
        written = 0
        with pytest.raises(BlockingIOError):
            for _ in range(1024):
                written += os.write(write_end, b"x" * 4096)
        assert written > 0
        # Releasing while a writer is still there cannot finish, and says so
        # rather than blocking teardown or reporting a pipe that is gone.
        held.release()
        assert held.resolve(timeout=0.5) is False
        assert held.read_end >= 0
    finally:
        os.close(write_end)
    # With the last writer gone the drain reaches end of file and closes.
    assert held.resolve(timeout=30.0) is True
    assert held.read_end == -1
    # And a second teardown is a no-op rather than a close of whatever
    # descriptor number has since been handed out.
    held.release()
    held.close()
    assert held.read_end == -1


def test_teardown_reports_a_held_stdout_that_still_has_a_writer() -> None:
    read_end, write_end = os.pipe()
    server = SimpleNamespace(held_stdout=HeldPipe(read_end))
    release_held_stdout(server)
    # A writer that outlived the group kill still holds the write end, so the
    # drain never reaches end of file. That is a survivor by another name and
    # has to be said out loud, not swallowed with the thread.
    unresolved = harness.unresolved_stdout(server, timeout=0.5)
    assert "still holds the write end" in unresolved
    survivors = harness.report_teardown(
        TeardownReport(True, unresolved=unresolved),
        harness.DescendantCensus({}, True),
        300,
        parse_process_snapshot(SERVER_PS_OUTPUT),
    )
    assert survivors.trusted is False
    assert survivors.reason == unresolved

    os.close(write_end)
    assert harness.unresolved_stdout(server, timeout=30.0) == ""
    # A server whose stdout was a file holds no pipe, and teardown says so by
    # doing nothing rather than by failing.
    release_held_stdout(SimpleNamespace(held_stdout=None))
    assert harness.unresolved_stdout(SimpleNamespace(held_stdout=None), timeout=0.0) == ""


def test_the_no_capture_kernelspec_turns_ipykernels_fd_capture_off(tmp_path: Path) -> None:
    name = write_no_capture_kernelspec(tmp_path)
    assert name == NO_FD_CAPTURE_KERNEL_NAME
    spec = json.loads((tmp_path / "kernels" / name / "kernel.json").read_text(encoding="utf-8"))
    # The same interpreter as the default spec, so the capture setting is the
    # only thing that differs between a run with it and a run without.
    assert spec["argv"][0] == sys.executable
    assert "{connection_file}" in spec["argv"]
    assert "--IPKernelApp.capture_fd_output=False" in spec["argv"]


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
        held_stdout=None,
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
