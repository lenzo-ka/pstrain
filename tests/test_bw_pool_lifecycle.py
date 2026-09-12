"""Real BW pool shutdown, observed before fingerprint-verified test cleanup."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Subprocesses execute this file directly rather than inheriting an editable
# install that may point at a different checkout.
if __name__ == "__main__":
    sys.meta_path[:] = [
        f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_pstrain"
    ]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.clib import requires_c_library
from tests.test_procctl import Launcher

pytest_plugins = ["tests.test_procctl"]

SETUP_TIMEOUT = 45.0
SHUTDOWN_TIMEOUT = 5.0
POLL_INTERVAL = 0.05
WORKER_COUNT = 2
FEATURE_REPEATS = 32
BUSY_UTTERANCES = 512


def _write_status(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def _members(group: int) -> list[dict[str, object]]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,pgid=,ppid=,stat="],
        check=True,
        capture_output=True,
        text=True,
        timeout=SHUTDOWN_TIMEOUT,
    )
    rows = []
    for line in result.stdout.splitlines():
        pid, pgid, ppid, state = line.split()
        if int(pgid) == group:
            rows.append({"pid": int(pid), "ppid": int(ppid), "state": state})
    return rows


def _live_workers(group: int, workers: list[int]) -> list[int]:
    return [
        int(row["pid"])
        for row in _members(group)
        if row["pid"] in workers and not str(row["state"]).startswith("Z")
    ]


def _coordinate(root: Path, mode: str) -> None:
    import numpy as np

    from pstrain.lib.bw import BWConfig
    from pstrain.lib.features import _write_sphinx_mfc, read_sphinx_mfc
    from pstrain.lib.steps.train import run_bw_training
    from tests.numeric_harness import create_project

    ctx = create_project(root / "project")
    features = root / "features"
    features.mkdir()
    values = np.tile(read_sphinx_mfc(ctx.features_dir / "arctic_a0001.mfc"), (FEATURE_REPEATS, 1))
    _write_sphinx_mfc(values, features / "base.mfc")
    # Busy cases must still be doing real native work when the signal lands.
    count = WORKER_COUNT if mode == "normal" else BUSY_UTTERANCES
    identities = [f"work-{i}" for i in range(count)]
    for key in identities:
        os.link(features / "base.mfc", features / f"{key}.mfc")
    (root / "fileids").write_text("\n".join(identities) + "\n")
    (root / "transcription").write_text("".join(f"{key} a and\n" for key in identities))
    _write_status(root / "ready.json", _members(os.getpgrp()))
    try:
        run_bw_training(
            model_dir=ctx.model_dir("flat"),
            output_dir=root / "output",
            features_dir=features,
            train_fileids=root / "fileids",
            transcription=root / "transcription",
            dictionary=ctx.shared_dir / "dictionary.dict",
            filler_dict=ctx.filler_dict,
            config=BWConfig(
                pass2var=True,
                unobserved_gaussian_policy="zero",
                a_beam=1e-200,
                b_beam=1e-200,
                optional_final_silence=False,
            ),
            first_pass_2passvar=True,
            multipron=True,
            n_shards=WORKER_COUNT,
            n_iter=1,
            min_iterations=1,
            project_dir=root,
            stage="lifecycle",
        )
    except BaseException as error:
        _write_status(root / "exception.json", {"type": type(error).__name__})
        raise


def _observe(root: Path, mode: str) -> dict[str, object]:
    with (root / "coordinator.log").open("w") as log:
        # The separate procctl-owned supervisor remains alive after coordinator
        # death, so teardown can verify the group owner's fingerprint.
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "coordinate", str(root), mode],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    group = os.getpgrp()
    if mode == "normal":
        returncode = process.wait(timeout=SETUP_TIMEOUT)
        rows = json.loads((root / "output/bw_telemetry.json").read_text())["passes"]
        workers = rows[0]["performance"]["worker_pids"]
    else:
        deadline = time.monotonic() + SETUP_TIMEOUT
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"coordinator exited before signal: {process.returncode}")
            if (root / "ready.json").exists():
                known = {row["pid"] for row in json.loads((root / "ready.json").read_text())}
                workers = [
                    int(row["pid"])
                    for row in _members(group)
                    if row["ppid"] == process.pid and row["pid"] not in known
                ]
                logs = list((root / ".pstrain/bw/lifecycle").glob("*.log"))
                if (
                    len(workers) == WORKER_COUNT
                    and len(logs) == WORKER_COUNT
                    and all(b"utt>" in path.read_bytes() for path in logs)
                ):
                    break
            time.sleep(POLL_INTERVAL)
        else:
            raise RuntimeError("native workers did not become active within setup bound")
        if mode == "worker_failure":
            os.kill(workers[0], signal.SIGKILL)
        else:
            os.kill(process.pid, signal.SIGINT if mode == "interrupt" else signal.SIGKILL)
        deadline = time.monotonic() + SHUTDOWN_TIMEOUT
        while time.monotonic() < deadline:
            if process.poll() is not None and not _live_workers(group, workers):
                break
            time.sleep(POLL_INTERVAL)
        returncode = process.poll()
    exception = root / "exception.json"
    return {
        "returncode": returncode,
        "worker_pids": workers,
        "remaining_workers": _live_workers(group, workers),
        "exception": json.loads(exception.read_text())["type"] if exception.exists() else None,
    }


@requires_c_library
@pytest.mark.skipif(
    os.name != "posix", reason="coordinator signal witness uses POSIX process groups"
)
@pytest.mark.parametrize("mode", ["normal", "interrupt", "death", "worker_failure"])
def test_real_bw_pool_shutdown(launcher: Launcher, tmp_path: Path, mode: str) -> None:
    root = tmp_path / "witness"
    root.mkdir()
    launcher(sys.executable, str(Path(__file__).resolve()), "observe", str(root), mode)
    deadline = time.monotonic() + SETUP_TIMEOUT + SHUTDOWN_TIMEOUT
    while time.monotonic() < deadline and not (root / "result.json").exists():
        time.sleep(POLL_INTERVAL)
    assert (root / "result.json").exists(), (root / "coordinator.log").read_text()
    result = json.loads((root / "result.json").read_text())
    assert "observer_error" not in result, result
    assert len(set(result["worker_pids"])) == WORKER_COUNT
    # These assertions run before launcher teardown; procctl cleanup does not
    # count as the production pool honoring its shutdown contract.
    assert result["remaining_workers"] == [], result
    assert result["returncode"] is not None, result
    if mode == "normal":
        assert result["returncode"] == 0, result
    elif mode == "interrupt":
        assert result["exception"] == "KeyboardInterrupt", result
    elif mode == "worker_failure":
        assert result["exception"] == "BrokenProcessPool", result
    else:
        assert result["returncode"] == -signal.SIGKILL, result


if __name__ == "__main__":
    root, mode = Path(sys.argv[2]), sys.argv[3]
    if sys.argv[1] == "coordinate":
        _coordinate(root, mode)
    else:
        try:
            observed = _observe(root, mode)
        except BaseException as error:
            observed = {"observer_error": f"{type(error).__name__}: {error}"}
        _write_status(root / "result.json", observed)
        # Keep the fingerprinted test-cleanup anchor alive. This is outside the
        # measured coordinator/pool and does not implement production cleanup.
        while True:
            time.sleep(SETUP_TIMEOUT)
