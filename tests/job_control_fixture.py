"""Spawn-safe subprocess fixture for pipeline job-control tests."""

from __future__ import annotations

import functools
import os
import sys
import time
from pathlib import Path

from pstrain.lib import native_worker
from pstrain.lib.pipeline import Pipeline, Task


def work(output: Path, pids: Path, delay: float) -> None:
    worker = native_worker._owned_worker()
    worker._start()
    assert worker.pid is not None
    with pids.open("a", encoding="utf-8") as stream:
        stream.write(f"{os.getpid()} {worker.pid}\n")
        stream.flush()
    time.sleep(0.1 if output.name == "out-0" else delay)
    output.write_text("done", encoding="utf-8")


def main() -> int:
    directory = Path(sys.argv[1])
    delay = float(sys.argv[2])
    pids = directory / "pids"
    outputs = [directory / f"out-{index}" for index in range(4)]
    sentinel = directory / "sentinel"
    pipeline = Pipeline(directory, worker_nice=0)
    for index, output in enumerate(outputs):
        pipeline.add(
            Task(
                f"work:{index}",
                functools.partial(work, output, pids, delay),
                outputs=(output,),
                parallel_group="work",
            )
        )
    pipeline.add(
        Task(
            "finish",
            functools.partial(Path.write_text, sentinel, "done", encoding="utf-8"),
            inputs=tuple(outputs),
            outputs=(sentinel,),
        )
    )
    pipeline.register_target("all", sentinel)
    return pipeline.run("all", jobs=2)


if __name__ == "__main__":
    raise SystemExit(main())
