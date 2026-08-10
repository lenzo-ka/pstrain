"""Per-task pipeline timing records, persistence, and presentation."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SUMMARY_THRESHOLD_SECONDS = 3.0


@dataclass(frozen=True)
class TaskTiming:
    """Timing captured in the process that executed one task."""

    task: str
    stage: str
    group: str
    wall: float
    cpu_user: float
    cpu_sys: float
    cpu_children_user: float
    cpu_children_sys: float
    start: str
    end: str
    outcome: str

    @property
    def cpu_total(self) -> float:
        return self.cpu_user + self.cpu_sys + self.cpu_children_user + self.cpu_children_sys


def task_stage(name: str, group: str) -> str:
    """Return a compact rollup label without adding pipeline semantics."""
    return group or name.split(":", 1)[0]


@dataclass(frozen=True)
class MeasuredTask:
    """A task's observational record and its own execution error, if any."""

    timing: TaskTiming | None
    error: Exception | None


def measure(task: str, group: str, fn: Any) -> MeasuredTask:
    """Run ``fn`` while keeping measurement strictly observational."""
    started_at = datetime.now(UTC)
    wall_start = time.monotonic()
    cpu_start = None
    cpu_end = None
    measurement_error: Exception | None = None
    try:
        cpu_start = os.times()
    except Exception as exc:
        measurement_error = exc
    error: Exception | None = None
    try:
        fn()
    except Exception as exc:
        error = exc
    finally:
        try:
            cpu_end = os.times()
        except Exception as exc:
            if measurement_error is None:
                measurement_error = exc
    if measurement_error is not None or cpu_start is None or cpu_end is None:
        logger.warning("Could not measure task %s: %s", task, measurement_error)
        return MeasuredTask(timing=None, error=error)
    wall = time.monotonic() - wall_start
    timing = TaskTiming(
        task=task,
        stage=task_stage(task, group),
        group=group,
        wall=wall,
        cpu_user=cpu_end.user - cpu_start.user,
        cpu_sys=cpu_end.system - cpu_start.system,
        cpu_children_user=cpu_end.children_user - cpu_start.children_user,
        cpu_children_sys=cpu_end.children_system - cpu_start.children_system,
        start=started_at.isoformat(),
        end=datetime.now(UTC).isoformat(),
        outcome="failed" if error is not None else "ok",
    )
    return MeasuredTask(timing=timing, error=error)


def rollup(records: list[TaskTiming]) -> list[dict[str, float | str]]:
    """Sum task measurements by stage."""
    stages: dict[str, dict[str, float | str]] = {}
    for record in records:
        item = stages.setdefault(record.stage, {"stage": record.stage, "wall": 0.0, "cpu": 0.0})
        item["wall"] = float(item["wall"]) + record.wall
        item["cpu"] = float(item["cpu"]) + record.cpu_total
    for item in stages.values():
        wall = float(item["wall"])
        item["cpu_wall_ratio"] = float(item["cpu"]) / wall if wall else 0.0
    return list(stages.values())


def build_document(
    records: list[TaskTiming],
    *,
    run_id: str,
    target: str,
    started: str,
    ended: str,
    status: str,
) -> dict[str, Any]:
    """Build the stable JSON representation of a pipeline run."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "target": target,
        "start": started,
        "end": ended,
        "status": status,
        "tasks_recorded": len(records),
        "tasks_failed": sum(record.outcome == "failed" for record in records),
        "tasks": [asdict(record) for record in records],
        "stages": rollup(records),
    }


def new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def timings_dir(project_dir: Path) -> Path:
    return project_dir / ".pstrain" / "timings"


def write_document(project_dir: Path, document: dict[str, Any]) -> Path | None:
    """Atomically persist timings; observation failures are warnings only."""
    destination = timings_dir(project_dir) / f"{document['run_id']}.json"
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        for stale in destination.parent.glob(".*.json.tmp-*"):
            stale.unlink(missing_ok=True)
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except Exception as exc:
        logger.warning("Could not write pipeline timings to %s: %s", destination, exc)
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        return None
    return destination


def load_document(project_dir: Path, run_id: str | None = None) -> tuple[Path, dict[str, Any]]:
    directory = timings_dir(project_dir)
    path = directory / f"{run_id}.json" if run_id else max(directory.glob("*.json"))
    return path, json.loads(path.read_text(encoding="utf-8"))


def format_summary(document: dict[str, Any]) -> str:
    """Pretty-print a compact per-stage rollup table."""
    lines = ["Pipeline timings", "Stage                         Wall       CPU  CPU/wall"]
    for item in document["stages"]:
        lines.append(
            f"{item['stage']:<27} {item['wall']:>7.2f}s {item['cpu']:>8.2f}s"
            f" {item['cpu_wall_ratio']:>8.2f}x"
        )
    return "\n".join(lines)
