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

    @property
    def cpu_total(self) -> float:
        return self.cpu_user + self.cpu_sys + self.cpu_children_user + self.cpu_children_sys


def task_stage(name: str, group: str) -> str:
    """Return a compact rollup label without adding pipeline semantics."""
    return group or name.split(":", 1)[0]


def measure(task: str, group: str, fn: Any) -> TaskTiming:
    """Run ``fn`` and measure wall and process-family CPU around it."""
    started_at = datetime.now(UTC)
    wall_start = time.monotonic()
    cpu_start = os.times()
    fn()
    cpu_end = os.times()
    wall = time.monotonic() - wall_start
    ended_at = datetime.now(UTC)
    return TaskTiming(
        task=task,
        stage=task_stage(task, group),
        group=group,
        wall=wall,
        cpu_user=cpu_end.user - cpu_start.user,
        cpu_sys=cpu_end.system - cpu_start.system,
        cpu_children_user=cpu_end.children_user - cpu_start.children_user,
        cpu_children_sys=cpu_end.children_system - cpu_start.children_system,
        start=started_at.isoformat(),
        end=ended_at.isoformat(),
    )


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
    records: list[TaskTiming], *, run_id: str, target: str, started: str, ended: str
) -> dict[str, Any]:
    """Build the stable JSON representation of a pipeline run."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "target": target,
        "start": started,
        "end": ended,
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
