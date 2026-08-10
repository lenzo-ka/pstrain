from __future__ import annotations

import json
import sys
from pathlib import Path

from pstrain.cli.cli import main


def test_timings_command_prints_named_run(tmp_path: Path, monkeypatch, capsys) -> None:
    directory = tmp_path / ".pstrain" / "timings"
    directory.mkdir(parents=True)
    document = {
        "schema_version": 1,
        "run_id": "run-one",
        "target": "features",
        "start": "start",
        "end": "end",
        "status": "failed",
        "tasks_recorded": 0,
        "tasks_failed": 0,
        "tasks": [],
        "stages": [{"stage": "features", "wall": 1.0, "cpu": 0.5, "cpu_wall_ratio": 0.5}],
    }
    (directory / "run-one.json").write_text(json.dumps(document))
    monkeypatch.setattr(
        sys, "argv", ["pstrain", "timings", "--project-dir", str(tmp_path), "run-one"]
    )
    assert main() == 0
    output = capsys.readouterr().out
    assert "run-one" in output
    assert "features" in output
    assert "STATUS: FAILED" in output
