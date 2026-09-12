"""Inspect and explicitly restore retained BW updates without selecting a model."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

from pstrain.lib.model import (
    MODEL_FILES_REQUIRED,
    MODEL_PARAMETER_FILES,
    file_sha256,
    fingerprint_model,
    staged_model_update,
)
from pstrain.lib.telemetry import load_bw_telemetry

COUNTS_FILE = "gauden_counts"


def model_snapshot(model_dir: Path) -> dict[str, str | None]:
    """Bind parameters and optional counts independently of scoring identity."""
    return {
        name: file_sha256(model_dir / name) if (model_dir / name).is_file() else None
        for name in (*MODEL_FILES_REQUIRED, COUNTS_FILE)
    }


def evaluation_health(
    *,
    total_log_lik: float,
    average: float,
    frames: int,
    utterances: int,
    processed: int,
    skipped: int,
    inputs: int,
    max_skip_fraction: float,
) -> bool:
    """Report configured alignment/statistics health, not model quality."""
    return (
        frames > 0
        and utterances > 0
        and processed > 0
        and inputs > 0
        and math.isfinite(total_log_lik)
        and math.isfinite(average)
        and skipped / inputs <= max_skip_fraction
    )


def list_checkpoints(model_dir: Path) -> list[dict[str, Any]]:
    """Update N is evaluated by pass N+1; old unbound reports remain unverified."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    try:
        passes = load_bw_telemetry(model_dir)["passes"]
    except FileNotFoundError:
        passes = []
    result: list[dict[str, Any]] = []
    directory = model_dir / "iterations"
    if not directory.exists():
        return result
    for checkpoint in sorted(directory.iterdir()):
        if not checkpoint.is_dir() or not checkpoint.name.isdecimal():
            continue
        number = int(checkpoint.name)
        rows = [row for row in passes if row["pass"] == number + 1]
        row = rows[0] if len(rows) == 1 else None
        status = "unevaluated" if not rows else "reported-unverified"
        evidence = row.get("input_model_evaluation") if row is not None else None
        snapshot = model_snapshot(checkpoint)
        if (
            isinstance(evidence, dict)
            and isinstance(evidence.get("healthy"), bool)
            and isinstance(evidence.get("max_skip_fraction"), (int, float))
            and all(snapshot[name] for name in MODEL_FILES_REQUIRED)
        ):
            if (
                evidence.get("model_fingerprint") == fingerprint_model(checkpoint)
                and evidence.get("snapshot") == snapshot
            ):
                status = (
                    "evaluated-healthy"
                    if evidence.get("healthy") is True
                    else "evaluated-unhealthy"
                )
            else:
                status = "evidence-mismatch"
        result.append(
            {
                "checkpoint": number,
                "path": str(checkpoint),
                "evaluation_pass": number + 1,
                "status": status,
                "snapshot": snapshot,
                "evaluation": evidence,
            }
        )
    return result


def restore_checkpoint(model_dir: Path, number: int, *, dry_run: bool = False) -> dict[str, Any]:
    """Restore an explicitly selected snapshot, retaining a reversible backup.

    Callers must stop concurrent training/readers first. This never chooses a
    checkpoint or certifies recognition quality. Completion/provenance remains
    invalidated even if publication fails; diagnostics and checkpoints survive.
    """
    model_dir = Path(model_dir).resolve()
    matches = [row for row in list_checkpoints(model_dir) if row["checkpoint"] == number]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one checkpoint numbered {number}")
    selected = matches[0]
    checkpoint = Path(selected["path"])
    if checkpoint.is_symlink() or checkpoint.resolve().parent != model_dir / "iterations":
        raise ValueError("Checkpoint must be a retained local directory")
    for name in MODEL_FILES_REQUIRED:
        if not (checkpoint / name).is_file() or (checkpoint / name).is_symlink():
            raise ValueError(f"Checkpoint has no regular model file: {name}")
    if (checkpoint / "mdef").read_bytes() != (model_dir / "mdef").read_bytes():
        raise ValueError("Checkpoint state mapping differs from the destination mdef")
    names = (*MODEL_PARAMETER_FILES,)
    if (checkpoint / COUNTS_FILE).is_file():
        if (checkpoint / COUNTS_FILE).is_symlink():
            raise ValueError("Checkpoint counts must be a regular local file")
        names += (COUNTS_FILE,)
    elif (checkpoint / COUNTS_FILE).exists():
        raise ValueError("Checkpoint counts are not a regular file")
    remove = (COUNTS_FILE,) if COUNTS_FILE not in names else ()
    invalidated = sorted(model_dir.glob(".mdef.*.complete"))
    invalidated += [model_dir / "provenance.json", model_dir / "sendump"]
    result = {
        "selected": selected,
        "dry_run": dry_run,
        "invalidated": [str(p) for p in invalidated],
    }
    if dry_run:
        return result
    history = model_dir / "recovery-history"
    history.mkdir(exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix="restore-", dir=history))
    # Complete a durable backup before invalidating or replacing anything.
    for path in [
        *(model_dir / name for name in (*MODEL_FILES_REQUIRED, COUNTS_FILE)),
        model_dir / "feat.params",
        model_dir / "bw_telemetry.json",
        *invalidated,
    ]:
        if path.is_file() or path.is_symlink():
            shutil.copy2(path, backup / path.name, follow_symlinks=False)
    result["backup"] = str(backup)
    (backup / "restore.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    # These are never restored as successful provenance after a failed mutation.
    for path in invalidated:
        path.unlink(missing_ok=True)
    with staged_model_update(model_dir, names, remove=remove) as staging:
        for name in names:
            shutil.copyfile(checkpoint / name, staging / name)
        if model_snapshot(checkpoint) != selected["snapshot"]:
            raise RuntimeError("Checkpoint changed while preparing restoration")
        for name in names:
            if file_sha256(staging / name) != selected["snapshot"][name]:
                raise RuntimeError("Staged restoration differs from selected checkpoint")
    return result
