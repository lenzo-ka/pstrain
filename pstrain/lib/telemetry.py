"""Readers for Baum-Welch training telemetry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pstrain.lib.pipeline.targets import TARGETS

_TELEMETRY_FILENAME = "bw_telemetry.json"
_SCHEMA_VERSIONS = {1, 2}
_REQUIRED_PASS_FIELDS = {
    "pass",
    "total_log_likelihood",
    "total_frames",
    "per_frame_log_likelihood",
    "signed_convergence_delta",
    "stop_decision",
}


def load_bw_telemetry(model_dir: Path) -> dict[str, Any]:
    """Load a Baum-Welch telemetry document from ``model_dir``.

    Supported schema versions 1 and 2 contain the same core ``passes`` fields
    used here. Each pass reports its number, total and per-frame likelihoods,
    frame count, signed convergence delta, and stop decision; performance,
    accounting, and shard details are optional. The returned dictionary is
    newly decoded on every call.

    Raises:
        FileNotFoundError: If ``model_dir/bw_telemetry.json`` does not exist.
        ValueError: If the file cannot be decoded, uses an unsupported schema,
            or lacks the required pass fields.
    """
    path = model_dir / _TELEMETRY_FILENAME
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Baum-Welch telemetry file not found: {path}") from None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read Baum-Welch telemetry file {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"Malformed Baum-Welch telemetry file {path}: expected an object")
    if document.get("schema_version") not in _SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported Baum-Welch telemetry schema in {path}: "
            f"expected one of {sorted(_SCHEMA_VERSIONS)}, "
            f"got {document.get('schema_version')!r}"
        )
    if not isinstance(document.get("passes"), list):
        raise ValueError(f"Malformed Baum-Welch telemetry file {path}: expected a passes list")
    for index, row in enumerate(document["passes"], 1):
        if not isinstance(row, dict):
            raise ValueError(
                f"Malformed Baum-Welch telemetry file {path}: pass {index} must be an object"
            )
        missing = _REQUIRED_PASS_FIELDS - row.keys()
        if missing:
            raise ValueError(
                f"Malformed Baum-Welch telemetry file {path}: pass {index} is missing "
                f"{', '.join(sorted(missing))}"
            )
    return document


def learning_curves(
    project_dir: Path, config: str = "default"
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Return Baum-Welch pass records for the project's trained stages.

    ``config`` is the model config-directory name below each stage in
    ``shared/models``, not a request to resolve a profile. Stages without a
    telemetry file are skipped, so a project with no trained models returns an
    empty list. A telemetry file that exists but is unreadable or invalid raises
    the same contextual error as :func:`load_bw_telemetry`.

    Results are in canonical acoustic-ladder order from the pipeline's target
    declarations. This is not a record of historical execution order: stale or
    independently trained stages retain their canonical position. Recovering
    actual execution order would require run or stage order to be recorded in
    project provenance. Filesystem modification times are deliberately ignored.
    """
    curves: list[tuple[str, list[dict[str, Any]]]] = []
    models_dir = project_dir / "shared" / "models"
    for target in TARGETS:
        if target.kind not in {"ci", "cd"}:
            continue
        model_dir = models_dir / target.name / config
        if not (model_dir / _TELEMETRY_FILENAME).is_file():
            continue
        document = load_bw_telemetry(model_dir)
        curves.append((target.name, document["passes"]))
    return curves
