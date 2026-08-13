"""Strict provenance and identity-accounting gates for parallel BW artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pstrain.lib.steps.train import (
    _SHARD_METADATA,
    _SHARD_SCHEMA,
    _ShardResult,
    _validate_shard_artifacts,
)


def _artifact(
    root: Path,
    index: int,
    span: tuple[int, int],
    assigned: list[str],
    *,
    processed: list[str] | None = None,
) -> _ShardResult:
    directory = root / f"shard-{index}-{len(list(root.iterdir()))}"
    directory.mkdir()
    processed = assigned if processed is None else processed
    skipped = [
        {"utterance": utterance, "reason": "alignment_failure"}
        for utterance in assigned
        if utterance not in processed
    ]
    metadata: dict[str, object] = {
        "schema_version": _SHARD_SCHEMA,
        "pass_number": 2,
        "model_fingerprint": "model",
        "config_fingerprint": "config",
        "manifest_fingerprint": "manifest",
        "parameter_shapes": {"means": [2, 1, 1, 3]},
        "shard_index": index,
        "serial_span": list(span),
        "assigned_ids": assigned,
        "processed_ids": processed,
        "retried_ids": [],
        "skipped": skipped,
        "attempted_count": len(assigned),
        "accumulated_count": len(processed),
        "skipped_count": len(skipped),
        "log_lik_contributions": [0.0] * len(processed),
    }
    (directory / _SHARD_METADATA).write_text(json.dumps(metadata), encoding="utf-8")
    return _ShardResult(index, directory, metadata)


def _validate(results: list[_ShardResult]) -> list[_ShardResult]:
    return _validate_shard_artifacts(
        results,
        expected_ids=["a", "b"],
        pass_number=2,
        model_fingerprint="model",
        config_fingerprint="config",
        manifest_fingerprint="manifest",
        parameter_shapes={"means": [2, 1, 1, 3]},
    )


def test_artifacts_accept_exact_identity_partition_and_empty_shard(tmp_path: Path) -> None:
    results = [
        _artifact(tmp_path, 0, (0, 1), ["a"]),
        _artifact(tmp_path, 1, (1, 2), ["b"]),
        _artifact(tmp_path, 2, (2, 2), []),
    ]
    assert _validate(list(reversed(results))) == results


def test_artifacts_reject_missing_shard_coverage(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="incomplete BW shard coverage"):
        _validate([_artifact(tmp_path, 0, (0, 1), ["a"])])


def test_artifacts_reject_duplicate_shard(tmp_path: Path) -> None:
    first = _artifact(tmp_path, 0, (0, 1), ["a"])
    duplicate = _artifact(tmp_path, 0, (1, 2), ["b"])
    with pytest.raises(RuntimeError, match="duplicate or invalid BW shard index"):
        _validate([first, duplicate])


def test_artifacts_reject_overlapping_identity_coverage(tmp_path: Path) -> None:
    first = _artifact(tmp_path, 0, (0, 1), ["a"])
    second = _artifact(tmp_path, 1, (1, 2), ["a"])
    with pytest.raises(RuntimeError, match="overlapping BW shard coverage"):
        _validate([first, second])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pass_number", 1, "pass_number"),
        ("model_fingerprint", "stale", "model_fingerprint"),
        ("config_fingerprint", "other", "config_fingerprint"),
        ("manifest_fingerprint", "other", "manifest_fingerprint"),
        ("parameter_shapes", {"means": [99]}, "parameter_shapes"),
    ],
)
def test_artifacts_reject_wrong_pass_stale_or_incompatible_fingerprints(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    first = _artifact(tmp_path, 0, (0, 1), ["a"])
    second = _artifact(tmp_path, 1, (1, 2), ["b"])
    second.metadata[field] = value
    (second.accum_dir / _SHARD_METADATA).write_text(json.dumps(second.metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        _validate([first, second])


def test_artifacts_reject_metadata_tampering_after_completion(tmp_path: Path) -> None:
    result = _artifact(tmp_path, 0, (0, 2), ["a", "b"])
    on_disk = dict(result.metadata)
    on_disk["processed_ids"] = ["a"]
    (result.accum_dir / _SHARD_METADATA).write_text(json.dumps(on_disk), encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after worker completion"):
        _validate([result])
