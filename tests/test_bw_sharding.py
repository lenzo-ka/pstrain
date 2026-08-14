"""Contracts for deterministic BW partitioning and accumulator artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pstrain.lib.compare import compare_models
from pstrain.lib.contract_docs import contract_scope
from pstrain.lib.steps.train import (
    _ACCUMULATOR_FILES,
    _effective_bw_shard_count,
    _partition_manifest,
    _ShardResult,
    _validate_shard_artifacts,
    _write_shard_metadata,
)


@contract_scope(
    order=4,
    kind="provenance-comparison",
    file=("provenance.json",),
    shard_counts=(1, 2),
)
def test_model_comparison_surfaces_effective_bw_shard_count(tmp_path: Path) -> None:
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    (one / "provenance.json").write_text(
        json.dumps({"execution": {"requested_jobs": 2, "bw_shard_count": 1}})
    )
    (two / "provenance.json").write_text(
        json.dumps({"execution": {"requested_jobs": 2, "bw_shard_count": 2}})
    )

    result = compare_models(one, two)

    assert not result.all_match
    assert not result.components["provenance.json"].match
    assert "provenance.json: DIFFER (text)" in result.summary()


@contract_scope(
    order=5,
    kind="multipron-fallback",
    requested_shards=(4,),
    effective_shards=(1,),
    reason=("fallback_senone",),
)
def test_multipron_multiple_shards_falls_back_loudly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Certify fallback_senone safety; this is blind to multipron model correctness."""
    with caplog.at_level("WARNING"):
        assert _effective_bw_shard_count(4, multipron=True) == 1
    assert "fallback_senone" in caplog.text
    assert _effective_bw_shard_count(4, multipron=False) == 4


def test_partition_manifest_varies_boundaries_and_keeps_empty_shards() -> None:
    fileids = ["early", "skip", "middle", "late"]
    assert _partition_manifest(fileids, 1) == [fileids]
    assert _partition_manifest(fileids, 3) == [["early", "skip"], ["middle"], ["late"]]
    assert _partition_manifest(fileids, 5) == [
        ["early"],
        ["skip"],
        ["middle"],
        ["late"],
        [],
    ]


def _artifacts(root: Path) -> tuple[list[Path], dict[str, Any]]:
    common: dict[str, Any] = {
        "iteration": 2,
        "model_fingerprint": "model",
        "config_fingerprint": "config",
        "manifest_fingerprint": "manifest",
        "shapes": {"means": [2, 1, 3]},
    }
    directories: list[Path] = []
    for shard, assigned in enumerate((("a",), ("b",))):
        directory = root / str(shard)
        directory.mkdir(parents=True)
        for filename in _ACCUMULATOR_FILES:
            (directory / filename).write_bytes(f"{shard}-{filename}".encode())
        result = _ShardResult(
            shard=shard,
            assigned_ids=assigned,
            processed_ids=assigned,
            retried_ids=(),
            skipped=(),
            total_log_lik=-1.0,
            total_frames=1,
            accum_dir=directory,
        )
        _write_shard_metadata(result, **common)
        directories.append(directory)
    return directories, common


def _validate(directories: list[Path], common: dict[str, Any]) -> None:
    _validate_shard_artifacts(directories, fileids=["a", "b"], **common)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda dirs: (dirs[0] / "artifact.json").unlink(), "Missing BW shard metadata"),
        (
            lambda dirs: (dirs[0] / "gauden_counts").write_bytes(b"stale"),
            "payload digest mismatch",
        ),
    ],
)
@contract_scope(
    order=6,
    kind="artifact-validation",
    mutations=("missing metadata", "stale accumulator payload"),
)
def test_artifacts_reject_missing_or_stale_payload(
    tmp_path: Path, mutation: Callable[[list[Path]], object], match: str
) -> None:
    directories, common = _artifacts(tmp_path)
    mutation(directories)
    with pytest.raises(RuntimeError, match=match):
        _validate(directories, common)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("pass", 3, "pass"),
        ("model_fingerprint", "wrong", "model_fingerprint"),
        ("config_fingerprint", "wrong", "config_fingerprint"),
        ("manifest_fingerprint", "wrong", "manifest_fingerprint"),
        ("parameter_shapes", {"means": [99]}, "parameter_shapes"),
    ],
)
def test_artifacts_reject_wrong_pass_or_incompatible_metadata(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    directories, common = _artifacts(tmp_path)
    path = directories[0] / "artifact.json"
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match=match):
        _validate(directories, common)


def test_artifacts_reject_duplicate_overlap_and_missing_coverage(tmp_path: Path) -> None:
    directories, common = _artifacts(tmp_path)
    duplicate = directories[1] / "artifact.json"
    row = json.loads(duplicate.read_text())
    row["shard"] = 0
    duplicate.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="Duplicate BW shard"):
        _validate(directories, common)

    directories, common = _artifacts(tmp_path / "overlap")
    path = directories[1] / "artifact.json"
    row = json.loads(path.read_text())
    row["assigned_ids"] = ["a"]
    row["processed_ids"] = ["a"]
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="Overlapping BW shard coverage"):
        _validate(directories, common)

    directories, common = _artifacts(tmp_path / "missing")
    with pytest.raises(RuntimeError, match="Missing BW shard coverage"):
        _validate(directories[:1], common)


def test_duplicate_manifest_identity_is_rejected(tmp_path: Path) -> None:
    directories, common = _artifacts(tmp_path)
    with pytest.raises(RuntimeError, match="duplicate utterance IDs"):
        _validate_shard_artifacts(directories, fileids=["a", "a"], **common)
