"""Contracts for deterministic BW partitioning and accumulator artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip(
    "resource", reason="POSIX-only training resource accounting requires the resource module"
)

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
    """Gate shard-count provenance against two synthetic pstrain records.

    REFERENCE: the one-shard synthetic ``provenance.json`` parsed and compared
    by pstrain's model-comparison apparatus; no live model, aligner, decoder, or
    scorer is used. AXIS: declared effective shard count (one versus two), with
    requested jobs held at two. SILENT ON: training correctness, provenance
    fields not varied here, architecture, arithmetic, and defects shared by the
    record writer and comparison path.
    """
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

    assert not result.all_compared_components_match
    assert not result.components["provenance.json"].match
    assert result.components["provenance.json"].structured_diff == [
        {
            "path": "/execution/bw_shard_count",
            "operation": "changed",
            "first": 1,
            "second": 2,
        }
    ]
    assert "provenance.json: DIFFER (text)" in result.summary()


@contract_scope(
    order=5,
    kind="multipron-fallback",
    requested_shards=(4,),
    effective_shards=(1,),
    reason=("fallback_senone",),
)
def test_multipron_multiple_shards_falls_back_loudly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Gate multipron shard selection against pstrain's serial fallback.

    REFERENCE: the declared pstrain ``fallback_senone`` policy, with no live
    model, aligner, decoder, or scorer. AXIS: multipron disabled versus enabled
    at four requested shards. SILENT ON: multipron model correctness, execution
    after selection, architecture, arithmetic, and defects shared with serial.
    """
    from pstrain.lib.bw import BWConfig
    from pstrain.lib.steps.train import run_bw_training

    assert _effective_bw_shard_count(4, multipron=True) == 1
    assert _effective_bw_shard_count(4, multipron=False) == 4
    with pytest.raises(FileNotFoundError):
        run_bw_training(
            model_dir=tmp_path / "model",
            output_dir=tmp_path / "output",
            features_dir=tmp_path / "features",
            train_fileids=tmp_path / "fileids",
            transcription=tmp_path / "transcription",
            dictionary=tmp_path / "dictionary",
            first_pass_2passvar=False,
            config=BWConfig(pass2var=False, unobserved_gaussian_policy="zero"),
            n_shards=4,
        )
    assert capsys.readouterr().out.splitlines()[0] == (
        "bw-parallelism\tserial (multipron_training is on)"
    )


def test_partition_manifest_varies_boundaries_and_keeps_empty_shards() -> None:
    """Gate pstrain's partitioner against explicit synthetic manifests.

    REFERENCE: literal expected contiguous manifests; no model, aligner,
    decoder, scorer, or numeric apparatus participates. AXIS: shard count
    (one, three, and five) for a fixed ordered identity list. SILENT ON: BW
    execution and reduction, other list sizes/orderings, architecture, and
    arithmetic.
    """
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


def test_partition_positions_reproduce_measured_1042_item_shapes() -> None:
    fileids = [f"utt-{index:04d}" for index in range(1042)]

    pstrain = _partition_manifest(fileids, 8, "remainder-first")
    upstream = _partition_manifest(fileids, 8, "remainder-last")

    assert [len(part) for part in pstrain] == [131, 131, 130, 130, 130, 130, 130, 130]
    assert [len(part) for part in upstream] == [130, 130, 130, 130, 130, 130, 130, 132]
    assert [item for part in pstrain for item in part] == fileids
    assert [item for part in upstream for item in part] == fileids


def test_upstream_partition_manifest_matches_stock_reference_bytes() -> None:
    fileids = [f"utt-{index:04d}" for index in range(1042)]
    part_len = len(fileids) // 8
    stock = [fileids[index * part_len : (index + 1) * part_len] for index in range(7)]
    stock.append(fileids[7 * part_len :])

    pstrain = _partition_manifest(fileids, 8, "remainder-last")
    serialize = lambda parts: b"".join(  # noqa: E731
        f"{shard}\t{fileid}\n".encode() for shard, part in enumerate(parts) for fileid in part
    )

    assert serialize(pstrain) == serialize(stock)


def test_production_reducer_receives_shard_dirs_in_index_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from pstrain.lib.bw import BWConfig
    from pstrain.lib.model import MODEL_FILES_REQUIRED
    from pstrain.lib.steps.train import run_bw_training

    class ReductionObserved(Exception):
        pass

    restored: list[Path] = []

    class Trainer:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def set_dict(self, *_args: object) -> None:
            pass

        def restore_accumulators(self, directories: list[Path]) -> None:
            restored.extend(directories)
            raise ReductionObserved

    class CompletedFuture:
        def __init__(self, shard: int, accum_dir: Path) -> None:
            self.shard = shard
            self.accum_dir = accum_dir

        def result(self) -> _ShardResult:
            return _ShardResult(
                shard=self.shard,
                assigned_ids=(f"utt-{self.shard}",),
                processed_ids=(f"utt-{self.shard}",),
                retried_ids=(),
                skipped=(),
                total_log_lik=-1.0,
                total_frames=1,
                accum_dir=self.accum_dir,
            )

    class ImmediatePool:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> ImmediatePool:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def submit(self, _fn: object, *args: object) -> CompletedFuture:
            return CompletedFuture(int(args[0]), Path(args[11]))

    model = tmp_path / "model"
    model.mkdir()
    for filename in MODEL_FILES_REQUIRED:
        (model / filename).write_bytes(b"model")
    fileids = tmp_path / "train.fileids"
    fileids.write_text("utt-0\nutt-1\nutt-2\n")
    transcription = tmp_path / "train.transcription"
    transcription.write_text("utt-0 ZERO\nutt-1 ONE\nutt-2 TWO\n")
    dictionary = tmp_path / "dictionary.dict"
    dictionary.write_text("ZERO Z\nONE W\nTWO T\n")

    monkeypatch.setattr("pstrain.lib.steps.train.BWTrainer", Trainer)
    monkeypatch.setattr("pstrain.lib.steps.train.ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr("pstrain.lib.steps.train._write_shard_metadata", lambda *a, **k: None)
    monkeypatch.setattr("pstrain.lib.steps.train._validate_shard_artifacts", lambda *a, **k: [])
    monkeypatch.setattr("pstrain.lib.steps.train._fingerprint_model", lambda _model: "model")
    monkeypatch.setattr("pstrain.lib.steps.train._fingerprint_config", lambda _config: "config")
    monkeypatch.setattr("pstrain.lib.steps.train._fingerprint_manifest", lambda _ids: "manifest")
    arrays = SimpleNamespace(shape=(1,))
    monkeypatch.setattr(
        "pstrain.lib.steps.train.HMM.load",
        lambda _model: SimpleNamespace(means=arrays, variances=arrays, mixw=arrays, tmat=arrays),
    )

    with pytest.raises(ReductionObserved):
        run_bw_training(
            model_dir=model,
            output_dir=tmp_path / "output",
            features_dir=tmp_path / "features",
            train_fileids=fileids,
            transcription=transcription,
            dictionary=dictionary,
            first_pass_2passvar=False,
            config=BWConfig(
                pass2var=False,
                unobserved_gaussian_policy="zero",
                multipron=False,
            ),
            multipron=False,
            n_iter=1,
            n_shards=3,
        )

    assert [directory.name for directory in restored] == [
        "shard-00000",
        "shard-00001",
        "shard-00002",
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
    """Gate shard validation against pstrain-authored synthetic artifacts.

    REFERENCE: intact metadata and payload digests emitted by pstrain's artifact
    writer; no live model, aligner, decoder, or scorer. AXIS: intact versus
    missing metadata or a stale accumulator payload. SILENT ON: unmutated
    fields, BW numerical correctness, architecture, arithmetic, and defects
    shared by writer and validator.
    """
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
    """Gate shard metadata against pstrain-authored compatible metadata.

    REFERENCE: intact synthetic metadata emitted by pstrain's artifact writer;
    no live model, aligner, decoder, or scorer. AXIS: one pass, fingerprint, or
    shape field changed at a time. SILENT ON: unvaried fields, BW numerical
    correctness, architecture, arithmetic, and writer/validator shared defects.
    """
    directories, common = _artifacts(tmp_path)
    path = directories[0] / "artifact.json"
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match=match):
        _validate(directories, common)


def test_artifacts_reject_duplicate_overlap_and_missing_coverage(tmp_path: Path) -> None:
    """Gate shard coverage against pstrain-authored complete coverage.

    REFERENCE: two intact synthetic shards covering the declared manifest once;
    no live model, aligner, decoder, or scorer. AXIS: duplicate shard identity,
    overlapping utterance assignment, or missing shard coverage. SILENT ON:
    payload numerics, other coverage defects, architecture, arithmetic, and
    writer/validator shared defects.
    """
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
    """Gate manifest identity against a unique synthetic pstrain manifest.

    REFERENCE: the unique ``a,b`` manifest expected by pstrain's shard validator;
    no live model, aligner, decoder, or scorer. AXIS: unique versus duplicated
    utterance identity. SILENT ON: other manifest defects, BW numerics,
    architecture, arithmetic, and defects shared by producers and validator.
    """
    directories, common = _artifacts(tmp_path)
    with pytest.raises(RuntimeError, match="duplicate utterance IDs"):
        _validate_shard_artifacts(directories, fileids=["a", "a"], **common)
