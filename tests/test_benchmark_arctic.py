from __future__ import annotations

import json
import os
import tarfile
from dataclasses import asdict, fields, replace
from pathlib import Path

import pytest

from pstrain.benchmarks.arctic import (
    ARCHIVES,
    DATA_DIR,
    DECODER_CONDITIONS,
    PIN_CONFIGS,
    RECORD_SCHEMA_VERSION,
    audit_monotonicity,
    benchmark_conditions,
    bootstrap_ci,
    compare_results,
    extract_archive,
    load_transcripts,
    main,
    make_record,
    paired_delta_ci,
    validate_record,
)


def test_archive_manifest_matches_runtime_config() -> None:
    manifest = json.loads((DATA_DIR.parent / "archives.json").read_text())
    assert manifest["archives"] == [asdict(archive) for archive in ARCHIVES]
    assert {archive.voice for archive in ARCHIVES} == {"slt", "bdl", "rms", "clb"}
    assert all(
        archive.url.startswith("http://festvox.org/cmu_arctic/packed/") for archive in ARCHIVES
    )
    assert all(len(archive.sha256) == 64 for archive in ARCHIVES)


def test_committed_transcripts_are_normalized_and_complete() -> None:
    expected = {"train.transcription": 1132, "slt55.transcription": 55, "big.transcription": 3395}
    for name, count in expected.items():
        transcripts = load_transcripts(DATA_DIR / name)
        assert len(transcripts) == count
        assert all(text == text.lower() and text.strip() == text for text in transcripts.values())
        assert all("<s>" not in text and "</s>" not in text for text in transcripts.values())
    big = load_transcripts(DATA_DIR / "big.transcription")
    assert {utterance.split("/", 1)[0] for utterance in big} == {"bdl", "rms", "clb"}

    train = set(load_transcripts(DATA_DIR / "train.transcription"))
    assert set(load_transcripts(DATA_DIR / "slt55.transcription")) <= train
    assert not (set(big) & train)


def test_transcript_serializations_are_enforced() -> None:
    train_lines = (DATA_DIR / "train.transcription").read_text().splitlines()
    assert all(line.startswith("arctic_") and "<s>" not in line for line in train_lines)
    for name in ("slt55.transcription", "big.transcription"):
        lines = (DATA_DIR / name).read_text().splitlines()
        assert all(
            line.startswith("<s> ") and " </s> (" in line and line.endswith(")") for line in lines
        )


def test_pin_configs_resolve_ratified_conditions() -> None:
    off = PIN_CONFIGS["off"]["training"]
    on = PIN_CONFIGS["on"]["training"]
    assert off["multipron_training"] is False
    assert off["untied_inventory"] == "linear"
    assert {off[stage]["convergence_ratio"] for stage in ("ci", "untied", "tied")} == {0.1}
    assert on["multipron_training"] is True
    assert on["untied_inventory"] == "transcript-reachable"
    assert {on[stage]["convergence_ratio"] for stage in ("ci", "untied", "tied")} == {0.001}
    assert PIN_CONFIGS["off"]["split"]["test_count"] == 0
    assert PIN_CONFIGS["on"]["split"]["test_count"] == 0
    from pstrain.lib.pipeline.context import FeatParams, TrainParams

    assert all(
        {field.name for field in fields(TrainParams)} == set(PIN_CONFIGS[mode]["training"])
        for mode in ("off", "on")
    )
    assert all(
        {field.name for field in fields(FeatParams)} == set(PIN_CONFIGS[mode]["features"])
        for mode in ("off", "on")
    )
    assert PIN_CONFIGS["off"]["training"]["exclusion_schedule"] == {}
    assert PIN_CONFIGS["on"]["training"]["exclusion_schedule"] == {}
    assert set(DECODER_CONDITIONS) == {
        "beam",
        "wbeam",
        "pl_window",
        "lw",
        "wip",
        "pbeam",
        "lpbeam",
        "lponlybeam",
        "fwdflatbeam",
        "fwdflatwbeam",
    }


def _cell(errors: tuple[int, ...], *, recorded: bool) -> dict[str, object]:
    rows = [[f"voice/u{index}", 10, error] for index, error in enumerate(errors)]
    total_errors = sum(errors)
    cell: dict[str, object] = {
        "wer": total_errors / (10 * len(errors)) * (100 if recorded else 1),
        "errors": total_errors,
        "ref_words": 10 * len(errors),
        "utterances": len(errors),
        "decoded": len(errors),
        "oov_tokens": 0,
    }
    if recorded:
        cell["utterance_rows"] = rows
    else:
        cell["matched_pairs"] = {row[0]: {"ref_words": row[1], "errors": row[2]} for row in rows}
    return cell


def _comparison_documents() -> tuple[dict[str, object], dict[str, object]]:
    results = {
        mode: {dataset: _cell((1, 1, 1, 1), recorded=False) for dataset in ("slt55", "big")}
        for mode in ("off", "on")
    }
    actual = {
        "results": results,
        "resources": {"x": "y"},
        "conditions": {
            "band": "BM1",
            "bootstrap": {"resamples": 500, "seed": 11},
        },
        "engine": {"version": "1"},
    }
    record = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "resources": actual["resources"],
        "conditions": actual["conditions"],
        "engine": actual["engine"],
        "results": {
            mode: {dataset: _cell((1, 1, 1, 1), recorded=True) for dataset in ("slt55", "big")}
            for mode in ("off", "on")
        },
        "off_big_floor_plus_1": True,
    }
    return actual, record


def test_comparison_uses_true_cross_run_matched_pairs() -> None:
    actual, record = _comparison_documents()
    rows = compare_results(actual, record)
    assert [row["delta"] for row in rows] == [0.0] * 4
    assert [row["paired_delta_ci_95"] for row in rows] == [[0.0, 0.0]] * 4
    assert all(row["pass"] for row in rows)

    actual["results"]["on"]["big"] = _cell((2, 2, 2, 2), recorded=False)  # type: ignore[index]
    rows = compare_results(actual, record)
    assert rows[-1]["paired_delta_ci_95"] == [10.0, 10.0]
    assert not rows[-1]["pass"]


def test_comparison_authenticates_inputs_and_pair_ids() -> None:
    actual, record = _comparison_documents()

    for key in ("resources", "conditions", "engine"):
        drifted = json.loads(json.dumps(actual))
        drifted[key][next(iter(drifted[key]))] = "drift"
        with pytest.raises(RuntimeError, match=key.replace("resources", "resources")):
            compare_results(drifted, record)
    assert compare_results(
        {**actual, "engine": {"version": "other"}}, record, allow_engine_drift=True
    )
    pairs = actual["results"]["off"]["slt55"]["matched_pairs"]  # type: ignore[index]
    pairs["voice/extra"] = pairs.pop("voice/u0")  # type: ignore[union-attr]
    with pytest.raises(RuntimeError, match="utterance ID mismatch"):
        compare_results(actual, record)


def test_record_schema_and_bootstrap_smoke() -> None:
    conditions = benchmark_conditions()
    assert conditions["cells"]["slt55"] == "same-speaker resubstitution cell"
    assert conditions["bootstrap"]["resamples"] == 100_000
    pairs = {
        "a/1": {"errors": 0, "ref_words": 10},
        "a/2": {"errors": 2, "ref_words": 10},
        "b/1": {"errors": 1, "ref_words": 10},
    }
    low, high = bootstrap_ci(pairs, resamples=200, seed=1)
    assert 0 <= low <= high <= 20
    delta_low, delta_high = paired_delta_ci(
        [["a/1", 10, 0], ["a/2", 10, 2], ["b/1", 10, 1]],
        pairs,
        speaker_stratified=True,
        resamples=200,
        seed=1,
    )
    assert delta_low == delta_high == 0
    with pytest.raises(RuntimeError, match="missing required field: engine"):
        validate_record({"schema_version": RECORD_SCHEMA_VERSION})


def test_incomplete_extraction_is_recovered(tmp_path: Path) -> None:
    archive = ARCHIVES[0]
    source = tmp_path / "source"
    wav = source / "cmu_us_slt_arctic" / "wav"
    wav.mkdir(parents=True)
    (wav / "one.wav").write_bytes(b"wav")
    packed = tmp_path / "voice.tar.bz2"
    with tarfile.open(packed, "w:bz2") as tar:
        tar.add(source / "cmu_us_slt_arctic", arcname="cmu_us_slt_arctic")
    archive = replace(archive, expected_wavs=1)
    corpus = tmp_path / "corpus"
    stale = corpus / "cmu_us_slt_arctic"
    stale.mkdir(parents=True)
    (stale / ".pstrain-extraction.json").write_text('{"source_archive_sha256":"bad"}')
    destination = extract_archive(packed, archive, corpus)
    assert (destination / "wav" / "one.wav").is_file()
    assert (
        json.loads((destination / ".pstrain-extraction.json").read_text())["source_archive_sha256"]
        == archive.sha256
    )
    marker = json.loads((destination / ".pstrain-extraction.json").read_text())
    assert marker["wav_manifest"] == [
        {"path": "wav/one.wav", "size": 3, "sha256": marker["wav_manifest"][0]["sha256"]}
    ]

    (destination / "wav" / "one.wav").write_bytes(b"bad")
    extract_archive(packed, archive, corpus)
    assert (destination / "wav" / "one.wav").read_bytes() == b"wav"


def test_emitted_record_round_trip_comparison(tmp_path: Path) -> None:
    actual, _record = _comparison_documents()
    record = make_record(actual)
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record))
    loaded = json.loads(path.read_text())
    validate_record(loaded)
    assert all(row["pass"] for row in compare_results(actual, loaded))


def test_gate_failure_blocks_comparison(tmp_path: Path) -> None:
    path = tmp_path / "shared/models/stage/bw_telemetry.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"passes": [{"pass": 1, "signed_convergence_delta": None}]}))
    with pytest.raises(RuntimeError, match="telemetry gate failed"):
        audit_monotonicity(tmp_path)


@pytest.mark.benchmark
def test_arctic_bm1_turnkey(tmp_path: Path) -> None:
    record = os.environ.get("PSTRAIN_BENCH_RECORD")
    args = ["--work-dir", str(tmp_path)]
    args.extend(["--record", record] if record else ["--no-compare"])
    assert main(args) == 0
