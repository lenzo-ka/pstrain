from __future__ import annotations

import json
import os
import tarfile
from dataclasses import asdict, replace
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
    frozen = {
        "a_beam",
        "b_beam",
        "max_skip_fraction",
        "retry_beam_factor",
        "tree_state_weights",
        "tree_ssplitmax",
        "tree_ssplitthr",
        "tree_csplitmax",
        "tree_csplitthr",
        "tree_mwfloor",
        "question_npermute",
        "question_quests_per_state",
        "question_niter",
    }
    assert all(frozen <= set(PIN_CONFIGS[mode]["training"]) for mode in ("off", "on"))
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


def test_comparison_arithmetic_and_off_big_floor_clause() -> None:
    results = {
        mode: {
            dataset: {"wer": value / 100} for dataset, value in {"slt55": 20.2, "big": 30.8}.items()
        }
        for mode in ("off", "on")
    }
    actual = {
        "results": results,
        "resources": {"x": "y"},
        "conditions": {"band": "BM1"},
        "engine": {"version": "1"},
    }
    record = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "resources": actual["resources"],
        "conditions": actual["conditions"],
        "engine": actual["engine"],
        "results": {
            mode: {
                dataset: {"wer": value, "bootstrap_ci_95": [value - 0.25, value + 0.25]}
                for dataset, value in {"slt55": 20.0, "big": 30.0}.items()
            }
            for mode in ("off", "on")
        },
        "off_big_floor_plus_1": True,
    }
    rows = compare_results(actual, record)
    assert [round(row["delta"], 3) for row in rows] == [0.2, 0.8, 0.2, 0.8]
    assert [row["pass"] for row in rows] == [True, True, True, False]

    for key in ("resources", "conditions", "engine"):
        drifted = json.loads(json.dumps(actual))
        drifted[key][next(iter(drifted[key]))] = "drift"
        with pytest.raises(RuntimeError, match=key.replace("resources", "resources")):
            compare_results(drifted, record)
    assert compare_results(
        {**actual, "engine": {"version": "other"}}, record, allow_engine_drift=True
    )


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
