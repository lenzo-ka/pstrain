from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pstrain.benchmarks.arctic import (
    ARCHIVES,
    DATA_DIR,
    PIN_CONFIGS,
    compare_results,
    load_transcripts,
    main,
)


def test_archive_manifest_matches_runtime_config() -> None:
    manifest = json.loads((DATA_DIR.parent / "archives.json").read_text())
    assert manifest["archives"] == [
        {"voice": archive.voice, "url": archive.url, "sha256": archive.sha256}
        for archive in ARCHIVES
    ]
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


def test_comparison_arithmetic_and_off_big_floor_clause() -> None:
    actual = {
        mode: {
            dataset: {"wer": value / 100} for dataset, value in {"slt55": 20.2, "big": 30.8}.items()
        }
        for mode in ("off", "on")
    }
    record = {
        "results": {
            mode: {
                dataset: {"wer": value} for dataset, value in {"slt55": 20.0, "big": 30.0}.items()
            }
            for mode in ("off", "on")
        },
        "tolerances": {mode: {"slt55": 0.25, "big": 0.25} for mode in ("off", "on")},
        "off_big_floor_plus_1": True,
    }
    rows = compare_results(actual, record)
    assert [round(row["delta"], 3) for row in rows] == [0.2, 0.8, 0.2, 0.8]
    assert [row["pass"] for row in rows] == [True, True, True, False]


@pytest.mark.benchmark
def test_arctic_bm1_turnkey(tmp_path: Path) -> None:
    record = os.environ.get("PSTRAIN_BENCH_RECORD")
    args = ["--work-dir", str(tmp_path)]
    args.extend(["--record", record] if record else ["--no-compare"])
    assert main(args) == 0
