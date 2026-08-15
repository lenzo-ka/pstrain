from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tarfile
from collections.abc import Callable
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import pytest

from pstrain.benchmarks.arctic import (
    ARCHIVES,
    DATA_DIR,
    DECODER_CONDITIONS,
    FILLER_DICTIONARY,
    KNOWN_SKIPS,
    PIN_CONFIGS,
    PINNED_RESOURCE_HASHES,
    RECORD_SCHEMA_VERSION,
    _run_trusted_child,
    adopt_uncovered_conditions,
    audit_monotonicity,
    authenticate_conditions,
    authenticate_pin_resources,
    band_resources,
    benchmark_conditions,
    bootstrap_ci,
    compare_results,
    configuration_provenance,
    engine_identity,
    extract_archive,
    fetch_archive,
    load_transcripts,
    main,
    make_record,
    paired_delta_ci,
    render_configuration_provenance,
    render_results_report,
    resolve_data_dir,
    resolved_configuration_provenance,
    run,
    sha256,
    training_corpus_identity,
    validate_record,
    write_project,
)
from pstrain.lib.config.models import Profile
from pstrain.lib.config.resolver import resolve_config
from pstrain.lib.corpus.split import train_test_split
from pstrain.lib.lm import build_lm
from pstrain.lib.transcription import parse_transcription_file
from tests.clib import requires_c_library


def test_archive_manifest_matches_runtime_config() -> None:
    manifest = json.loads((DATA_DIR.parent / "archives.json").read_text())
    assert manifest["archives"] == [asdict(archive) for archive in ARCHIVES]
    assert {archive.voice for archive in ARCHIVES} == {"slt", "bdl", "rms", "clb", "awb", "jmk"}
    assert all(
        archive.url.startswith("http://festvox.org/cmu_arctic/packed/") for archive in ARCHIVES
    )
    assert all(len(archive.sha256) == 64 for archive in ARCHIVES)


def test_network_and_benchmark_children_use_safe_launch_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> None:
        calls.append((command, kwargs))

    def reject_in_process_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("the harness parent must never access the network")

    monkeypatch.setattr("pstrain.benchmarks.arctic.subprocess.run", fake_run)
    monkeypatch.setattr(
        "pstrain.benchmarks.arctic.urllib.request.urlopen", reject_in_process_network
    )

    destination = fetch_archive(ARCHIVES[0], tmp_path)
    assert destination == tmp_path / Path(ARCHIVES[0].url).name
    assert calls[0][0][2] == "_fetch-archive"
    _run_trusted_child(["/absolute/python", "/absolute/child.py"])
    assert len(calls) == 2
    for _command, kwargs in calls:
        assert "cwd" not in kwargs
        assert kwargs["close_fds"] is False
        assert kwargs["check"] is True


def test_committed_transcripts_are_normalized_and_complete() -> None:
    expected = {
        "pin-train.transcription": 1043,
        "full-slt.transcription": 1132,
        "slt55.transcription": 55,
        "big.transcription": 3395,
    }
    for name, count in expected.items():
        transcripts = load_transcripts(DATA_DIR / name)
        assert len(transcripts) == count
        assert all(text == text.lower() and text.strip() == text for text in transcripts.values())
        assert all("<s>" not in text and "</s>" not in text for text in transcripts.values())
    big = load_transcripts(DATA_DIR / "big.transcription")
    assert {utterance.split("/", 1)[0] for utterance in big} == {"bdl", "rms", "clb"}

    train = set(load_transcripts(DATA_DIR / "pin-train.transcription"))
    fileids = (DATA_DIR / "pin-train.fileids").read_text().splitlines()
    assert len(fileids) == len(set(fileids)) == 1043
    assert fileids == list(load_transcripts(DATA_DIR / "pin-train.transcription"))
    assert not (set(load_transcripts(DATA_DIR / "slt55.transcription")) & train)
    assert not (set(big) & train)

    assert training_corpus_identity() == {
        "utterances": 1043,
        "transcription": {
            "name": "pin-train.transcription",
            "sha256": "28788cd1ce2269d344b50420d74007fa8c443778680724f6334e2712ea110959",
        },
        "fileids": {
            "name": "pin-train.fileids",
            "sha256": "8ce9a55c5929f6f86579ee1b244c38fd4d0a9d41e436e2057337f74c1bb4d631",
        },
    }


def test_data_resolves_from_wheel_and_repo_layouts(tmp_path: Path) -> None:
    wheel_root = tmp_path / "wheel"
    repo_root = tmp_path / "repo"
    wheel_data = wheel_root / "benchmarks/arctic/data"
    repo_data = repo_root / "benchmarks/arctic/data"
    wheel_data.mkdir(parents=True)
    repo_data.mkdir(parents=True)
    (wheel_data / "pin-train.transcription").write_text("wheel text\n")
    (repo_data / "pin-train.transcription").write_text("repo text\n")
    assert resolve_data_dir(package_root=wheel_root, repo_root=repo_root) == wheel_data
    (wheel_data / "pin-train.transcription").unlink()
    assert resolve_data_dir(package_root=wheel_root, repo_root=repo_root) == repo_data


@requires_c_library
def test_engine_identity_is_in_default_test_suite() -> None:
    """The default suite authenticates the engine fields used by benchmark records."""
    dictionary = DATA_DIR / "cmu_arctic_slt.dict"
    identity = engine_identity(dictionary)
    assert identity["native_library_sha256"] != "absent"
    assert identity["native_library_fp_contract_declared"] == "off"
    assert identity["decode_dictionary_sha256"] == sha256(dictionary)
    assert any(key.startswith("native_program_") for key in identity)


def test_pin_band_resources_and_hashes() -> None:
    dictionary, lm = band_resources("pin")
    assert dictionary == DATA_DIR / "cmu_arctic_slt.dict"
    assert lm == DATA_DIR / "training-unigram.lm"
    assert sha256(dictionary) == PINNED_RESOURCE_HASHES["dictionary_sha256"]
    assert sha256(lm) == PINNED_RESOURCE_HASHES["lm_sha256"]
    assert PINNED_RESOURCE_HASHES["filler_dictionary_sha256"].startswith("fb508839")
    authenticate_pin_resources(dictionary, lm, FILLER_DICTIONARY.encode())


def test_unigram_builder_difference_from_canonical_is_known(tmp_path: Path) -> None:
    built = tmp_path / "training-unigram.lm"
    build_lm(load_transcripts(DATA_DIR / "pin-train.transcription"), built, max_order=1)
    assert sha256(built) == "2c75cacb19b45c442fd857b791a2abaef7645eb9a1da510907546ac8503179c6"
    assert built.read_bytes() != (DATA_DIR / "training-unigram.lm").read_bytes()


def test_pin_training_transcript_is_consumable_by_split_flat_and_bw(tmp_path: Path) -> None:
    source = DATA_DIR / "pin-train.transcription"
    expected_ids = (DATA_DIR / "pin-train.fileids").read_text().splitlines()

    # Baum-Welch uses this reader, while split emits the control list consumed
    # by flat initialization. Exercise both sides without starting training.
    source_transcripts = parse_transcription_file(source)
    source_ids = list(source_transcripts)
    assert len(source_ids) == len(set(source_ids)) == 1043
    assert source_ids == expected_ids
    assert "<s>" not in source_ids
    assert all(re.fullmatch(r"arctic_[ab]\d{4}", fileid) for fileid in source_ids)

    split = train_test_split(source, tmp_path, test_count=0)
    split_ids = split.train_fileids.read_text().splitlines()
    split_transcripts = parse_transcription_file(split.train_transcription)
    assert split.n_train == 1043
    assert split.n_test == 0
    assert split_ids == list(split_transcripts) == expected_ids
    assert split_transcripts == source_transcripts


def test_eval_transcript_serializations_are_enforced() -> None:
    full_lines = (DATA_DIR / "full-slt.transcription").read_text().splitlines()
    assert all(line.startswith("arctic_") and "<s>" not in line for line in full_lines)
    for name in ("slt55.transcription", "big.transcription"):
        lines = (DATA_DIR / name).read_text().splitlines()
        assert all(
            line.startswith("<s> ") and " </s> (" in line and line.endswith(")") for line in lines
        )


def test_pin_configs_resolve_ratified_conditions(tmp_path: Path) -> None:
    assert benchmark_conditions()["known_skips"] == KNOWN_SKIPS
    project = tmp_path / "bare-checkout" / "benchmark-project"
    write_project(project, tmp_path / "corpus", DATA_DIR / "cmu_arctic_slt.dict")
    resolved = {
        mode: resolve_config(
            project, profile_name=mode, user_config_path=tmp_path / "absent-user.yaml"
        )
        for mode in ("off", "on")
    }
    for mode in ("off", "on"):
        expected = Profile.model_validate(PIN_CONFIGS[mode]).model_dump(mode="json")
        assert resolved[mode].as_dict() == expected
        assert resolved[mode].profile.split.test_count == 0

    off = PIN_CONFIGS["off"]["training"]
    on = PIN_CONFIGS["on"]["training"]
    assert off["multipron_training"] is False
    assert off["untied_inventory"] == "linear"
    assert {off[stage]["convergence_ratio"] for stage in ("ci", "untied", "tied")} == {0.1}
    assert on["multipron_training"] is True
    assert on["untied_inventory"] == "all-triphone"
    assert on["optional_final_silence"] is True
    assert on["accept_arctic_a0587_known_skip"] is False
    assert on["arctic_a0302_zero_codebook_band"] is None
    assert {on[stage]["convergence_ratio"] for stage in ("ci", "untied", "tied")} == {0.001}
    assert on["untied"]["max_iterations"] == 6
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
    assert resolved["off"].profile.training.untied_inventory == "linear"
    assert resolved["on"].profile.training.untied_inventory == "all-triphone"
    assert [
        getattr(resolved[mode].profile.training, stage).convergence_ratio
        for mode in ("off", "on")
        for stage in ("ci", "untied", "tied")
    ] == [0.1, 0.1, 0.1, 0.001, 0.001, 0.001]
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


def test_configuration_provenance_renders_default_and_non_default_cells() -> None:
    shipped = Profile()
    default = configuration_provenance(shipped, source_kind="shipped defaults")
    non_default = configuration_provenance(
        Profile(training={"tree_directional_questions": False}),
        source_kind="explicit overrides",
        override_reason="isolate directional-question behavior",
    )

    assert default["diff_from_shipped_defaults"] == []
    assert render_configuration_provenance(default) == (
        "Source: shipped defaults\n\nDiff from shipped schema defaults: (empty)"
    )
    assert non_default["diff_from_shipped_defaults"] == [
        {
            "setting": "training.tree_directional_questions",
            "shipped_default": True,
            "value": False,
        }
    ]
    rendered = render_configuration_provenance(non_default)
    assert "Source: explicit overrides" in rendered
    assert "Reason: isolate directional-question behavior" in rendered
    assert "| `training.tree_directional_questions` | `true` | `false` |" in rendered


def test_cell_provenance_includes_resolved_cli_override(tmp_path: Path) -> None:
    project = tmp_path / "benchmark-project"
    write_project(project, tmp_path / "corpus", DATA_DIR / "cmu_arctic_slt.dict")
    resolved = resolve_config(
        project,
        profile_name="off",
        user_config_path=tmp_path / "absent-user.yaml",
        cli_overrides={"runner": {"jobs": 3}},
    )

    provenance = resolved_configuration_provenance(resolved.benchmark_document())
    jobs = next(
        row for row in provenance["diff_from_shipped_defaults"] if row["setting"] == "runner.jobs"
    )
    assert jobs["value"] == resolved.profile.runner.jobs == 3
    assert jobs["source"]["kind"] == resolved.fields["runner.jobs"].winner.source_kind == "cli"


def test_emitted_record_uses_child_resolved_cli_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dictionary = tmp_path / "dictionary.dict"
    lm = tmp_path / "language-model.lm"
    dictionary.write_text("WORD W ER D\n")
    lm.write_text("language model\n")
    monkeypatch.setattr("pstrain.benchmarks.arctic.ARCHIVES", ())
    monkeypatch.setattr("pstrain.benchmarks.arctic.band_resources", lambda _band: (dictionary, lm))
    monkeypatch.setattr("pstrain.benchmarks.arctic.authenticate_pin_resources", lambda *_args: None)
    monkeypatch.setattr("pstrain.benchmarks.arctic.training_corpus_identity", lambda: {})
    monkeypatch.setattr("pstrain.benchmarks.arctic.audit_monotonicity", lambda _project: [])
    monkeypatch.setattr(
        "pstrain.benchmarks.arctic.score_model", lambda *_args: _cell((1,), recorded=False)
    )
    monkeypatch.setattr("pstrain.benchmarks.arctic.engine_identity", lambda _dictionary: {})

    def surface_resolved(command: list[str], **_kwargs: object) -> None:
        project = Path(command[command.index("--project-dir") + 1])
        mode = command[command.index("--config") + 1]
        jobs = int(command[command.index("--jobs") + 1])
        output = Path(command[command.index("--resolved-config-output") + 1])
        resolved = resolve_config(
            project,
            profile_name=mode,
            user_config_path=tmp_path / "absent-user.yaml",
            cli_overrides={"runner": {"jobs": jobs}},
        )
        output.write_text(json.dumps(resolved.benchmark_document()))

    monkeypatch.setattr("pstrain.benchmarks.arctic._run_trusted_child", surface_resolved)
    emitted = tmp_path / "record.json"
    historical = _comparison_documents()[1]
    historical_path = tmp_path / "historical.json"
    historical_path.write_text(json.dumps(historical))
    run(
        tmp_path / "work",
        None,
        3,
        emit_record=emitted,
        historical_record_path=historical_path,
    )
    record = json.loads(emitted.read_text())

    for mode in ("on",):
        assert record["conditions"]["pin_conditions"][mode]["runner"]["jobs"] == 3
        for dataset in ("slt55", "big"):
            rows = record["results"][mode][dataset]["configuration_provenance"][
                "diff_from_shipped_defaults"
            ]
            jobs = next(row for row in rows if row["setting"] == "runner.jobs")
            assert jobs == {
                "setting": "runner.jobs",
                "shipped_default": None,
                "value": 3,
                "source": {"kind": "cli"},
            }


def test_record_rejects_provenance_that_disagrees_with_conditions() -> None:
    actual, _record = _comparison_documents()
    actual["conditions"] = benchmark_conditions()
    for mode in ("on",):
        document = {
            "profile": actual["conditions"]["pin_conditions"][mode],
            "profile_name": mode,
            "field_source_kinds": actual["conditions"]["pin_condition_source_kinds"][mode],
        }
        provenance = resolved_configuration_provenance(document)
        for dataset in ("slt55", "big"):
            actual["results"][mode][dataset]["configuration_provenance"] = provenance
    record = make_record(actual)
    record["results"]["on"]["slt55"]["configuration_provenance"]["diff_from_shipped_defaults"].pop()
    with pytest.raises(RuntimeError, match="provenance/conditions consistency"):
        validate_record(record)


def test_adopt_uncovered_keeps_cell_provenance_consistent(tmp_path: Path) -> None:
    source = Path("docs/benchmarks/arctic-pin/record.json")
    record = json.loads(source.read_text())
    setting = "training.optional_final_silence"
    for mode in ("off", "on"):
        del record["conditions"]["pin_conditions"][mode]["training"]["optional_final_silence"]
        for dataset in ("slt55", "big"):
            provenance = record["results"][mode][dataset]["configuration_provenance"]
            provenance["diff_from_shipped_defaults"] = [
                row for row in provenance["diff_from_shipped_defaults"] if row["setting"] != setting
            ]
    validate_record(record)
    measurements = {
        mode: {
            dataset: {
                key: record["results"][mode][dataset][key] for key in ("wer", "errors", "ref_words")
            }
            for dataset in ("slt55", "big")
        }
        for mode in ("off", "on")
    }
    adopted = tmp_path / "record.json"
    adopted.write_text(json.dumps(record))

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_arctic_pin.py",
            "--record",
            str(adopted),
            "--adopt-uncovered",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "adopted 2 previously uncovered" in completed.stdout
    updated = json.loads(adopted.read_text())
    validate_record(updated)
    assert (
        updated["conditions"]["pin_conditions"]["off"]["training"]["optional_final_silence"]
        is False
    )
    assert (
        updated["conditions"]["pin_conditions"]["on"]["training"]["optional_final_silence"] is False
    )
    assert {
        mode: {
            dataset: {
                key: updated["results"][mode][dataset][key]
                for key in ("wer", "errors", "ref_words")
            }
            for dataset in ("slt55", "big")
        }
        for mode in ("off", "on")
    } == measurements


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda record: record["results"]["off"]["slt55"]["configuration_provenance"][
                "diff_from_shipped_defaults"
            ].pop(),
            "provenance/conditions consistency mismatch",
        ),
        (
            lambda record: record["conditions"]["decoder"].__setitem__("wip", 0.3),
            "conditions.decoder.wip mismatch",
        ),
    ],
)
def test_adopt_uncovered_refuses_existing_drift(
    tmp_path: Path, mutation: Callable[[dict[str, Any]], object], message: str
) -> None:
    record = json.loads(Path("docs/benchmarks/arctic-pin/record.json").read_text())
    mutation(record)
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record))
    before = path.read_bytes()

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_arctic_pin.py",
            "--record",
            str(path),
            "--adopt-uncovered",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert message in completed.stderr
    assert path.read_bytes() == before


def _cell(errors: tuple[int, ...], *, recorded: bool) -> dict[str, object]:
    rows = [[f"voice/u{index}", 10, error] for index, error in enumerate(errors)]
    total_errors = sum(errors)
    cell: dict[str, object] = {
        "wer": total_errors / (10 * len(errors)) * (100 if recorded else 1),
        "errors": total_errors,
        "ref_words": 10 * len(errors),
        "utterances": len(errors),
        "decoded": len(errors),
        "decode_denominator": len(errors),
        "oov_tokens": 0,
        "known_skips": [],
        "configuration_provenance": configuration_provenance(
            Profile(), source_kind="shipped defaults"
        ),
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
        "basis": {
            "name": "MULTIPRON-ONLY",
            "live_cells": ["on/slt55", "on/big"],
            "retired_cells": ["off/slt55", "off/big"],
        },
        "resources": actual["resources"],
        "conditions": actual["conditions"],
        "engine": actual["engine"],
        "results": {
            mode: {dataset: _cell((1, 1, 1, 1), recorded=True) for dataset in ("slt55", "big")}
            for mode in ("off", "on")
        },
    }
    for mode, cells in record["results"].items():
        for cell in cells.values():
            cell["status"] = "live" if mode == "on" else "retired/historical"
    return actual, record


def test_comparison_uses_true_cross_run_matched_pairs() -> None:
    actual, record = _comparison_documents()
    rows = compare_results(actual, record)
    assert [row["delta"] for row in rows] == [0.0] * 2
    assert [row["paired_delta_ci_95"] for row in rows] == [[0.0, 0.0]] * 2
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
    pairs = actual["results"]["on"]["slt55"]["matched_pairs"]  # type: ignore[index]
    pairs["voice/extra"] = pairs.pop("voice/u0")
    with pytest.raises(RuntimeError, match="utterance ID mismatch"):
        compare_results(actual, record)


def test_condition_authentication_rejects_changed_pinned_value() -> None:
    recorded = {"pin_conditions": {"on": {"training": {"multipron_training": True}}}}
    actual = json.loads(json.dumps(recorded))
    actual["pin_conditions"]["on"]["training"]["multipron_training"] = False
    with pytest.raises(RuntimeError, match="multipron_training mismatch"):
        authenticate_conditions(actual, recorded)


def test_condition_authentication_reports_schema_addition_without_failing() -> None:
    recorded = {
        "frozen_dataclass_fields": {"features": ["samprate"]},
        "pin_conditions": {"off": {"features": {"samprate": 16_000}}},
    }
    actual = json.loads(json.dumps(recorded))
    actual["frozen_dataclass_fields"]["features"].append("new_schema_field")
    actual["pin_conditions"]["off"]["features"]["new_schema_field"] = "default"

    assert authenticate_conditions(actual, recorded) == [
        "conditions.frozen_dataclass_fields.features.new_schema_field",
        "conditions.pin_conditions.off.features.new_schema_field",
    ]
    assert adopt_uncovered_conditions(actual, recorded) == actual


def test_committed_record_covers_and_authenticates_live_conditions() -> None:
    record_path = Path(__file__).parents[1] / "docs/benchmarks/arctic-pin/record.json"
    record = json.loads(record_path.read_text())
    assert authenticate_conditions(benchmark_conditions(), record["conditions"]) == []


def test_record_schema_and_bootstrap_smoke() -> None:
    conditions = benchmark_conditions()
    assert conditions["cells"]["slt55"] == "same-speaker held-out cell"
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
    with pytest.raises(RuntimeError, match="missing required field: basis"):
        validate_record({"schema_version": RECORD_SCHEMA_VERSION})


def test_headline_reports_decode_shortfall_denominator() -> None:
    cell = _cell((1, 2, 3), recorded=False)
    cell["utterances"] = 5
    cell["decode_denominator"] = 5
    report = render_results_report({"results": {"on": {"big": cell}}})
    assert "WER: 20.0000% over 3/5 decoded" in report


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


def _write_skip_telemetry(
    project: Path,
    *,
    utterance: str | None,
    reason: str = "alignment_failure",
    stage: str = "cd-2g",
    pass_numbers: tuple[int, ...] = (1,),
    occupancy: int | None = None,
) -> None:
    path = project / f"shared/models/{stage}/{project.name}/bw_telemetry.json"
    path.parent.mkdir(parents=True)
    terminal = [] if utterance is None else [{"utterance": utterance, "reason": reason}]
    path.write_text(
        json.dumps(
            {
                "passes": [
                    {
                        "pass": pass_number,
                        "signed_convergence_delta": None,
                        "accounting": {
                            "skipped_utts": len(terminal),
                            "skip_reasons": {
                                "excluded_by_schedule": 0,
                                "alignment_failure": len(terminal),
                            },
                            "terminal_skips": terminal,
                            "accepted_exceptions": (
                                [
                                    {
                                        "sentinel": utterance,
                                        "quantity": "exact_zero_codebooks",
                                        "value": occupancy,
                                        "band": [4548, 4623],
                                    }
                                ]
                                if occupancy is not None
                                else []
                            ),
                        },
                    }
                    for pass_number in pass_numbers
                ]
            }
        )
    )


def test_retired_a0587_exception_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "on"
    utterance = "arctic_a0587"
    _write_skip_telemetry(project, utterance=utterance, stage="cd-1g", pass_numbers=(6,))
    with pytest.raises(RuntimeError, match="unlisted terminal skip"):
        audit_monotonicity(project)


def test_retired_a0302_band_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "on"
    _write_skip_telemetry(
        project,
        utterance="arctic_a0302",
        reason="accepted_exception_band",
        stage="cd-untied",
        pass_numbers=tuple(range(3, 11)),
        occupancy=4600,
    )
    with pytest.raises(RuntimeError, match="invalid accepted-exception"):
        audit_monotonicity(project)


@pytest.mark.parametrize(
    ("mode", "utterance", "stage", "pass_numbers"),
    [
        ("off", "arctic_a0302", "cd-untied", (3,)),
        ("on", "arctic_a0587", "cd-2g", (1,)),
    ],
)
def test_known_skip_manifest_is_scoped_per_mode(
    tmp_path: Path,
    mode: str,
    utterance: str,
    stage: str,
    pass_numbers: tuple[int, ...],
) -> None:
    project = tmp_path / mode
    _write_skip_telemetry(project, utterance=utterance, stage=stage, pass_numbers=pass_numbers)
    with pytest.raises(RuntimeError, match="unlisted terminal skip"):
        audit_monotonicity(project)


def test_unlisted_terminal_skip_fails(tmp_path: Path) -> None:
    project = tmp_path / "off"
    _write_skip_telemetry(project, utterance="arctic_a0001")
    with pytest.raises(RuntimeError, match="unlisted terminal skip"):
        audit_monotonicity(project)


def test_non_manifest_failure_reason_still_fails(tmp_path: Path) -> None:
    project = tmp_path / "off"
    _write_skip_telemetry(project, utterance="arctic_a0587", reason="exception")
    with pytest.raises(RuntimeError, match="unlisted terminal skip"):
        audit_monotonicity(project)


@pytest.mark.benchmark
def test_arctic_bm1_turnkey(tmp_path: Path) -> None:
    record = os.environ.get("PSTRAIN_BENCH_RECORD")
    args = ["--work-dir", str(tmp_path)]
    args.extend(["--record", record] if record else ["--no-compare"])
    assert main(args) == 0
