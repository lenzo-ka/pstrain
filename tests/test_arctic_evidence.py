"""Checks that the Arctic pin's comparison arms stay on the same resources."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[1]
EVIDENCE = ROOT / "evidence/arctic-pin"


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifacts() -> tuple[dict, dict]:
    record = json.loads((EVIDENCE / "record.json").read_text())
    oracle = json.loads((EVIDENCE / "oracle-sidecar.json").read_text())
    return record, oracle


def test_live_oracle_cells_are_resource_matched_to_the_record() -> None:
    record, oracle = _artifacts()

    _script("regenerate_arctic_oracle").verify_sidecar(oracle, record)

    assert oracle["resources"]["dictionary_sha256"] == record["resources"]["dictionary_sha256"]
    assert oracle["resources"]["lm_sha256"] == record["resources"]["lm_sha256"]
    assert oracle["resources"]["record_resources_match"] == {"dictionary": True, "lm": True}


def test_retired_oracle_cells_keep_the_dictionary_they_were_decoded_with() -> None:
    record, oracle = _artifacts()
    retained = oracle["historical_provenance"]

    assert sorted(retained["cells"]) == sorted(record["basis"]["retired_cells"])
    assert (
        retained["resources"]["dictionary_sha256"]
        == (record["historical_provenance"]["resources"]["dictionary_sha256"])
    )
    assert retained["resources"]["dictionary_sha256"] != oracle["resources"]["dictionary_sha256"]
    assert oracle["results"]["off"] != oracle["results"]["on"]


@pytest.mark.parametrize("field", ["dictionary_sha256", "lm_sha256"])
def test_sidecar_check_rejects_a_false_resource_match(field: str) -> None:
    record, oracle = _artifacts()
    oracle["resources"][field] = "0" * 64

    with pytest.raises(SystemExit, match="record_resources_match"):
        _script("regenerate_arctic_oracle").verify_sidecar(oracle, record)


def test_sidecar_check_rejects_a_false_comparability_classification() -> None:
    record, oracle = _artifacts()
    oracle["pstrain_vs_oracle"][0]["comparability"]["paired_decode"]["status"] = "NOT COMPARABLE"

    with pytest.raises(SystemExit, match="invalid comparability"):
        _script("regenerate_arctic_oracle").verify_sidecar(oracle, record)


def test_front_end_control_digests_recompute_from_the_emitted_rows() -> None:
    _, oracle = _artifacts()
    module = _script("regenerate_arctic_oracle")
    control = oracle["front_end_reconstruction"]["neutrality_control"]

    assert control["results_identical"] is True
    assert control["dictionary_sha256"] == oracle["resources"]["dictionary_sha256"]
    for dataset in ("slt55", "big"):
        emitted = oracle["results"]["on"][dataset]
        assert control["arms"]["completed"][dataset] == control["arms"]["preserved"][dataset]
        assert control["arms"]["completed"][dataset]["utterance_rows_sha256"] == module.rows_digest(
            emitted["utterance_rows"]
        )
        assert control["arms"]["completed"][dataset]["results_sha256"] == module.canonical_sha256(
            emitted
        )


def test_historical_dictionary_control_retains_the_rows_it_compares_against() -> None:
    record, oracle = _artifacts()
    module = _script("regenerate_arctic_oracle")
    control = oracle["historical_dictionary_control"]

    assert (
        control["dictionary_sha256"]
        == (record["historical_provenance"]["resources"]["dictionary_sha256"])
    )
    assert control["rows_reproduce_historical"] is False
    for dataset in ("slt55", "big"):
        current = control["current_engine"][dataset]
        historical = control["historical_decode"][dataset]
        # The claim rests on rows this file carries, not on a quoted count.
        assert historical["utterance_rows_sha256"] == module.rows_digest(
            historical["utterance_rows"]
        )
        assert sum(row[2] for row in historical["utterance_rows"]) == historical["errors"]
        assert current["utterance_rows_sha256"] != historical["utterance_rows_sha256"]
        assert current["errors"] > historical["errors"]
    module.verify_sidecar(oracle, record)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda oracle: oracle["front_end_reconstruction"]["neutrality_control"].update(
                {"results_identical": False}
            ),
            "non-neutral completion",
            id="false-neutrality-flag",
        ),
        pytest.param(
            lambda oracle: oracle["front_end_reconstruction"]["neutrality_control"]["arms"][
                "preserved"
            ]["slt55"].update({"errors": 999}),
            "not neutral",
            id="arms-disagree",
        ),
        pytest.param(
            lambda oracle: oracle["results"]["on"]["slt55"]["utterance_rows"].append(
                ["voice/ghost", 1, 1]
            ),
            "does not describe the emitted",
            id="stale-completed-digest",
        ),
        pytest.param(
            lambda oracle: oracle["historical_dictionary_control"].update(
                {"rows_reproduce_historical": True}
            ),
            "misstates whether it reproduced",
            id="false-reproduction-claim",
        ),
        pytest.param(
            lambda oracle: oracle["historical_dictionary_control"].update(
                {"dictionary_sha256": "0" * 64}
            ),
            "does not use the dictionary the retired cells",
            id="wrong-control-dictionary",
        ),
        pytest.param(
            lambda oracle: oracle["historical_dictionary_control"]["historical_decode"]["slt55"][
                "utterance_rows"
            ].append(["voice/ghost", 1, 1]),
            "do not hash to the digest",
            id="tampered-historical-rows",
        ),
        pytest.param(
            lambda oracle: oracle["historical_dictionary_control"]["historical_decode"][
                "slt55"
            ].update({"errors": 1}),
            "errors, not the",
            id="historical-count-disagrees-with-rows",
        ),
        pytest.param(
            lambda oracle: oracle["front_end_reconstruction"]["added_fields"].update(
                {"-remove_dc": "yes"}
            ),
            "not the ones the sidecar was",
            id="defaults-changed-without-rerun",
        ),
        pytest.param(
            lambda oracle: oracle["front_end_reconstruction"]["staged_model_identity"].update(
                {"replaced_files": ["feat.params", "means"]}
            ),
            "replaced more than the front-end record",
            id="undeclared-staged-replacement",
        ),
        pytest.param(
            lambda oracle: oracle["historical_dictionary_control"]["current_engine"][
                "slt55"
            ].update({"errors": 138}),
            "summary disagrees with the emitted cell",
            id="current-engine-errors-edited",
        ),
        pytest.param(
            lambda oracle: oracle["historical_dictionary_control"]["current_engine"][
                "slt55"
            ].update({"ref_words": 512}),
            "summary disagrees with the emitted cell",
            id="current-engine-ref-words-edited",
        ),
        pytest.param(
            lambda oracle: oracle["historical_dictionary_control"]["current_engine"][
                "slt55"
            ].update({"wer": 26.9}),
            "summary disagrees with the emitted cell",
            id="current-engine-wer-edited",
        ),
        pytest.param(
            lambda oracle: oracle["historical_dictionary_control"]["current_engine"][
                "slt55"
            ].update({"results_sha256": "0" * 64}),
            "summary disagrees with the emitted cell",
            id="current-engine-results-digest-edited",
        ),
        pytest.param(
            lambda oracle: oracle["historical_dictionary_control"]["historical_decode"][
                "slt55"
            ].update({"ref_words": 512}),
            "reference words, not the",
            id="historical-ref-words-disagrees-with-rows",
        ),
        pytest.param(
            lambda oracle: oracle["historical_dictionary_control"]["historical_decode"][
                "slt55"
            ].update({"wer": 99.0}),
            "states a WER of",
            id="historical-wer-disagrees-with-counts",
        ),
    ],
)
def test_sidecar_check_rejects_an_unsupported_control_claim(mutate: object, message: str) -> None:
    record, oracle = _artifacts()
    mutate(oracle)  # type: ignore[operator]

    with pytest.raises(SystemExit, match=message):
        _script("regenerate_arctic_oracle").verify_sidecar(oracle, record)


def test_staging_starts_from_an_empty_directory(tmp_path: Path) -> None:
    module = _script("regenerate_arctic_oracle")
    source = tmp_path / "preserved"
    source.mkdir()
    (source / "feat.params").write_text("-nfilt 25\n-transform dct\n")
    (source / "means").write_bytes(b"means")

    staged = module.stage_model(tmp_path / "work", source, "completed", complete=True)
    stale = staged / "sendump"
    stale.write_bytes(b"left over from an earlier run")
    assert module.model_directory_identity(staged)["files"]

    staged = module.stage_model(tmp_path / "work", source, "completed", complete=True)

    assert not stale.exists()
    identity = module.staged_identity(staged, source, replaced=("feat.params",))
    assert identity["replaced_files"] == ["feat.params"]
    assert sorted(row["path"] for row in identity["files"]) == ["feat.params", "means"]


def test_staged_identity_refuses_an_undeclared_difference(tmp_path: Path) -> None:
    module = _script("regenerate_arctic_oracle")
    source = tmp_path / "preserved"
    source.mkdir()
    (source / "feat.params").write_text("-nfilt 25\n")
    (source / "means").write_bytes(b"means")
    staged = module.stage_model(tmp_path / "work", source, "completed", complete=True)
    (staged / "means").write_bytes(b"tampered")

    with pytest.raises(SystemExit, match="only"):
        module.staged_identity(staged, source, replaced=("feat.params",))

    (staged / "extra").write_bytes(b"extra")
    with pytest.raises(SystemExit, match="authenticated file set"):
        module.staged_identity(staged, source, replaced=("feat.params",))


@pytest.mark.parametrize("field", ["dictionary_sha256", "lm_sha256"])
def test_paired_analysis_refuses_arms_on_different_resources(tmp_path: Path, field: str) -> None:
    record, oracle = _artifacts()
    oracle["resources"][field] = "0" * 64
    oracle["resources"]["record_resources_match"] = {"dictionary": True, "lm": True}
    oracle_path = tmp_path / "oracle-sidecar.json"
    oracle_path.write_text(json.dumps(oracle))

    with pytest.raises(SystemExit, match=f"arms disagree on {field}"):
        _script("regenerate_arctic_paired_analysis").regenerate(
            EVIDENCE / "record.json", oracle_path
        )


def test_paired_analysis_matches_its_generated_form() -> None:
    module = _script("regenerate_arctic_paired_analysis")
    rendered = module.regenerate(EVIDENCE / "record.json", EVIDENCE / "oracle-sidecar.json")

    assert rendered == (EVIDENCE / "paired-analysis.json").read_text()


def test_paired_analysis_rejects_a_false_comparability_classification() -> None:
    analysis = json.loads((EVIDENCE / "paired-analysis.json").read_text())
    analysis["pstrain_vs_oracle"][0]["comparability"]["implementation_attribution"]["status"] = (
        "COMPARABLE"
    )

    with pytest.raises(SystemExit, match="invalid comparability"):
        _script("regenerate_arctic_paired_analysis").validate_analysis(analysis)
