"""Explicit recovery preserves model bytes and binds health to evaluated inputs."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from pstrain.api.checkpoints import list_checkpoints, restore_checkpoint
from pstrain.cli.cli import create_parser
from pstrain.lib import _pstrainc
from pstrain.lib.bw import BWConfig
from pstrain.lib.checkpoints import evaluation_health, model_snapshot
from pstrain.lib.model import MODEL_PARAMETER_FILES, fingerprint_model
from pstrain.lib.steps.train import _fingerprint_model, _sha256_files, run_bw_training
from tests.clib import requires_c_library

FIXTURE = Path(__file__).parent / "fixtures" / "multipron_final_state"


@requires_c_library
def test_real_updates_have_next_pass_hash_bound_health(tmp_path: Path) -> None:
    ids = tmp_path / "ids"
    ids.write_text("arctic_a0257\n")
    transcription = tmp_path / "transcription"
    transcription.write_text("arctic_a0257 they are coming ashore whoever they are\n")
    output = tmp_path / "model"
    run_bw_training(
        model_dir=FIXTURE / "model",
        output_dir=output,
        features_dir=FIXTURE,
        train_fileids=ids,
        transcription=transcription,
        dictionary=FIXTURE / "dictionary.dict",
        filler_dict=FIXTURE / "filler.dict",
        first_pass_2passvar=False,
        config=BWConfig(pass2var=False, unobserved_gaussian_policy="zero", a_beam=1e-200),
        n_iter=2,
        min_iterations=2,
        checkpoint_iterations=True,
    )
    rows = list_checkpoints(output)
    assert [(r["checkpoint"], r["evaluation_pass"], r["status"]) for r in rows] == [
        (1, 2, "evaluated-healthy"),
        (2, 3, "unevaluated"),
    ]
    checkpoint = output / "iterations" / "01"
    assert rows[0]["evaluation"]["snapshot"] == model_snapshot(checkpoint)
    assert _fingerprint_model(checkpoint) == fingerprint_model(checkpoint)
    assert fingerprint_model(checkpoint) == _sha256_files(
        [checkpoint / name for name in ["mdef", *MODEL_PARAMETER_FILES]]
    )
    # Counts are not scoring input, but are bound to the retained generation.
    (checkpoint / "gauden_counts").write_bytes(b"changed")
    assert list_checkpoints(output)[0]["status"] == "evidence-mismatch"


@pytest.fixture
def retained_model(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    shutil.copytree(FIXTURE / "model", model)
    checkpoint = model / "iterations" / "01"
    checkpoint.mkdir(parents=True)
    for name in ["mdef", *MODEL_PARAMETER_FILES]:
        shutil.copyfile(model / name, checkpoint / name)
    # Counts need not be interpreted or normalized by a raw byte restore.
    (model / "gauden_counts").write_bytes(b"new-counts")
    (checkpoint / "gauden_counts").write_bytes(b"old-counts")
    (model / "provenance.json").write_text("{}")
    (model / ".mdef.012345abcdef.complete").write_text("{}")
    (model / "bw_telemetry.json").write_text(json.dumps({"schema_version": 2, "passes": []}))
    return model


@requires_c_library
def test_restore_is_explicit_reversible_and_invalidates_success(retained_model: Path) -> None:
    model = retained_model
    old_snapshot = model_snapshot(model)
    values = _pstrainc.read_gau(str(model / "variances"))[0]
    assert _pstrainc.write_gau(str(model / "variances"), values + np.float32(1)) == 0
    before = model_snapshot(model)
    evidence = (model / "bw_telemetry.json").read_bytes()
    result = restore_checkpoint(model, 1)
    assert model_snapshot(model) == {
        **old_snapshot,
        "gauden_counts": model_snapshot(model / "iterations" / "01")["gauden_counts"],
    }
    backup = Path(result["backup"])
    assert model_snapshot(backup) == before
    assert (model / "bw_telemetry.json").read_bytes() == evidence
    assert not (model / "provenance.json").exists()
    assert not list(model.glob(".mdef.*.complete"))
    assert (backup / "provenance.json").is_file()
    assert list_checkpoints(model)[0]["status"] == "unevaluated"


@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_restore_failure_rolls_back_data_but_never_success_markers(
    retained_model: Path,
    monkeypatch: pytest.MonkeyPatch,
    after: bool,
    error_type: type[BaseException],
) -> None:
    model = retained_model
    before = model_snapshot(model)
    real_replace = Path.replace
    fired = False

    def fail(source: Path, destination: Path) -> Path:
        nonlocal fired
        if (
            source.name == "gauden_counts"
            and source.parent.name.startswith(".hmm-save-")
            and not fired
        ):
            fired = True
            if after:
                real_replace(source, destination)
            raise error_type("injected publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(error_type):
        restore_checkpoint(model, 1)
    assert model_snapshot(model) == before
    assert not list(model.glob(".mdef.*.complete"))
    assert not (model / "provenance.json").exists()
    assert len(list((model / "recovery-history").iterdir())) == 1


def test_restore_absent_counts_removes_new_generation_and_dry_run_is_read_only(
    retained_model: Path,
) -> None:
    model = retained_model
    (model / "iterations" / "01" / "gauden_counts").unlink()
    before = {p.relative_to(model): p.read_bytes() for p in model.rglob("*") if p.is_file()}
    assert restore_checkpoint(model, 1, dry_run=True)["dry_run"]
    assert {p.relative_to(model): p.read_bytes() for p in model.rglob("*") if p.is_file()} == before
    restore_checkpoint(model, 1)
    assert not (model / "gauden_counts").exists()


def test_legacy_report_is_not_certified_and_unknown_index_refuses(retained_model: Path) -> None:
    model = retained_model
    row = {
        "pass": 2,
        "total_log_likelihood": -100,
        "total_frames": 10,
        "per_frame_log_likelihood": -10,
        "signed_convergence_delta": -1,
        "stop_decision": "converged",
    }
    (model / "bw_telemetry.json").write_text(json.dumps({"schema_version": 2, "passes": [row]}))
    assert list_checkpoints(model)[0]["status"] == "reported-unverified"
    with pytest.raises(ValueError, match="exactly one"):
        restore_checkpoint(model, 2)
    assert (model / "provenance.json").exists()


def test_health_uses_guard_and_finite_statistics_not_likelihood_sign() -> None:
    values = {
        "total_log_lik": -100.0,
        "average": -10.0,
        "frames": 10,
        "utterances": 2,
        "processed": 2,
        "skipped": 1,
        "inputs": 3,
        "max_skip_fraction": 0.5,
    }
    assert evaluation_health(**values)
    assert not evaluation_health(**{**values, "max_skip_fraction": 0.1})
    assert not evaluation_health(**{**values, "frames": 0})
    assert not evaluation_health(**{**values, "average": float("nan")})


def test_checkpoint_cli_requires_explicit_restore_number() -> None:
    args = create_parser().parse_args(
        ["checkpoints", "/tmp/model", "--restore", "2", "--dry-run", "--json"]
    )
    assert args.restore == 2 and args.dry_run and args.json


@requires_c_library
def test_guard_failure_retains_bound_evidence_without_restoring(tmp_path: Path) -> None:
    ids = tmp_path / "ids"
    ids.write_text("arctic_a0257\n")
    transcription = tmp_path / "transcription"
    transcription.write_text("arctic_a0257 they are coming ashore whoever they are\n")
    output = tmp_path / "model"
    with pytest.raises(RuntimeError, match="above configured limit"):
        run_bw_training(
            model_dir=FIXTURE / "model",
            output_dir=output,
            features_dir=FIXTURE,
            train_fileids=ids,
            transcription=transcription,
            dictionary=FIXTURE / "dictionary.dict",
            filler_dict=FIXTURE / "filler.dict",
            first_pass_2passvar=False,
            config=BWConfig(pass2var=False, unobserved_gaussian_policy="zero", a_beam=1e-200),
            n_iter=2,
            min_iterations=2,
            checkpoint_iterations=True,
            exclusion_schedule={2: ["arctic_a0257"]},
            max_skip_fraction=0.1,
        )
    assert list_checkpoints(output)[0]["status"] == "evaluated-unhealthy"
    assert model_snapshot(output) == model_snapshot(output / "iterations" / "01")
    assert not (output / "recovery-history").exists()
    row = json.loads((output / "bw_telemetry.json").read_text())["passes"][-1]
    assert row["stop_decision"] == "failed"
    assert row["input_model_evaluation"]["max_skip_fraction"] == 0.1


def test_interrupted_absent_count_removal_restores_previous_generation(
    retained_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = retained_model
    (model / "iterations" / "01" / "gauden_counts").unlink()
    before = model_snapshot(model)
    real_unlink = Path.unlink
    fired = False

    def interrupt(path: Path, missing_ok: bool = False) -> None:
        nonlocal fired
        real_unlink(path, missing_ok=missing_ok)
        if path == model / "gauden_counts" and not fired:
            fired = True
            raise KeyboardInterrupt("removed before interruption")

    monkeypatch.setattr(Path, "unlink", interrupt)
    with pytest.raises(KeyboardInterrupt):
        restore_checkpoint(model, 1)
    assert model_snapshot(model) == before
    assert not (model / "provenance.json").exists()


def test_state_mapping_mismatch_refuses_before_invalidation(retained_model: Path) -> None:
    (retained_model / "iterations" / "01" / "mdef").write_text("different mapping")
    with pytest.raises(ValueError, match="state mapping"):
        restore_checkpoint(retained_model, 1)
    assert (retained_model / "provenance.json").exists()
    assert not (retained_model / "recovery-history").exists()
