"""Pre-pin numerical-correctness program for the five BASIS choke points."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest

from pstrain.lib import _pstrainc
from pstrain.lib.bw import HMM, BWConfig, BWTrainer
from pstrain.lib.contract_docs import contract_check_fields, contract_check_files, contract_scope
from pstrain.lib.features import read_sphinx_mfc
from pstrain.lib.pipeline import PipelineContext
from pstrain.lib.pipeline.tasks import TARGETS
from pstrain.lib.steps.cd_pipeline import run_init_cd_untied
from pstrain.lib.steps.train import run_bw_training
from tests.clib import requires_c_library
from tests.numeric_harness import (
    GOLDEN,
    GOLDEN_FILEIDS,
    SEED,
    create_project,
    golden_payload,
    read_model_arrays,
    sha256,
    strict_golden_enabled,
    train_golden,
    write_golden_subset,
)

_CHECKPOINT_MODEL_FILES = {
    "mdef",
    "means",
    "variances",
    "mixture_weights",
    "transition_matrices",
    "gauden_counts",
}
_CONTRACT_MODEL_FILES = ("mdef", "means", "variances", "mixture_weights", "transition_matrices")
_CONTRACT_ACCUMULATOR_FILES = ("artifact.json", "gauden_counts", "mixw_counts", "tmat_counts")
_CONTRACT_DISCRETE_FIELDS = (
    "assigned_ids",
    "processed_ids",
    "retried_ids",
    "skipped",
    "total_frames",
    "stop_decision",
)
_CONTRACT_REFERENCE_FILES = (
    "gauden_counts",
    "mdef",
    "means",
    "mixture_weights",
    "transition_matrices",
    "variances",
)
_CONTRACT_TELEMETRY_FIELDS = ("total_frames", "stop_decision")
_M4_FIXTURE = Path(__file__).parent / "fixtures" / "multipron_final_state"


def _cd_rows(path: Path) -> list[tuple[str, str, str, str]]:
    return [tuple(row[:4]) for row in _mdef_rows(path) if row[1] != "-"]


def _runtime_contexts(
    trainer: BWTrainer, mdef_path: Path, transcript: str
) -> set[tuple[str, str, str, str]]:
    rows = list(_mdef_rows(mdef_path))
    states = trainer.inspect_state_seq(transcript)
    return {
        tuple(rows[state.phn][:4])
        for state in states
        if state.m_state == 0 and rows[state.phn][1] != "-"
    }


@pytest.fixture(scope="module")
def flat_project(tmp_path_factory: pytest.TempPathFactory) -> PipelineContext:
    """One fixed flat model shared by the BW-level invariants."""
    return create_project(tmp_path_factory.mktemp("numeric-flat") / "project")


@pytest.fixture(scope="module")
def full_project(tmp_path_factory: pytest.TempPathFactory) -> PipelineContext:
    """One full 1→2→4→8 run shared by split and tree invariants."""
    return create_project(
        tmp_path_factory.mktemp("numeric-full") / "project",
        "cd-8g",
        checkpoint_iterations=True,
    )


def _trainer(ctx: PipelineContext, *, multipron: bool = True) -> BWTrainer:
    model = ctx.model_dir("flat")
    trainer = BWTrainer(
        model / "mdef",
        model / "means",
        model / "variances",
        model / "mixture_weights",
        model / "transition_matrices",
        BWConfig(
            pass2var=True,
            unobserved_gaussian_policy="zero",
            a_beam=1e-200,
            multipron=multipron,
        ),
    )
    trainer.set_dict(ctx.shared_dir / "dictionary.dict", ctx.filler_dict)
    return trainer


@requires_c_library
def test_bw_unobserved_policy_and_raw_variance_artifact(
    flat_project: PipelineContext, tmp_path: Path
) -> None:
    """Zero/retain differ only on empty cells; occupied raw variance is lossless."""
    model = tmp_path / "input"
    shutil.copytree(flat_project.model_dir("flat"), model)
    prior_means_raw, n_cb, n_feat, n_density, veclens = _pstrainc.read_gau(str(model / "means"))
    assert n_feat == 1
    veclen = veclens[0]
    prior_means = prior_means_raw.reshape(n_cb, n_density, veclen)
    prior_vars = _pstrainc.read_gau(str(model / "variances"))[0].reshape(n_cb, n_density, veclen)
    prior_tmat = _pstrainc.read_tmat_counts(str(model / "transition_matrices"))[0]
    # Split the one-density flat model into two identical densities so top-N
    # pruning leaves genuine zero-posterior cells in otherwise active senones.
    prior_means = np.repeat(prior_means, 2, axis=1)
    prior_vars = np.repeat(prior_vars, 2, axis=1)
    n_density = 2
    _pstrainc.write_gau(str(model / "means"), prior_means)
    mixw = _pstrainc.read_mixw_counts(str(model / "mixture_weights"))[0]
    mixw = np.repeat(mixw.reshape(-1, 1, 1), n_density, axis=2)
    # Raw upstream norm output contains occupancy counts, not probabilities.
    # Give every row a distinct, non-unit sum so a runtime-normalized copy is
    # observably different from retaining the serialized input.
    row = np.arange(mixw.shape[0], dtype=np.float32).reshape(-1, 1)
    mixw[:, 0, 0] = np.float32(3.0) + row[:, 0]
    mixw[:, 0, 1] = np.float32(11.0) + np.float32(2.0) * row[:, 0]
    _pstrainc.write_mixw(str(model / "mixture_weights"), mixw)
    # Preserve the transition topology while likewise turning each emitting
    # row into a distinct raw-count row whose sum is well away from one.
    tmat_scale = (
        np.float32(17.0)
        + np.arange(prior_tmat.shape[0], dtype=np.float32)[:, None, None]
        + np.float32(3.0) * np.arange(prior_tmat.shape[1], dtype=np.float32)[None, :, None]
    )
    prior_tmat = prior_tmat * tmat_scale
    tmat_with_exit = np.concatenate(
        [prior_tmat, np.zeros((prior_tmat.shape[0], 1, prior_tmat.shape[2]), dtype=np.float32)],
        axis=1,
    )
    _pstrainc.write_tmat(str(model / "transition_matrices"), tmat_with_exit)
    # Exercise both sides of the evaluation floor on every codebook. Retain
    # must serialize these exact input floats for cells with zero occupancy.
    prior_vars[..., 0::2] = np.float32(5e-5)
    prior_vars[..., 1::2] = np.float32(0.0)
    _pstrainc.write_gau(str(model / "variances"), prior_vars)

    # Public HMM loading remains a probability API even though BW artifacts
    # now store upstream-compatible raw occupancy counts.
    hmm = HMM.load(model)
    np.testing.assert_allclose(hmm.mixw.sum(axis=-1), np.float32(1.0), rtol=1e-6)
    nonzero_tmat_rows = prior_tmat.sum(axis=-1) > 0
    np.testing.assert_allclose(hmm.tmat.sum(axis=-1)[nonzero_tmat_rows], np.float32(1.0), rtol=1e-6)

    # Saving without a pass must be a byte-representation-preserving model
    # container operation for upstream raw-count inputs.
    untouched = BWTrainer(
        model / "mdef",
        model / "means",
        model / "variances",
        model / "mixture_weights",
        model / "transition_matrices",
        BWConfig(pass2var=False, unobserved_gaussian_policy="retain", multipron=False),
    )
    untouched_dir = tmp_path / "untouched"
    untouched_dir.mkdir()
    assert untouched.save(
        untouched_dir / "means",
        untouched_dir / "variances",
        untouched_dir / "mixture_weights",
        untouched_dir / "transition_matrices",
    )
    np.testing.assert_array_equal(
        _pstrainc.read_mixw_counts(str(untouched_dir / "mixture_weights"))[0], mixw
    )
    np.testing.assert_array_equal(
        _pstrainc.read_tmat_counts(str(untouched_dir / "transition_matrices"))[0], prior_tmat
    )

    features = np.full((60, 39), 0.25, dtype=np.float32)
    phones = np.array([0], dtype=np.uint32)
    outputs: dict[str, dict[str, np.ndarray[Any, Any]]] = {}
    accumulators: dict[str, dict[str, np.ndarray[Any, Any]]] = {}

    policies: tuple[Literal["zero", "retain"], ...] = ("zero", "retain")
    for policy in policies:
        trainer = BWTrainer(
            model / "mdef",
            model / "means",
            model / "variances",
            model / "mixture_weights",
            model / "transition_matrices",
            BWConfig(
                pass2var=False,
                unobserved_gaussian_policy=policy,
                a_beam=1e-200,
                topn=n_density - 1,
                multipron=False,
            ),
        )
        assert trainer.process_utterance(features, phones)
        # Two identical updates make raw occupancy row sums observably differ
        # from normalized probabilities even for a one-state path.
        assert trainer.process_utterance(features, phones)
        assert trainer.count_active_fallback_senones() == 0
        accum_dir = tmp_path / f"{policy}-accum"
        trainer.dump_accumulators(accum_dir)
        accumulators[policy] = {
            "mixture_weights": _pstrainc.read_mixw_counts(str(accum_dir / "mixw_counts"))[0],
            "transition_matrices": _pstrainc.read_tmat_counts(str(accum_dir / "tmat_counts"))[0],
        }
        assert trainer.normalize()
        out = tmp_path / policy
        out.mkdir()
        assert trainer.save(
            out / "means",
            out / "variances",
            out / "mixture_weights",
            out / "transition_matrices",
        )
        outputs[policy] = read_model_arrays(out)
        outputs[policy]["means"] = outputs[policy]["means"].reshape(n_cb, n_density, veclen)
        outputs[policy]["variances"] = outputs[policy]["variances"].reshape(n_cb, n_density, veclen)

    # A pass with mixw/tmat re-estimation disabled must also carry the raw
    # incoming representation through normalization and serialization.
    disabled = BWTrainer(
        model / "mdef",
        model / "means",
        model / "variances",
        model / "mixture_weights",
        model / "transition_matrices",
        BWConfig(
            pass2var=False,
            unobserved_gaussian_policy="retain",
            a_beam=1e-200,
            topn=n_density - 1,
            mixw_reest=False,
            tmat_reest=False,
            multipron=False,
        ),
    )
    assert disabled.process_utterance(features, phones)
    assert disabled.normalize()
    disabled_dir = tmp_path / "disabled-reest"
    disabled_dir.mkdir()
    assert disabled.save(
        disabled_dir / "means",
        disabled_dir / "variances",
        disabled_dir / "mixture_weights",
        disabled_dir / "transition_matrices",
    )
    np.testing.assert_array_equal(
        _pstrainc.read_mixw_counts(str(disabled_dir / "mixture_weights"))[0], mixw
    )
    np.testing.assert_array_equal(
        _pstrainc.read_tmat_counts(str(disabled_dir / "transition_matrices"))[0], prior_tmat
    )

    occupied = np.any(outputs["zero"]["means"] != 0.0, axis=-1)
    assert occupied.any()
    empty = ~occupied
    assert empty.any(), "fixture must contain a genuinely unobserved codebook"

    # Upstream norm's fresh output allocation leaves empty Gaussian cells zero.
    assert np.count_nonzero(outputs["zero"]["means"][empty]) == 0
    assert np.count_nonzero(outputs["zero"]["variances"][empty]) == 0
    np.testing.assert_array_equal(outputs["retain"]["means"][empty], prior_means[empty])
    np.testing.assert_array_equal(outputs["retain"]["variances"][empty], prior_vars[empty])
    assert np.any(outputs["retain"]["variances"][empty] == np.float32(5e-5))
    assert np.any(outputs["retain"]["variances"][empty] == np.float32(0.0))

    # Upstream norm writes fresh accumulators directly, so unobserved mixture
    # and transition rows remain zero. Retain preserves their input values.
    zero_mixw = outputs["zero"]["mixture_weights"]
    retain_mixw = outputs["retain"]["mixture_weights"]
    mixw_empty = zero_mixw.sum(axis=-1) == 0
    assert mixw_empty.any()
    assert np.count_nonzero(zero_mixw[mixw_empty]) == 0
    np.testing.assert_array_equal(retain_mixw[mixw_empty], mixw[mixw_empty])
    mixw_occupied = ~mixw_empty
    np.testing.assert_array_equal(zero_mixw[mixw_occupied], retain_mixw[mixw_occupied])
    np.testing.assert_array_equal(zero_mixw, accumulators["zero"]["mixture_weights"])

    zero_tmat = outputs["zero"]["transition_matrices"]
    retain_tmat = outputs["retain"]["transition_matrices"]
    tmat_empty = zero_tmat.sum(axis=-1) == 0
    assert tmat_empty.any()
    assert np.count_nonzero(zero_tmat[tmat_empty]) == 0
    np.testing.assert_array_equal(retain_tmat[tmat_empty], prior_tmat[tmat_empty])
    tmat_occupied = ~tmat_empty
    np.testing.assert_array_equal(zero_tmat[tmat_occupied], retain_tmat[tmat_occupied])
    np.testing.assert_array_equal(zero_tmat, accumulators["zero"]["transition_matrices"])

    # Every occupied cell saw the same value, so direct one-pass V/N-E[x]^2
    # is exactly zero in float32.  The saved artifact must not contain the
    # evaluation-time 1e-4 floor or a reciprocal round-trip perturbation.
    direct = np.float32(0.25 * 0.25) - np.float32(0.25) * np.float32(0.25)
    np.testing.assert_array_equal(outputs["zero"]["variances"][occupied], direct)
    np.testing.assert_array_equal(
        outputs["zero"]["variances"][occupied], outputs["retain"]["variances"][occupied]
    )
    np.testing.assert_array_equal(
        outputs["zero"]["means"][occupied], outputs["retain"]["means"][occupied]
    )

    # Reloading applies the evaluation floor.  The phone used above reaches
    # only occupied states, so both policies must produce the same score.
    scores = {}
    for policy in policies:
        out = tmp_path / policy
        evaluator = BWTrainer(
            model / "mdef",
            out / "means",
            out / "variances",
            out / "mixture_weights",
            out / "transition_matrices",
            BWConfig(
                pass2var=False,
                unobserved_gaussian_policy=policy,
                a_beam=1e-200,
                multipron=False,
            ),
        )
        assert evaluator.process_utterance(features, phones)
        scores[policy] = evaluator.get_stats().total_log_lik
    assert scores["zero"] == scores["retain"]

    # The training loader matches upstream senone loading: an all-zero row is
    # floored cell-wise and normalized for evaluation. Saving without a new BW
    # pass preserves the serialized raw row rather than leaking that runtime
    # normalization back into the artifact.
    reloaded = BWTrainer(
        model / "mdef",
        tmp_path / "zero" / "means",
        tmp_path / "zero" / "variances",
        tmp_path / "zero" / "mixture_weights",
        tmp_path / "zero" / "transition_matrices",
        BWConfig(
            pass2var=False,
            unobserved_gaussian_policy="zero",
            a_beam=1e-200,
            multipron=False,
        ),
    )
    reloaded_dir = tmp_path / "zero-reloaded"
    reloaded_dir.mkdir()
    assert reloaded.save(
        reloaded_dir / "means",
        reloaded_dir / "variances",
        reloaded_dir / "mixture_weights",
        reloaded_dir / "transition_matrices",
    )
    reloaded_mixw = _pstrainc.read_mixw_counts(str(reloaded_dir / "mixture_weights"))[0]
    assert np.count_nonzero(reloaded_mixw[mixw_empty]) == 0


@requires_c_library
def test_feature_frames_finiteness_and_golden_checksum(flat_project: PipelineContext) -> None:
    """Choke point A: portable feature shape/envelope, plus optional bytes."""
    expected = json.loads(GOLDEN.read_text())["feature"]
    paths = sorted(flat_project.features_dir.glob("*.mfc"))
    assert len(paths) == 10
    for path in paths:
        features = read_sphinx_mfc(path)
        assert features.shape[0] > 0
        assert features.shape[1] == 13
        assert np.isfinite(features).all()
    anchor = flat_project.features_dir / f"{expected['fileid']}.mfc"
    values = read_sphinx_mfc(anchor)
    assert values.shape[0] == expected["frames"]
    assert values.size == expected["values"]
    observed = [values.min(), values.max(), values.mean(), values.std(), np.linalg.norm(values)]
    reference = [
        expected["minimum"],
        expected["maximum"],
        expected["mean"],
        expected["stddev"],
        expected["l2_norm"],
    ]
    tolerance = json.loads(GOLDEN.read_text())["feature_tolerance"]
    np.testing.assert_allclose(observed, reference, **tolerance)
    if strict_golden_enabled():
        assert sha256(anchor) == expected["sha256"]


@requires_c_library
def test_bw_golden_trajectory_and_accounting(
    flat_project: PipelineContext, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Choke points B/C: BW numerics and utterance conservation cannot drift."""
    expected = json.loads(GOLDEN.read_text())
    output_dir = tmp_path / "trained"
    caplog.set_level("INFO", logger="pstrain.lib.steps.train")
    pinned = replace(
        flat_project,
        train=replace(flat_project.train, optional_final_silence=False),
    )
    result = train_golden(pinned, output_dir)
    actual = golden_payload(pinned, result)
    tolerance = expected["strict_tolerance" if strict_golden_enabled() else "portable_tolerance"]
    assert len(actual["trajectory"]) == len(expected["trajectory"]) == 3
    for observed, golden in zip(actual["trajectory"], expected["trajectory"], strict=True):
        for key in (
            "iteration",
            "frames",
            "input_utts",
            "processed_utts",
            "retried_utts",
            "skipped_utts",
        ):
            assert observed[key] == golden[key]
        assert (
            observed["processed_utts"] + observed["retried_utts"] + observed["skipped_utts"]
            == observed["input_utts"]
        )
        assert observed["skipped_utts"] == 0
        np.testing.assert_allclose(
            [observed["total_log_lik"], observed["avg_log_prob"]],
            [golden["total_log_lik"], golden["avg_log_prob"]],
            **tolerance,
        )
        if golden["per_frame_delta"] is None:
            assert observed["per_frame_delta"] is None
        else:
            assert observed["per_frame_delta"] == pytest.approx(
                golden["per_frame_delta"], rel=tolerance["rtol"], abs=tolerance["atol"]
            )

    artifact = json.loads((output_dir / "bw_telemetry.json").read_text(encoding="utf-8"))
    assert json.loads(json.dumps(artifact, allow_nan=False)) == artifact
    assert artifact["schema_version"] == 2
    rows = artifact["passes"]
    assert len(rows) == 3
    assert [row["pass"] for row in rows] == [1, 2, 3]
    assert [row["stop_decision"] for row in rows] == ["continued", "continued", "cap"]
    assert rows[0]["signed_convergence_delta"] is None
    for row, observed in zip(rows, actual["trajectory"], strict=True):
        assert row["total_frames"] == observed["frames"]
        assert row["total_log_likelihood"] == observed["total_log_lik"]
        assert row["per_frame_log_likelihood"] == observed["avg_log_prob"]
        assert row["signed_convergence_delta"] == observed["per_frame_delta"]

    log_rows = [
        record.message for record in caplog.records if record.message.startswith("BW telemetry:")
    ]
    assert len(log_rows) == 3
    assert all("total_log_likelihood=" in row and "stop=" in row for row in log_rows)
    assert log_rows[-1].endswith("stop=cap")


@requires_c_library
def test_bw_exclusion_schedule_targets_named_passes_and_wildcard(
    flat_project: PipelineContext, tmp_path: Path
) -> None:
    """Scheduled utterances are accounted for without reaching accumulation."""
    fileids, transcription = write_golden_subset(flat_project)
    common: dict[str, Any] = {
        "model_dir": flat_project.model_dir("flat"),
        "features_dir": flat_project.features_dir,
        "train_fileids": fileids,
        "transcription": transcription,
        "dictionary": flat_project.shared_dir / "dictionary.dict",
        "filler_dict": flat_project.filler_dict,
        "first_pass_2passvar": False,
        "n_iter": 1,
        "max_skip_fraction": 1.0,
        "config": BWConfig(pass2var=True, unobserved_gaussian_policy="zero", a_beam=1e-200),
    }
    cases = (
        ("named", {1: ["arctic_a0001"], 2: ["arctic_a0002"]}),
        ("wildcard", {"*": ["arctic_a0003"]}),
    )
    for name, schedule in cases:
        output_dir = tmp_path / name
        result = run_bw_training(output_dir=output_dir, exclusion_schedule=schedule, **common)

        row = result.trajectory[0]
        assert (row.input_utts, row.processed_utts, row.retried_utts) == (3, 2, 0)
        assert (row.skipped_utts, row.excluded_by_schedule) == (1, 1)
        telemetry = json.loads((output_dir / "bw_telemetry.json").read_text())
        assert telemetry["schema_version"] == 2
        assert telemetry["passes"][0]["accounting"]["skip_reasons"] == {
            "alignment_failure": 0,
            "exception": 0,
            "excluded_by_schedule": 1,
            "feature_dimension": 0,
            "feature_not_found": 0,
            "transcript_not_found": 0,
        }


@requires_c_library
def test_per_utterance_aggregation_conserves_totals(flat_project: PipelineContext) -> None:
    """Choke point B: utterance contributions sum to the batch accumulator."""
    transcripts = {
        line.split(maxsplit=1)[0]: line.split(maxsplit=1)[1]
        for line in (flat_project.project_dir / "etc" / "all.transcription")
        .read_text()
        .splitlines()
    }
    individual = []
    for fileid in GOLDEN_FILEIDS:
        trainer = _trainer(flat_project)
        mfcc = read_sphinx_mfc(flat_project.features_dir / f"{fileid}.mfc")
        assert trainer.process_utterance_mfcc(mfcc, f"<s> {transcripts[fileid]} </s>")
        individual.append(trainer.get_stats())

    combined = _trainer(flat_project)
    for fileid in GOLDEN_FILEIDS:
        mfcc = read_sphinx_mfc(flat_project.features_dir / f"{fileid}.mfc")
        assert combined.process_utterance_mfcc(mfcc, f"<s> {transcripts[fileid]} </s>")
    total = combined.get_stats()
    assert total.total_utts == sum(item.total_utts for item in individual)
    assert total.total_frames == sum(item.total_frames for item in individual)
    assert total.total_log_lik == pytest.approx(
        sum(item.total_log_lik for item in individual), rel=1e-14, abs=1e-8
    )


@requires_c_library
def test_real_training_retry_is_accounted_once_and_conserves_stats(
    flat_project: PipelineContext, tmp_path: Path
) -> None:
    """A recovered real retry is one input and one accumulated contribution."""
    fileid = "arctic_a0001"
    fileids = tmp_path / "retry.fileids"
    transcription = tmp_path / "retry.transcription"
    fileids.write_text(f"{fileid}\n")
    text = "author of the danger trail philip steels etc"
    transcription.write_text(f"{fileid} {text}\n")
    result = run_bw_training(
        flat_project.model_dir("flat"),
        tmp_path / "retried-model",
        flat_project.features_dir,
        fileids,
        transcription,
        flat_project.shared_dir / "dictionary.dict",
        first_pass_2passvar=True,
        filler_dict=flat_project.filler_dict,
        n_iter=1,
        config=BWConfig(pass2var=True, unobserved_gaussian_policy="zero", a_beam=1e-1),
        retry_beam_factor=1e199,
    )
    row = result.trajectory[0]
    assert (row.input_utts, row.processed_utts, row.retried_utts, row.skipped_utts) == (1, 0, 1, 0)
    assert row.processed_utts + row.retried_utts + row.skipped_utts == row.input_utts

    direct = _trainer(flat_project)
    mfcc = read_sphinx_mfc(flat_project.features_dir / f"{fileid}.mfc")
    assert direct.process_utterance_mfcc(mfcc, f"<s> {text} </s>")
    expected = direct.get_stats()
    assert (row.frames, result.final_frames, result.final_utts) == (
        expected.total_frames,
        expected.total_frames,
        expected.total_utts,
    )
    assert row.total_log_lik == pytest.approx(expected.total_log_lik, rel=1e-14, abs=1e-8)


def _assert_bw_model(model_dir: Path) -> None:
    arrays = read_model_arrays(model_dir)
    for name, values in arrays.items():
        assert np.isfinite(values).all(), name
    # Gaussian artifacts contain direct, unfloored normalization output.  Mixw
    # and tmat artifacts follow upstream and retain raw BW accumulators; the
    # next engine load normalizes and applies the evaluation floors.
    assert (np.maximum(arrays["variances"], np.float32(1e-4)) >= 1e-4).all()
    mixw_sums = arrays["mixture_weights"].sum(axis=-1)
    tmat_sums = arrays["transition_matrices"].sum(axis=-1)
    assert np.any(mixw_sums > 1.0)
    assert np.any(tmat_sums > 1.0)


@requires_c_library
def test_iteration_checkpoints_are_opt_in_and_replace_stale_passes(
    flat_project: PipelineContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normal training emits no checkpoints; the diagnostic mode is stage-clean."""
    fileids, transcription = write_golden_subset(flat_project)
    kwargs = {
        "model_dir": flat_project.model_dir("flat"),
        "features_dir": flat_project.features_dir,
        "train_fileids": fileids,
        "transcription": transcription,
        "dictionary": flat_project.shared_dir / "dictionary.dict",
        "filler_dict": flat_project.filler_dict,
        "min_iterations": 4,
        "config": BWConfig(pass2var=True, unobserved_gaussian_policy="zero", a_beam=1e-200),
        "first_pass_2passvar": False,
    }

    monkeypatch.delenv("PSTRAIN_BW_CHECKPOINTS", raising=False)
    default_output = tmp_path / "default-model"
    run_bw_training(output_dir=default_output, n_iter=1, **kwargs)
    assert not (default_output / "iterations").exists()

    enabled_output = tmp_path / "checkpointed-model"
    stale = enabled_output / "iterations" / "99"
    stale.mkdir(parents=True)
    (stale / "stale").write_text("stale")
    monkeypatch.setenv("PSTRAIN_BW_CHECKPOINTS", "1")
    run_bw_training(output_dir=enabled_output, n_iter=2, **kwargs)
    assert [path.name for path in sorted((enabled_output / "iterations").iterdir())] == [
        "01",
        "02",
    ]
    for checkpoint in (enabled_output / "iterations").iterdir():
        assert {path.name for path in checkpoint.iterdir()} == set(_CHECKPOINT_MODEL_FILES)


@requires_c_library
def test_updates_and_split_schedule_preserve_invariants(full_project: PipelineContext) -> None:
    """Choke points D/E: each pass/split stays valid at exactly 1→2→4→8."""
    exercised_specs = TARGETS[
        : next(i for i, spec in enumerate(TARGETS) if spec.name == "cd-8g") + 1
    ]
    bw_stages = [
        spec.name for spec in exercised_specs if spec.kind in {"ci", "cd"} and spec.name != "flat"
    ]
    senones: int | None = None
    for stage in bw_stages:
        model = full_project.model_dir(stage)
        checkpoints = sorted((model / "iterations").iterdir())
        assert checkpoints, f"no per-pass checkpoints for {stage}"
        for checkpoint in checkpoints:
            _assert_bw_model(checkpoint)
        _assert_bw_model(model)

    for density in (1, 2, 4, 8):
        model = full_project.model_dir(f"cd-{density}g")
        mixw, n_mixw, _, actual_density = _pstrainc.read_mixw_counts(str(model / "mixture_weights"))
        assert actual_density == density
        if senones is None:
            senones = n_mixw
        assert n_mixw == senones
        counts, n_cb, _, count_density = _pstrainc.read_dnom(str(model / "gauden_counts"))
        assert (n_cb, count_density) == (n_mixw, density)
        observed = counts > 0
        means, n_gau, _, gau_density, veclens = _pstrainc.read_gau(str(model / "means"))
        variances = _pstrainc.read_gau(str(model / "variances"))[0]
        assert (n_gau, gau_density) == (n_cb, density)
        means = means.reshape(n_cb, 1, density, veclens[0])
        variances = variances.reshape(n_cb, 1, density, veclens[0])
        # Empty densities are expected in this sparse fixture. Under the
        # parity-stage ZERO policy their artifact cells must be exact zeros.
        assert np.count_nonzero(means[~observed]) == 0
        assert np.count_nonzero(variances[~observed]) == 0


def _ci_state_ids(mdef: Path, phone: str) -> list[int]:
    for line in mdef.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[:4] == [phone, "-", "-", "-"]:
            return [int(value) for value in fields[6:] if value != "N"]
    raise AssertionError(f"missing CI phone {phone}")


@requires_c_library
def test_multipron_second_variant_receives_occupancy(
    flat_project: PipelineContext, tmp_path: Path
) -> None:
    """M2: a matching second pronunciation changes stable phone occupancy."""
    engineered = tmp_path / "engineered.dict"
    base = (flat_project.shared_dir / "dictionary.dict").read_text()
    base = re.sub(r"^author .*$", "author K", base, flags=re.MULTILINE)
    engineered.write_text(base + "author(2) AO TH ER\n")
    mfcc = read_sphinx_mfc(flat_project.features_dir / "arctic_a0001.mfc")

    occupancies: dict[bool, np.ndarray[Any, Any]] = {}
    for enabled in (False, True):
        trainer = _trainer(flat_project, multipron=enabled)
        trainer.set_dict(engineered, flat_project.filler_dict)
        assert trainer.process_utterance_mfcc(mfcc, "<s> author </s>")
        counts_path = tmp_path / f"counts-{enabled}"
        assert trainer.save_density_counts(counts_path)
        occupancies[enabled] = _pstrainc.read_dnom(str(counts_path))[0]

    ao_states = _ci_state_ids(flat_project.model_dir("flat") / "mdef", "AO")
    off = float(occupancies[False][ao_states].sum())
    on = float(occupancies[True][ao_states].sum())
    assert off == 0.0
    assert on > 1.0


@requires_c_library
def test_multipron_variants_share_utterance_final_state(
    flat_project: PipelineContext, tmp_path: Path
) -> None:
    """M4: every last-word pronunciation can terminate through one </s>."""
    engineered = tmp_path / "terminal-variants.dict"
    base = (flat_project.shared_dir / "dictionary.dict").read_text()
    # The recording says the normal AO TH ER pronunciation. Put a deliberately
    # bad pronunciation last: before M4, context expansion duplicated </s> and
    # made only this last (K-ending) branch lead to n_state - 1.
    engineered.write_text(base + f"author(2) {' '.join(['K'] * 150)}\n")
    mfcc = read_sphinx_mfc(flat_project.features_dir / "arctic_a0001.mfc")

    trainer = _trainer(flat_project)
    trainer.set_dict(engineered, flat_project.filler_dict)
    states = trainer.inspect_state_seq("<s> author </s>")
    terminal_exits = [index for index, state in enumerate(states) if not state.next_state]
    assert terminal_exits == [len(states) - 1]

    terminal_exit = terminal_exits[0]
    terminal_phone = states[terminal_exit].phn
    terminal_starts = [
        index
        for index, state in enumerate(states)
        if state.phn == terminal_phone and state.m_state == 0
    ]
    assert len(terminal_starts) == 2  # initial <s> and one terminal </s>
    terminal_start = terminal_starts[-1]
    branch_count = 2  # author and author(2)
    fan_in = [
        predecessor
        for predecessor in states[terminal_start].prior_state
        if states[predecessor].phn != terminal_phone
    ]
    assert len(fan_in) == branch_count

    reachable = {terminal_exit}
    pending = [terminal_exit]
    while pending:
        state_id = pending.pop()
        for predecessor in states[state_id].prior_state:
            if predecessor not in reachable:
                reachable.add(predecessor)
                pending.append(predecessor)
    assert reachable == set(range(len(states)))

    assert trainer.process_utterance_mfcc(mfcc, "<s> author </s>")
    assert not trainer.final_state_not_reached


@requires_c_library
def test_cd_variant_boundaries_expand_both_triphone_contexts(
    full_project: PipelineContext,
) -> None:
    """M4b: fan-out and fan-in both select the matching untied triphone."""
    model = full_project.model_dir("cd-untied")
    trainer = BWTrainer(
        model / "mdef",
        model / "means",
        model / "variances",
        model / "mixture_weights",
        model / "transition_matrices",
        BWConfig(
            pass2var=True,
            unobserved_gaussian_policy="zero",
            a_beam=1e-200,
            topn=1,
            multipron=True,
        ),
    )
    trainer.set_dict(full_project.shared_dir / "dictionary.dict", full_project.filler_dict)

    mdef_rows = list(_mdef_rows(model / "mdef"))
    expected_boundary_models = {
        ("AH", "SIL", "AH", "s"),
        ("AH", "SIL", "AE", "s"),
        ("EY", "SIL", "AH", "s"),
        ("EY", "SIL", "AE", "s"),
    }
    assert expected_boundary_models <= {tuple(row[:4]) for row in mdef_rows}

    # `a` has AH/EY variants and `and` has AH/AE initial phones.  Thus the
    # word boundary requires two right-context copies of each `a` variant and
    # two left-context copies of each `and` variant.  There are 14 phone HMMs:
    # two shared SILs, four `a` copies, four first-phone `and` copies, and the
    # four remaining phones in the two `and` variants.
    states = trainer.inspect_state_seq("<s> a and </s>")
    phone_starts = [index for index, state in enumerate(states) if state.m_state == 0]
    assert len(phone_starts) == 14
    graph_models = [tuple(mdef_rows[states[index].phn][:4]) for index in phone_starts]
    assert {model for model in graph_models if model in expected_boundary_models} == (
        expected_boundary_models
    )
    assert all(graph_models.count(model) == 1 for model in expected_boundary_models)
    terminal_exits = [index for index, state in enumerate(states) if not state.next_state]
    assert terminal_exits == [len(states) - 1]


@requires_c_library
@pytest.mark.parametrize(
    "words",
    [
        "ONE",
        "LEFT RIGHT",
        "LEFT <sil> RIGHT",
        "LEFT <sil>",
        "DUP RIGHT",
    ],
    ids=[
        "one-word-single-phone",
        "variant-boundary",
        "filler-between-lexical",
        "variant-adjacent-filler",
        "duplicate-variants",
    ],
)
@pytest.mark.parametrize("multipron", [False, True], ids=["linear", "multipron"])
def test_inventory_policy_equals_runtime_contexts_by_mode(
    full_project: PipelineContext,
    tmp_path: Path,
    words: str,
    multipron: bool,
) -> None:
    """Each supported inventory policy matches its BW runtime domain."""
    from pstrain.lib.mdef import generate_alltriphones_mdef, generate_untied_mdef

    phones = tmp_path / "phones.txt"
    phones.write_text((full_project.shared_dir / "phoneset.txt").read_text())
    dictionary = tmp_path / "dictionary.dict"
    dictionary.write_text(
        "ONE G\nLEFT L K\nLEFT(2) M IY\nRIGHT P N\nRIGHT(2) D OW\nDUP B AE\nDUP(2) B AE\n"
    )
    filler = tmp_path / "filler.dict"
    filler.write_text("<s> SIL\n</s> SIL\n<sil> SIL\n")
    transcript_text = f"<s> {words} </s>"
    transcript = tmp_path / "train.transcription"
    transcript.write_text(f"{transcript_text} (utt1)\n")
    inventory_mdef = tmp_path / "inventory.mdef"
    runtime_mdef = tmp_path / "runtime.mdef"
    generate_untied_mdef(
        phones,
        dictionary,
        transcript,
        inventory_mdef,
        filler_dict=filler,
        inventory_policy="transcript-reachable" if multipron else "linear",
        multipron=multipron,
    )
    generate_alltriphones_mdef(phones, dictionary, runtime_mdef, filler_dict=filler)
    runtime_model = tmp_path / "runtime-model"
    run_init_cd_untied(full_project.model_dir("ci-1g"), runtime_mdef, runtime_model)
    trainer = BWTrainer(
        runtime_model / "mdef",
        runtime_model / "means",
        runtime_model / "variances",
        runtime_model / "mixture_weights",
        runtime_model / "transition_matrices",
        BWConfig(pass2var=True, unobserved_gaussian_policy="zero", multipron=multipron),
    )
    trainer.set_dict(dictionary, filler)

    inventory = set(_cd_rows(inventory_mdef))
    runtime = _runtime_contexts(trainer, runtime_mdef, transcript_text)
    assert inventory == runtime
    if words == "LEFT RIGHT":
        second_variant = {row for row in runtime if row[0] in {"M", "IY", "D", "OW"}}
        assert bool(second_variant) is multipron


@requires_c_library
def test_complete_cd_inventory_leaves_ci_fallback_accumulators_zero(
    full_project: PipelineContext, tmp_path: Path
) -> None:
    """A complete CD runtime never enters the CI fallback prior path."""
    from pstrain.lib.mdef import generate_alltriphones_mdef

    model = tmp_path / "complete-model"
    complete_mdef = tmp_path / "complete.mdef"
    generate_alltriphones_mdef(
        full_project.shared_dir / "phoneset.txt",
        full_project.shared_dir / "dictionary.dict",
        complete_mdef,
        filler_dict=full_project.filler_dict,
    )
    run_init_cd_untied(full_project.model_dir("ci-2g"), complete_mdef, model)
    trainer = BWTrainer(
        model / "mdef",
        model / "means",
        model / "variances",
        model / "mixture_weights",
        model / "transition_matrices",
        BWConfig(
            pass2var=True,
            unobserved_gaussian_policy="zero",
            a_beam=1e-200,
            multipron=True,
        ),
    )
    trainer.set_dict(full_project.shared_dir / "dictionary.dict", full_project.filler_dict)
    mfcc = read_sphinx_mfc(full_project.features_dir / "arctic_a0001.mfc")
    assert trainer.process_utterance_mfcc(mfcc, "<s> a and </s>")
    assert trainer.count_active_fallback_senones() == 0
    non_filler_ci_senones = [
        int(state)
        for row in _mdef_rows(complete_mdef)
        if row[1] == "-" and row[4] != "filler"
        for state in row[6:-1]
    ]
    assert non_filler_ci_senones
    assert all(not trainer.fallback_senone_active(senone) for senone in non_filler_ci_senones)


@requires_c_library
def test_withheld_context_uses_live_ci_fallback_across_passes(
    full_project: PipelineContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An inventory miss trains through CI fallback for every re-estimation pass."""
    from pstrain.lib.mdef import generate_untied_mdef
    from pstrain.lib.steps.cd_pipeline import run_init_cd_untied

    transcript = tmp_path / "fallback.transcription"
    transcript.write_text("<s> a and </s> (arctic_a0001)\n")
    fileids = tmp_path / "fallback.fileids"
    fileids.write_text("arctic_a0001\n")
    linear_mdef = tmp_path / "linear.mdef"
    generate_untied_mdef(
        full_project.shared_dir / "phoneset.txt",
        full_project.shared_dir / "dictionary.dict",
        transcript,
        linear_mdef,
        filler_dict=full_project.filler_dict,
        inventory_policy="linear",
    )
    rows = {tuple(row[:4]) for row in _mdef_rows(linear_mdef)}
    withheld = ("EY", "SIL", "AE", "s")
    assert withheld not in rows

    initial = tmp_path / "initial"
    run_init_cd_untied(full_project.model_dir("ci-2g"), linear_mdef, initial)
    probe = BWTrainer(
        initial / "mdef",
        initial / "means",
        initial / "variances",
        initial / "mixture_weights",
        initial / "transition_matrices",
        BWConfig(pass2var=True, unobserved_gaussian_policy="zero", multipron=True),
    )
    probe.set_dict(full_project.shared_dir / "dictionary.dict", full_project.filler_dict)
    states = probe.inspect_state_seq("<s> a and </s>")
    ci_rows = [row for row in _mdef_rows(linear_mdef) if row[1] == "-"]
    sil_id = next(index for index, row in enumerate(ci_rows) if row[0] == "SIL")
    fallback_senones = np.asarray(
        sorted(
            {
                state.mixw
                for state in states
                if state.mixw != 0xFFFFFFFF and state.phn < 36 and state.phn != sil_id
            }
        )
    )
    assert fallback_senones.size
    fallback_raw_mixw = _pstrainc.read_mixw_counts(str(initial / "mixture_weights"))[0]
    fallback_scale = (
        np.float32(29.0) + np.arange(fallback_raw_mixw.shape[0], dtype=np.float32)[:, None, None]
    )
    fallback_raw_mixw = fallback_raw_mixw * fallback_scale
    _pstrainc.write_mixw(str(initial / "mixture_weights"), fallback_raw_mixw)
    assert np.all(fallback_raw_mixw.sum(axis=-1) > np.float32(1.0))
    initial_arrays = read_model_arrays(initial)
    output = tmp_path / "trained"
    monkeypatch.setenv("PSTRAIN_BW_CHECKPOINTS", "1")
    result = run_bw_training(
        initial,
        output,
        full_project.features_dir,
        fileids,
        transcript,
        full_project.shared_dir / "dictionary.dict",
        filler_dict=full_project.filler_dict,
        n_iter=3,
        min_iterations=3,
        convergence_ratio=-1e30,
        first_pass_2passvar=False,
        config=BWConfig(
            pass2var=True,
            unobserved_gaussian_policy="zero",
            a_beam=1e-200,
            b_beam=1e-200,
            multipron=True,
            optional_final_silence=False,
        ),
    )
    assert result.iterations == 3
    assert result.total_skipped == 0
    assert all(row.skipped_utts == 0 for row in result.trajectory)

    previous = initial_arrays
    for iteration in range(1, 4):
        checkpoint = output / "iterations" / f"{iteration:02d}"
        checkpoint_arrays = read_model_arrays(checkpoint)
        posterior_counts = _pstrainc.read_dnom(str(checkpoint / "gauden_counts"))[0]
        fallback_counts = posterior_counts.reshape(-1, 2)[fallback_senones]
        positive_mass = fallback_counts.sum(axis=1)
        positive_mass = positive_mass[positive_mass > 0]
        assert positive_mass.min() < 1.1  # comparable to the unit-mass prior
        zero_mass_senones = fallback_senones[fallback_counts.sum(axis=1) == 0]
        assert zero_mass_senones.size, "fixture must include an unselected fallback branch"
        for name in ("means", "variances", "mixture_weights"):
            current_values = checkpoint_arrays[name].reshape(
                -1, 2, *(() if name == "mixture_weights" else (39,))
            )[fallback_senones]
            previous_values = previous[name].reshape(
                -1, 2, *(() if name == "mixture_weights" else (39,))
            )[fallback_senones]
            assert np.any(current_values != previous_values), (iteration, name)
            if name == "mixture_weights":
                zero_current = checkpoint_arrays[name].reshape(-1, 2)[zero_mass_senones]
                zero_previous = previous[name].reshape(-1, 2)[zero_mass_senones]
                assert np.all(zero_previous.sum(axis=-1) > np.float32(1.0))
                np.testing.assert_array_equal(zero_current, zero_previous)
            else:
                zero_current = checkpoint_arrays[name].reshape(-1, 2, 39)[zero_mass_senones]
                zero_previous = previous[name].reshape(-1, 2, 39)[zero_mass_senones]
                np.testing.assert_array_equal(zero_current, zero_previous)
        previous = checkpoint_arrays

    mfcc = read_sphinx_mfc(full_project.features_dir / "arctic_a0001.mfc")
    for iteration in range(1, 4):
        checkpoint = output / "iterations" / f"{iteration:02d}"
        trainer = BWTrainer(
            checkpoint / "mdef",
            checkpoint / "means",
            checkpoint / "variances",
            checkpoint / "mixture_weights",
            checkpoint / "transition_matrices",
            BWConfig(
                pass2var=True,
                unobserved_gaussian_policy="zero",
                a_beam=1e-200,
                multipron=True,
            ),
        )
        trainer.set_dict(full_project.shared_dir / "dictionary.dict", full_project.filler_dict)
        assert trainer.process_utterance_mfcc(mfcc, "<s> a and </s>")
        assert not trainer.final_state_not_reached


@requires_c_library
@pytest.mark.parametrize("fileid", ["arctic_a0257", "arctic_a0336", "arctic_b0424"])
def test_m4_real_utterances_reach_shared_final_state(
    fileid: str,
) -> None:
    """M4: representative shared and pstrain-only SLT failures train."""
    model = _M4_FIXTURE / "model"
    transcripts = {
        fields[0]: fields[1]
        for line in (_M4_FIXTURE / "transcription.txt").read_text().splitlines()
        if (fields := line.split(maxsplit=1))
    }
    trainer = BWTrainer(
        model / "mdef",
        model / "means",
        model / "variances",
        model / "mixture_weights",
        model / "transition_matrices",
        BWConfig(
            pass2var=True,
            unobserved_gaussian_policy="zero",
            a_beam=1e-90,
            multipron=True,
        ),
    )
    trainer.set_dict(_M4_FIXTURE / "dictionary.dict", _M4_FIXTURE / "filler.dict")
    mfcc = read_sphinx_mfc(_M4_FIXTURE / f"{fileid}.mfc")

    assert trainer.process_utterance_mfcc(mfcc, f"<s> {transcripts[fileid]} </s>")
    assert not trainer.final_state_not_reached


def _mdef_rows(path: Path) -> Iterator[list[str]]:
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 8 and not fields[0].isdigit() and not fields[0].startswith("#"):
            yield fields


def _tree_leaf(
    tree_path: Path, questions: dict[str, set[str]], base: str, left: str, right: str, pos: str
) -> int:
    lines = tree_path.read_text().splitlines()[1:]
    nodes = {int(fields[0]): fields for fields in map(str.split, lines)}
    leaf_labels: dict[int, int] = {}

    def label(node_id: int) -> None:
        fields = nodes[node_id]
        if fields[1] == "-":
            leaf_labels[node_id] = len(leaf_labels)
            return
        label(int(fields[1]))
        label(int(fields[2]))

    label(0)
    phone_by_context = {-1: left, 0: base, 1: right}
    position_by_name = {"WDBNDRY_B": "b", "WDBNDRY_E": "e", "WDBNDRY_S": "s", "WDBNDRY_I": "i"}
    node = 0
    while nodes[node][1] != "-":
        fields = nodes[node]
        expression = " ".join(fields[5:])
        terms = re.findall(r"(!?[A-Za-z0-9_]+) (-?\d+)", expression)
        matched = True
        for raw_name, raw_context in terms:
            negated = raw_name.startswith("!")
            name = raw_name.removeprefix("!")
            if name in position_by_name:
                value = pos == position_by_name[name]
            else:
                value = phone_by_context[int(raw_context)] in questions[name]
            matched &= not value if negated else value
        node = int(fields[1] if matched else fields[2])
    return leaf_labels[node]


@requires_c_library
def test_tied_assignments_match_independent_tree_walk(full_project: PipelineContext) -> None:
    """Declined-A1 debt: sampled mdef assignments equal independently walked leaves."""
    question_path = full_project.trees_dir / "questions"
    questions = {
        fields[0]: set(fields[1:])
        for fields in map(str.split, question_path.read_text().splitlines())
        if fields and not fields[0].startswith("WDBNDRY_")
    }
    ci_count = len(
        [
            state
            for row in _mdef_rows(full_project.model_dir("ci-1g") / "mdef")
            for state in row[6:]
            if state != "N"
        ]
    )
    offsets: dict[tuple[str, int], int] = {}
    next_id = ci_count
    for phone in (full_project.shared_dir / "phoneset.txt").read_text().splitlines():
        if phone.startswith("+") or phone == "SIL":
            continue
        for state in range(3):
            tree = full_project.trees_dir / "pruned" / f"{phone}-{state}.dtree"
            offsets[(phone, state)] = next_id
            next_id += sum(
                1 for line in tree.read_text().splitlines()[1:] if line.split()[1] == "-"
            )

    checked = 0
    for row in _mdef_rows(full_project.model_dir("cd-1g") / "mdef"):
        base, left, right, pos = row[:4]
        if left == "-" or (base, 0) not in offsets:
            continue
        for state, assigned in enumerate(row[6:-1]):
            leaf = _tree_leaf(
                full_project.trees_dir / "pruned" / f"{base}-{state}.dtree",
                questions,
                base,
                left,
                right,
                pos,
            )
            assert int(assigned) == offsets[(base, state)] + leaf
            checked += 1
        if checked >= 60:
            break
    assert checked >= 60


def _bw_contract_state(pass_row: dict[str, Any]) -> dict[str, object]:
    shards = pass_row["shards"]
    return {
        "total_frames": pass_row["total_frames"],
        "stop_decision": pass_row["stop_decision"],
        "assigned_ids": [item for shard in shards for item in shard["assigned_ids"]],
        "processed_ids": [item for shard in shards for item in shard["processed_ids"]],
        "retried_ids": [item for shard in shards for item in shard["retried_ids"]],
        "skipped": [item for shard in shards for item in shard["skipped"]],
    }


def _file_bytes(root: Path, relative: str) -> bytes:
    return (root / relative).read_bytes()


def _assert_bw_discrete_contract(
    reference_pass: dict[str, Any], candidate_pass: dict[str, Any]
) -> None:
    reference = _bw_contract_state(reference_pass)
    candidate = _bw_contract_state(candidate_pass)
    assert reference == candidate, f"BW discrete-state mismatch: {reference!r} != {candidate!r}"


def test_bw_discrete_contract_negative_control_rejects_dropped_identity() -> None:
    reference = {
        "total_frames": 9,
        "stop_decision": "continued",
        "shards": [
            {
                "assigned_ids": ["a", "b"],
                "processed_ids": ["a", "b"],
                "retried_ids": [],
                "skipped": [],
            }
        ],
    }
    dropped = json.loads(json.dumps(reference))
    dropped["shards"][0]["assigned_ids"].remove("b")
    dropped["shards"][0]["processed_ids"].remove("b")
    with pytest.raises(AssertionError, match="BW discrete-state mismatch"):
        _assert_bw_discrete_contract(reference, dropped)


@requires_c_library
@contract_scope(
    order=1,
    kind="fixed-count-reproducibility",
    shard_counts=(2,),
    passes=3,
)
@contract_scope(
    order=2,
    kind="cross-count-discrete-state",
    shard_counts=(1, 2),
    passes=3,
)
def test_seeded_bw_shards_are_reproducible_and_discrete_state_is_partition_independent(
    flat_project: PipelineContext, tmp_path: Path
) -> None:
    """Certify our one-shard serial reference; defects shared with it remain invisible."""
    assert flat_project.split.seed == SEED
    fileids, transcription = write_golden_subset(flat_project)
    outputs = [tmp_path / "one", tmp_path / "two-a", tmp_path / "two-b"]
    for output, n_shards in zip(outputs, (1, 2, 2), strict=True):
        run_bw_training(
            flat_project.model_dir("flat"),
            output,
            flat_project.features_dir,
            fileids,
            transcription,
            flat_project.shared_dir / "dictionary.dict",
            first_pass_2passvar=False,
            filler_dict=flat_project.filler_dict,
            n_iter=3,
            convergence_ratio=float("-inf"),
            config=BWConfig(
                pass2var=True,
                unobserved_gaussian_policy="zero",
                a_beam=1e-200,
                multipron=False,
            ),
            multipron=False,
            n_shards=n_shards,
        )

    contract_check_files(
        left=outputs[1], right=outputs[2], artifacts=_CONTRACT_MODEL_FILES, scope=1
    )
    for pass_number in range(1, 4):
        for shard_number in range(2):
            relative = f".bw-accum/pass-{pass_number:02d}/shard-{shard_number:05d}"
            contract_check_files(
                left=outputs[1] / relative,
                right=outputs[2] / relative,
                artifacts=_CONTRACT_ACCUMULATOR_FILES,
                scope=1,
            )

    telemetry = [json.loads((output / "bw_telemetry.json").read_text()) for output in outputs]
    assert all(len(item["passes"]) == 3 for item in telemetry)
    for serial_pass, sharded_pass in zip(
        telemetry[0]["passes"], telemetry[1]["passes"], strict=True
    ):
        contract_check_fields(
            left=_bw_contract_state(serial_pass),
            right=_bw_contract_state(sharded_pass),
            artifacts=_CONTRACT_DISCRETE_FIELDS,
            scope=2,
        )


@requires_c_library
@contract_scope(
    order=3,
    kind="one-shard-reference",
    shard_counts=(1,),
    passes=3,
)
def test_one_shard_reducer_matches_established_in_process_bw(
    flat_project: PipelineContext, tmp_path: Path
) -> None:
    """Gate dump/restore/reduction against the pre-sharding accumulation path."""
    fileids, transcription = write_golden_subset(flat_project)
    common = {
        "model_dir": flat_project.model_dir("flat"),
        "features_dir": flat_project.features_dir,
        "train_fileids": fileids,
        "transcription": transcription,
        "dictionary": flat_project.shared_dir / "dictionary.dict",
        "first_pass_2passvar": False,
        "filler_dict": flat_project.filler_dict,
        "n_iter": 3,
        "convergence_ratio": float("-inf"),
        "config": BWConfig(
            pass2var=True,
            unobserved_gaussian_policy="zero",
            a_beam=1e-200,
            multipron=False,
        ),
        "multipron": False,
        "n_shards": 1,
    }
    established = tmp_path / "established-in-process"
    reduced = tmp_path / "one-shard-reducer"
    run_bw_training(output_dir=established, _in_process_reference=True, **common)
    run_bw_training(output_dir=reduced, **common)

    contract_check_files(
        left=established, right=reduced, artifacts=_CONTRACT_REFERENCE_FILES, scope=3
    )
    established_telemetry = json.loads((established / "bw_telemetry.json").read_text())
    reduced_telemetry = json.loads((reduced / "bw_telemetry.json").read_text())
    for established_pass, reduced_pass in zip(
        established_telemetry["passes"], reduced_telemetry["passes"], strict=True
    ):
        contract_check_fields(
            left=established_pass, right=reduced_pass, artifacts=_CONTRACT_TELEMETRY_FIELDS, scope=3
        )
