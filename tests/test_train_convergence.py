"""Signed Baum-Welch deltas with bounded, nonregressing convergence."""

from pathlib import Path

import pytest

pytest.importorskip(
    "resource", reason="POSIX-only training resource accounting requires the resource module"
)

from pstrain.lib.steps.train import _convergence_delta, _has_converged
from tests.clib import requires_c_library


def test_convergence_uses_signed_absolute_delta() -> None:
    assert _convergence_delta(-70.0, -70.1) == 0.09999999999999432
    assert _convergence_delta(-70.2, -70.1) < 0


def test_threshold_equality_converges() -> None:
    assert _has_converged(-9.999, -10.0, 2, 0.001, 1)


def test_strictly_greater_delta_runs_one_more_iteration() -> None:
    assert not _has_converged(-9.998, -10.0, 2, 0.001, 1)


def test_minimum_iterations_override_convergence() -> None:
    assert not _has_converged(-10.0, -10.0, 2, 0.001, 3)
    assert _has_converged(-10.0, -10.0, 3, 0.001, 3)


def test_zero_previous_matches_upstream_sign_fallback() -> None:
    assert _convergence_delta(2.0, 0.0) == 1.0
    assert _convergence_delta(-2.0, 0.0) == -1.0
    assert _convergence_delta(0.0, 0.0) == 0.0


@pytest.mark.parametrize(
    "current,previous",
    [(-70.2, -70.1), (-100.2, -100.0), (-1.0, 0.0)],
)
def test_likelihood_regression_never_converges(current: float, previous: float) -> None:
    assert _convergence_delta(current, previous) < 0
    assert not _has_converged(current, previous, 6, 0.1, 1)


@pytest.mark.parametrize(
    "current,previous",
    [
        (float("nan"), -10.0),
        (-float("inf"), -10.0),
        (float("inf"), -10.0),
        (-10.0, float("inf")),
        (-10.0, -float("inf")),
        (float("inf"), 0.0),
        (-1e308, 1e308),
    ],
)
def test_nonfinite_likelihood_or_delta_never_converges(current: float, previous: float) -> None:
    assert not _has_converged(current, previous, 6, 2.0, 1)


@pytest.mark.parametrize("n_iter", [2, 3])
@requires_c_library
def test_real_training_loop_does_not_stop_at_likelihood_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_iter: int
) -> None:
    from dataclasses import replace

    from pstrain.lib.bw import BWConfig, BWResult, BWTrainer
    from pstrain.lib.steps.train import run_bw_training
    from pstrain.lib.telemetry import load_bw_telemetry

    fixture = Path(__file__).parent / "fixtures" / "multipron_final_state"
    fileids = tmp_path / "train.fileids"
    transcript = tmp_path / "train.transcription"
    fileids.write_text("arctic_a0257\n")
    transcript.write_text("arctic_a0257 they are coming ashore whoever they are\n")
    likelihoods = [-100.0, -100.2, -100.1995]
    observed_frames: list[int] = []
    real_get_stats = BWTrainer.get_stats

    def controlled_likelihood(trainer: BWTrainer) -> BWResult:
        stats = real_get_stats(trainer)
        assert stats.total_frames > 0
        assert stats.total_utts == 1
        average = likelihoods[len(observed_frames)]
        observed_frames.append(stats.total_frames)
        return replace(stats, avg_log_prob=average, total_log_lik=average * stats.total_frames)

    # Only convergence evidence is controlled. Accumulation, normalization,
    # model saving and checkpoints use the real native trainer and fixture.
    monkeypatch.setattr(BWTrainer, "get_stats", controlled_likelihood)
    output = tmp_path / "trained"
    result = run_bw_training(
        model_dir=fixture / "model",
        output_dir=output,
        features_dir=fixture,
        train_fileids=fileids,
        transcription=transcript,
        dictionary=fixture / "dictionary.dict",
        filler_dict=fixture / "filler.dict",
        first_pass_2passvar=False,
        config=BWConfig(pass2var=False, unobserved_gaussian_policy="zero", a_beam=1e-200),
        n_iter=n_iter,
        convergence_ratio=0.1,
        checkpoint_iterations=True,
    )
    assert len(observed_frames) == n_iter
    assert result.iterations == n_iter
    assert result.converged is (n_iter == 3)
    assert result.final_likelihood == likelihoods[n_iter - 1]
    rows = load_bw_telemetry(output)["passes"]
    assert rows[1]["signed_convergence_delta"] == pytest.approx(-0.2)
    assert rows[1]["stop_decision"] == ("cap" if n_iter == 2 else "continued")
    assert (output / "iterations" / "02" / "means").is_file()
    if n_iter == 3:
        assert rows[2]["stop_decision"] == "converged"
        assert rows[2]["signed_convergence_delta"] == pytest.approx(0.0005)
