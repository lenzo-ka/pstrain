"""Tests for SphinxTrain-compatible Baum-Welch convergence semantics."""

from pstrain.lib.steps.train import _convergence_delta, _has_converged


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
