from math import isinf

import pytest

from pstrain.benchmarks.boundaries import PhoneInterval, score_phone_boundaries


def _phones(labels: str, boundaries: list[float]) -> list[PhoneInterval]:
    return [
        PhoneInterval(label, boundaries[i], boundaries[i + 1])
        for i, label in enumerate(labels.split())
    ]


def test_exact_phone_sequence_reports_timing_error() -> None:
    ref = _phones("a b c", [0.0, 0.10, 0.20, 0.30])
    hyp = _phones("a b c", [0.0, 0.11, 0.18, 0.30])
    score = score_phone_boundaries(ref, hyp)

    assert score.mean_absolute_error_ms == pytest.approx(15.0)
    assert score.median_absolute_error_ms == pytest.approx(15.0)
    assert score.coverage == 1.0
    assert score.recall_within[10.0] == 0.5
    assert score.recall_within[20.0] == 1.0


def test_insertion_is_not_silently_dropped_from_accuracy() -> None:
    ref = _phones("a b c", [0.0, 0.10, 0.20, 0.30])
    hyp = _phones("a x b c", [0.0, 0.05, 0.10, 0.20, 0.30])
    score = score_phone_boundaries(ref, hyp)

    assert score.insertions == 1
    assert score.comparable_boundaries == 1
    assert score.coverage == 0.5
    assert score.recall_within[10.0] == 0.5
    assert score.precision_within[10.0] == pytest.approx(1 / 3)


def test_no_comparable_boundaries_has_infinite_timing_error() -> None:
    score = score_phone_boundaries(
        _phones("a b", [0.0, 0.1, 0.2]), _phones("x y", [0.0, 0.1, 0.2])
    )

    assert score.substitutions == 2
    assert isinf(score.mean_absolute_error_ms)
    assert score.recall_within[50.0] == 0.0
