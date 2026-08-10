"""Tests for retrying recoverable forward-beam update failures."""

import numpy as np
import pytest

from pstrain.lib.steps.train import _process_with_final_state_retry


class TightBeamTrainer:
    """Synthetic trainer whose tight-beam attempt fails and widened retry succeeds."""

    def __init__(self) -> None:
        self.beam = 1e-90
        self.attempt_beams: list[float] = []
        self._retry_transaction_active = False

    def process_utterance_mfcc(self, mfcc: np.ndarray, transcript: str) -> bool:
        self.attempt_beams.append(self.beam)
        return self.beam < 1e-90

    @property
    def final_state_not_reached(self) -> bool:
        return self.beam == 1e-90

    def set_a_beam(self, beam: float) -> float:
        previous = self.beam
        self.beam = beam
        return previous


def test_tight_beam_failure_succeeds_once_on_widened_retry() -> None:
    trainer = TightBeamTrainer()
    success = _process_with_final_state_retry(
        trainer,  # type: ignore[arg-type]
        np.zeros((4, 13), dtype=np.float32),
        "<s> TEST </s>",
        normal_beam=1e-90,
        retry_beam_factor=1e10,
        fileid="tight-beam",
    )

    assert success
    assert trainer.attempt_beams == [pytest.approx(1e-90), pytest.approx(1e-100)]
    assert trainer.beam == pytest.approx(1e-90)


def test_non_final_state_failure_is_not_retried() -> None:
    trainer = TightBeamTrainer()
    trainer.beam = 1e-80

    assert not _process_with_final_state_retry(
        trainer,  # type: ignore[arg-type]
        np.zeros((4, 13), dtype=np.float32),
        "<s> TEST </s>",
        normal_beam=1e-80,
        retry_beam_factor=1e10,
        fileid="other-failure",
    )
    assert trainer.attempt_beams == [pytest.approx(1e-80)]


class CountingTrainer(TightBeamTrainer):
    """Count only successful passes, mirroring the native accumulator contract."""

    def __init__(self, beam: float) -> None:
        super().__init__()
        self.beam = beam
        self.counts = np.zeros(3, dtype=np.float64)

    def process_utterance_mfcc(self, mfcc: np.ndarray, transcript: str) -> bool:
        self.attempt_beams.append(self.beam)
        if self.beam >= 1e-90:
            return False
        self.counts += np.array([mfcc.shape[0], len(transcript), 1.0])
        return True


def test_failed_pass_then_retry_matches_clean_wide_beam_counts() -> None:
    mfcc = np.zeros((4, 13), dtype=np.float32)
    transcript = "<s> TEST </s>"
    retried = CountingTrainer(1e-90)
    clean = CountingTrainer(1e-100)

    assert _process_with_final_state_retry(
        retried,  # type: ignore[arg-type]
        mfcc,
        transcript,
        normal_beam=1e-90,
        retry_beam_factor=1e10,
        fileid="retried",
    )
    assert _process_with_final_state_retry(
        clean,  # type: ignore[arg-type]
        mfcc,
        transcript,
        normal_beam=1e-100,
        retry_beam_factor=1.0,
        fileid="clean",
    )

    np.testing.assert_array_equal(retried.counts, clean.counts)
