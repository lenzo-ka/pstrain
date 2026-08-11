"""Tests for retrying recoverable forward-beam update failures."""

import logging
from pathlib import Path

import numpy as np
import pytest

from pstrain.lib.steps.train import _process_with_final_state_retry
from tests.clib import requires_c_library

FIXTURE = Path(__file__).parent / "fixtures" / "mini_arctic"


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


@requires_c_library
def test_native_failed_pass_then_retry_matches_clean_wide_beam_model(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed native pass must leave no residue in the BW accumulators."""
    from pstrain.lib import _pstrainc
    from pstrain.lib.bw import BWConfig
    from pstrain.lib.pipeline import PipelineContext
    from pstrain.lib.pipeline.tasks import build_pipeline
    from pstrain.lib.setup import setup_project
    from pstrain.lib.steps.train import run_bw_training

    project = tmp_path / "project"
    setup_project(
        project,
        transcription_path=FIXTURE / "transcription.txt",
        audio_path=FIXTURE / "wav",
        dictionary_path=FIXTURE / "dictionary.dict",
        phoneset_path=FIXTURE / "phoneset.txt",
        filler_dict_path=FIXTURE / "filler.dict",
    )
    context = PipelineContext.from_config(project)
    assert build_pipeline(context).run("flat", jobs=1) == 0

    # Use one fixed utterance so every update in the retried run takes the
    # same effective beam as its counterpart in the clean run.
    fileid = "arctic_a0001"
    fileids = context.etc_dir / "retry.fileids"
    transcription = context.etc_dir / "retry.transcription"
    fileids.write_text(f"{fileid}\n")
    transcription.write_text(f"{fileid} author of the danger trail philip steels etc\n")

    tight_beam = 1e-1
    wide_beam = 1e-200

    def train(output: Path, beam: float, retry_factor: float) -> None:
        result = run_bw_training(
            model_dir=context.model_dir("flat"),
            output_dir=output,
            features_dir=context.features_dir,
            train_fileids=fileids,
            transcription=transcription,
            dictionary=context.shared_dir / "dictionary.dict",
            first_pass_2passvar=True,
            filler_dict=context.filler_dict,
            n_iter=1,
            config=BWConfig(pass2var=True, unobserved_gaussian_policy="zero", a_beam=beam),
            retry_beam_factor=retry_factor,
        )
        assert result.final_utts == 1

    caplog.set_level(logging.WARNING, logger="pstrain.lib.steps.train")
    retried = tmp_path / "retried"
    clean = tmp_path / "clean"
    train(retried, tight_beam, tight_beam / wide_beam)
    assert sum("retrying once" in record.message for record in caplog.records) == 1
    train(clean, wide_beam, 1.0)

    readers = {
        "means": _pstrainc.read_gau,
        "variances": _pstrainc.read_gau,
        "mixture_weights": _pstrainc.read_mixw,
        "transition_matrices": _pstrainc.read_tmat,
    }
    for filename, reader in readers.items():
        retried_values = reader(str(retried / filename))[0]
        clean_values = reader(str(clean / filename))[0]
        np.testing.assert_array_equal(retried_values, clean_values, err_msg=filename)
