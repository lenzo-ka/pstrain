"""Tests for retrying recoverable forward-beam update failures."""

import logging
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip(
    "resource", reason="POSIX-only training resource accounting requires the resource module"
)

from pstrain.lib.steps.train import (
    TerminalAlignmentError,
    _accept_arctic_a0302_exception,
    _process_with_final_state_retry,
)
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
        failed_alignment="recover",
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


def test_final_state_retry_exhaustion_omits_the_utterance_and_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An utterance that cannot align must not end a build that aligned the rest.

    ``recover`` used to raise here, so one genuinely unalignable recording
    failed the whole run and left the operator to read its id out of the log,
    add it to an exclusion list, and start over -- a loop paid once per bad
    file. It now reports the utterance and continues without it. The caller
    records that as an ``alignment_failure`` skip, and ``max_skip_fraction``
    still fails the run once skips stop being incidental, so this tolerates a
    bad recording without tolerating a broken corpus.
    """
    trainer = TightBeamTrainer()
    # Fails at every beam, but records each attempt so the test can prove the
    # widened retry actually ran rather than the call returning False at once.
    trainer.process_utterance_mfcc = (  # type: ignore[method-assign]
        lambda mfcc, transcript: trainer.attempt_beams.append(trainer.beam) or False
    )
    caplog.set_level(logging.ERROR, logger="pstrain.lib.steps.train")

    assert not _process_with_final_state_retry(
        trainer,  # type: ignore[arg-type]
        np.zeros((4, 13), dtype=np.float32),
        "<s> TEST </s>",
        normal_beam=1e-90,
        retry_beam_factor=1e10,
        fileid="malformed",
        failed_alignment="recover",
    )

    # Two attempts, the second at the widened beam: this is what distinguishes
    # "recover" from "omit", which gives up after the first.
    assert trainer.attempt_beams == [pytest.approx(1e-90), pytest.approx(1e-100)]
    # The drop is reported, names the utterance, and names the beam it failed at.
    assert "malformed" in caplog.text
    assert "1e-100" in caplog.text
    assert "omitting" in caplog.text
    # The beam is restored for the next utterance.
    assert trainer.beam == pytest.approx(1e-90)


def test_abort_still_fails_the_run_for_callers_that_want_that() -> None:
    """The strict behavior remains available; it is no longer the default."""
    trainer = TightBeamTrainer()
    trainer.process_utterance_mfcc = lambda mfcc, transcript: False  # type: ignore[method-assign]

    with pytest.raises(TerminalAlignmentError):
        _process_with_final_state_retry(
            trainer,  # type: ignore[arg-type]
            np.zeros((4, 13), dtype=np.float32),
            "<s> TEST </s>",
            normal_beam=1e-90,
            retry_beam_factor=1e10,
            fileid="malformed",
            failed_alignment="abort",
        )


def test_abort_position_does_not_retry_and_fails_loudly() -> None:
    trainer = TightBeamTrainer()

    with pytest.raises(TerminalAlignmentError, match="Final state not reached for abort-me"):
        _process_with_final_state_retry(
            trainer,  # type: ignore[arg-type]
            np.zeros((4, 13), dtype=np.float32),
            "<s> TEST </s>",
            normal_beam=1e-90,
            retry_beam_factor=1e10,
            fileid="abort-me",
            failed_alignment="abort",
        )

    assert trainer.attempt_beams == [pytest.approx(1e-90)]


def test_omit_position_reports_failure_and_does_not_retry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    trainer = TightBeamTrainer()
    caplog.set_level(logging.ERROR, logger="pstrain.lib.steps.train")

    assert not _process_with_final_state_retry(
        trainer,  # type: ignore[arg-type]
        np.zeros((4, 13), dtype=np.float32),
        "<s> TEST </s>",
        normal_beam=1e-90,
        retry_beam_factor=1e10,
        fileid="omit-me",
        failed_alignment="omit",
    )

    assert trainer.attempt_beams == [pytest.approx(1e-90)]
    assert "omit-me ignored after failed alignment" in caplog.text


def test_retry_beam_factor_flip_changes_the_accept_or_skip_outcome() -> None:
    """Prove-the-treatment: flipping ``retry_beam_factor`` changes whether the
    SAME failing utterance is rescued or skipped.

    The shipped ``1e10`` rescues a final-state-not-reached utterance on a
    widened retry; the parity/stock setting of ``1`` must instead disable the
    retry and refuse the utterance (the ``retry_beam_factor <= 1.0`` branch in
    ``_process_with_final_state_retry``). No other retry test exercises that
    branch, so a regression that ignored the factor and retried regardless
    would pass every case above yet be caught here.
    """
    mfcc = np.zeros((4, 13), dtype=np.float32)
    transcript = "<s> TEST </s>"

    rescued = TightBeamTrainer()
    assert _process_with_final_state_retry(
        rescued,  # type: ignore[arg-type]
        mfcc,
        transcript,
        normal_beam=1e-90,
        retry_beam_factor=1e10,
        fileid="knob-flip",
        failed_alignment="recover",
    )
    assert rescued.attempt_beams == [pytest.approx(1e-90), pytest.approx(1e-100)]

    disabled = TightBeamTrainer()
    with pytest.raises(TerminalAlignmentError, match="retry is disabled"):
        _process_with_final_state_retry(
            disabled,  # type: ignore[arg-type]
            mfcc,
            transcript,
            normal_beam=1e-90,
            retry_beam_factor=1.0,
            fileid="knob-flip",
            failed_alignment="recover",
        )
    assert disabled.attempt_beams == [pytest.approx(1e-90)]


def test_a0302_exception_reports_current_value_inside_band(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    monkeypatch.setattr("pstrain.lib.steps.train._exact_zero_codebooks", lambda model: 4600)
    caplog.set_level(logging.WARNING, logger="pstrain.lib.steps.train")

    assert (
        _accept_arctic_a0302_exception(fileid="arctic_a0302", model_dir=tmp_path, band=(4548, 4623))
        == 4600
    )
    assert "exact_zero_codebooks=4600 is inside inclusive band [4548, 4623]" in caplog.text


@pytest.mark.parametrize("value", [4547, 4624])
def test_a0302_exception_halts_outside_either_side(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: int
) -> None:
    monkeypatch.setattr("pstrain.lib.steps.train._exact_zero_codebooks", lambda model: value)
    with pytest.raises(
        TerminalAlignmentError,
        match=rf"exact_zero_codebooks={value} is outside inclusive band \[4548, 4623\]",
    ):
        _accept_arctic_a0302_exception(fileid="arctic_a0302", model_dir=tmp_path, band=(4548, 4623))


def test_a0302_band_does_not_accept_another_utterance(tmp_path: Path) -> None:
    assert (
        _accept_arctic_a0302_exception(fileid="arctic_a0587", model_dir=tmp_path, band=(4548, 4623))
        is None
    )


@requires_c_library
def test_optional_final_silence_flip_changes_native_bw_artifact(tmp_path: Path) -> None:
    """Prove the native graph builder consumes ``optional_final_silence``."""
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

    fileid = "arctic_a0001"
    fileids = context.etc_dir / "optional-final-silence.fileids"
    transcription = context.etc_dir / "optional-final-silence.transcription"
    fileids.write_text(f"{fileid}\n")
    transcription.write_text(f"{fileid} author of the danger trail philip steels etc\n")

    results = {}
    for enabled in (True, False):
        results[enabled] = run_bw_training(
            model_dir=context.model_dir("flat"),
            output_dir=tmp_path / f"optional-final-silence-{enabled}",
            features_dir=context.features_dir,
            train_fileids=fileids,
            transcription=transcription,
            dictionary=context.shared_dir / "dictionary.dict",
            first_pass_2passvar=True,
            filler_dict=context.filler_dict,
            n_iter=1,
            config=BWConfig(
                pass2var=True,
                unobserved_gaussian_policy="zero",
                optional_final_silence=enabled,
            ),
        )
        assert results[enabled].final_utts == 1

    assert results[True].final_likelihood != results[False].final_likelihood


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
        "mixture_weights": _pstrainc.read_mixw_counts,
        "transition_matrices": _pstrainc.read_tmat_counts,
    }
    for filename, reader in readers.items():
        retried_values = reader(str(retried / filename))[0]
        clean_values = reader(str(clean / filename))[0]
        np.testing.assert_array_equal(retried_values, clean_values, err_msg=filename)
