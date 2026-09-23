"""Tests for retrying recoverable forward-beam update failures."""

import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip(
    "resource", reason="POSIX-only training resource accounting requires the resource module"
)

from pstrain.lib.steps.train import (
    TerminalAlignmentError,
    _accept_arctic_a0302_exception,
    _process_with_final_state_retry,
    _redirect_stdout_fd,
)
from tests.clib import requires_c_library

FIXTURE = Path(__file__).parent / "fixtures" / "mini_arctic"


def test_fd_stdout_redirect_restores_after_exception(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    diagnostic = tmp_path / "diagnostic.log"

    with pytest.raises(RuntimeError, match="native failure"), _redirect_stdout_fd(diagnostic):
        os.write(1, b"native bytes before failure\n")
        raise RuntimeError("native failure")

    os.write(1, b"stdout restored\n")
    captured = capfd.readouterr()
    assert "native bytes before failure" not in captured.out
    assert "stdout restored\n" in captured.out
    assert diagnostic.read_bytes() == b"native bytes before failure\n"


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


def test_repeated_omission_output_collapses_to_stage_summary(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    from pstrain.lib.steps.train import _record_omission, _report_skip_summary

    trainer = TightBeamTrainer()
    trainer.process_utterance_mfcc = (  # type: ignore[method-assign]
        lambda mfcc, transcript: trainer.attempt_beams.append(trainer.beam) or False
    )
    caplog.set_level(logging.WARNING, logger="pstrain.lib.steps.train")
    for report_retry in (True, False, False, False):
        assert not _process_with_final_state_retry(
            trainer,  # type: ignore[arg-type]
            np.zeros((4, 13), dtype=np.float32),
            "<s> TEST </s>",
            normal_beam=1e-90,
            retry_beam_factor=1e10,
            fileid="arctic_a0135",
            failed_alignment="recover",
            report_retry=report_retry,
        )

    omissions: dict[tuple[str, str], list[int]] = {}
    reason = "final state not reached even at a_beam=1e-100"
    for iteration in range(3, 7):
        _record_omission(omissions, "arctic_a0135", reason, iteration)
    _report_skip_summary(omissions)
    output = capsys.readouterr().out
    assert output == (
        "omitted\tarctic_a0135\tfinal state not reached even at a_beam=1e-100\tpasses 3-6\n"
    )
    assert caplog.text.count("retrying once") == 1
    assert caplog.text.count("omitting it from this pass") == 1
    assert "iteration 3 skipped" not in caplog.text
    assert "Failed to process" not in caplog.text


def test_skip_limit_failure_reports_changes_and_omissions_first(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    from pstrain.lib.steps.train import _account_and_enforce_skips, _record_omission

    omissions: dict[tuple[str, str], list[int]] = {}
    skipped_by_pass: list[tuple[int, int]] = []
    reason = "final state not reached even at a_beam=1e-100"
    caplog.set_level(logging.WARNING, logger="pstrain.lib.steps.train")
    for iteration, skipped in ((1, 0), (2, 0), (3, 1)):
        if skipped:
            _record_omission(omissions, "arctic_a0135", reason, iteration)
        _account_and_enforce_skips(
            iteration=iteration,
            skipped=skipped,
            input_utts=10,
            max_skip_fraction=0.15,
            skipped_by_pass=skipped_by_pass,
            omitted_passes=omissions,
        )
    _record_omission(omissions, "arctic_a0135", reason, 4)
    with pytest.raises(RuntimeError, match="Iteration 4: skipped 2/10"):
        _account_and_enforce_skips(
            iteration=4,
            skipped=2,
            input_utts=10,
            max_skip_fraction=0.15,
            skipped_by_pass=skipped_by_pass,
            omitted_passes=omissions,
        )

    assert "iteration 3 skipped 1/10" in caplog.text
    assert "iteration 4 skipped 2/10" in caplog.text
    assert capsys.readouterr().out == f"omitted\tarctic_a0135\t{reason}\tpasses 3-4\n"


@pytest.mark.parametrize(
    ("counts", "reported"),
    [
        ((1, 1), ((1, 1),)),
        ((1, 0), ((1, 1), (2, 0))),
        ((0, 0, 1), ((3, 1),)),
        ((2, 2, 3), ((1, 2), (3, 3))),
    ],
)
def test_skip_reporting_prints_first_nonzero_and_each_change(
    caplog: pytest.LogCaptureFixture,
    counts: tuple[int, ...],
    reported: tuple[tuple[int, int], ...],
) -> None:
    from pstrain.lib.steps.train import _report_changed_skip

    history: list[tuple[int, int]] = []
    caplog.set_level(logging.WARNING, logger="pstrain.lib.steps.train")
    for iteration, skipped in enumerate(counts, start=1):
        history.append((iteration, skipped))
        _report_changed_skip(history, input_utts=10)

    messages = [record.message for record in caplog.records]
    assert len(messages) == len(reported)
    for message, (iteration, skipped) in zip(messages, reported, strict=True):
        assert f"iteration {iteration} skipped {skipped}/10" in message


def test_one_utterance_with_two_omission_reasons_gets_two_summaries(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    from pstrain.lib.steps.train import _record_omission, _report_skip_summary

    omissions: dict[tuple[str, str], list[int]] = {}
    first = "final state not reached even at a_beam=1e-100"
    second = "final state not reached even at a_beam=1e-110"
    caplog.set_level(logging.WARNING, logger="pstrain.lib.steps.train")

    class AlwaysFinalStateFailure(TightBeamTrainer):
        @property
        def final_state_not_reached(self) -> bool:
            return True

    for iteration, beam, reason in (
        (3, 1e-90, first),
        (4, 1e-90, first),
        (5, 1e-100, second),
    ):
        trainer = AlwaysFinalStateFailure()
        trainer.beam = beam
        trainer.process_utterance_mfcc = lambda mfcc, transcript: False  # type: ignore[method-assign]
        assert not _process_with_final_state_retry(
            trainer,  # type: ignore[arg-type]
            np.zeros((4, 13), dtype=np.float32),
            "<s> TEST </s>",
            normal_beam=beam,
            retry_beam_factor=1e10,
            fileid="arctic_a0135",
            report_retry=("arctic_a0135", reason) not in omissions,
        )
        _record_omission(omissions, "arctic_a0135", reason, iteration)
    _report_skip_summary(omissions)

    assert capsys.readouterr().out.splitlines() == [
        f"omitted\tarctic_a0135\t{first}\tpasses 3-4",
        f"omitted\tarctic_a0135\t{second}\tpasses 5",
    ]
    assert caplog.text.count("retrying once") == 2
    assert caplog.text.count("omitting it from this pass") == 2


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
def test_real_bw_shard_writes_complete_diagnostics_to_log(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
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
    fileids = context.etc_dir / "diagnostic.fileids"
    transcription = context.etc_dir / "diagnostic.transcription"
    fileids.write_text(f"{fileid}\n")
    transcription.write_text(f"{fileid} author of the danger trail philip steels etc\n")
    capfd.readouterr()

    result = run_bw_training(
        model_dir=context.model_dir("flat"),
        output_dir=tmp_path / "diagnostic-model",
        features_dir=context.features_dir,
        train_fileids=fileids,
        transcription=transcription,
        dictionary=context.shared_dir / "dictionary.dict",
        filler_dict=context.filler_dict,
        first_pass_2passvar=False,
        n_iter=1,
        config=BWConfig(
            pass2var=False,
            unobserved_gaussian_policy="zero",
            multipron=False,
        ),
        multipron=False,
        n_shards=1,
        project_dir=project,
        stage="diagnostic-stage",
    )
    assert result.final_utts == 1
    print("stdout restored")

    captured = capfd.readouterr()
    assert "column defns" not in captured.out
    assert "stdout restored" in captured.out
    assert f"bw-logs\t{project / '.pstrain' / 'bw' / 'diagnostic-stage'}" in captured.out
    diagnostic = project / ".pstrain" / "bw" / "diagnostic-stage" / "pass-01-shard-00.log"
    content = diagnostic.read_text(encoding="utf-8")
    assert content.startswith("column defns\n\t<seq>\n\t<id>\n")
    assert "\t<avg_posterior_prune>\n" in content
    assert "\t... timing info ... \n" in content
    native_row = re.compile(r"(?m)^utt>\s+\d+\s+arctic_a0001\s+\d+.*[-+]?\d+\.\d+e[-+]\d+")
    assert native_row.search(content)
    assert not native_row.search(captured.out)


@requires_c_library
def test_real_multipron_bw_writes_native_diagnostics_to_log(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
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
    fileids = context.etc_dir / "multipron-diagnostic.fileids"
    transcription = context.etc_dir / "multipron-diagnostic.transcription"
    fileids.write_text(f"{fileid}\n")
    transcription.write_text(f"{fileid} author of the danger trail philip steels etc\n")
    capfd.readouterr()

    result = run_bw_training(
        model_dir=context.model_dir("flat"),
        output_dir=tmp_path / "multipron-diagnostic-model",
        features_dir=context.features_dir,
        train_fileids=fileids,
        transcription=transcription,
        dictionary=context.shared_dir / "dictionary.dict",
        filler_dict=context.filler_dict,
        first_pass_2passvar=False,
        n_iter=1,
        config=BWConfig(
            pass2var=False,
            unobserved_gaussian_policy="zero",
            multipron=True,
        ),
        multipron=True,
        n_shards=1,
        project_dir=project,
        stage="multipron-diagnostic-stage",
    )
    assert result.final_utts == 1
    print("stdout restored")

    captured = capfd.readouterr()
    diagnostic = project / ".pstrain" / "bw" / "multipron-diagnostic-stage" / "pass-01-shard-00.log"
    content = diagnostic.read_text(encoding="utf-8")
    native_row = re.compile(r"(?m)^utt>\s+\d+\s+arctic_a0001\s+\d+.*[-+]?\d+\.\d+e[-+]\d+")
    assert native_row.search(content)
    assert not native_row.search(captured.out)
    assert "stdout restored" in captured.out


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


@requires_c_library
@pytest.mark.parametrize("failed_method", ["save_density_counts", "normalize", "save"])
def test_training_stops_on_native_false_return_before_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_method: str
) -> None:
    import json

    from pstrain.lib.bw import BWConfig, BWTrainer
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
    fileids = context.etc_dir / "failure.fileids"
    transcription = context.etc_dir / "failure.transcription"
    fileids.write_text("arctic_a0001\n")
    transcription.write_text("arctic_a0001 author of the danger trail philip steels etc\n")
    failures: list[str] = []

    def fail_native_status(*_args: object, **_kwargs: object) -> bool:
        failures.append(failed_method)
        return False

    # All extraction, initialization, accumulation and statistics use the real
    # native machinery; only the selected failure return is injected.
    monkeypatch.setattr(BWTrainer, failed_method, fail_native_status)
    output = tmp_path / "failed-model"
    expected = {
        "save_density_counts": "density count save failed",
        "normalize": "normalization failed",
        "save": "model save failed",
    }[failed_method]
    with pytest.raises(RuntimeError, match=f"Iteration 1: BW {expected}"):
        run_bw_training(
            model_dir=context.model_dir("flat"),
            output_dir=output,
            features_dir=context.features_dir,
            train_fileids=fileids,
            transcription=transcription,
            dictionary=context.shared_dir / "dictionary.dict",
            filler_dict=context.filler_dict,
            first_pass_2passvar=False,
            n_iter=1,
            config=BWConfig(pass2var=False, unobserved_gaussian_policy="zero"),
            checkpoint_iterations=True,
        )
    assert failures == [failed_method]
    assert not (output / "iterations" / "01").exists()
    row = json.loads((output / "bw_telemetry.json").read_text())["passes"][-1]
    assert row["stop_decision"] == "failed"
    assert expected in row["error"]
    assert row["total_frames"] > 0


def _truncated_copy(source: Path, destination: Path, seconds: float) -> None:
    """Write the first ``seconds`` of a mono WAV, leaving the transcript alone."""
    import wave

    with wave.open(str(source), "rb") as reader:
        frame_rate = reader.getframerate()
        payload = reader.readframes(int(frame_rate * seconds))
        with wave.open(str(destination), "wb") as writer:
            writer.setnchannels(reader.getnchannels())
            writer.setsampwidth(reader.getsampwidth())
            writer.setframerate(frame_rate)
            writer.writeframes(payload)


@requires_c_library
def test_audio_too_short_for_its_transcript_is_named_not_searched_for(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """A transcript that cannot fit in the audio is arithmetic, not a search failure.

    An utterance HMM spends at least one frame in every emitting state it
    passes through, so audio shorter than the shortest path through the graph
    can never reach the final state. Widening the beam cannot help, and the
    generic "final state not reached" message sends the reader looking for a
    pruning problem that is not there.
    """
    import json

    from pstrain.lib.bw import BW_CEPSTRAL_LENGTH, BWConfig
    from pstrain.lib.features import read_sphinx_mfc
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
    # Genuinely short audio, extracted by the same front end as the rest of the
    # corpus: half a second of speech carrying a nine-word transcript.
    _truncated_copy(
        context.audio_dir / "arctic_a0001.wav",
        context.audio_dir / "arctic_short.wav",
        seconds=0.5,
    )
    assert build_pipeline(context).run("flat", jobs=1) == 0

    text = "author of the danger trail philip steels etc"
    fileids = context.etc_dir / "short.fileids"
    transcription = context.etc_dir / "short.transcription"
    fileids.write_text("arctic_a0001\narctic_short\n")
    transcription.write_text(f"arctic_a0001 {text}\narctic_short {text}\n")
    frames = read_sphinx_mfc(
        context.features_dir / "arctic_short.mfc", veclen=BW_CEPSTRAL_LENGTH
    ).shape[0]

    caplog.set_level(logging.WARNING, logger="pstrain.lib.steps.train")
    output = tmp_path / "short-audio"
    result = run_bw_training(
        model_dir=context.model_dir("flat"),
        output_dir=output,
        features_dir=context.features_dir,
        train_fileids=fileids,
        transcription=transcription,
        dictionary=context.shared_dir / "dictionary.dict",
        filler_dict=context.filler_dict,
        first_pass_2passvar=True,
        n_iter=1,
        config=BWConfig(pass2var=True, unobserved_gaussian_policy="zero"),
        max_skip_fraction=0.6,
    )
    assert result.final_utts == 1
    assert result.total_skipped == 1

    # The diagnosis names the condition and both numbers, and the numbers are
    # the real ones: the minimum exceeds what this audio actually supplies.
    match = re.search(
        r"arctic_short cannot be aligned at any beam width: its transcript needs at "
        r"least (\d+) frames to reach the final state, and the audio has (\d+)",
        caplog.text,
    )
    assert match is not None, caplog.text
    required, available = int(match.group(1)), int(match.group(2))
    assert available == frames
    assert required > available
    assert "no wider beam can change that, so the retry was skipped" in caplog.text

    # No second forward pass was spent on an utterance no beam could recover.
    assert "retrying once" not in caplog.text
    row = json.loads((output / "bw_telemetry.json").read_text())["passes"][-1]
    assert row["accounting"]["retry_attempted_utts"] == 0
    assert row["accounting"]["retry_recovered_utts"] == 0

    # The stage summary carries the condition rather than a beam width.
    assert (
        "omitted\tarctic_short\taudio has fewer frames than the transcript requires\tpasses 1"
        in capsys.readouterr().out
    )


@requires_c_library
def test_widened_beam_retry_reports_what_it_bought(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """The retry costs a forward pass per failure, so it reports its own yield."""
    import json

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

    fileids = context.etc_dir / "yield.fileids"
    transcription = context.etc_dir / "yield.transcription"
    fileids.write_text("arctic_a0001\n")
    transcription.write_text("arctic_a0001 author of the danger trail philip steels etc\n")

    caplog.set_level(logging.WARNING, logger="pstrain.lib.steps.train")
    output = tmp_path / "retry-yield"
    # 1e-1 prunes this utterance out at the normal beam; 1e-200 recovers it.
    result = run_bw_training(
        model_dir=context.model_dir("flat"),
        output_dir=output,
        features_dir=context.features_dir,
        train_fileids=fileids,
        transcription=transcription,
        dictionary=context.shared_dir / "dictionary.dict",
        filler_dict=context.filler_dict,
        first_pass_2passvar=True,
        n_iter=1,
        config=BWConfig(pass2var=True, unobserved_gaussian_policy="zero", a_beam=1e-1),
        retry_beam_factor=1e-1 / 1e-200,
    )
    assert result.final_utts == 1

    assert "iteration 1 widened the beam on 1 utterance(s) and recovered 1" in caplog.text
    assert "retry\tsecond-pass attempts 1\trecovered 1" in capsys.readouterr().out
    row = json.loads((output / "bw_telemetry.json").read_text())["passes"][-1]
    assert row["accounting"]["retry_attempted_utts"] == 1
    assert row["accounting"]["retry_recovered_utts"] == 1


def test_retry_yield_reports_attempts_that_recovered_nothing(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """A retry that buys nothing is the reading the feature has to be able to give."""
    from pstrain.lib.steps.train import _account_and_enforce_skips, _record_omission

    omissions: dict[tuple[str, str], list[int]] = {}
    for fileid in ("a", "b", "c"):
        _record_omission(omissions, fileid, "final state not reached even at a_beam=1e-100", 1)
    caplog.set_level(logging.WARNING, logger="pstrain.lib.steps.train")

    _account_and_enforce_skips(
        iteration=1,
        skipped=3,
        input_utts=100,
        max_skip_fraction=0.5,
        skipped_by_pass=[],
        omitted_passes=omissions,
        retry_attempts=3,
        retry_recoveries=0,
        retry_yield=(3, 0),
    )
    assert "iteration 1 widened the beam on 3 utterance(s) and recovered 0" in caplog.text
    # Below the skip limit, so no summary is printed; the per-pass line stands alone.
    assert "retry\tsecond-pass attempts" not in capsys.readouterr().out

    from pstrain.lib.steps.train import _report_skip_summary

    _report_skip_summary(omissions, (3, 0))
    assert "retry\tsecond-pass attempts 3\trecovered 0" in capsys.readouterr().out


def test_retry_yield_is_silent_when_the_retry_never_ran(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    from pstrain.lib.steps.train import _account_and_enforce_skips, _report_skip_summary

    caplog.set_level(logging.WARNING, logger="pstrain.lib.steps.train")
    _account_and_enforce_skips(
        iteration=1,
        skipped=0,
        input_utts=10,
        max_skip_fraction=0.5,
        skipped_by_pass=[],
        omitted_passes={},
        retry_attempts=0,
        retry_recoveries=0,
    )
    _report_skip_summary({}, (0, 0))
    assert "widened the beam" not in caplog.text
    assert capsys.readouterr().out == ""


@requires_c_library
def test_converged_clean_run_does_not_claim_skips_it_did_not_have(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Printing the retry summary must not drag the skip warning along with it.

    The converged tail prints both the stage omission summary and the total-skip
    warning. Admitting the retry summary when nothing was skipped widened the
    guard around both, so a clean run that used the retry announced skips it
    never had.
    """
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

    fileids = context.etc_dir / "converged.fileids"
    transcription = context.etc_dir / "converged.transcription"
    fileids.write_text("arctic_a0001\n")
    transcription.write_text("arctic_a0001 author of the danger trail philip steels etc\n")

    caplog.set_level(logging.WARNING, logger="pstrain.lib.steps.train")
    result = run_bw_training(
        model_dir=context.model_dir("flat"),
        output_dir=tmp_path / "converged",
        features_dir=context.features_dir,
        train_fileids=fileids,
        transcription=transcription,
        dictionary=context.shared_dir / "dictionary.dict",
        filler_dict=context.filler_dict,
        first_pass_2passvar=True,
        n_iter=4,
        # A tight beam so every pass pays for the retry, and a threshold wide
        # enough that the run converges and leaves through the converged tail.
        config=BWConfig(pass2var=True, unobserved_gaussian_policy="zero", a_beam=1e-1),
        retry_beam_factor=1e199,
        convergence_ratio=1e6,
    )
    assert result.converged
    assert result.total_skipped == 0

    # The retry summary is still printed, because the retry did run.
    assert "retry\tsecond-pass attempts" in capsys.readouterr().out
    assert "widened the beam" in caplog.text
    # But nothing was skipped, so nothing claims otherwise.
    assert "skipped" not in caplog.text


@requires_c_library
def test_sharded_retry_attempts_are_summed_across_shards(tmp_path: Path) -> None:
    """The merge path must carry the counter, not just the serial path.

    ``multipron`` with one shard runs in this process; more than one shard goes
    through ``_ShardResult`` and a pool merge, which is a different accounting
    route for the same number.
    """
    import json

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

    text = "author of the danger trail philip steels etc"
    fileids = context.etc_dir / "sharded.fileids"
    transcription = context.etc_dir / "sharded.transcription"
    ids = ["arctic_a0001", "arctic_a0002"]
    fileids.write_text("".join(f"{fileid}\n" for fileid in ids))
    transcription.write_text("".join(f"{fileid} {text}\n" for fileid in ids))

    output = tmp_path / "sharded"
    result = run_bw_training(
        model_dir=context.model_dir("flat"),
        output_dir=output,
        features_dir=context.features_dir,
        train_fileids=fileids,
        transcription=transcription,
        dictionary=context.shared_dir / "dictionary.dict",
        filler_dict=context.filler_dict,
        first_pass_2passvar=True,
        n_iter=1,
        config=BWConfig(pass2var=True, unobserved_gaussian_policy="zero", a_beam=1e-1),
        retry_beam_factor=1e199,
        n_shards=2,
    )
    assert result.final_utts == len(ids)

    row = json.loads((output / "bw_telemetry.json").read_text())["passes"][-1]
    assert row["accounting"]["retry_attempted_utts"] == len(ids)
    assert row["accounting"]["retry_recovered_utts"] == len(ids)

    # Each shard's own artifact records its share, and the shares add up.
    artifacts = sorted((output / ".bw-accum" / "pass-01").glob("shard-*/artifact.json"))
    assert len(artifacts) == 2
    per_shard = [json.loads(path.read_text())["retry_attempts"] for path in artifacts]
    assert sum(per_shard) == len(ids)
    assert per_shard == [1, 1]


# --- Retry ladder -------------------------------------------------------------


class LadderTrainer:
    """Synthetic trainer that reaches its final state only at or below ``threshold``.

    Every failure is a final-state failure, so the only thing that ends a climb
    early is a success. ``threshold=None`` never succeeds.
    """

    def __init__(self, beam: float, threshold: float | None) -> None:
        self.beam = beam
        self.threshold = threshold
        self.attempt_beams: list[float] = []
        self.final_state_not_reached = False
        self._retry_transaction_active = False

    def process_utterance_mfcc(self, mfcc: np.ndarray, transcript: str) -> bool:
        self.attempt_beams.append(self.beam)
        success = self.threshold is not None and self.beam <= self.threshold
        self.final_state_not_reached = not success
        return success

    def set_a_beam(self, beam: float) -> float:
        previous = self.beam
        self.beam = beam
        return previous


def _climb(trainer: LadderTrainer, factors: list[float], fileid: str = "ladder") -> bool:
    return _process_with_final_state_retry(
        trainer,  # type: ignore[arg-type]
        np.zeros((4, 13), dtype=np.float32),
        "<s> TEST </s>",
        normal_beam=1e-90,
        retry_beam_factor=factors,
        fileid=fileid,
        failed_alignment="recover",
    )


def test_ladder_recovers_at_a_wider_rung_and_runs_no_rung_after_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    trainer = LadderTrainer(1e-90, threshold=1e-110)
    caplog.set_level(logging.WARNING, logger="pstrain.lib.steps.train")

    assert _climb(trainer, [1e10, 1e20, 1e30])

    # The first rung fails, the second recovers, and the third never runs.
    assert trainer.attempt_beams == [
        pytest.approx(1e-90),
        pytest.approx(1e-100),
        pytest.approx(1e-110),
    ]
    assert trainer._last_retry_rungs == 2  # type: ignore[attr-defined]
    assert trainer._last_process_retried  # type: ignore[attr-defined]
    assert trainer.beam == pytest.approx(1e-90)
    assert "retrying with a_beam=1e-100 (rung 1 of 3)" in caplog.text
    assert "retrying with a_beam=1e-110 (rung 2 of 3)" in caplog.text
    assert "rung 3 of 3" not in caplog.text


def test_ladder_exhaustion_omits_the_utterance_at_the_widest_beam(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pstrain.lib.steps import train

    measurements: list[str] = []

    def measure(trainer: object, transcript: str, frames: int) -> None:
        measurements.append(transcript)

    monkeypatch.setattr(train, "_infeasible_frame_budget", measure)
    trainer = LadderTrainer(1e-90, threshold=None)
    caplog.set_level(logging.WARNING, logger="pstrain.lib.steps.train")

    assert not _climb(trainer, [1e10, 1e20, 1e30], fileid="unalignable")

    assert trainer.attempt_beams == [
        pytest.approx(1e-90),
        pytest.approx(1e-100),
        pytest.approx(1e-110),
        pytest.approx(1e-120),
    ]
    assert trainer._last_retry_rungs == 3  # type: ignore[attr-defined]
    assert trainer.beam == pytest.approx(1e-90)
    assert "Final state not reached for unalignable even at a_beam=1e-120" in caplog.text
    # The frame budget is measured once for the failure, not once per rung.
    assert measurements == ["<s> TEST </s>"]


def test_infeasible_utterance_runs_no_rung_of_the_ladder(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pstrain.lib.steps import train

    measurements: list[int] = []

    def too_short(trainer: object, transcript: str, frames: int) -> tuple[int, int]:
        measurements.append(frames)
        return (40, frames)

    monkeypatch.setattr(train, "_infeasible_frame_budget", too_short)
    trainer = LadderTrainer(1e-90, threshold=1e-110)
    caplog.set_level(logging.WARNING, logger="pstrain.lib.steps.train")

    assert not _climb(trainer, [1e10, 1e20, 1e30], fileid="short")

    assert trainer.attempt_beams == [pytest.approx(1e-90)]
    assert trainer._last_retry_rungs == 0  # type: ignore[attr-defined]
    assert not trainer._last_process_retried  # type: ignore[attr-defined]
    assert trainer._last_infeasible_frames == (40, 4)  # type: ignore[attr-defined]
    assert measurements == [4]
    assert "retrying" not in caplog.text


def test_ladder_yield_reports_every_rung_and_a_single_rung_reports_as_before(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    from pstrain.lib.steps.train import _report_retry_yield, _report_skip_summary

    caplog.set_level(logging.WARNING, logger="pstrain.lib.steps.train")
    _report_retry_yield(1, 3, 2, ((1e48, 3, 1), (1e72, 2, 1)))
    _report_skip_summary({}, (3, 2), ((1e48, 3, 1), (1e72, 2, 1)))
    assert (
        "iteration 1 widened the beam on 3 utterance(s) and recovered 2 "
        "(rung 1 factor 1e+48: 3 attempted, 1 recovered; "
        "rung 2 factor 1e+72: 2 attempted, 1 recovered)"
    ) in caplog.text
    assert capsys.readouterr().out == (
        "retry\tsecond-pass attempts 3\trecovered 2\n"
        "retry\trung 1 factor 1e+48\tattempts 3\trecovered 1\n"
        "retry\trung 2 factor 1e+72\tattempts 2\trecovered 1\n"
    )

    caplog.clear()
    _report_retry_yield(1, 3, 2, ((1e36, 3, 2),))
    _report_skip_summary({}, (3, 2), ((1e36, 3, 2),))
    assert caplog.messages == ["iteration 1 widened the beam on 3 utterance(s) and recovered 2"]
    assert capsys.readouterr().out == "retry\tsecond-pass attempts 3\trecovered 2\n"


def _ladder_project(tmp_path: Path, short_audio: bool = False) -> Any:
    from pstrain.lib.pipeline import PipelineContext
    from pstrain.lib.pipeline.tasks import build_pipeline
    from pstrain.lib.setup import setup_project

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
    if short_audio:
        _truncated_copy(
            context.audio_dir / "arctic_a0001.wav",
            context.audio_dir / "arctic_short.wav",
            seconds=0.5,
        )
    assert build_pipeline(context).run("flat", jobs=1) == 0
    return context


def _train_ladder(
    context: Any,
    output: Path,
    ids: list[str],
    factors: list[float],
    **options: Any,
) -> Any:
    from pstrain.lib.bw import BWConfig
    from pstrain.lib.steps.train import run_bw_training

    text = "author of the danger trail philip steels etc"
    fileids = context.etc_dir / f"{output.name}.fileids"
    transcription = context.etc_dir / f"{output.name}.transcription"
    fileids.write_text("".join(f"{fileid}\n" for fileid in ids))
    transcription.write_text("".join(f"{fileid} {text}\n" for fileid in ids))
    return run_bw_training(
        model_dir=context.model_dir("flat"),
        output_dir=output,
        features_dir=context.features_dir,
        train_fileids=fileids,
        transcription=transcription,
        dictionary=context.shared_dir / "dictionary.dict",
        filler_dict=context.filler_dict,
        first_pass_2passvar=True,
        n_iter=1,
        config=BWConfig(
            pass2var=True, unobserved_gaussian_policy="zero", a_beam=options.pop("a_beam", 1e-1)
        ),
        retry_beam_factor=factors,
        **options,
    )


@requires_c_library
@pytest.mark.parametrize("n_shards", [1, 2])
def test_native_ladder_recovers_at_a_wider_rung_and_reports_each_rung(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    n_shards: int,
) -> None:
    """Real training: the first rung recovers one utterance, the second the other.

    At a_beam=1e-1 both fixture utterances miss the final state. A factor of 10
    recovers arctic_a0002 but not arctic_a0001, and 1e199 recovers arctic_a0001.
    So arctic_a0002 must stop at the first rung and arctic_a0001 must reach the
    second, and the per-rung counts say exactly that on both accounting routes:
    in process with one shard, and merged from a pool with two.
    """
    import json

    context = _ladder_project(tmp_path)
    ids = ["arctic_a0001", "arctic_a0002"]
    caplog.set_level(logging.WARNING, logger="pstrain.lib.steps.train")
    output = tmp_path / "ladder"
    result = _train_ladder(context, output, ids, [10.0, 1e199], n_shards=n_shards)
    assert result.final_utts == len(ids)
    assert result.total_skipped == 0

    row = json.loads((output / "bw_telemetry.json").read_text())["passes"][-1]
    accounting = row["accounting"]
    assert accounting["retry_attempted_utts"] == 2
    assert accounting["retry_recovered_utts"] == 2
    assert accounting["retry_rungs"] == [
        {"rung": 1, "factor": 10.0, "attempted_utts": 2, "recovered_utts": 1},
        {"rung": 2, "factor": 1e199, "attempted_utts": 1, "recovered_utts": 1},
    ]
    assert (
        "iteration 1 widened the beam on 2 utterance(s) and recovered 2 "
        "(rung 1 factor 10: 2 attempted, 1 recovered; "
        "rung 2 factor 1e+199: 1 attempted, 1 recovered)"
    ) in caplog.text
    assert (
        "retry\tsecond-pass attempts 2\trecovered 2\n"
        "retry\trung 1 factor 10\tattempts 2\trecovered 1\n"
        "retry\trung 2 factor 1e+199\tattempts 1\trecovered 1\n"
    ) in capsys.readouterr().out

    if n_shards > 1:
        # Each shard records its own rungs, and the merge is their sum.
        artifacts = sorted((output / ".bw-accum" / "pass-01").glob("shard-*/artifact.json"))
        shards = [json.loads(path.read_text()) for path in artifacts]
        assert [shard["retry_rung_attempts"] for shard in shards] == [[1, 1], [1, 0]]
        assert [shard["retry_rung_recoveries"] for shard in shards] == [[0, 1], [1, 0]]


@requires_c_library
def test_native_irrecoverable_ladder_keeps_every_rung_diagnostic(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every rung fails, the native record of each attempt survives.

    The engine writes one line per forward pass to the pass diagnostic log, so
    the first attempt and every rung all remain there, and the omission names
    the widest beam that ran. The frame budget is measured once per failure,
    not once per rung.
    """
    import json

    from pstrain.lib.bw import BWTrainer

    context = _ladder_project(tmp_path)
    measured: list[str] = []
    inspect = BWTrainer.inspect_state_seq

    def counting_inspect(self: BWTrainer, transcript: str) -> Any:
        measured.append(transcript)
        return inspect(self, transcript)

    monkeypatch.setattr(BWTrainer, "inspect_state_seq", counting_inspect)
    caplog.set_level(logging.WARNING, logger="pstrain.lib.steps.train")
    output = tmp_path / "irrecoverable"
    with pytest.raises(RuntimeError, match="No utterances processed successfully"):
        _train_ladder(context, output, ["arctic_a0001"], [1.5, 3.0], max_skip_fraction=1.0)

    assert len(measured) == 1
    log = (tmp_path / ".pstrain" / "bw" / "irrecoverable" / "pass-01-shard-00.log").read_text()
    attempts = [line for line in log.splitlines() if "arctic_a0001" in line]
    assert len(attempts) == 3
    assert all(line.rstrip().endswith("failed") for line in attempts)
    assert "Final state not reached for arctic_a0001 even at a_beam=0.0333" in caplog.text

    row = json.loads((output / "bw_telemetry.json").read_text())["passes"][-1]
    assert row["stop_decision"] == "failed"
    assert row["accounting"]["retry_rungs"] == [
        {"rung": 1, "factor": 1.5, "attempted_utts": 1, "recovered_utts": 0},
        {"rung": 2, "factor": 3.0, "attempted_utts": 1, "recovered_utts": 0},
    ]
    out = capsys.readouterr().out
    assert "omitted\tarctic_a0001\tfinal state not reached even at a_beam=0.0333" in out
    assert "retry\trung 2 factor 3\tattempts 1\trecovered 0\n" in out


@requires_c_library
def test_native_infeasible_utterance_runs_no_rung_of_the_ladder(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import json

    context = _ladder_project(tmp_path, short_audio=True)
    caplog.set_level(logging.WARNING, logger="pstrain.lib.steps.train")
    output = tmp_path / "short-ladder"
    result = _train_ladder(
        context,
        output,
        ["arctic_a0001", "arctic_short"],
        [1e10, 1e40],
        a_beam=1e-90,
        max_skip_fraction=0.6,
    )
    assert result.total_skipped == 1
    assert "arctic_short cannot be aligned at any beam width" in caplog.text
    assert "retrying" not in caplog.text
    row = json.loads((output / "bw_telemetry.json").read_text())["passes"][-1]
    assert row["accounting"]["retry_attempted_utts"] == 0
    assert row["accounting"]["retry_rungs"] == [
        {"rung": 1, "factor": 1e10, "attempted_utts": 0, "recovered_utts": 0},
        {"rung": 2, "factor": 1e40, "attempted_utts": 0, "recovered_utts": 0},
    ]


def test_numpy_scalar_factor_is_one_rung_as_before() -> None:
    from pstrain.lib.retry_ladder import retry_ladder

    assert retry_ladder(np.float32(1e10)) == (pytest.approx(1e10),)
    assert retry_ladder(np.int64(100)) == (100.0,)
    assert retry_ladder(np.float64(0.5)) == ()

    trainer = TightBeamTrainer()
    assert _process_with_final_state_retry(
        trainer,  # type: ignore[arg-type]
        np.zeros((4, 13), dtype=np.float32),
        "<s> TEST </s>",
        normal_beam=1e-90,
        retry_beam_factor=np.float64(1e10),  # type: ignore[arg-type]
        fileid="numpy-factor",
    )
    assert trainer.attempt_beams == [pytest.approx(1e-90), pytest.approx(1e-100)]


@requires_c_library
def test_omission_names_the_widest_beam_that_ran_not_the_widest_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A climb that stops early must not record a beam nothing tried.

    The update below misses its final state, and the first rung then fails for
    a reason no wider beam answers, so the climb stops after one rung: a_beam
    1e-90 / 10 = 1e-91 ran, and 1e-93 never did.
    """
    import json

    from pstrain.lib.steps import train

    def stopped_after_one_rung(trainer: Any, *args: object, **kwargs: object) -> bool:
        trainer._last_process_retried = True
        trainer._last_retry_rungs = 1
        trainer._last_infeasible_frames = None
        return False

    context = _ladder_project(tmp_path)
    monkeypatch.setattr(train, "_process_with_final_state_retry", stopped_after_one_rung)
    output = tmp_path / "stopped-early"
    with pytest.raises(RuntimeError, match="No utterances processed successfully"):
        _train_ladder(
            context,
            output,
            ["arctic_a0001"],
            [10.0, 100.0, 1000.0],
            a_beam=1e-90,
            max_skip_fraction=1.0,
        )

    out = capsys.readouterr().out
    assert "omitted\tarctic_a0001\tfinal state not reached even at a_beam=1e-91\tpasses 1" in out
    assert "1e-93" not in out
    row = json.loads((output / "bw_telemetry.json").read_text())["passes"][-1]
    assert row["accounting"]["retry_rungs"][0]["attempted_utts"] == 1
    assert row["accounting"]["retry_rungs"][1]["attempted_utts"] == 0
