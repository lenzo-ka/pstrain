"""The acceptance check on alignments a wider-beam retry recovered.

Every test here aligns with a one-Gaussian CI model trained on the mini corpus,
never the flat fixture model: the flat model scores every senone alike, so it
cannot tell a true transcript from a wrong one.

On that model, at a nominal beam of 1e-20 with the retry at 1e-200, arctic_a0003
fails the first pass and is recovered with a normal score, about -2.1 nats per
speech frame. arctic_a0002, the utterance the split held out of training, is
recovered at about -8.4. A transcript moved onto another utterance's audio
scores -10 or lower. The first-pass alignments of the training utterances score
between about -1.8 and -3.0.
"""

from __future__ import annotations

import importlib.util
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from pstrain.lib.alignment import (
    AlignmentRejectedError,
    RungYield,
    align_corpus,
)
from pstrain.lib.alignment.acceptance import (
    LOGS3_NATS,
    RETRY_ACCEPTANCE_MIN_SAMPLES,
    read_filler_words,
    speech_score,
)
from pstrain.lib.alignment.native import Aligner

_FIXTURES = Path(__file__).parent / "fixtures" / "mini_arctic"
_DICT = _FIXTURES / "dictionary.dict"
_FILLER = _FIXTURES / "filler.dict"
_IDS = [f"arctic_a{n:04d}" for n in range(1, 11)]
# Nominal beam and single retry factor putting the retry at exactly 1e-200.
_NOMINAL = 1e-20
_TO_1E200 = 1e180


@pytest.fixture(scope="module")
def trained_ci(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """The mini corpus's trained one-Gaussian CI model, and its feature directory."""
    if importlib.util.find_spec("fcntl") is None:
        pytest.skip("building the mini-corpus model requires POSIX provenance locking")
    from tests.numeric_harness import create_project

    ctx = create_project(tmp_path_factory.mktemp("retry-acceptance") / "project", "ci-1g")
    return ctx.model_dir("ci-1g"), ctx.features_dir


@pytest.fixture
def in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the aligner in this process, as the other aligner tests do."""
    from pstrain.lib import native_worker

    monkeypatch.setattr(native_worker, "in_worker", lambda: True)


def _transcripts() -> dict[str, str]:
    lines = (_FIXTURES / "transcription.txt").read_text().splitlines()
    return dict(line.split(maxsplit=1) for line in lines)


def _mfcc(features: Path, utterance_id: str) -> np.ndarray:
    from pstrain.lib.features import read_sphinx_mfc

    mfcc = read_sphinx_mfc(features / f"{utterance_id}.mfc", veclen=13)
    return np.ascontiguousarray(mfcc, dtype=np.float32)


def _aligner(model: Path, **kwargs: object) -> Aligner:
    kwargs.setdefault("beam", _NOMINAL)
    kwargs.setdefault("retry_beam_factor", _TO_1E200)
    return Aligner(model, _DICT, filler_dict=_FILLER, **kwargs)  # type: ignore[arg-type]


def _corpus(tmp_path: Path, copies: int, rotated: bool = False) -> tuple[Path, dict[str, str]]:
    """Copies of the mini corpus under distinct ids, and optionally rotated transcripts.

    A rotated id pairs each utterance's audio with the next utterance's
    transcript: a whole wrong transcript.
    """
    audio = tmp_path / "audio"
    audio.mkdir(exist_ok=True)
    texts = _transcripts()
    transcripts: dict[str, str] = {}
    for copy in range(copies):
        for utterance_id in _IDS:
            name = f"{utterance_id}_{copy}"
            shutil.copy(_FIXTURES / "wav" / f"{utterance_id}.wav", audio / f"{name}.wav")
            transcripts[name] = texts[utterance_id]
    if rotated:
        for index, utterance_id in enumerate(_IDS):
            name = f"{utterance_id}_rotated"
            shutil.copy(_FIXTURES / "wav" / f"{utterance_id}.wav", audio / f"{name}.wav")
            transcripts[name] = texts[_IDS[(index + 1) % len(_IDS)]]
    return audio, transcripts


def _corpus_job(model: Path, audio: Path, transcripts: dict[str, str], **kwargs: object):  # type: ignore[no-untyped-def]
    kwargs.setdefault("beam", _NOMINAL)
    kwargs.setdefault("retry_beam_factor", _TO_1E200)
    return align_corpus(
        transcripts=transcripts,
        audio_dir=audio,
        model_dir=model,
        dict_path=_DICT,
        filler_dict=_FILLER,
        **kwargs,  # type: ignore[arg-type]
    )


class TestDefault:
    def test_default_is_one_retry_at_1e_200_with_the_check_on(self) -> None:
        import inspect

        from pstrain.lib.alignment import align_utterance
        from pstrain.lib.alignment.core import DEFAULT_RETRY_BEAM_FACTOR
        from pstrain.lib.config.models import AlignmentConfig

        config = AlignmentConfig()
        assert config.retry_beam_factor == 1e136
        assert config.retry_acceptance_target == 0.05
        assert DEFAULT_RETRY_BEAM_FACTOR == 1e136
        for function in (Aligner.__init__, align_corpus, align_utterance):
            parameters = inspect.signature(function).parameters
            assert parameters["retry_beam_factor"].default == 1e136
            assert parameters["retry_acceptance_target"].default == 0.05

    def test_default_retry_is_exactly_a_direct_alignment_at_1e_200(
        self, trained_ci: tuple[Path, Path], in_process: None
    ) -> None:
        """1e-64 / 1e136 is one ulp from 1e-200, and must search exactly as 1e-200 does.

        arctic_a0002 fails at the default beam and at 1e-100, and aligns at
        1e-200. A threshold low enough to accept anything isolates the search.
        """
        model, features = trained_ci
        mfcc = _mfcc(features, "arctic_a0002")
        transcript = _transcripts()["arctic_a0002"]
        with Aligner(
            model, _DICT, filler_dict=_FILLER, include_states=True, retry_acceptance_threshold=-1e9
        ) as aligner:
            retried = aligner.align_mfcc(mfcc, transcript, "arctic_a0002")
        with Aligner(
            model,
            _DICT,
            filler_dict=_FILLER,
            include_states=True,
            beam=1e-200,
            retry_beam_factor=1.0,
        ) as aligner:
            direct = aligner.align_mfcc(mfcc, transcript, "arctic_a0002")

        assert retried.retry is not None
        assert retried.retry.beam == 1e-64 / 1e136
        retried.retry = None
        assert retried == direct


class TestSpeechScore:
    @pytest.mark.parametrize("utterance_id", ["arctic_a0002", "arctic_a0007", "arctic_a0001"])
    def test_word_score_is_the_per_frame_state_score_over_speech(
        self, trained_ci: tuple[Path, Path], in_process: None, utterance_id: str
    ) -> None:
        """The statistic the rulings were measured with, from words alone.

        The measurement summed per-frame state scores over non-filler phones.
        a0002 and a0007 align with an inserted <sil>, which must be left out.
        """
        model, features = trained_ci
        mfcc = _mfcc(features, utterance_id)
        transcript = _transcripts()[utterance_id]
        with Aligner(
            model, _DICT, filler_dict=_FILLER, beam=0.0, retry_beam_factor=1.0, include_states=True
        ) as aligner:
            full = aligner.align_mfcc(mfcc, transcript, utterance_id)
        with Aligner(
            model, _DICT, filler_dict=_FILLER, beam=0.0, retry_beam_factor=1.0, include_phones=False
        ) as aligner:
            words_only = aligner.align_mfcc(mfcc, transcript, utterance_id)

        frames = np.array([state.score for state in full.states], dtype=np.float64)
        speech = [phone for phone in full.phones if phone.name.split()[0] != "SIL"]
        n_speech = sum(phone.duration_frames for phone in speech)
        expected = (
            sum(frames[p.start_frame : p.end_frame + 1].sum() for p in speech)
            * LOGS3_NATS
            / n_speech
        )
        fillers = read_filler_words(_FILLER)
        assert not words_only.phones
        assert speech_score(full, fillers) == pytest.approx(expected, abs=1e-12)
        assert speech_score(words_only, fillers) == pytest.approx(expected, abs=1e-12)
        if utterance_id != "arctic_a0001":
            assert any(word.name == "<sil>" for word in full.words)
            with_silence = sum(w.score for w in full.words) * LOGS3_NATS / full.n_frames
            assert not math.isclose(with_silence, expected)


class TestSingleCall:
    def test_supplied_threshold_accepts_a_normal_recovery_and_rejects_a_poor_one(
        self, trained_ci: tuple[Path, Path], in_process: None
    ) -> None:
        model, features = trained_ci
        texts = _transcripts()
        with _aligner(model, retry_acceptance_threshold=-4.0) as aligner:
            accepted = aligner.align_mfcc(
                _mfcc(features, "arctic_a0003"), texts["arctic_a0003"], "arctic_a0003"
            )
            with pytest.raises(AlignmentRejectedError) as rejected:
                aligner.align_mfcc(
                    _mfcc(features, "arctic_a0002"), texts["arctic_a0002"], "arctic_a0002"
                )
            assert aligner.retry_yield() == (RungYield(_TO_1E200, 2, 2, 1),)

        assert accepted.retry is not None
        assert (accepted.retry.rung, accepted.retry.beam) == (1, 1e-200)
        assert accepted.retry.score == pytest.approx(-2.08, abs=0.01)
        assert accepted.retry.threshold == -4.0
        assert isinstance(rejected.value, RuntimeError)
        message = str(rejected.value)
        assert "retry at beam 1e-200 aligned it" in message
        assert "speech score -8.37 nats per frame is below the threshold -4.00" in message
        assert rejected.value.outcome.score == pytest.approx(-8.37, abs=0.01)

    def test_without_a_threshold_the_retry_does_not_run_and_says_why(
        self, trained_ci: tuple[Path, Path], in_process: None
    ) -> None:
        model, features = trained_ci
        with _aligner(model) as aligner:
            with pytest.raises(RuntimeError, match="the wider-beam retry was not run") as failed:
                aligner.align_mfcc(
                    _mfcc(features, "arctic_a0003"), _transcripts()["arctic_a0003"], "arctic_a0003"
                )
            assert aligner._last_retry_rungs == 0
            assert aligner.retry_yield() == (RungYield(_TO_1E200, 0, 0, 0),)
        assert "retry_acceptance_threshold" in str(failed.value)
        assert "calibrate_retry_acceptance" in str(failed.value)

    def test_check_off_returns_the_recovery_unchecked(
        self, trained_ci: tuple[Path, Path], in_process: None
    ) -> None:
        model, features = trained_ci
        with _aligner(model, retry_acceptance_target=None) as aligner:
            result = aligner.align_mfcc(
                _mfcc(features, "arctic_a0002"), _transcripts()["arctic_a0002"], "arctic_a0002"
            )
            assert aligner.retry_yield() == (RungYield(_TO_1E200, 1, 1, 0),)
        assert result.retry is not None
        assert result.retry.threshold is None
        assert result.retry.basis == "acceptance check off"

    def test_first_pass_alignment_is_never_checked(
        self, trained_ci: tuple[Path, Path], in_process: None
    ) -> None:
        """A wrong transcript that aligns on the first pass passes, as ruled."""
        model, features = trained_ci
        with Aligner(model, _DICT, filler_dict=_FILLER, retry_acceptance_threshold=-4.0) as aligner:
            result = aligner.align_mfcc(
                _mfcc(features, "arctic_a0003"), _transcripts()["arctic_a0004"], "rotated"
            )
        assert result.retry is None
        assert speech_score(result, read_filler_words(_FILLER)) < -10.0  # type: ignore[operator]

    def test_calibration_helper_needs_enough_alignments_and_feeds_the_threshold(
        self, trained_ci: tuple[Path, Path], in_process: None
    ) -> None:
        model, features = trained_ci
        texts = _transcripts()
        good = [(_mfcc(features, u), texts[u]) for u in _IDS if u not in {"arctic_a0002"}] * 3
        with _aligner(model) as aligner:
            with pytest.raises(ValueError, match=f"{RETRY_ACCEPTANCE_MIN_SAMPLES} needed"):
                aligner.calibrate_retry_acceptance(good[:5])
            (threshold,) = aligner.calibrate_retry_acceptance(good)
        with Aligner(
            model, _DICT, filler_dict=_FILLER, beam=1e-200, retry_beam_factor=1.0
        ) as direct:
            fillers = read_filler_words(_FILLER)
            scores = [
                speech_score(direct.align_mfcc(mfcc, text, "direct"), fillers)
                for mfcc, text in good
            ]
        assert threshold == pytest.approx(
            float(np.quantile([s for s in scores if s is not None], 0.05)), abs=1e-12
        )
        with _aligner(model, retry_acceptance_threshold=threshold) as aligner:
            assert (
                aligner.align_mfcc(
                    _mfcc(features, "arctic_a0003"), texts["arctic_a0003"], "arctic_a0003"
                ).retry
                is not None
            )

    def test_supplied_thresholds_must_match_the_ladder(self, trained_ci: tuple[Path, Path]) -> None:
        model, _ = trained_ci
        with pytest.raises(ValueError, match="one threshold per retry rung"):
            _aligner(model, retry_beam_factor=[1e80, 1e180], retry_acceptance_threshold=-4.0)

    @pytest.mark.parametrize("target", [0.0, 0.5, -0.1])
    def test_target_out_of_range_is_refused(
        self, trained_ci: tuple[Path, Path], target: float
    ) -> None:
        model, _ = trained_ci
        with pytest.raises(ValueError, match="retry_acceptance_target"):
            _aligner(model, retry_acceptance_target=target)

    def test_ladder_does_not_climb_past_a_rejected_rung(
        self, trained_ci: tuple[Path, Path], in_process: None
    ) -> None:
        """a0007's audio with a0008's transcript first aligns at 1e-100, and is rejected there."""
        model, features = trained_ci
        with _aligner(
            model, retry_beam_factor=[1e80, 1e180], retry_acceptance_threshold=[-4.0, -4.0]
        ) as aligner:
            with pytest.raises(AlignmentRejectedError) as rejected:
                aligner.align_mfcc(
                    _mfcc(features, "arctic_a0007"), _transcripts()["arctic_a0008"], "rotated"
                )
            assert aligner.retry_yield() == (
                RungYield(1e80, 1, 1, 1),
                RungYield(1e180, 0, 0, 0),
            )
        assert rejected.value.outcome.beam == pytest.approx(1e-100, rel=1e-12)

    def test_rejection_crosses_the_native_worker_boundary(
        self, trained_ci: tuple[Path, Path]
    ) -> None:
        """Judged once, in the caller's process, beside the counts it updates."""
        model, features = trained_ci
        texts = _transcripts()
        with _aligner(model, retry_acceptance_threshold=-4.0) as aligner:
            assert hasattr(aligner, "_proxy")
            accepted = aligner.align_mfcc(
                _mfcc(features, "arctic_a0003"), texts["arctic_a0003"], "arctic_a0003"
            )
            with pytest.raises(AlignmentRejectedError):
                aligner.align_mfcc(
                    _mfcc(features, "arctic_a0002"), texts["arctic_a0002"], "arctic_a0002"
                )
            assert aligner.retry_yield() == (RungYield(_TO_1E200, 2, 2, 1),)
        assert accepted.retry is not None
        assert accepted.retry.basis == "supplied threshold"


class TestCorpus:
    def test_corpus_calibrates_from_its_first_pass_alignments(
        self, trained_ci: tuple[Path, Path], tmp_path: Path
    ) -> None:
        model, features = trained_ci
        audio, transcripts = _corpus(tmp_path, copies=3)
        job = _corpus_job(model, audio, transcripts)

        recovered = {f"arctic_a0002_{c}" for c in range(3)} | {
            f"arctic_a0003_{c}" for c in range(3)
        }
        first_pass = [u for u in transcripts if u not in recovered]
        assert len(first_pass) == 24
        (calibration,) = job.retry_calibration
        assert (calibration.rung, calibration.beam, calibration.n_scored) == (1, 1e-200, 24)

        # The threshold, recomputed independently: the same utterances aligned
        # directly at 1e-200.
        fillers = read_filler_words(_FILLER)
        texts = _transcripts()
        with Aligner(
            model, _DICT, filler_dict=_FILLER, beam=1e-200, retry_beam_factor=1.0
        ) as direct:
            scores = [
                speech_score(direct.align_audio(audio / f"{u}.wav", transcripts[u]), fillers)
                for u in first_pass
            ]
        # This model's front end dithers, so every extraction of the same audio
        # differs slightly; the scores agree to hundredths, not exactly.
        assert calibration.threshold == pytest.approx(
            float(np.quantile([s for s in scores if s is not None], 0.05)), abs=0.05
        )

        for copy in range(3):
            accepted = job.results[f"arctic_a0003_{copy}"]
            assert accepted.retry is not None
            assert accepted.retry.threshold == calibration.threshold
            assert f"arctic_a0002_{copy}" not in job.results
            reason = job.errors[f"arctic_a0002_{copy}"]
            assert "the acceptance check rejected the alignment" in reason
            assert "5% quantile of 24 first-pass alignments, of at most 200" in reason
            assert f"arctic_a0002_{copy}" in job.retry_rejections
        assert job.retry_yield == (RungYield(_TO_1E200, 6, 6, 3),)
        assert (job.n_aligned, job.n_failed) == (27, 3)
        # Judging after the pass keeps the corpus order.
        assert list(job.results) == [u for u in transcripts if u in job.results]
        assert texts  # the fixture transcripts were read

    def test_nothing_recovered_means_no_calibration(
        self, trained_ci: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model, _ = trained_ci
        audio, transcripts = _corpus(tmp_path, copies=1)
        del transcripts["arctic_a0002_0"]
        calls: list[int] = []
        original = Aligner.calibrate_rung

        def spy(self: Aligner, *args: object, **kwargs: object) -> object:
            calls.append(1)
            return original(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Aligner, "calibrate_rung", spy)
        job = _corpus_job(model, audio, transcripts, beam=1e-64, retry_beam_factor=1e136)

        assert job.n_aligned == 9
        assert job.retry_calibration == ()
        assert calls == []
        assert job.retry_yield == (RungYield(1e136, 0, 0, 0),)

    def test_too_few_first_pass_alignments_reject_every_recovery(
        self, trained_ci: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """At 1e-10 most of the corpus needs the retry, leaving too few to calibrate."""
        model, _ = trained_ci
        audio, transcripts = _corpus(tmp_path, copies=1)
        job = _corpus_job(model, audio, transcripts, beam=1e-10, retry_beam_factor=1e190)

        first_pass = [u for u, result in job.results.items() if result.retry is None]
        assert set(job.results) == set(first_pass)
        assert 0 < len(first_pass) < RETRY_ACCEPTANCE_MIN_SAMPLES
        (calibration,) = job.retry_calibration
        assert calibration.threshold is None
        assert calibration.n_scored == len(first_pass)
        recovered = len(transcripts) - len(first_pass)
        assert len(job.retry_rejections) == recovered >= 5
        plural = "" if len(first_pass) == 1 else "s"
        for utterance_id, reason in job.errors.items():
            assert utterance_id in job.retry_rejections
            assert (
                f"the threshold could not be calibrated: {len(first_pass)} first-pass "
                f"alignment{plural} realigned at beam 1e-200, 20 needed "
                "(RETRY_ACCEPTANCE_MIN_SAMPLES)"
            ) in reason
        assert job.retry_yield == (RungYield(1e190, recovered, recovered, recovered),)

    def test_wrong_transcripts_recovered_by_the_retry_are_rejected(
        self, trained_ci: tuple[Path, Path], tmp_path: Path
    ) -> None:
        model, _ = trained_ci
        audio, transcripts = _corpus(tmp_path, copies=3, rotated=True)
        job = _corpus_job(model, audio, transcripts)

        rotated = [u for u in transcripts if u.endswith("_rotated")]
        recovered_rotated = [u for u in rotated if u in job.retry_rejections]
        assert len(recovered_rotated) >= 6
        for utterance_id in rotated:
            if utterance_id in job.results:
                # Only a first-pass alignment, which the check never sees.
                assert job.results[utterance_id].retry is None
        for copy in range(3):
            assert f"arctic_a0003_{copy}" in job.results

    def test_check_off_is_said_and_accepts_recoveries(
        self, trained_ci: tuple[Path, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        model, _ = trained_ci
        audio, transcripts = _corpus(tmp_path, copies=1)
        with caplog.at_level("WARNING"):
            job = _corpus_job(model, audio, transcripts, retry_acceptance_target=None)
        assert "retry acceptance check is off" in caplog.text
        assert job.n_aligned == 10
        assert job.retry_calibration == ()
        assert job.retry_acceptance_target is None

    def test_a_misconfigured_threshold_is_raised_not_recorded_per_utterance(
        self, trained_ci: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """A threshold list that does not match the ladder is the caller's error."""
        model, _ = trained_ci
        audio, transcripts = _corpus(tmp_path, copies=1)
        with pytest.raises(ValueError, match="one threshold per retry rung"):
            _corpus_job(
                model,
                audio,
                transcripts,
                retry_beam_factor=[1e80, 1e180],
                retry_acceptance_threshold=-4.0,
            )


class TestCli:
    @pytest.mark.parametrize("supplied", [False, True], ids=["calibrated", "supplied"])
    def test_cli_reports_rejections_and_their_reason(
        self,
        trained_ci: tuple[Path, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        in_process: None,
        supplied: bool,
    ) -> None:
        from pstrain.cli.cli import main

        model, _ = trained_ci
        project = tmp_path / "project"
        (project / "etc").mkdir(parents=True)
        (project / "etc" / "config.yaml").write_text(
            "config_version: 1\nalignment:\n  beam: 1.0e-20\n  retry_beam_factor: 1.0e+180\n"
        )
        transcript_file = project / "all.transcription"
        texts = _transcripts()
        transcript_file.write_text("".join(f"<s> {texts[u]} </s> ({u})\n" for u in _IDS))
        argv = [
            "pstrain",
            "align",
            str(model),
            "--project-dir",
            str(project),
            "--transcripts",
            str(transcript_file),
            "--audio-dir",
            str(_FIXTURES / "wav"),
            "--dict",
            str(_DICT),
            "--filler-dict",
            str(_FILLER),
        ]
        if supplied:
            argv += ["--retry-acceptance-threshold", "-4.0"]
        monkeypatch.setattr(sys, "argv", argv)

        assert main() != 0
        combined = "".join(capsys.readouterr())
        line = "Retry rung 1 (factor 1e+180, beam 1e-200): 2 attempted, 2 recovered"
        if supplied:
            assert "Retry check: supplied threshold -4" in combined
            assert f"{line}, 1 rejected by the acceptance check (supplied threshold)" in combined
            assert "Aligned 9/10" in combined
        else:
            assert "Retry check: calibrated at the 5% quantile" in combined
            assert f"{line}, 2 rejected by the acceptance check" in combined
            assert "8 first-pass alignments realigned at beam 1e-200, 20 needed" in combined
            assert "Aligned 8/10" in combined
        assert "arctic_a0002: the retry at beam 1e-200 aligned it" in combined


class TestSuppliedThresholdMustBeFinite:
    """NaN compares false with every score, so it would accept every recovery.

    An infinity would accept or reject every recovery. Either would turn the
    check into a no-op or a blanket refusal while reporting that it ran.
    """

    @pytest.mark.parametrize("threshold", [math.nan, math.inf, -math.inf])
    def test_aligner_refuses_a_non_finite_threshold(
        self, trained_ci: tuple[Path, Path], threshold: float
    ) -> None:
        model, _ = trained_ci
        with pytest.raises(ValueError, match="must be a finite number"):
            _aligner(model, retry_acceptance_threshold=threshold)
        with pytest.raises(ValueError, match="must be a finite number"):
            _aligner(
                model,
                retry_beam_factor=[1e80, 1e180],
                retry_acceptance_threshold=[-4.0, threshold],
            )

    def test_corpus_refuses_a_non_finite_threshold_before_aligning(
        self, trained_ci: tuple[Path, Path], tmp_path: Path
    ) -> None:
        model, _ = trained_ci
        audio, transcripts = _corpus(tmp_path, copies=1, rotated=True)
        with pytest.raises(ValueError, match="must be a finite number"):
            _corpus_job(model, audio, transcripts, retry_acceptance_threshold=math.nan)

    @pytest.mark.parametrize("text", ["nan", "inf", "-inf"])
    def test_cli_refuses_a_non_finite_threshold(
        self,
        trained_ci: tuple[Path, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        text: str,
    ) -> None:
        from pstrain.cli.cli import main

        model, _ = trained_ci
        transcript_file = tmp_path / "all.transcription"
        texts = _transcripts()
        transcript_file.write_text(f"<s> {texts['arctic_a0003']} </s> (arctic_a0003)\n")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "pstrain",
                "align",
                str(model),
                "--project-dir",
                str(tmp_path),
                "--transcripts",
                str(transcript_file),
                "--audio-dir",
                str(_FIXTURES / "wav"),
                "--dict",
                str(_DICT),
                # The joined form: argparse reads a bare "-inf" as an option.
                f"--retry-acceptance-threshold={text}",
            ],
        )
        with pytest.raises(SystemExit) as exited:
            main()
        assert exited.value.code != 0
        assert "must be a finite number of nats per speech frame" in capsys.readouterr().err


def test_calibration_lost_with_the_aligner_process_says_so_and_rejects(
    trained_ci: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The helper dies just before calibration: every recovery is rejected, and the reason
    names the lost process rather than suggesting a threshold."""
    import os
    import signal

    from pstrain.lib import native_worker

    model, _ = trained_ci
    audio, transcripts = _corpus(tmp_path, copies=3)
    calibrate_rung = Aligner.calibrate_rung

    def killed_first(self: Aligner, *args: object, **kwargs: object) -> object:
        worker = native_worker._owned_worker()
        assert worker.pid is not None
        os.kill(worker.pid, signal.SIGKILL)
        assert worker._process is not None
        worker._process.join()
        return calibrate_rung(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Aligner, "calibrate_rung", killed_first)
    job = _corpus_job(model, audio, transcripts)

    (calibration,) = job.retry_calibration
    assert calibration.threshold is None
    assert calibration.aligner_lost
    assert len(job.retry_rejections) == 6
    for utterance_id in job.retry_rejections:
        reason = job.errors[utterance_id]
        assert "could not be calibrated because the aligner process was lost" in reason
        assert "every recovery at this beam is rejected" in reason
        assert "pass a retry acceptance threshold" not in reason
        assert len(reason) < 400
    assert job.retry_yield == (RungYield(_TO_1E200, 6, 6, 6),)
