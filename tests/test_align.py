"""Tests for the forced alignment package (pstrain.lib.alignment)."""

from __future__ import annotations

import shutil
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from pstrain.cli.cli import main
from pstrain.lib._cffi import read_gau, write_gau
from pstrain.lib.alignment import (
    AlignedSegment,
    Aligner,
    AlignmentJob,
    AlignmentResult,
    align_corpus,
    load_transcripts,
    save_ctm,
    save_textgrid,
    to_ctm,
    to_sphinx_segments,
    to_textgrid,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_ALIGNMENT_TRANSCRIPT = "author of the danger trail philip steels etc"


def _alignment_model(tmp_path: Path, **updates: str) -> Path:
    """Copy the real acoustic fixture and update its validated front-end record."""
    model = tmp_path / "model"
    shutil.copytree(_FIXTURES / "multipron_final_state" / "model", model)
    record = dict(
        line.split(maxsplit=1) for line in (model / "feat.params").read_text().splitlines()
    )
    record.update(updates)
    (model / "feat.params").write_text(
        "".join(f"{name} {value}\n" for name, value in record.items())
    )
    return model


def _final_state_case(tmp_path: Path, utterance_id: str) -> tuple[Path, str, Path]:
    """The final-state fixture, one utterance's transcript, and a model copy."""
    fixture = _FIXTURES / "multipron_final_state"
    transcript = next(
        line.split(maxsplit=1)[1]
        for line in (fixture / "transcription.txt").read_text().splitlines()
        if line.startswith(f"{utterance_id} ")
    )
    return fixture, transcript, _alignment_model(tmp_path)


def _downsample_to_8khz(source: Path, output: Path) -> Path:
    """Create the matching 8 kHz waveform used by the real-boundary construction."""
    with wave.open(str(source), "rb") as source_wav:
        samples = np.frombuffer(source_wav.readframes(source_wav.getnframes()), dtype=np.int16)
    with wave.open(str(output), "wb") as output_wav:
        output_wav.setnchannels(1)
        output_wav.setsampwidth(2)
        output_wav.setframerate(8000)
        output_wav.writeframes(samples[::2].tobytes())
    return output


def _sample_result(utterance_id: str = "utt-1") -> AlignmentResult:
    """Build a representative alignment result for export/format tests."""
    words = [
        AlignedSegment(name="hello", start_frame=0, end_frame=9, score=-100),
        AlignedSegment(name="world", start_frame=10, end_frame=24, score=-90),
    ]
    phones = [
        AlignedSegment(name="HH", start_frame=0, end_frame=2, score=-30),
        AlignedSegment(name="AH", start_frame=3, end_frame=9, score=-70),
        AlignedSegment(name="W", start_frame=10, end_frame=14, score=-40),
        AlignedSegment(name="ER", start_frame=15, end_frame=20, score=-25),
        AlignedSegment(name="L", start_frame=21, end_frame=22, score=-15),
        AlignedSegment(name="D", start_frame=23, end_frame=24, score=-10),
    ]
    return AlignmentResult(
        utterance_id=utterance_id,
        words=words,
        phones=phones,
        states=[],
        total_score=-190,
        n_frames=25,
        transcript="hello world",
    )


class TestAlignedSegment:
    def test_duration_frames_is_inclusive(self) -> None:
        seg = AlignedSegment(name="hello", start_frame=10, end_frame=20, score=-100)
        assert seg.duration_frames == 11

    def test_times_use_frame_shift(self) -> None:
        seg = AlignedSegment(name="x", start_frame=0, end_frame=9, score=0)
        assert seg.start_time() == pytest.approx(0.0)
        assert seg.end_time() == pytest.approx(0.10)
        assert seg.duration_time() == pytest.approx(0.10)
        assert seg.duration_time(frame_shift=0.02) == pytest.approx(0.20)


class TestAlignmentResult:
    def test_duration_time(self) -> None:
        result = _sample_result()
        assert result.duration_time() == pytest.approx(0.25)
        assert result.duration_time(frame_shift=0.02) == pytest.approx(0.50)

    def test_optional_states_default(self) -> None:
        result = AlignmentResult(
            utterance_id="u",
            words=[],
            phones=[],
            states=[],
            total_score=0,
            n_frames=0,
        )
        assert result.transcript == ""
        assert result.states == []


class TestTextGridExport:
    def test_contains_two_tiers_by_default(self) -> None:
        text = to_textgrid(_sample_result())
        assert 'class = "IntervalTier"' in text
        assert 'name = "words"' in text
        assert 'name = "phones"' in text
        assert 'name = "states"' not in text
        assert "size = 2" in text

    def test_states_tier_optional(self) -> None:
        result = _sample_result()
        result.states = [AlignedSegment("s1", 0, 24, 0)]
        text = to_textgrid(result, include_states=True)
        assert 'name = "states"' in text
        assert "size = 3" in text

    def test_xmax_matches_n_frames(self) -> None:
        text = to_textgrid(_sample_result())
        assert "xmax = 0.2500" in text

    def test_save_writes_file(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "utt.TextGrid"
        save_textgrid(_sample_result(), out)
        assert out.exists()
        assert out.read_text().startswith('File type = "ooTextFile"')


class TestCTMExport:
    def test_words_level_rows(self) -> None:
        text = to_ctm(_sample_result(), channel="1")
        lines = text.strip().splitlines()
        assert lines == [
            "utt-1 1 0.000 0.100 hello",
            "utt-1 1 0.100 0.150 world",
        ]

    def test_phones_level(self) -> None:
        text = to_ctm(_sample_result(), level="phones")
        assert "HH" in text
        assert "AH" in text
        assert len(text.strip().splitlines()) == 6

    def test_invalid_level_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported CTM level"):
            to_ctm(_sample_result(), level="states")

    def test_empty_result_returns_empty_string(self) -> None:
        empty = AlignmentResult(
            utterance_id="u",
            words=[],
            phones=[],
            states=[],
            total_score=0,
            n_frames=0,
        )
        assert to_ctm(empty) == ""

    def test_save_writes_file(self, tmp_path: Path) -> None:
        out = tmp_path / "utt.ctm"
        save_ctm(_sample_result(), out)
        assert out.exists()
        assert "hello" in out.read_text()


class TestSphinxSegmentsExport:
    def test_header_and_total(self) -> None:
        text = to_sphinx_segments(_sample_result())
        lines = text.splitlines()
        assert lines[0].strip().split() == ["SFrm", "EFrm", "SegScore", "Word"]
        assert lines[-1].endswith(str(-190))
        assert any("hello" in line for line in lines)

    def test_invalid_level_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported sphinx segment level"):
            to_sphinx_segments(_sample_result(), level="states")


class TestAligner:
    def test_class_is_importable(self) -> None:
        assert Aligner is not None
        assert callable(Aligner)

    def test_missing_model_dir_raises(self, tmp_path: Path) -> None:
        dict_path = tmp_path / "dict"
        dict_path.write_text("")
        with pytest.raises(FileNotFoundError):
            Aligner(tmp_path / "does-not-exist", dict_path)

    def test_non_final_state_failure_is_not_retried(self) -> None:
        aligner = object.__new__(Aligner)
        aligner._beam = 1e-64
        aligner._retry_beam_factor = 1e36
        aligner._failed_alignment = "recover"

        # A non-final-state failure (rc != -3) is never retried.
        assert aligner._final_state_retry_beam(-2) is None
        assert aligner._last_alignment_retried is False

        # A final-state failure under "recover" widens the beam once.
        assert aligner._final_state_retry_beam(-3) == 1e-64 / 1e36
        assert aligner._last_alignment_retried is True

        # A retry factor at or below 1 disables the retry.
        aligner._retry_beam_factor = 1.0
        assert aligner._final_state_retry_beam(-3) is None
        assert aligner._last_alignment_retried is False

    @pytest.mark.parametrize("utterance_id", ["arctic_a0336", "arctic_b0424"])
    def test_final_state_failure_gets_one_wider_beam_retry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        utterance_id: str,
    ) -> None:
        from pstrain.lib import native_worker

        monkeypatch.setattr(native_worker, "in_worker", lambda: True)
        fixture = _FIXTURES / "multipron_final_state"
        transcript = next(
            line.split(maxsplit=1)[1]
            for line in (fixture / "transcription.txt").read_text().splitlines()
            if line.startswith(f"{utterance_id} ")
        )
        model = _alignment_model(tmp_path)
        kwargs = {
            "filler_dict": fixture / "filler.dict",
            "include_phones": True,
            # Deliberately tight to induce the same final-state failure that a
            # one-shot retry widening the beam to 1e-80 recovers.
            "beam": 1e-40,
        }

        with (
            Aligner(model, fixture / "dictionary.dict", retry_beam_factor=1.0, **kwargs) as aligner,
            pytest.raises(RuntimeError, match=r"rc=-3"),
        ):
            aligner.align_mfc_file(fixture / f"{utterance_id}.mfc", transcript)

        with Aligner(
            model,
            fixture / "dictionary.dict",
            retry_beam_factor=1e40,
            failed_alignment="recover",
            **kwargs,
        ) as aligner:
            result = aligner.align_mfc_file(fixture / f"{utterance_id}.mfc", transcript)
            aligner._retry_beam_factor = 1.0
            with pytest.raises(RuntimeError, match=r"rc=-3"):
                aligner.align_mfc_file(fixture / f"{utterance_id}.mfc", transcript)

        assert result.phones
        assert result.words[-1].end_frame == result.n_frames - 1

    @pytest.mark.parametrize("utterance_id", ["arctic_a0336"])
    def test_audio_too_short_for_its_transcript_is_named_not_retried(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        utterance_id: str,
    ) -> None:
        """Alignment states the arithmetic instead of widening the beam in vain.

        The aligner builds its own sentence HMM, so it measures its own minimum:
        Baum-Welch builds a different utterance HMM for the same transcript and
        arrives at a different number.
        """
        from pstrain.lib import native_worker
        from pstrain.lib.features import read_sphinx_mfc

        monkeypatch.setattr(native_worker, "in_worker", lambda: True)
        fixture = _FIXTURES / "multipron_final_state"
        transcript = next(
            line.split(maxsplit=1)[1]
            for line in (fixture / "transcription.txt").read_text().splitlines()
            if line.startswith(f"{utterance_id} ")
        )
        model = _alignment_model(tmp_path)
        mfcc = read_sphinx_mfc(fixture / f"{utterance_id}.mfc", veclen=13)

        with Aligner(
            model,
            fixture / "dictionary.dict",
            filler_dict=fixture / "filler.dict",
            retry_beam_factor=1e36,
            failed_alignment="recover",
        ) as aligner:
            required = aligner.minimum_frames(transcript)
            # The measurement is exact, so one frame short fails and the
            # minimum itself succeeds.
            assert required <= mfcc.shape[0]
            assert aligner.align_mfcc(mfcc[:required], transcript, utterance_id)

            with pytest.raises(RuntimeError) as failure:
                aligner.align_mfcc(mfcc[: required - 1], transcript, utterance_id)
            message = str(failure.value)
            assert f"{utterance_id} cannot be aligned at any beam width" in message
            assert f"at least {required} frames" in message
            assert f"the audio has {required - 1}" in message
            assert "the retry was skipped" in message
            assert aligner._last_alignment_retried is False

    def test_ladder_rung_beams_are_relative_to_the_nominal_beam(self) -> None:
        aligner = object.__new__(Aligner)
        aligner._beam = 1e-64
        aligner._retry_beam_factor = [1e36, 1e48]
        aligner._failed_alignment = "recover"

        assert aligner._final_state_retry_beam(-3, 0) == 1e-64 / 1e36
        assert aligner._final_state_retry_beam(-3, 1) == 1e-64 / 1e48
        assert aligner._final_state_retry_beam(-3, 2) is None
        assert aligner._last_alignment_retried is True
        assert aligner._final_state_retry_beam(-2, 0) is None
        assert aligner._last_alignment_retried is False

    @pytest.mark.parametrize("utterance_id", ["arctic_a0336", "arctic_b0424"])
    def test_ladder_recovers_at_a_wider_rung(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        utterance_id: str,
    ) -> None:
        """At beam 1e-40 a factor of 1e10 still fails and 1e40 recovers."""
        from pstrain.lib import native_worker

        monkeypatch.setattr(native_worker, "in_worker", lambda: True)
        fixture, transcript, model = _final_state_case(tmp_path, utterance_id)

        with Aligner(
            model,
            fixture / "dictionary.dict",
            filler_dict=fixture / "filler.dict",
            beam=1e-40,
            retry_beam_factor=[1e10, 1e40],
        ) as aligner:
            result = aligner.align_mfc_file(fixture / f"{utterance_id}.mfc", transcript)
            assert aligner._last_retry_rungs == 2
            assert aligner.retry_yield() == ((1e10, 1, 0), (1e40, 1, 1))
            # The nominal beam is back in force after the climb.
            assert aligner.set_beam(1e-40) == pytest.approx(1e-40)

        assert result.words[-1].end_frame == result.n_frames - 1

    def test_ladder_runs_no_rung_after_the_first_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pstrain.lib import native_worker

        monkeypatch.setattr(native_worker, "in_worker", lambda: True)
        fixture, transcript, model = _final_state_case(tmp_path, "arctic_a0336")

        with Aligner(
            model,
            fixture / "dictionary.dict",
            filler_dict=fixture / "filler.dict",
            beam=1e-40,
            retry_beam_factor=[1e40, 1e60],
        ) as aligner:
            beams: list[float] = []
            set_beam = aligner.set_beam

            def recording_set_beam(beam: float) -> float:
                beams.append(beam)
                return set_beam(beam)

            monkeypatch.setattr(aligner, "set_beam", recording_set_beam)
            aligner.align_mfc_file(fixture / "arctic_a0336.mfc", transcript)
            assert aligner._last_retry_rungs == 1
            assert aligner.retry_yield() == ((1e40, 1, 1), (1e60, 0, 0))

        # One rung, then the nominal beam restored; 1e-100 was never tried.
        assert beams == [pytest.approx(1e-80), pytest.approx(1e-40)]

    def test_irrecoverable_ladder_reports_the_native_diagnostic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every rung fails: the caller gets the engine's own final-state error.

        The native error left behind is the last rung's, the widest search that
        ran, and it is the same message a single retry reports. The frame budget
        is measured once for the failure, not once per rung.
        """
        from pstrain.lib import native_worker

        monkeypatch.setattr(native_worker, "in_worker", lambda: True)
        fixture, transcript, model = _final_state_case(tmp_path, "arctic_a0336")

        with Aligner(
            model,
            fixture / "dictionary.dict",
            filler_dict=fixture / "filler.dict",
            beam=1e-40,
            retry_beam_factor=[1e2, 1e5, 1e10],
        ) as aligner:
            measured: list[str] = []
            minimum_frames = aligner.minimum_frames

            def counting_minimum_frames(text: str) -> int:
                measured.append(text)
                return minimum_frames(text)

            monkeypatch.setattr(aligner, "minimum_frames", counting_minimum_frames)
            with pytest.raises(RuntimeError) as failure:
                aligner.align_mfc_file(fixture / "arctic_a0336.mfc", transcript)
            assert str(failure.value) == (
                "pstrain_align_mfc_file failed: "
                "pstrain_align_mfc_file: align_utt_capture failed (rc=-3)"
            )
            assert measured == [transcript]
            assert aligner._last_retry_rungs == 3
            assert aligner.retry_yield() == ((1e2, 1, 0), (1e5, 1, 0), (1e10, 1, 0))
            assert aligner.set_beam(1e-40) == pytest.approx(1e-40)

    def test_infeasible_utterance_runs_no_rung_of_the_ladder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pstrain.lib import native_worker
        from pstrain.lib.features import read_sphinx_mfc

        monkeypatch.setattr(native_worker, "in_worker", lambda: True)
        fixture, transcript, model = _final_state_case(tmp_path, "arctic_a0336")
        mfcc = read_sphinx_mfc(fixture / "arctic_a0336.mfc", veclen=13)

        with Aligner(
            model,
            fixture / "dictionary.dict",
            filler_dict=fixture / "filler.dict",
            retry_beam_factor=[1e36, 1e48],
        ) as aligner:
            required = aligner.minimum_frames(transcript)
            measured: list[str] = []
            minimum_frames = aligner.minimum_frames

            def counting_minimum_frames(text: str) -> int:
                measured.append(text)
                return minimum_frames(text)

            monkeypatch.setattr(aligner, "minimum_frames", counting_minimum_frames)
            with pytest.raises(RuntimeError, match="the retry was skipped"):
                aligner.align_mfcc(mfcc[: required - 1], transcript, "arctic_a0336")
            assert measured == [transcript]
            assert aligner._last_alignment_retried is False
            assert aligner._last_retry_rungs == 0
            assert aligner.retry_yield() == ((1e36, 0, 0), (1e48, 0, 0))

    def test_ladder_yield_crosses_the_native_worker_boundary(self, tmp_path: Path) -> None:
        """The default aligner runs in a contained worker; its yield must come back."""
        fixture, transcript, model = _final_state_case(tmp_path, "arctic_a0336")

        with Aligner(
            model,
            fixture / "dictionary.dict",
            filler_dict=fixture / "filler.dict",
            beam=1e-40,
            retry_beam_factor=[1e10, 1e40],
        ) as aligner:
            assert hasattr(aligner, "_proxy")
            aligner.align_mfc_file(fixture / "arctic_a0336.mfc", transcript)
            assert aligner.retry_yield() == ((1e10, 1, 0), (1e40, 1, 1))

    def test_invalid_ladder_is_refused_before_loading_anything(self, tmp_path: Path) -> None:
        dict_path = tmp_path / "dict"
        dict_path.write_text("")
        with pytest.raises(ValueError, match="strictly ascend"):
            Aligner(tmp_path / "does-not-exist", dict_path, retry_beam_factor=[1e48, 1e36])

    def test_missing_model_files_raises(self, tmp_path: Path) -> None:
        empty_model = tmp_path / "model"
        empty_model.mkdir()
        dict_path = tmp_path / "dict"
        dict_path.write_text("")
        with pytest.raises(FileNotFoundError, match="Model file missing"):
            Aligner(empty_model, dict_path)

    def test_missing_feat_params_explains_front_end_mismatch(self, tmp_path: Path) -> None:
        model = tmp_path / "model"
        model.mkdir()
        for name in ("mdef", "means", "variances", "mixture_weights", "transition_matrices"):
            (model / name).write_text(name)
        dict_path = tmp_path / "dict"
        dict_path.write_text("")

        with pytest.raises(
            FileNotFoundError,
            match=(
                rf"feat\.params.*{model}.*decode-time front end.*"
                r"silently differ.*feature shape and basis"
            ),
        ):
            Aligner(model, dict_path)

    def test_real_alignment_rejects_12_cepstral_record_with_39d_model(self, tmp_path: Path) -> None:
        model = _alignment_model(tmp_path, **{"-ncep": "12", "-ceplen": "12"})

        with pytest.raises(
            RuntimeError,
            match=r"Feature dimension 36 does not match Gaussian dimension 39",
        ):
            Aligner(
                model,
                _FIXTURES / "mini_arctic" / "dictionary.dict",
                filler_dict=_FIXTURES / "mini_arctic" / "filler.dict",
                beam=1e-200,
            )

    def test_real_alignment_defaults_to_recorded_live_cmn(self, tmp_path: Path) -> None:
        """Exercise CMN precedence against a score-sensitive real model path.

        The copied one-density fixture is made score-sensitive without changing
        its topology or dimensions: all 39-D means are zeroed, then codebooks
        are spread from -1 to 1 on coefficient zero.  Existing variances and
        mixture weights are retained coherently.
        """
        model = _alignment_model(tmp_path, **{"-cmn": "live", "-cmninit": "0,0,0"})
        means = read_gau(str(model / "means"))[0]
        means.fill(0.0)
        means[:, 0, 0, 0] = np.linspace(-1.0, 1.0, means.shape[0])
        assert write_gau(str(model / "means"), means) == 0
        scores = []
        for overrides in ({}, {"cmn": "live", "cmninit": "0,0,0"}, {"cmninit": "40,3,-1"}):
            with Aligner(
                model,
                _FIXTURES / "mini_arctic" / "dictionary.dict",
                filler_dict=_FIXTURES / "mini_arctic" / "filler.dict",
                beam=1e-200,
                **overrides,
            ) as aligner:
                result = aligner.align_audio(
                    _FIXTURES / "mini_arctic" / "wav" / "arctic_a0001.wav",
                    _ALIGNMENT_TRANSCRIPT,
                )
            scores.append(result.total_score)

        assert scores[0] == scores[1] == -773494
        assert scores[2] != scores[0]

    def test_real_alignment_batch_record_preserves_previous_defaults(self, tmp_path: Path) -> None:
        model = _alignment_model(tmp_path)
        scores = []
        for overrides in ({}, {"cmn": "batch", "cmninit": "40,3,-1"}):
            with Aligner(
                model,
                _FIXTURES / "mini_arctic" / "dictionary.dict",
                filler_dict=_FIXTURES / "mini_arctic" / "filler.dict",
                beam=1e-200,
                **overrides,
            ) as aligner:
                result = aligner.align_audio(
                    _FIXTURES / "mini_arctic" / "wav" / "arctic_a0001.wav",
                    _ALIGNMENT_TRANSCRIPT,
                )
            scores.append(result.total_score)

        assert scores[0] == scores[1]

    def test_real_alignment_honors_8khz_profile(self, tmp_path: Path) -> None:
        model = _alignment_model(
            tmp_path,
            **{
                "-samprate": "8000",
                "-nfft": "256",
                "-lowerf": "200.5",
                "-upperf": "3500.5",
            },
        )
        audio = _downsample_to_8khz(
            _FIXTURES / "mini_arctic" / "wav" / "arctic_a0001.wav",
            tmp_path / "arctic_a0001-8khz.wav",
        )

        with Aligner(
            model,
            _FIXTURES / "mini_arctic" / "dictionary.dict",
            filler_dict=_FIXTURES / "mini_arctic" / "filler.dict",
            beam=1e-200,
        ) as aligner:
            result = aligner.align_audio(audio, _ALIGNMENT_TRANSCRIPT)

        assert result.words
        assert result.n_frames > 0

    def test_real_80hz_alignment_preserves_time_in_result_and_exports(self, tmp_path: Path) -> None:
        model = _alignment_model(tmp_path, **{"-frate": "80"})
        audio = _FIXTURES / "mini_arctic" / "wav" / "arctic_a0001.wav"

        with Aligner(
            model,
            _FIXTURES / "mini_arctic" / "dictionary.dict",
            filler_dict=_FIXTURES / "mini_arctic" / "filler.dict",
            beam=1e-200,
        ) as aligner:
            result = aligner.align_audio(audio, _ALIGNMENT_TRANSCRIPT)

        assert result.frame_rate == 80
        assert result.duration_time() == pytest.approx(result.n_frames / 80)
        assert f"xmax = {result.duration_time():.4f}" in to_textgrid(result)
        final_ctm = to_ctm(result).splitlines()[-1].split()
        assert float(final_ctm[2]) + float(final_ctm[3]) == pytest.approx(
            result.duration_time(), abs=0.002
        )

    def test_default_explicit_variant_matches_prechange_collapsed_output(
        self, tmp_path: Path
    ) -> None:
        """The opt-in leaves the recorded default output byte-for-byte unchanged."""
        model = _alignment_model(tmp_path)
        kwargs = {
            "filler_dict": _FIXTURES / "mini_arctic" / "filler.dict",
            "beam": 1e-200,
        }
        transcript = "author of the(2) danger trail philip steels etc"
        with Aligner(model, _FIXTURES / "mini_arctic" / "dictionary.dict", **kwargs) as aligner:
            implicit_default = aligner.align_audio(
                _FIXTURES / "mini_arctic" / "wav" / "arctic_a0001.wav", transcript
            )
        with Aligner(
            model,
            _FIXTURES / "mini_arctic" / "dictionary.dict",
            verbatim_tokens=False,
            **kwargs,
        ) as aligner:
            explicit_default = aligner.align_audio(
                _FIXTURES / "mini_arctic" / "wav" / "arctic_a0001.wav", transcript
            )

        assert implicit_default == explicit_default
        assert implicit_default.total_score == -700195
        assert [
            (word.name, word.start_frame, word.end_frame) for word in implicit_default.words
        ] == [
            ("<s>", 0, 2),
            ("author", 3, 11),
            ("of", 12, 17),
            ("the", 18, 23),
            ("danger", 24, 38),
            ("trail", 39, 50),
            ("philip", 51, 65),
            ("steels", 66, 80),
            ("etc", 81, 101),
            ("</s>", 102, 333),
        ]
        assert [phone.name for phone in implicit_default.phones[6:8]] == ["DH", "AH"]

    def test_verbatim_variant_forces_nonfirst_pronunciation(self, tmp_path: Path) -> None:
        model = _alignment_model(tmp_path)
        with Aligner(
            model,
            _FIXTURES / "mini_arctic" / "dictionary.dict",
            filler_dict=_FIXTURES / "mini_arctic" / "filler.dict",
            beam=1e-200,
            verbatim_tokens=True,
        ) as aligner:
            result = aligner.align_audio(
                _FIXTURES / "mini_arctic" / "wav" / "arctic_a0001.wav",
                "author of the(2) danger trail philip steels etc",
            )

        assert result.words[3].name == "the(2)"
        assert [phone.name for phone in result.phones[6:8]] == ["DH", "IY"]

    def test_verbatim_unknown_variant_names_token(self, tmp_path: Path) -> None:
        model = _alignment_model(tmp_path)
        with (
            Aligner(
                model,
                _FIXTURES / "mini_arctic" / "dictionary.dict",
                filler_dict=_FIXTURES / "mini_arctic" / "filler.dict",
                beam=1e-200,
                verbatim_tokens=True,
            ) as aligner,
            pytest.raises(RuntimeError, match=r"the\(9\).*not in the dictionary"),
        ):
            aligner.align_audio(
                _FIXTURES / "mini_arctic" / "wav" / "arctic_a0001.wav",
                "author of the(9) danger trail philip steels etc",
            )


class TestLoadTranscripts:
    def test_parses_sphinx_format(self, tmp_path: Path) -> None:
        trans_file = tmp_path / "all.transcription"
        trans_file.write_text(
            "<s> hello world </s> (utt-1)\n"
            "<s> goodbye </s> (utt-2)\n"
            "\n"
            "<s> trailing whitespace </s> (utt-3)  \n"
        )
        loaded = load_transcripts(trans_file)
        assert loaded == {
            "utt-1": "<s> hello world </s>",
            "utt-2": "<s> goodbye </s>",
            "utt-3": "<s> trailing whitespace </s>",
        }


class TestAlignCorpus:
    def test_retry_configuration_is_passed_to_aligner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audio = tmp_path / "audio"
        audio.mkdir()
        (audio / "utt.wav").touch()
        captured: dict[str, object] = {}

        class FakeAligner:
            def __init__(self, *args: object, **kwargs: object) -> None:
                captured.update(kwargs)

            def align_audio(self, *args: object, **kwargs: object) -> AlignmentResult:
                return _sample_result("utt")

            def retry_yield(self) -> tuple[tuple[float, int, int], ...]:
                return ()

            def close(self) -> None:
                pass

        monkeypatch.setattr("pstrain.lib.alignment.batch.Aligner", FakeAligner)
        job = align_corpus(
            {"utt": "hello world"},
            audio,
            tmp_path,
            tmp_path / "dict",
            beam=1e-80,
            retry_beam_factor=1e20,
            failed_alignment="omit",
        )

        assert job.n_aligned == 1
        assert captured["beam"] == 1e-80
        assert captured["retry_beam_factor"] == 1e20
        assert captured["failed_alignment"] == "omit"

    @pytest.mark.parametrize(
        ("factor", "expected_yield"),
        [
            (1e20, ((1e20, 0, 0),)),
            ([1e5, 1e10, 1e20], ((1e5, 1, 0), (1e10, 1, 1), (1e20, 0, 0))),
        ],
    )
    def test_worker_death_mid_corpus_keeps_the_partial_job_and_its_yield(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        factor: float | list[float],
        expected_yield: tuple[tuple[float, int, int], ...],
    ) -> None:
        """A helper killed mid-corpus must not discard the alignments already done.

        The later utterances fail, because the replacement helper no longer
        holds the aligner, and the job returns with the first one aligned. The
        per-rung counts live in this process, so the first utterance's climb
        is still reported.
        """
        import os
        import signal

        from pstrain.lib import native_worker

        model = _alignment_model(tmp_path)
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        ids = ["utt1", "utt2", "utt3"]
        for utterance_id in ids:
            shutil.copy(
                _FIXTURES / "mini_arctic" / "wav" / "arctic_a0001.wav",
                audio_dir / f"{utterance_id}.wav",
            )

        calls: list[str] = []
        align_audio = Aligner.align_audio

        def killing_align_audio(self: Aligner, *args: object, **kwargs: object) -> object:
            calls.append("call")
            if len(calls) == 2:
                worker = native_worker._owned_worker()
                assert worker.pid is not None
                os.kill(worker.pid, signal.SIGKILL)
                assert worker._process is not None
                worker._process.join()
            return align_audio(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Aligner, "align_audio", killing_align_audio)
        job = align_corpus(
            transcripts=dict.fromkeys(ids, _ALIGNMENT_TRANSCRIPT),
            audio_dir=audio_dir,
            model_dir=model,
            dict_path=_FIXTURES / "mini_arctic" / "dictionary.dict",
            filler_dict=_FIXTURES / "mini_arctic" / "filler.dict",
            beam=1e-40 if isinstance(factor, list) else 1e-64,
            retry_beam_factor=factor,
        )

        assert (job.n_aligned, job.n_failed) == (1, 2)
        assert set(job.results) == {"utt1"}
        assert set(job.errors) == {"utt2", "utt3"}
        assert job.retry_yield == expected_yield

    def test_corpus_reports_what_each_rung_of_the_ladder_bought(self, tmp_path: Path) -> None:
        """At beam 1e-40 a factor of 1e5 fails this utterance and 1e10 recovers it."""
        model = _alignment_model(tmp_path)
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        shutil.copy(
            _FIXTURES / "mini_arctic" / "wav" / "arctic_a0001.wav",
            audio_dir / "arctic_a0001.wav",
        )

        job = align_corpus(
            transcripts={"arctic_a0001": _ALIGNMENT_TRANSCRIPT},
            audio_dir=audio_dir,
            model_dir=model,
            dict_path=_FIXTURES / "mini_arctic" / "dictionary.dict",
            filler_dict=_FIXTURES / "mini_arctic" / "filler.dict",
            beam=1e-40,
            retry_beam_factor=[1e5, 1e10, 1e20],
        )

        assert job.n_aligned == 1
        assert job.retry_yield == ((1e5, 1, 0), (1e10, 1, 1), (1e20, 0, 0))

    def test_verbatim_unknown_variant_preserves_token(self, tmp_path: Path) -> None:
        model = _alignment_model(tmp_path)
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        shutil.copy(
            _FIXTURES / "mini_arctic" / "wav" / "arctic_a0001.wav",
            audio_dir / "bad.wav",
        )

        job = align_corpus(
            transcripts={"bad": "author of the(9) danger trail philip steels etc"},
            audio_dir=audio_dir,
            model_dir=model,
            dict_path=_FIXTURES / "mini_arctic" / "dictionary.dict",
            filler_dict=_FIXTURES / "mini_arctic" / "filler.dict",
            verbatim_tokens=True,
        )

        assert job.n_failed == 1
        assert "the(9)" in job.errors["bad"]

    def test_unsupported_phone_is_reported_before_the_run(self, tmp_path: Path) -> None:
        # A pronunciation using a phone the model never trained on is dropped
        # by the native lexicon reader. Report it up front, and say so again
        # when the word's utterance then fails.
        model = _alignment_model(tmp_path)
        dict_path = tmp_path / "dictionary.dict"
        source = (_FIXTURES / "mini_arctic" / "dictionary.dict").read_text()
        dict_path.write_text(f"{source}boeuf B OE F\n")
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        shutil.copy(
            _FIXTURES / "mini_arctic" / "wav" / "arctic_a0001.wav",
            audio_dir / "bad.wav",
        )

        job = align_corpus(
            transcripts={"bad": "<s> boeuf </s>"},
            audio_dir=audio_dir,
            model_dir=model,
            dict_path=dict_path,
            filler_dict=_FIXTURES / "mini_arctic" / "filler.dict",
        )

        assert job.phone_report is not None
        assert job.phone_report.words == ("boeuf",)
        assert job.phone_report.missing_phones == ("OE",)
        assert "boeuf" in job.phone_report.format()
        assert job.n_failed == 1
        assert "phone inventory" in job.errors["bad"]
        assert "boeuf (OE)" in job.errors["bad"]

    def test_dropped_base_resolves_to_its_surviving_variant(self, tmp_path: Path) -> None:
        # "boeuf" uses a phone the model does not define and is dropped;
        # "boeuf(2)" survives. The aligner resolves the word to it, as
        # training does, rather than ending the run.
        model = _alignment_model(tmp_path)
        dict_path = tmp_path / "dictionary.dict"
        source = (_FIXTURES / "mini_arctic" / "dictionary.dict").read_text()
        dict_path.write_text(f"{source}boeuf B OE F\nboeuf(2) B AH F\n")
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        for name in ("unrelated", "uses_word"):
            shutil.copy(
                _FIXTURES / "mini_arctic" / "wav" / "arctic_a0001.wav",
                audio_dir / f"{name}.wav",
            )

        job = align_corpus(
            transcripts={
                "unrelated": f"<s> {_ALIGNMENT_TRANSCRIPT} </s>",
                "uses_word": f"<s> {_ALIGNMENT_TRANSCRIPT} boeuf </s>",
            },
            audio_dir=audio_dir,
            model_dir=model,
            dict_path=dict_path,
            filler_dict=_FIXTURES / "mini_arctic" / "filler.dict",
        )

        assert job.phone_report is not None
        assert job.phone_report.resolved_to_alternative == frozenset({"boeuf"})
        assert "will not start" not in job.phone_report.format()

        assert job.errors == {}
        assert job.n_aligned == 2
        result = job.results["uses_word"]
        (word,) = [segment for segment in result.words if segment.name == "boeuf(2)"]
        phones = [
            segment.name
            for segment in result.phones
            if word.start_frame <= segment.start_frame and segment.end_frame <= word.end_frame
        ]
        assert phones == ["B", "AH", "F"]

    def test_dropped_base_aligns_over_every_surviving_variant(self, tmp_path: Path) -> None:
        # The first survivor only becomes the word's base entry; the aligner
        # still considers every surviving alternative. "boeuf(2)" needs more
        # frames than the audio has, so only "boeuf(3)" can align.
        model = _alignment_model(tmp_path)
        dict_path = tmp_path / "dictionary.dict"
        source = (_FIXTURES / "mini_arctic" / "dictionary.dict").read_text()
        too_long = " ".join(["S", "IY"] * 50)
        dict_path.write_text(f"{source}boeuf B OE F\nboeuf(2) {too_long}\nboeuf(3) B AH F\n")
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        shutil.copy(
            _FIXTURES / "mini_arctic" / "wav" / "arctic_a0001.wav",
            audio_dir / "uses_word.wav",
        )

        job = align_corpus(
            transcripts={"uses_word": f"<s> {_ALIGNMENT_TRANSCRIPT} boeuf </s>"},
            audio_dir=audio_dir,
            model_dir=model,
            dict_path=dict_path,
            filler_dict=_FIXTURES / "mini_arctic" / "filler.dict",
        )

        assert job.phone_report is not None
        assert job.phone_report.resolved_to_alternative == frozenset({"boeuf"})
        prose = " ".join(job.phone_report.format().split())
        assert "aligned over all of its surviving alternatives" in prose

        assert job.errors == {}
        names = [segment.name for segment in job.results["uses_word"].words]
        assert "boeuf(3)" in names
        assert "boeuf(2)" not in names

    def test_dropped_base_does_not_promote_a_filler_variant(self, tmp_path: Path) -> None:
        # A base dropped from the main dictionary never promotes an
        # alternative read from the filler dictionary: that would place the
        # word in the filler range, where it could be inserted between words.
        model = _alignment_model(tmp_path)
        dict_path = tmp_path / "dictionary.dict"
        source = (_FIXTURES / "mini_arctic" / "dictionary.dict").read_text()
        dict_path.write_text(f"{source}boeuf B OE F\n")
        filler_path = tmp_path / "filler.dict"
        filler = (_FIXTURES / "mini_arctic" / "filler.dict").read_text()
        filler_path.write_text(f"{filler}boeuf(2) B AH F\n")
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        shutil.copy(
            _FIXTURES / "mini_arctic" / "wav" / "arctic_a0001.wav",
            audio_dir / "unrelated.wav",
        )

        job = align_corpus(
            transcripts={"unrelated": f"<s> {_ALIGNMENT_TRANSCRIPT} </s>"},
            audio_dir=audio_dir,
            model_dir=model,
            dict_path=dict_path,
            filler_dict=filler_path,
        )

        # The aligner ended the native helper along with itself; start a fresh
        # one on this module's import route (see the no-base-line test).
        with Aligner(
            model,
            _FIXTURES / "mini_arctic" / "dictionary.dict",
            filler_dict=_FIXTURES / "mini_arctic" / "filler.dict",
        ):
            pass

        assert job.phone_report is not None
        assert job.phone_report.resolved_to_alternative == frozenset()
        assert job.n_aligned == 0
        assert "Missing base word for 'boeuf(2)'" in job.errors["unrelated"]

    def test_variant_with_no_base_line_still_stops_the_aligner(self, tmp_path: Path) -> None:
        # A dictionary that never had an unsuffixed line for a word is a
        # separate case from a dropped one, and is left as it was: the
        # aligner does not start.
        model = _alignment_model(tmp_path)
        dict_path = tmp_path / "dictionary.dict"
        source = (_FIXTURES / "mini_arctic" / "dictionary.dict").read_text()
        dict_path.write_text(f"{source}boeuf(2) B AH F\n")
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        shutil.copy(
            _FIXTURES / "mini_arctic" / "wav" / "arctic_a0001.wav",
            audio_dir / "unrelated.wav",
        )

        job = align_corpus(
            transcripts={"unrelated": f"<s> {_ALIGNMENT_TRANSCRIPT} </s>"},
            audio_dir=audio_dir,
            model_dir=model,
            dict_path=dict_path,
            filler_dict=_FIXTURES / "mini_arctic" / "filler.dict",
        )

        # The aligner ended the native helper along with itself. Start a fresh
        # one here, on this module's own import route, before asserting: a
        # later test that had to spawn one from inside a CLI command would
        # trip the CLI-to-library boundary guard, because multiprocessing
        # imports the helper module to pickle its entry point.
        with Aligner(
            model,
            _FIXTURES / "mini_arctic" / "dictionary.dict",
            filler_dict=_FIXTURES / "mini_arctic" / "filler.dict",
        ):
            pass

        assert job.n_aligned == 0
        assert "Missing base word for 'boeuf(2)'" in job.errors["unrelated"]

    def test_cli_prints_the_collected_phone_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project = tmp_path / "project"
        (project / "etc").mkdir(parents=True)
        (project / "etc" / "config.yaml").write_text("config_version: 1\n")
        model = _alignment_model(tmp_path)
        dict_path = tmp_path / "dictionary.dict"
        source = (_FIXTURES / "mini_arctic" / "dictionary.dict").read_text()
        dict_path.write_text(f"{source}boeuf B OE F\n")
        transcript_file = project / "bad.transcription"
        transcript_file.write_text("<s> boeuf </s> (bad)\n")
        audio_dir = project / "audio"
        audio_dir.mkdir()
        shutil.copy(
            _FIXTURES / "mini_arctic" / "wav" / "arctic_a0001.wav",
            audio_dir / "bad.wav",
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "pstrain",
                "align",
                str(model),
                "--project-dir",
                str(project),
                "--transcripts",
                str(transcript_file),
                "--audio-dir",
                str(audio_dir),
                "--dict",
                str(dict_path),
                "--filler-dict",
                str(_FIXTURES / "mini_arctic" / "filler.dict"),
            ],
        )

        assert main() != 0
        output = capsys.readouterr()
        combined = output.out + output.err
        assert "Pronunciations using phones the model does not define" in combined
        assert "no pronunciation survived the model's phone inventory for: boeuf (OE)" in combined
        assert "boeuf" in combined
        assert "Undefined: OE" in combined
        # The report precedes the corpus pass rather than trailing it.
        assert combined.index("Undefined: OE") < combined.index("Aligning...")

    def test_cli_reports_each_rung_of_a_configured_ladder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from pstrain.lib import native_worker

        # Keep the aligner in this process, as the other aligner tests do.
        monkeypatch.setattr(native_worker, "in_worker", lambda: True)
        project = tmp_path / "project"
        (project / "etc").mkdir(parents=True)
        (project / "etc" / "config.yaml").write_text(
            "config_version: 1\n"
            "alignment:\n"
            "  beam: 1.0e-40\n"
            "  retry_beam_factor: [1.0e+10, 1.0e+20]\n"
        )
        model = _alignment_model(tmp_path)
        transcript_file = project / "ladder.transcription"
        transcript_file.write_text(f"<s> {_ALIGNMENT_TRANSCRIPT} </s> (arctic_a0001)\n")
        audio_dir = project / "audio"
        audio_dir.mkdir()
        shutil.copy(
            _FIXTURES / "mini_arctic" / "wav" / "arctic_a0001.wav",
            audio_dir / "arctic_a0001.wav",
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "pstrain",
                "align",
                str(model),
                "--project-dir",
                str(project),
                "--transcripts",
                str(transcript_file),
                "--audio-dir",
                str(audio_dir),
                "--dict",
                str(_FIXTURES / "mini_arctic" / "dictionary.dict"),
                "--filler-dict",
                str(_FIXTURES / "mini_arctic" / "filler.dict"),
            ],
        )

        assert main() == 0
        output = capsys.readouterr()
        combined = output.out + output.err
        assert "Aligned 1/1" in combined
        assert "Retry rung 1 (factor 1e+10): 1 attempted, 0 recovered" in combined
        assert "Retry rung 2 (factor 1e+20): 1 attempted, 1 recovered" in combined

    def test_profile_cli_unknown_variant_exits_nonzero_and_names_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        project = tmp_path / "project"
        (project / "etc").mkdir(parents=True)
        (project / "etc" / "config.yaml").write_text(
            "config_version: 1\nalignment:\n  verbatim_tokens: true\n"
        )
        model = _alignment_model(tmp_path)
        transcript_file = project / "bad.transcription"
        transcript_file.write_text(
            "<s> author of the(9) danger trail philip steels etc </s> (bad)\n"
        )
        audio_dir = project / "audio"
        audio_dir.mkdir()
        shutil.copy(
            _FIXTURES / "mini_arctic" / "wav" / "arctic_a0001.wav",
            audio_dir / "bad.wav",
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "pstrain",
                "align",
                str(model),
                "--project-dir",
                str(project),
                "--transcripts",
                str(transcript_file),
                "--audio-dir",
                str(audio_dir),
                "--dict",
                str(_FIXTURES / "mini_arctic" / "dictionary.dict"),
                "--filler-dict",
                str(_FIXTURES / "mini_arctic" / "filler.dict"),
            ],
        )

        assert main() != 0
        output = capsys.readouterr()
        assert "the(9)" in output.out + output.err

    def test_missing_model_records_per_utt_error(self, tmp_path: Path) -> None:
        # Aligner init fails (model files missing) -> every utterance is
        # marked failed with the same init error.
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        dict_path = tmp_path / "dict"
        dict_path.write_text("")

        job = align_corpus(
            transcripts={"missing": "hello"},
            audio_dir=audio_dir,
            model_dir=model_dir,
            dict_path=dict_path,
        )
        assert isinstance(job, AlignmentJob)
        assert job.n_utterances == 1
        assert job.n_aligned == 0
        assert job.n_failed == 1
        assert "missing" in job.errors
        assert job.success_rate == 0.0

    def test_success_rate_zero_when_empty(self, tmp_path: Path) -> None:
        job = align_corpus(
            transcripts={},
            audio_dir=tmp_path,
            model_dir=tmp_path,
            dict_path=tmp_path / "dict",
        )
        assert job.success_rate == 0.0
        assert job.n_utterances == 0
