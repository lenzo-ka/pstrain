"""Tests for the forced alignment package (pstrain.lib.alignment)."""

from __future__ import annotations

from pathlib import Path

import pytest

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

    def test_missing_model_files_raises(self, tmp_path: Path) -> None:
        empty_model = tmp_path / "model"
        empty_model.mkdir()
        dict_path = tmp_path / "dict"
        dict_path.write_text("")
        with pytest.raises(FileNotFoundError, match="Model file missing"):
            Aligner(empty_model, dict_path)


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
