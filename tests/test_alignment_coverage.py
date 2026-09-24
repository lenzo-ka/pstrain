"""The mass-and-coverage report on a corpus alignment.

The aggregation and flag rules are tested directly on synthetic outcomes. The
acoustic tests align with a one-Gaussian CI model trained on the mini corpus,
never the flat fixture model, which scores every senone alike and so cannot
produce the rejections the report has to count.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import shutil
import sys
import wave
from pathlib import Path

import pytest

from pstrain.lib.alignment import AlignmentJob, align_corpus, alignment_coverage
from pstrain.lib.alignment.core import AlignedSegment, AlignmentResult, RetryOutcome
from pstrain.lib.alignment.coverage import (
    NO_SPEAKER,
    THIN_MIN_CARRIERS,
    THIN_RATE_RATIO,
    THIN_TOKENS,
    AlignmentCoverage,
    Outcome,
    UtteranceUnits,
    build_coverage,
    job_outcomes,
    speaker_from_id,
    triphones,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "mini_arctic"
_DICT = _FIXTURES / "dictionary.dict"
_FILLER = _FIXTURES / "filler.dict"
_IDS = [f"arctic_a{n:04d}" for n in range(1, 11)]
_FILLERS = frozenset({"SIL", "+NOISE+"})


def _utt(
    utterance_id: str,
    outcome: Outcome,
    phones: str = "",
    seconds: float | None = 1.0,
    speaker: str | None = None,
) -> UtteranceUnits:
    return UtteranceUnits(
        utterance_id=utterance_id,
        outcome=outcome,
        speaker=speaker if speaker is not None else speaker_from_id(utterance_id),
        seconds=seconds,
        phones=tuple(phones.split()),
    )


class TestSpeakerAndTriphones:
    def test_speaker_is_the_text_before_the_first_slash(self) -> None:
        assert speaker_from_id("bdl/arctic_a0001") == "bdl"
        assert speaker_from_id("a/b/c") == "a"
        assert speaker_from_id("arctic_a0001") == NO_SPEAKER
        assert speaker_from_id("/arctic_a0001") == NO_SPEAKER

    def test_triphones_drop_fillers_before_taking_neighbors(self) -> None:
        phones = ["SIL", "HH", "AH", "+NOISE+", "L", "OW", "SIL"]
        assert triphones(phones, _FILLERS) == ["SIL-HH+AH", "HH-AH+L", "AH-L+OW", "L-OW+SIL"]
        # A pause between two phones leaves their contexts as they are without it.
        assert triphones(["AE", "T", "SIL", "DH", "AH"], _FILLERS) == triphones(
            ["AE", "T", "DH", "AH"], _FILLERS
        )
        assert triphones(["AH"], _FILLERS) == ["SIL-AH+SIL"]
        assert triphones([], _FILLERS) == []


class TestMass:
    def test_utterances_and_seconds_split_by_outcome_overall_and_per_speaker(self) -> None:
        coverage = build_coverage(
            [
                _utt("a/1", "first_pass", seconds=2.0),
                _utt("a/2", "retry_accepted", seconds=3.0),
                _utt("b/1", "retry_rejected", seconds=4.0),
                _utt("b/2", "not_recovered", seconds=None),
                _utt("c/1", "not_recovered", seconds=5.0),
            ],
            _FILLERS,
        )

        assert coverage.utterances.as_dict() == {
            "first_pass": 1,
            "retry_accepted": 1,
            "retry_rejected": 1,
            "not_recovered": 2,
        }
        assert coverage.seconds.as_dict() == {
            "first_pass": 2.0,
            "retry_accepted": 3.0,
            "retry_rejected": 4.0,
            "not_recovered": 5.0,
        }
        assert coverage.n_duration_unknown == 1
        assert coverage.run_failure_rate == pytest.approx(3 / 5)
        a_utts, a_seconds = coverage.speakers["a"]
        assert (a_utts.aligned, a_utts.failed, a_seconds.aligned) == (2, 0, 5.0)
        b_utts, b_seconds = coverage.speakers["b"]
        assert (b_utts.retry_rejected, b_utts.not_recovered, b_seconds.failed) == (1, 1, 4.0)
        assert coverage.outcomes == {
            "a/1": "first_pass",
            "a/2": "retry_accepted",
            "b/1": "retry_rejected",
            "b/2": "not_recovered",
            "c/1": "not_recovered",
        }

    def test_speakers_whose_every_utterance_failed_are_flagged(self) -> None:
        coverage = build_coverage(
            [
                _utt("kept/1", "retry_accepted"),
                _utt("kept/2", "not_recovered"),
                _utt("lost/1", "retry_rejected"),
                _utt("lost/2", "not_recovered"),
                _utt("fine/1", "first_pass"),
            ],
            _FILLERS,
        )
        assert coverage.speakers_without_aligned_audio == ("lost",)

    def test_an_empty_run_reports_nothing_and_flags_nothing(self) -> None:
        coverage = build_coverage([], _FILLERS)
        assert coverage.utterances.total == 0
        assert coverage.n_flags == 0
        assert coverage.run_failure_rate == 0.0
        assert "No speaker or unit flagged." in coverage.format()


class TestUnits:
    def test_tokens_are_counted_by_outcome_and_fillers_are_not_units(self) -> None:
        coverage = build_coverage(
            [
                _utt("u1", "first_pass", "SIL AH B SIL AH SIL"),
                _utt("u2", "retry_accepted", "SIL AH SIL"),
            ],
            _FILLERS,
        )
        assert set(coverage.phones) == {"AH", "B"}
        assert coverage.phones["AH"].as_dict() == {
            "first_pass": 2,
            "retry_accepted": 1,
            "retry_rejected": 0,
            "not_recovered": 0,
        }
        # Carriers count utterances, not tokens.
        assert coverage.phone_carriers["AH"].first_pass == 1
        # The pause inside u1 is dropped before contexts are taken: AH B AH.
        assert set(coverage.triphones) == {"SIL-AH+B", "AH-B+AH", "B-AH+SIL", "SIL-AH+SIL"}
        assert coverage.triphones["B-AH+SIL"].first_pass == 1
        assert coverage.triphones["SIL-AH+SIL"].as_dict()["first_pass"] == 0
        assert coverage.triphones["SIL-AH+SIL"].as_dict()["retry_accepted"] == 1

    def test_retry_only_and_lost_units_are_flagged(self) -> None:
        coverage = build_coverage(
            [
                _utt("u1", "first_pass", "SIL AH B SIL"),
                # ZH aligns only through an accepted retry.
                _utt("u2", "retry_accepted", "SIL AH ZH SIL"),
                # OY appears only where nothing aligned; B also appears in failures
                # but has first-pass tokens, so it is neither.
                _utt("u3", "retry_rejected", "SIL OY B SIL"),
                _utt("u4", "not_recovered", "SIL OY SIL"),
            ],
            _FILLERS,
        )
        assert coverage.retry_only_phones == ("ZH",)
        assert coverage.lost_phones == ("OY",)
        assert coverage.retry_only_triphones == ("AH-ZH+SIL", "SIL-AH+ZH")
        # OY-B+SIL is lost even though B is not: the unit is the triphone.
        assert coverage.lost_triphones == ("OY-B+SIL", "SIL-OY+B", "SIL-OY+SIL")

    def test_a_unit_seen_in_accepted_retries_and_failures_is_retry_only_not_lost(self) -> None:
        coverage = build_coverage(
            [
                _utt("u1", "retry_accepted", "SIL ZH SIL"),
                _utt("u2", "not_recovered", "SIL ZH SIL"),
            ],
            _FILLERS,
        )
        assert coverage.retry_only_phones == ("ZH",)
        assert coverage.lost_phones == ()


class TestThinPhoneRule:
    """Fewer than N first-pass tokens, at least C carriers, carrier failure rate >= M x run's."""

    @staticmethod
    def _run(
        thin_first_pass: int,
        thin_failed: int,
        thin_ok_retry: int = 0,
        others_ok: int = 20,
        others_failed: int = 0,
        **rule: float,
    ) -> AlignmentCoverage:
        utts = []
        for i in range(thin_first_pass):
            utts.append(_utt(f"tf{i}", "first_pass", "SIL ZH SIL"))
        for i in range(thin_failed):
            utts.append(
                _utt(f"tx{i}", "not_recovered" if i % 2 else "retry_rejected", "SIL ZH SIL")
            )
        for i in range(thin_ok_retry):
            utts.append(_utt(f"tr{i}", "retry_accepted", "SIL ZH SIL"))
        for i in range(others_ok):
            utts.append(_utt(f"of{i}", "first_pass", "SIL AH SIL"))
        for i in range(others_failed):
            utts.append(_utt(f"ox{i}", "not_recovered", "SIL AH SIL"))
        return build_coverage(utts, _FILLERS, **rule)  # type: ignore[arg-type]

    def test_defaults(self) -> None:
        assert (THIN_TOKENS, THIN_MIN_CARRIERS, THIN_RATE_RATIO) == (50, 3, 2.0)
        coverage = build_coverage([], _FILLERS)
        assert (coverage.thin_tokens, coverage.thin_min_carriers, coverage.thin_rate_ratio) == (
            50,
            3,
            2.0,
        )

    def test_a_thin_phone_over_represented_among_failures_is_flagged(self) -> None:
        # ZH: 2 first-pass tokens, 3 of 5 carriers failed (60%); run: 3 of 25 failed (12%).
        coverage = self._run(thin_first_pass=2, thin_failed=3)
        (thin,) = coverage.thin_phones
        assert (thin.phone, thin.first_pass_tokens, thin.carriers, thin.failed_carriers) == (
            "ZH",
            2,
            5,
            3,
        )
        assert thin.failure_rate == pytest.approx(0.6)
        assert coverage.run_failure_rate == pytest.approx(3 / 25)
        assert "FLAG thin phones" in coverage.format()
        assert "ZH: 2 first-pass tokens; 3/5 carrying utterances failed (60%)" in (
            coverage.format()
        )

    def test_enough_first_pass_tokens_is_not_thin(self) -> None:
        assert self._run(thin_first_pass=2, thin_failed=3, thin_tokens=2).thin_phones == ()
        assert len(self._run(thin_first_pass=2, thin_failed=3, thin_tokens=3).thin_phones) == 1

    def test_too_few_carriers_is_not_flagged(self) -> None:
        # One first-pass carrier and one failed carrier: two carriers, under three.
        assert self._run(thin_first_pass=1, thin_failed=1).thin_phones == ()
        assert len(self._run(thin_first_pass=1, thin_failed=1, thin_min_carriers=2).thin_phones)

    def test_the_rate_bar_is_inclusive(self) -> None:
        # ZH: 1 of 4 carriers failed (25%). Others: 20 ok, 4 failed, so the run
        # fails 5 of 28. The bar is ratio x 5/28; 25% meets it at 1.4 exactly.
        coverage = self._run(thin_first_pass=3, thin_failed=1, others_failed=4, thin_rate_ratio=1.4)
        assert [t.phone for t in coverage.thin_phones] == ["ZH"]
        above = self._run(thin_first_pass=3, thin_failed=1, others_failed=4, thin_rate_ratio=1.41)
        assert above.thin_phones == ()

    def test_the_bar_is_exact_where_floating_point_is_not(self) -> None:
        # The run fails 10 of 100 (0.1); ZH fails 3 of its 10 carriers (0.3).
        # In floating point 3.0 * 0.1 is 0.30000000000000004, above 0.3.
        assert 3.0 * 0.1 > 3 / 10
        coverage = self._run(
            thin_first_pass=7, thin_failed=3, others_ok=83, others_failed=7, thin_rate_ratio=3.0
        )
        assert coverage.utterances.total == 100
        assert coverage.run_failure_rate == pytest.approx(0.1)
        (thin,) = coverage.thin_phones
        assert (thin.phone, thin.carriers, thin.failed_carriers) == ("ZH", 10, 3)

    def test_accepted_retries_are_not_failures(self) -> None:
        # ZH's carriers all needed the retry, but it accepted them: nothing lost.
        coverage = self._run(thin_first_pass=0, thin_failed=0, thin_ok_retry=5, others_failed=2)
        assert coverage.thin_phones == ()
        assert coverage.retry_only_phones == ("ZH",)

    def test_a_run_without_failures_flags_no_thin_phone(self) -> None:
        assert self._run(thin_first_pass=1, thin_failed=0, thin_ok_retry=4).thin_phones == ()


class TestReportForms:
    def test_to_dict_is_json_and_names_every_flagged_unit(self) -> None:
        utts = [_utt(f"s{i}/u", "retry_accepted", f"SIL P{i} SIL") for i in range(100)]
        utts.append(_utt("s0/v", "first_pass", "SIL AH SIL"))
        coverage = build_coverage(utts, _FILLERS)
        data = json.loads(json.dumps(coverage.to_dict()))
        assert len(data["flags"]["retry_only_phones"]) == 100
        assert data["utterances"]["retry_accepted"] == 100
        assert data["thin_phone_rule"] == {
            "thin_tokens": 50,
            "thin_min_carriers": 3,
            "thin_rate_ratio": 2.0,
            "run_failure_rate": 0.0,
        }
        assert set(data["speakers"]) == {f"s{i}" for i in range(100)}

    def test_text_is_bounded_for_large_inventories(self) -> None:
        utts = [_utt(f"s{i}/u", "retry_accepted", f"SIL P{i} SIL") for i in range(100)]
        text = build_coverage(utts, _FILLERS).format()
        assert "FLAG retry-only phones (100):" in text
        assert "... and 40 more" in text
        # 100 speakers: summarized, not tabulated.
        assert "Speakers: 100; 100 with accepted retries" in text
        assert "    s1: " not in text
        assert len(text.splitlines()) < 20

    def test_text_tabulates_a_few_speakers_and_says_when_ids_name_none(self) -> None:
        named = build_coverage(
            [_utt("a/1", "first_pass", "SIL AH SIL"), _utt("b/1", "retry_rejected")], _FILLERS
        ).format()
        assert "    a: 1 first pass, 0 retry accepted, 0 retry rejected, 0 not recovered" in named
        assert "FLAG speakers with no aligned audio (1): b" in named
        unnamed = build_coverage([_utt("u1", "first_pass", "SIL AH SIL")], _FILLERS).format()
        assert "Speakers: none named" in unnamed
        assert "reporting only; acceptance is unchanged" in unnamed


def _result(
    utterance_id: str,
    words: list[str],
    phones: list[str],
    n_frames: int = 150,
    retry: RetryOutcome | None = None,
    frame_rate: int = 100,
) -> AlignmentResult:
    return AlignmentResult(
        utterance_id=utterance_id,
        words=[AlignedSegment(w, i, i) for i, w in enumerate(words)],
        phones=[AlignedSegment(p, i, i) for i, p in enumerate(phones)],
        states=[AlignedSegment(f"p0.s{i}", i, i, -i) for i in range(3)],
        total_score=-12,
        n_frames=n_frames,
        transcript=" ".join(words),
        frame_rate=frame_rate,
        retry=retry,
    )


def _write_wav(path: Path, seconds: float, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\0\0" * int(seconds * rate))


class TestFromAJob:
    def test_outcomes_are_read_from_the_job_and_never_changed(self, tmp_path: Path) -> None:
        retry = RetryOutcome(rung=1, factor=1e136, beam=1e-200, score=-2.0, threshold=-3.0)
        rejected = RetryOutcome(rung=1, factor=1e136, beam=1e-200, score=-9.0, threshold=-3.0)
        job = AlignmentJob(
            model_dir=tmp_path,
            n_utterances=5,
            n_aligned=2,
            n_failed=3,
            results={
                # A context-dependent segment is named "base left right position".
                "spk/first": _result(
                    "spk/first",
                    ["<s>", "a(2)", "</s>"],
                    ["SIL", "EY SIL SIL s", "SIL"],
                    n_frames=200,
                ),
                # Phones not captured: the aligned words give them, with the
                # variant the aligner chose. At 50 frames a second, 150
                # frames are 3 seconds.
                "spk/retried": _result(
                    "spk/retried", ["<s>", "a(2)", "</s>"], [], retry=retry, frame_rate=50
                ),
            },
            errors={
                "other/rejected": "rejected",
                "other/unrecovered": "failed",
                "other/missing": "Audio file not found",
            },
            retry_rejections={"other/rejected": rejected},
        )
        transcripts = {
            "spk/first": "a",
            "spk/retried": "a",
            "other/rejected": "<s> again zzyzx </s>",
            "other/unrecovered": "a across",
            "other/missing": "a",
        }
        audio = tmp_path / "audio"
        _write_wav(audio / "other" / "rejected.wav", 2.5)
        _write_wav(audio / "other" / "unrecovered.wav", 1.5)
        before = copy.deepcopy(job)

        coverage = alignment_coverage(job, transcripts, _DICT, _FILLER, audio_dir=audio)

        assert dataclasses.asdict(job) == dataclasses.asdict(before)
        assert coverage.outcomes == {
            "spk/first": "first_pass",
            "spk/retried": "retry_accepted",
            "other/rejected": "retry_rejected",
            "other/unrecovered": "not_recovered",
            "other/missing": "not_recovered",
        }
        assert job_outcomes(job, transcripts) == coverage.outcomes
        assert coverage.seconds.first_pass == pytest.approx(2.0)
        assert coverage.seconds.retry_accepted == pytest.approx(3.0)
        assert coverage.seconds.retry_rejected == pytest.approx(2.5)
        assert coverage.seconds.not_recovered == pytest.approx(1.5)
        assert coverage.n_duration_unknown == 1
        assert coverage.speakers_without_aligned_audio == ("other",)
        assert coverage.phones_from_alignment is False
        # a(2) is EY in both aligned utterances.
        assert coverage.phones["EY"].aligned == 2
        # Failed transcripts take the variant the aligned output chose, a(2) ->
        # EY, and otherwise each word's first pronunciation: again -> AH G EH N,
        # across -> AH K R AO S.
        assert coverage.phones["EY"].as_dict() == {
            "first_pass": 1,
            "retry_accepted": 1,
            "retry_rejected": 0,
            "not_recovered": 2,
        }
        assert coverage.phones["AH"].as_dict() == {
            "first_pass": 0,
            "retry_accepted": 0,
            "retry_rejected": 1,
            "not_recovered": 1,
        }
        assert coverage.triphones["SIL-AH+G"].retry_rejected == 1
        assert coverage.triphones["EY-AH+K"].not_recovered == 1
        # "a" alone is SIL-EY+SIL in every outcome, so it is not lost.
        assert coverage.triphones["SIL-EY+SIL"].aligned == 2
        assert "SIL-EY+SIL" not in coverage.lost_triphones
        assert coverage.unexpanded_words == {"zzyzx": 1}
        assert "EY" not in coverage.lost_phones
        assert "AH" in coverage.lost_phones

    def test_a_report_that_cannot_be_built_does_not_fail_the_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        audio = tmp_path / "audio"
        audio.mkdir()
        (audio / "utt.wav").touch()

        class FakeAligner:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def align_audio(self, *args: object, **kwargs: object) -> AlignmentResult:
                return _result("utt", ["hello"], ["HH"])

            def retry_yield(self) -> tuple[tuple[float, int, int, int], ...]:
                return ()

            def close(self) -> None:
                pass

        def broken(*args: object, **kwargs: object) -> AlignmentCoverage:
            raise KeyError("boom")

        monkeypatch.setattr("pstrain.lib.alignment.batch.Aligner", FakeAligner)
        monkeypatch.setattr("pstrain.lib.alignment.batch.alignment_coverage", broken)
        with caplog.at_level("WARNING", logger="pstrain.lib.alignment.batch"):
            job = align_corpus({"utt": "hello"}, audio, tmp_path, _DICT)
        assert (job.n_aligned, job.coverage) == (1, None)
        assert "Not reporting alignment mass and coverage: KeyError: 'boom'" in caplog.text

    def test_no_report_when_the_aligner_did_not_start(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FailingAligner:
            def __init__(self, *args: object, **kwargs: object) -> None:
                raise RuntimeError("no model")

        def unexpected(*args: object, **kwargs: object) -> AlignmentCoverage:
            raise AssertionError("the report was built")

        monkeypatch.setattr("pstrain.lib.alignment.batch.Aligner", FailingAligner)
        monkeypatch.setattr("pstrain.lib.alignment.batch.alignment_coverage", unexpected)
        job = align_corpus({"utt": "a"}, tmp_path, tmp_path, _DICT)
        assert (job.n_failed, job.coverage) == (1, None)
        assert job.errors["utt"].startswith("Aligner init failed")

    def test_a_pause_at_a_word_boundary_is_not_a_lost_triphone(self, tmp_path: Path) -> None:
        """Aligned utterances pause between "at" and "the"; a failed one has the
        same words. The cross-word triphone T-DH+AH is the same unit in both."""
        aligned = {
            f"spk/ok{i}": _result(
                f"spk/ok{i}",
                ["<s>", "at", "<sil>", "the", "</s>"],
                ["SIL", "AE", "T", "SIL", "DH", "AH", "SIL"],
            )
            for i in range(2)
        }
        job = AlignmentJob(
            model_dir=tmp_path,
            n_utterances=3,
            n_aligned=2,
            n_failed=1,
            results=aligned,
            errors={"spk/failed": "failed"},
        )
        transcripts = dict.fromkeys([*aligned, "spk/failed"], "<s> at the </s>")

        coverage = alignment_coverage(job, transcripts, _DICT, _FILLER)

        assert coverage.triphones["AE-T+DH"].as_dict() == {
            "first_pass": 2,
            "retry_accepted": 0,
            "retry_rejected": 0,
            "not_recovered": 1,
        }
        assert "AE-T+SIL" not in coverage.triphones
        assert coverage.lost_triphones == ()
        assert coverage.lost_phones == ()

    def test_the_report_can_be_turned_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audio = tmp_path / "audio"
        audio.mkdir()

        class FakeAligner:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def retry_yield(self) -> tuple[tuple[float, int, int, int], ...]:
                return ()

            def close(self) -> None:
                pass

        monkeypatch.setattr("pstrain.lib.alignment.batch.Aligner", FakeAligner)
        on = align_corpus({"utt": "a"}, audio, tmp_path, _DICT)
        off = align_corpus({"utt": "a"}, audio, tmp_path, _DICT, coverage_report=False)
        assert off.coverage is None
        assert on.coverage is not None
        assert on.coverage.outcomes == {"utt": "not_recovered"}
        assert on.coverage.speakers_without_aligned_audio == (NO_SPEAKER,)


# Acoustic tests: a trained model producing first-pass, accepted, rejected and
# unrecovered utterances.

_NOMINAL = 1e-20
_TO_1E200 = 1e180


@pytest.fixture(scope="module")
def trained_ci(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The mini corpus's trained one-Gaussian CI model."""
    if importlib.util.find_spec("fcntl") is None:
        pytest.skip("building the mini-corpus model requires POSIX provenance locking")
    from tests.numeric_harness import create_project

    ctx = create_project(tmp_path_factory.mktemp("alignment-coverage") / "project", "ci-1g")
    return ctx.model_dir("ci-1g")


@pytest.fixture
def in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the aligner in this process, as the other aligner tests do."""
    from pstrain.lib import native_worker

    monkeypatch.setattr(native_worker, "in_worker", lambda: True)


def _speaker_corpus(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Three copies of the mini corpus, one speaker each, plus a speaker whose
    audio is paired with the wrong transcripts."""
    texts = dict(
        line.split(maxsplit=1)
        for line in (_FIXTURES / "transcription.txt").read_text().splitlines()
    )
    audio = tmp_path / "audio"
    transcripts: dict[str, str] = {}
    for speaker in ("s0", "s1", "s2"):
        (audio / speaker).mkdir(parents=True)
        for utterance_id in _IDS:
            shutil.copy(_FIXTURES / "wav" / f"{utterance_id}.wav", audio / speaker)
            transcripts[f"{speaker}/{utterance_id}"] = texts[utterance_id]
    (audio / "rot").mkdir()
    for index, utterance_id in enumerate(_IDS):
        shutil.copy(_FIXTURES / "wav" / f"{utterance_id}.wav", audio / "rot")
        transcripts[f"rot/{utterance_id}"] = texts[_IDS[(index + 1) % len(_IDS)]]
    return audio, transcripts


def _decisions(job: AlignmentJob) -> dict[str, object]:
    """Everything the acceptance check decided, with the report left out."""
    return {
        "results": {
            u: (r.retry, r.n_frames, r.words, r.phones, r.states, r.total_score, r.transcript)
            for u, r in job.results.items()
        },
        "errors": job.errors,
        "rejections": job.retry_rejections,
        "yield": job.retry_yield,
        "calibration": job.retry_calibration,
        "counts": (job.n_utterances, job.n_aligned, job.n_failed),
    }


def test_the_report_changes_no_outcome_and_no_acceptance_decision(
    trained_ci: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: None
) -> None:
    """The same corpus with the report off and on reaches the same decisions,
    and building the report leaves the job exactly as the pass left it."""
    audio, transcripts = _speaker_corpus(tmp_path)
    arguments = {
        "transcripts": transcripts,
        "audio_dir": audio,
        "model_dir": trained_ci,
        "dict_path": _DICT,
        "filler_dict": _FILLER,
        "beam": _NOMINAL,
        "retry_beam_factor": _TO_1E200,
    }
    off = align_corpus(**arguments, coverage_report=False)  # type: ignore[arg-type]

    snapshots: list[dict[str, object]] = []
    real = alignment_coverage

    def watched(job: AlignmentJob, *args: object, **kwargs: object) -> AlignmentCoverage:
        snapshots.append(copy.deepcopy(_decisions(job)))
        coverage = real(job, *args, **kwargs)  # type: ignore[arg-type]
        assert _decisions(job) == snapshots[-1]
        return coverage

    monkeypatch.setattr("pstrain.lib.alignment.batch.alignment_coverage", watched)
    on = align_corpus(**arguments)  # type: ignore[arg-type]

    assert off.coverage is None
    assert on.coverage is not None
    assert len(snapshots) == 1
    assert _decisions(on) == _decisions(off)

    coverage = on.coverage
    outcomes = coverage.outcomes
    assert list(outcomes) == list(transcripts)
    # The run exercises every outcome it can on this model.
    assert set(outcomes.values()) >= {"first_pass", "retry_accepted", "retry_rejected"}
    for utterance_id, outcome in outcomes.items():
        if outcome == "first_pass":
            assert on.results[utterance_id].retry is None
        elif outcome == "retry_accepted":
            assert on.results[utterance_id].retry is not None
        elif outcome == "retry_rejected":
            assert utterance_id in on.retry_rejections
            assert utterance_id not in on.results
        else:
            assert utterance_id in on.errors
            assert utterance_id not in on.retry_rejections
    assert coverage.utterances.aligned == on.n_aligned
    assert coverage.utterances.failed == on.n_failed
    assert coverage.utterances.retry_accepted + coverage.utterances.retry_rejected == sum(
        rung.recovered for rung in on.retry_yield
    )
    assert coverage.utterances.retry_rejected == sum(rung.rejected for rung in on.retry_yield)
    assert set(coverage.speakers) == {"s0", "s1", "s2", "rot"}
    assert coverage.phones_from_alignment
    assert coverage.n_duration_unknown == 0
    fixture_seconds = 0.0
    for utterance_id in _IDS:
        with wave.open(str(_FIXTURES / "wav" / f"{utterance_id}.wav")) as wf:
            fixture_seconds += wf.getnframes() / wf.getframerate()
    assert sum(coverage.seconds.as_dict().values()) == pytest.approx(4 * fixture_seconds, rel=0.02)
    json.dumps(coverage.to_dict())


@pytest.mark.parametrize("formats", [True, False], ids=["prints", "format-fails"])
def test_cli_prints_the_report_after_the_retry_line_and_after_writing_output(
    trained_ci: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    in_process: None,
    formats: bool,
) -> None:
    """The report comes after the retry line and the written segmentations, and
    a report that cannot be formatted warns without losing any output."""
    from pstrain.cli.cli import main

    project = tmp_path / "project"
    (project / "etc").mkdir(parents=True)
    (project / "etc" / "config.yaml").write_text(
        "config_version: 1\nalignment:\n  beam: 1.0e-20\n  retry_beam_factor: 1.0e+180\n"
    )
    audio, transcripts = _speaker_corpus(tmp_path)
    transcript_file = project / "all.transcription"
    transcript_file.write_text("".join(f"<s> {t} </s> ({u})\n" for u, t in transcripts.items()))
    out_dir = tmp_path / "textgrids"
    ctm = tmp_path / "all.ctm"
    if not formats:

        def broken(self: AlignmentCoverage) -> str:
            raise ValueError("cannot format")

        monkeypatch.setattr(AlignmentCoverage, "format", broken)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pstrain",
            "align",
            str(trained_ci),
            "--project-dir",
            str(project),
            "--transcripts",
            str(transcript_file),
            "--audio-dir",
            str(audio),
            "--dict",
            str(_DICT),
            "--filler-dict",
            str(_FILLER),
            "--output-dir",
            str(out_dir),
            "--ctm",
            str(ctm),
        ],
    )
    main()
    combined = "".join(capsys.readouterr())
    assert sorted(out_dir.rglob("*.TextGrid"))
    assert ctm.read_text()
    written = combined.index("CTM saved to:")
    assert combined.index("Retry rung 1") < written
    if formats:
        report = combined.index(
            "Mass and coverage by outcome (reporting only; acceptance is unchanged)"
        )
        assert written < report
        assert "Speakers: 4;" in combined
        assert "    rot: " in combined
        assert "  first pass " in combined
    else:
        assert "Mass and coverage by outcome" not in combined
        assert (
            "Warning: not printing the mass-and-coverage report: ValueError: cannot format"
            in combined
        )
