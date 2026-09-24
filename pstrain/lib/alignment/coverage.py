"""Mass and coverage of a corpus alignment, by outcome.

Every utterance a corpus pass sees ends in exactly one outcome:

* ``first_pass``: aligned at the nominal beam;
* ``retry_accepted``: a wider-beam retry recovered it and the acceptance check
  accepted the alignment (or the check was off);
* ``retry_rejected``: a retry recovered it and the acceptance check rejected it;
* ``not_recovered``: it did not align, for any other reason: the retry failed
  or was not run, the audio was too short for the transcript, the audio was
  missing, a word had no pronunciation, or the aligner did not start.

This module only counts. It reads the outcomes a pass already decided and
never changes one: which alignments are accepted, the threshold and its
calibration are the acceptance check's alone.

What it counts:

* **Mass**: utterances and seconds of audio per outcome, overall and per
  speaker. Seconds come from the alignment's frames when there is an
  alignment, and from the WAV header otherwise. An utterance whose audio is
  missing or unreadable counts with no duration.
* **Coverage**: phone and triphone tokens per outcome. For ``first_pass`` and
  ``retry_accepted`` the phones are the alignment's own phone segments (or,
  when phones were not captured, the aligned words expanded through the
  dictionary, using the pronunciation variant the aligner chose). For
  ``retry_rejected`` and ``not_recovered`` there is no alignment to read, so
  the transcript's words are expanded through the dictionary. Each word takes
  the pronunciation variant the aligned output chose most often for it, or the
  dictionary's first pronunciation when no aligned utterance contains it, so
  that a word the aligner realized as ``the(2)`` is not counted as a
  different unit in the utterances that failed. Filler phones (``SIL`` and the filler
  dictionary's phones) are never counted as units. A triphone is ``L-P+R``:
  the phone with its neighbors in the utterance's speech phones, across word
  boundaries, with ``SIL`` at the utterance edges. Fillers, including the
  pauses the aligner inserts between words, are removed before neighbors are
  taken. That puts aligned utterances and transcript expansions, which have no
  pauses, on the same basis, so a word boundary where the speaker paused does
  not count as a different triphone.

What it flags:

* **Speakers with no aligned audio**: every utterance was rejected or not
  recovered.
* **Retry-only units**: a phone or triphone whose aligned tokens all come
  from accepted retries, with none from the first pass.
* **Lost units**: a phone or triphone with no aligned tokens at all that
  appears only in rejected or unrecovered utterances.
* **Thin phones**: phones that may point at a lexicon or model problem. A
  phone is flagged when it has fewer than ``thin_tokens`` (default 50)
  first-pass tokens, appears in at least ``thin_min_carriers`` (default 3)
  utterances, and the share of those utterances that did not align (rejected
  or not recovered) is at least ``thin_rate_ratio`` (default 2.0) times the
  run's own share of utterances that did not align. A run with no failures
  flags none.

A speaker is the text before the first ``/`` in the utterance ID, the same
convention the Arctic benchmark uses for speaker-stratified statistics. pstrain
has no other speaker notion on the alignment path. IDs with no ``/`` share one
speaker, reported as ``(no speaker prefix)``, and so do IDs that start with
``/``, whose prefix is empty. A caller with a better notion
passes ``speaker_of`` to :func:`alignment_coverage`.
"""

from __future__ import annotations

import wave
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pstrain.lib.alignment.acceptance import read_filler_words
from pstrain.lib.lexicon_check import base_word, read_pronunciations

if TYPE_CHECKING:
    from pstrain.lib.alignment.batch import AlignmentJob
    from pstrain.lib.alignment.core import AlignmentResult

__all__ = [
    "NO_SPEAKER",
    "OUTCOMES",
    "THIN_MIN_CARRIERS",
    "THIN_RATE_RATIO",
    "THIN_TOKENS",
    "AlignmentCoverage",
    "Outcome",
    "OutcomeSplit",
    "ThinPhone",
    "UtteranceUnits",
    "alignment_coverage",
    "build_coverage",
    "job_outcomes",
    "speaker_from_id",
    "triphones",
]


Outcome = Literal["first_pass", "retry_accepted", "retry_rejected", "not_recovered"]
OUTCOMES: tuple[Outcome, ...] = ("first_pass", "retry_accepted", "retry_rejected", "not_recovered")
_OUTCOME_LABELS = {
    "first_pass": "first pass",
    "retry_accepted": "retry accepted",
    "retry_rejected": "retry rejected",
    "not_recovered": "not recovered",
}

# Thin-phone rule defaults.
THIN_TOKENS = 50
THIN_MIN_CARRIERS = 3
THIN_RATE_RATIO = 2.0

NO_SPEAKER = "(no speaker prefix)"
SILENCE = "SIL"

# How many units a flag list names in the text report before it counts the
# rest; the machine-readable form always lists every flagged unit.
_TEXT_UNIT_LIMIT = 60
# How many speakers the text report tabulates before it summarizes the rest.
_TEXT_SPEAKER_LIMIT = 20


def speaker_from_id(utterance_id: str) -> str:
    """The text before the first ``/``, or :data:`NO_SPEAKER` when there is none.

    An ID that starts with ``/`` has an empty prefix, and so no speaker prefix.
    """
    if "/" in utterance_id:
        speaker = utterance_id.split("/", 1)[0]
        if speaker:
            return speaker
    return NO_SPEAKER


@dataclass
class OutcomeSplit:
    """An amount split by outcome: utterances, seconds, or unit tokens."""

    first_pass: float = 0
    retry_accepted: float = 0
    retry_rejected: float = 0
    not_recovered: float = 0

    def add(self, outcome: Outcome, amount: float = 1) -> None:
        """Add ``amount`` to ``outcome``."""
        setattr(self, outcome, getattr(self, outcome) + amount)

    @property
    def aligned(self) -> float:
        """First pass plus accepted retries: what the output keeps."""
        return self.first_pass + self.retry_accepted

    @property
    def failed(self) -> float:
        """Rejected plus not recovered: what the output loses."""
        return self.retry_rejected + self.not_recovered

    @property
    def total(self) -> float:
        """Every outcome."""
        return self.aligned + self.failed

    def as_dict(self, digits: int | None = None) -> dict[str, float]:
        """The four outcomes, rounded to ``digits`` when given."""
        values: dict[str, float] = {outcome: getattr(self, outcome) for outcome in OUTCOMES}
        if digits is not None:
            return {key: round(value, digits) for key, value in values.items()}
        return values


@dataclass(frozen=True)
class ThinPhone:
    """A phone the thin-phone rule flagged.

    Attributes:
        phone: The phone.
        first_pass_tokens: Its tokens in first-pass alignments.
        carriers: Utterances containing it, in any outcome.
        failed_carriers: Those that were rejected or not recovered.
    """

    phone: str
    first_pass_tokens: int
    carriers: int
    failed_carriers: int

    @property
    def failure_rate(self) -> float:
        """Share of its carriers that did not align."""
        return self.failed_carriers / self.carriers if self.carriers else 0.0


@dataclass(frozen=True)
class UtteranceUnits:
    """One utterance's outcome, speaker, duration and phone sequence.

    ``phones`` is the utterance's phone sequence, fillers included, so that
    triphone contexts can see them. ``seconds`` is ``None`` when the audio's
    duration could not be read. ``unexpanded`` names transcript words with no
    pronunciation in the dictionary.
    """

    utterance_id: str
    outcome: Outcome
    speaker: str
    seconds: float | None
    phones: tuple[str, ...]
    unexpanded: tuple[str, ...] = ()


@dataclass
class AlignmentCoverage:
    """Mass and coverage of one corpus alignment, by outcome.

    Reporting only: nothing here feeds back into which alignments are
    accepted. :doc:`/alignment-coverage` defines each count and flag.
    """

    #: Each utterance's outcome, in corpus order.
    outcomes: dict[str, Outcome] = field(default_factory=dict)
    #: Utterance counts by outcome.
    utterances: OutcomeSplit = field(default_factory=OutcomeSplit)
    #: Audio seconds by outcome.
    seconds: OutcomeSplit = field(default_factory=OutcomeSplit)
    #: Utterances whose duration could not be read.
    n_duration_unknown: int = 0
    #: Per speaker, ``(utterances, seconds)`` by outcome.
    speakers: dict[str, tuple[OutcomeSplit, OutcomeSplit]] = field(default_factory=dict)
    #: Phone tokens by outcome.
    phones: dict[str, OutcomeSplit] = field(default_factory=dict)
    #: Triphone tokens by outcome.
    triphones: dict[str, OutcomeSplit] = field(default_factory=dict)
    #: Utterances containing each phone, by outcome.
    phone_carriers: dict[str, OutcomeSplit] = field(default_factory=dict)
    #: Transcript words with no pronunciation, over the utterances that did not align.
    unexpanded_words: dict[str, int] = field(default_factory=dict)
    #: Speakers whose every utterance was rejected or not recovered.
    speakers_without_aligned_audio: tuple[str, ...] = ()
    #: Phones whose aligned tokens all come from accepted retries.
    retry_only_phones: tuple[str, ...] = ()
    #: Triphones whose aligned tokens all come from accepted retries.
    retry_only_triphones: tuple[str, ...] = ()
    #: Phones that appear only in rejected or unrecovered utterances.
    lost_phones: tuple[str, ...] = ()
    #: Triphones that appear only in rejected or unrecovered utterances.
    lost_triphones: tuple[str, ...] = ()
    #: Phones the thin-phone rule flagged, worst first.
    thin_phones: tuple[ThinPhone, ...] = ()
    #: Share of utterances that did not align.
    run_failure_rate: float = 0.0
    #: Thin-phone rule: the first-pass token ceiling.
    thin_tokens: int = THIN_TOKENS
    #: Thin-phone rule: the minimum carrier count.
    thin_min_carriers: int = THIN_MIN_CARRIERS
    #: Thin-phone rule: the failure-rate multiple.
    thin_rate_ratio: float = THIN_RATE_RATIO
    #: Whether aligned utterances' phones came from phone segments (``True``) or from their
    #: words expanded through the dictionary (``False``).
    phones_from_alignment: bool = True

    @property
    def n_flags(self) -> int:
        """How many speakers and units are flagged, all kinds together."""
        return (
            len(self.speakers_without_aligned_audio)
            + len(self.retry_only_phones)
            + len(self.retry_only_triphones)
            + len(self.lost_phones)
            + len(self.lost_triphones)
            + len(self.thin_phones)
        )

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable form: every aggregate and every flagged unit.

        Per-utterance outcomes are left out; they are in :attr:`outcomes`.
        """
        return {
            "outcomes": list(OUTCOMES),
            "utterances": self.utterances.as_dict(),
            "seconds": self.seconds.as_dict(digits=2),
            "n_duration_unknown": self.n_duration_unknown,
            "speakers": {
                speaker: {
                    "utterances": utterances.as_dict(),
                    "seconds": seconds.as_dict(digits=2),
                }
                for speaker, (utterances, seconds) in self.speakers.items()
            },
            "phones": {unit: split.as_dict() for unit, split in self.phones.items()},
            "triphones": {unit: split.as_dict() for unit, split in self.triphones.items()},
            "phone_carriers": {
                unit: split.as_dict() for unit, split in self.phone_carriers.items()
            },
            "unexpanded_words": dict(self.unexpanded_words),
            "phones_from_alignment": self.phones_from_alignment,
            "flags": {
                "speakers_without_aligned_audio": list(self.speakers_without_aligned_audio),
                "retry_only_phones": list(self.retry_only_phones),
                "retry_only_triphones": list(self.retry_only_triphones),
                "lost_phones": list(self.lost_phones),
                "lost_triphones": list(self.lost_triphones),
                "thin_phones": [
                    {
                        "phone": thin.phone,
                        "first_pass_tokens": thin.first_pass_tokens,
                        "carriers": thin.carriers,
                        "failed_carriers": thin.failed_carriers,
                        "failure_rate": round(thin.failure_rate, 4),
                    }
                    for thin in self.thin_phones
                ],
            },
            "thin_phone_rule": {
                "thin_tokens": self.thin_tokens,
                "thin_min_carriers": self.thin_min_carriers,
                "thin_rate_ratio": self.thin_rate_ratio,
                "run_failure_rate": round(self.run_failure_rate, 4),
            },
        }

    def format(self) -> str:
        """A bounded, human-readable report.

        Speakers are tabulated in full up to a limit and summarized beyond it;
        flagged speakers are always named. Flag lists name units up to a limit
        and count the rest; :meth:`to_dict` names them all.
        """
        lines = ["Mass and coverage by outcome (reporting only; acceptance is unchanged):"]
        total_utts = int(self.utterances.total)
        for outcome in OUTCOMES:
            count = int(getattr(self.utterances, outcome))
            seconds = getattr(self.seconds, outcome)
            share = 100 * count / total_utts if total_utts else 0.0
            lines.append(
                f"  {_OUTCOME_LABELS[outcome]:<15} {count:>7} utts ({share:5.1f}%)  "
                f"{_duration(seconds)}"
            )
        if self.n_duration_unknown:
            lines.append(
                f"  ({self.n_duration_unknown} utterances with unreadable audio have no duration)"
            )
        lines.extend(self._format_speakers())
        lines.extend(self._format_units())
        return "\n".join(lines)

    def _format_speakers(self) -> list[str]:
        lines: list[str] = []
        n_speakers = len(self.speakers)
        if n_speakers == 0 or (n_speakers == 1 and NO_SPEAKER in self.speakers):
            lines.append("  Speakers: none named (utterance IDs have no 'speaker/' prefix)")
        else:
            with_retry = sum(1 for u, _ in self.speakers.values() if u.retry_accepted)
            with_loss = sum(1 for u, _ in self.speakers.values() if u.failed)
            lines.append(
                f"  Speakers: {n_speakers}; {with_retry} with accepted retries, "
                f"{with_loss} with rejected or unrecovered utterances"
            )
            if n_speakers <= _TEXT_SPEAKER_LIMIT:
                for speaker, (utterances, seconds) in sorted(self.speakers.items()):
                    lines.append(
                        f"    {speaker}: "
                        + ", ".join(
                            f"{int(getattr(utterances, o))} {_OUTCOME_LABELS[o]}" for o in OUTCOMES
                        )
                        + f"; aligned {_duration(seconds.aligned)}, "
                        f"lost {_duration(seconds.failed)}"
                    )
        if self.speakers_without_aligned_audio:
            lines.append(
                f"  FLAG speakers with no aligned audio "
                f"({len(self.speakers_without_aligned_audio)}): "
                + ", ".join(self.speakers_without_aligned_audio)
            )
        return lines

    def _format_units(self) -> list[str]:
        n_phones = len(self.phones)
        n_triphones = len(self.triphones)
        source = (
            "aligned phone segments"
            if self.phones_from_alignment
            else "aligned words expanded through the dictionary"
        )
        lines = [
            f"  Units: {n_phones} phones, {n_triphones} triphones "
            f"(aligned utterances from {source}; failed ones from their transcripts)"
        ]
        lines.extend(_flag_line("retry-only phones", self.retry_only_phones))
        lines.extend(_flag_line("retry-only triphones", self.retry_only_triphones))
        lines.extend(_flag_line("phones seen only in failed utterances", self.lost_phones))
        lines.extend(_flag_line("triphones seen only in failed utterances", self.lost_triphones))
        if self.thin_phones:
            lines.append(
                f"  FLAG thin phones, a possible lexicon or model problem "
                f"(< {self.thin_tokens} first-pass tokens, >= {self.thin_min_carriers} "
                f"carriers, failure rate >= {self.thin_rate_ratio:g}x the run's "
                f"{100 * self.run_failure_rate:.1f}%):"
            )
            for thin in self.thin_phones[:_TEXT_UNIT_LIMIT]:
                lines.append(
                    f"    {thin.phone}: {thin.first_pass_tokens} first-pass tokens; "
                    f"{thin.failed_carriers}/{thin.carriers} carrying utterances failed "
                    f"({100 * thin.failure_rate:.0f}%)"
                )
            if len(self.thin_phones) > _TEXT_UNIT_LIMIT:
                lines.append(f"    ... and {len(self.thin_phones) - _TEXT_UNIT_LIMIT} more")
        if self.unexpanded_words:
            words = sorted(self.unexpanded_words)
            named = ", ".join(words[:_TEXT_UNIT_LIMIT])
            more = (
                f" ... and {len(words) - _TEXT_UNIT_LIMIT} more"
                if len(words) > _TEXT_UNIT_LIMIT
                else ""
            )
            lines.append(
                f"  Words with no pronunciation in failed utterances ({len(words)}): {named}{more}"
            )
        if self.n_flags == 0:
            lines.append("  No speaker or unit flagged.")
        return lines


def _flag_line(label: str, units: tuple[str, ...]) -> list[str]:
    if not units:
        return []
    named = ", ".join(units[:_TEXT_UNIT_LIMIT])
    if len(units) > _TEXT_UNIT_LIMIT:
        named += f" ... and {len(units) - _TEXT_UNIT_LIMIT} more"
    return [f"  FLAG {label} ({len(units)}): {named}"]


def _duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s"


def triphones(phones: Iterable[str], filler_phones: frozenset[str]) -> list[str]:
    """The ``L-P+R`` triphones of a phone sequence's speech phones.

    Fillers are removed first, so a pause between two phones does not change
    their contexts; the utterance edges are ``SIL``.
    """
    speech = [phone for phone in phones if phone not in filler_phones]
    padded = [SILENCE, *speech, SILENCE]
    return [
        f"{padded[index - 1]}-{padded[index]}+{padded[index + 1]}"
        for index in range(1, len(padded) - 1)
    ]


def build_coverage(
    utterances: Iterable[UtteranceUnits],
    filler_phones: frozenset[str],
    *,
    thin_tokens: int = THIN_TOKENS,
    thin_min_carriers: int = THIN_MIN_CARRIERS,
    thin_rate_ratio: float = THIN_RATE_RATIO,
    phones_from_alignment: bool = True,
) -> AlignmentCoverage:
    """Aggregate per-utterance outcomes and phone sequences, and apply the flag rules."""
    coverage = AlignmentCoverage(
        thin_tokens=thin_tokens,
        thin_min_carriers=thin_min_carriers,
        thin_rate_ratio=thin_rate_ratio,
        phones_from_alignment=phones_from_alignment,
    )
    for utt in utterances:
        coverage.outcomes[utt.utterance_id] = utt.outcome
        coverage.utterances.add(utt.outcome)
        if utt.seconds is None:
            coverage.n_duration_unknown += 1
        else:
            coverage.seconds.add(utt.outcome, utt.seconds)
        speaker_utts, speaker_seconds = coverage.speakers.setdefault(
            utt.speaker, (OutcomeSplit(), OutcomeSplit())
        )
        speaker_utts.add(utt.outcome)
        if utt.seconds is not None:
            speaker_seconds.add(utt.outcome, utt.seconds)
        carried = set()
        for phone in utt.phones:
            if phone in filler_phones:
                continue
            coverage.phones.setdefault(phone, OutcomeSplit()).add(utt.outcome)
            carried.add(phone)
        for phone in carried:
            coverage.phone_carriers.setdefault(phone, OutcomeSplit()).add(utt.outcome)
        for unit in triphones(utt.phones, filler_phones):
            coverage.triphones.setdefault(unit, OutcomeSplit()).add(utt.outcome)
        if utt.outcome in ("retry_rejected", "not_recovered"):
            for word in utt.unexpanded:
                coverage.unexpanded_words[word] = coverage.unexpanded_words.get(word, 0) + 1

    coverage.speakers_without_aligned_audio = tuple(
        sorted(
            speaker
            for speaker, (utts, _) in coverage.speakers.items()
            if utts.total and not utts.aligned
        )
    )
    coverage.retry_only_phones = _retry_only(coverage.phones)
    coverage.retry_only_triphones = _retry_only(coverage.triphones)
    coverage.lost_phones = _lost(coverage.phones)
    coverage.lost_triphones = _lost(coverage.triphones)

    total = coverage.utterances.total
    coverage.run_failure_rate = coverage.utterances.failed / total if total else 0.0
    coverage.thin_phones = _thin_phones(coverage)
    return coverage


def _retry_only(units: Mapping[str, OutcomeSplit]) -> tuple[str, ...]:
    return tuple(
        sorted(
            unit for unit, split in units.items() if split.retry_accepted and not split.first_pass
        )
    )


def _lost(units: Mapping[str, OutcomeSplit]) -> tuple[str, ...]:
    return tuple(
        sorted(unit for unit, split in units.items() if split.failed and not split.aligned)
    )


def _thin_phones(coverage: AlignmentCoverage) -> tuple[ThinPhone, ...]:
    """Apply the thin-phone rule; see the module docstring."""
    run_failed = int(coverage.utterances.failed)
    run_total = int(coverage.utterances.total)
    if run_failed <= 0:
        return ()
    # failed / carriers >= ratio * run_failed / run_total, cross-multiplied and
    # exact, so the bar is inclusive whatever the ratio. The ratio is taken as
    # written in decimal: 3.0 is 3 and 1.4 is 7/5.
    ratio = Fraction(str(coverage.thin_rate_ratio))
    flagged = []
    for phone, carriers in coverage.phone_carriers.items():
        first_pass = int(coverage.phones[phone].first_pass)
        n_carriers = int(carriers.total)
        failed = int(carriers.failed)
        if first_pass >= coverage.thin_tokens or n_carriers < coverage.thin_min_carriers:
            continue
        if failed and failed * run_total >= ratio * run_failed * n_carriers:
            flagged.append(ThinPhone(phone, first_pass, n_carriers, failed))
    flagged.sort(key=lambda t: (-t.failure_rate, t.first_pass_tokens, t.phone))
    return tuple(flagged)


def job_outcomes(job: AlignmentJob, utterance_ids: Iterable[str]) -> dict[str, Outcome]:
    """Each utterance's outcome, read from what the pass already decided."""
    outcomes: dict[str, Outcome] = {}
    for utt_id in utterance_ids:
        result = job.results.get(utt_id)
        if result is not None:
            outcomes[utt_id] = "first_pass" if result.retry is None else "retry_accepted"
        elif utt_id in job.retry_rejections:
            outcomes[utt_id] = "retry_rejected"
        else:
            outcomes[utt_id] = "not_recovered"
    return outcomes


class _Lexicon:
    """Word to phones, keyed by the dictionary's own spellings."""

    def __init__(self, dict_path: Path, filler_dict: Path | None) -> None:
        self.exact: dict[str, tuple[str, ...]] = {}
        self.first: dict[str, tuple[str, ...]] = {}
        sources = [dict_path] + ([filler_dict] if filler_dict is not None else [])
        for path in sources:
            for word, phones in read_pronunciations(path):
                self.exact.setdefault(word, phones)
                self.first.setdefault(base_word(word), phones)
        filler_phones = {SILENCE}
        if filler_dict is not None:
            for _, phones in read_pronunciations(filler_dict):
                filler_phones.update(phones)
        self.filler_words = read_filler_words(filler_dict)
        self.filler_phones = frozenset(filler_phones)

    def expand(
        self,
        words: Iterable[str],
        *,
        chosen: bool,
        preferred: Mapping[str, str] | None = None,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Phones of ``words``, and the words that had no pronunciation.

        With ``chosen`` a word's own spelling (``word(2)``) picks its variant.
        Otherwise the spelling ``preferred`` gives for the base word is used,
        and failing that the base word's first pronunciation.
        """
        phones: list[str] = []
        missing: list[str] = []
        for word in words:
            base = base_word(word)
            if base in self.filler_words:
                continue
            if chosen:
                found = self.exact.get(word)
            else:
                spelling = preferred.get(base) if preferred is not None else None
                found = self.exact.get(spelling) if spelling is not None else None
            if found is None:
                found = self.first.get(base)
            if found is None:
                missing.append(base)
                continue
            phones.extend(found)
        return tuple(phones), tuple(dict.fromkeys(missing))


def _wav_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            return wf.getnframes() / rate if rate else None
    except (OSError, EOFError, wave.Error):
        return None


def _aligned_phones(result: AlignmentResult) -> tuple[str, ...] | None:
    """Base phones of the alignment's phone segments; ``None`` when it has none.

    A context-dependent segment is named ``base left right position``.
    """
    if result.phones:
        return tuple(segment.name.split()[0] for segment in result.phones if segment.name)
    return None


def alignment_coverage(
    job: AlignmentJob,
    transcripts: Mapping[str, str],
    dict_path: Path,
    filler_dict: Path | None = None,
    audio_dir: Path | None = None,
    audio_ext: str = ".wav",
    *,
    speaker_of: Callable[[str], str] = speaker_from_id,
    thin_tokens: int = THIN_TOKENS,
    thin_min_carriers: int = THIN_MIN_CARRIERS,
    thin_rate_ratio: float = THIN_RATE_RATIO,
) -> AlignmentCoverage:
    """Report a finished corpus alignment's mass and coverage by outcome.

    Reads ``job`` and never changes it.

    Args:
        job: The finished alignment.
        transcripts: The transcripts it aligned, by utterance ID.
        dict_path: The pronunciation dictionary it used.
        filler_dict: The filler dictionary it used, if any.
        audio_dir: Where the audio is, to read durations of utterances that
            did not align. Without it they have no duration.
        audio_ext: Audio file extension.
        speaker_of: Maps an utterance ID to its speaker. Defaults to the text
            before the first ``/``.
        thin_tokens: Thin-phone rule: fewer first-pass tokens than this.
        thin_min_carriers: Thin-phone rule: at least this many carrying utterances.
        thin_rate_ratio: Thin-phone rule: carrier failure rate at least this
            multiple of the run's.

    Returns:
        The :class:`AlignmentCoverage`.
    """
    lexicon = _Lexicon(Path(dict_path), Path(filler_dict) if filler_dict is not None else None)
    outcomes = job_outcomes(job, transcripts)
    aligned = [result for utt_id, result in job.results.items() if utt_id in outcomes]
    from_alignment = all(result.phones for result in aligned)
    preferred = _preferred_variants(aligned, lexicon.filler_words)

    def units() -> Iterator[UtteranceUnits]:
        for utt_id, outcome in outcomes.items():
            yield _utterance_units(
                job,
                transcripts,
                lexicon,
                preferred,
                utt_id,
                outcome,
                speaker_of,
                audio_dir,
                audio_ext,
            )

    return build_coverage(
        units(),
        lexicon.filler_phones,
        thin_tokens=thin_tokens,
        thin_min_carriers=thin_min_carriers,
        thin_rate_ratio=thin_rate_ratio,
        phones_from_alignment=from_alignment,
    )


def _preferred_variants(
    aligned: Iterable[AlignmentResult], filler_words: frozenset[str]
) -> dict[str, str]:
    """Each word's spelling the aligner chose most often, by base word.

    Ties go to the spelling seen first.
    """
    counts: dict[str, Counter[str]] = {}
    for result in aligned:
        for segment in result.words:
            base = base_word(segment.name)
            if base not in filler_words:
                counts.setdefault(base, Counter())[segment.name] += 1
    return {base: spellings.most_common(1)[0][0] for base, spellings in counts.items()}


def _utterance_units(
    job: AlignmentJob,
    transcripts: Mapping[str, str],
    lexicon: _Lexicon,
    preferred: Mapping[str, str],
    utt_id: str,
    outcome: Outcome,
    speaker_of: Callable[[str], str],
    audio_dir: Path | None,
    audio_ext: str,
) -> UtteranceUnits:
    """One utterance's outcome, speaker, duration and phones."""
    result = job.results.get(utt_id)
    unexpanded: tuple[str, ...] = ()
    if result is not None:
        seconds: float | None = result.duration_time()
        phones = _aligned_phones(result)
        if phones is None:
            phones, _ = lexicon.expand((w.name for w in result.words), chosen=True)
    else:
        seconds = (
            _wav_seconds(Path(audio_dir) / f"{utt_id}{audio_ext}")
            if audio_dir is not None
            else None
        )
        phones, unexpanded = lexicon.expand(
            transcripts[utt_id].split(), chosen=False, preferred=preferred
        )
    return UtteranceUnits(
        utterance_id=utt_id,
        outcome=outcome,
        speaker=speaker_of(utt_id),
        seconds=seconds,
        phones=phones,
        unexpanded=unexpanded,
    )
