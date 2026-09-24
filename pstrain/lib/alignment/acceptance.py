"""The acceptance check for alignments a wider-beam retry recovered.

A wider beam can force a transcript onto audio it does not match, so an
alignment the retry recovered is accepted only if it scores like the model's
normal alignments. First-pass alignments at the nominal beam are never checked.

The score is the mean path score per speech frame: the scores of the
alignment's non-filler words, summed, over their frames, in nats. The aligner
already normalizes every frame by that frame's best senone score, so this is a
normalized score. That normalizer grows with the active set on a CD model, so
the same path scores lower at a wider beam, and a threshold is only valid at
the beam it was calibrated at.

The threshold is a low quantile, the target (5% by default), of the scores of
normal alignments realigned directly at the retry beam. A corpus pass
calibrates it from its own first-pass successes, only when a retry recovered
something:

* at most :data:`RETRY_ACCEPTANCE_MAX_SAMPLES` (200) of them, evenly spaced
  through the corpus, are realigned;
* with fewer than :data:`RETRY_ACCEPTANCE_MIN_SAMPLES` (20) scored, no
  threshold is set, and every recovery at that beam is rejected with the
  reason stated. A rejected recovery leaves the utterance failed, as if the
  retry had not recovered it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TypeVar

import numpy as np

from pstrain.lib.alignment.core import (
    DEFAULT_RETRY_ACCEPTANCE_TARGET,
    AlignmentResult,
    RetryOutcome,
)

__all__ = [
    "DEFAULT_RETRY_ACCEPTANCE_TARGET",
    "LOGS3_NATS",
    "RETRY_ACCEPTANCE_MAX_SAMPLES",
    "RETRY_ACCEPTANCE_MIN_SAMPLES",
    "AlignmentRejectedError",
    "RetryCalibration",
    "RungYield",
    "calibration_from_scores",
    "evenly_spaced",
    "read_filler_words",
    "rejection_message",
    "speech_score",
    "threshold_from_scores",
]


RETRY_ACCEPTANCE_MIN_SAMPLES = 20
RETRY_ACCEPTANCE_MAX_SAMPLES = 200

# The aligner's scores are logs3 units, base 1.0003 (sphinx3_align -logbase).
LOGS3_NATS = math.log(1.0003)

# Sentence markers and the silence word are fillers whatever the filler
# dictionary says; the aligner inserts them itself.
_BUILTIN_FILLER_WORDS = frozenset({"<s>", "</s>", "<sil>"})

T = TypeVar("T")


class RungYield(NamedTuple):
    """What one rung of the retry ladder bought.

    ``attempted`` counts utterances that reached the rung and ``recovered``
    those it aligned. ``rejected`` counts the recoveries the acceptance check
    then rejected; it is part of ``recovered``, so the accepted recoveries are
    ``recovered - rejected``.
    """

    factor: float
    attempted: int
    recovered: int
    rejected: int


@dataclass(frozen=True)
class RetryCalibration:
    """The acceptance threshold calibrated for one retry rung.

    Attributes:
        rung: The ladder rung, counting from 1.
        factor: The rung's retry beam factor.
        beam: The beam the calibration alignments were realigned at.
        target: The quantile taken.
        threshold: The threshold, or ``None`` when too few alignments scored.
        n_scored: Alignments realigned at ``beam`` and scored.
        n_unaligned: Sampled alignments that did not align at ``beam``.
        error: Why calibration itself failed, when it did, in brief.
        aligner_lost: Calibration could not run because the aligner's native
            process was lost.
    """

    rung: int
    factor: float
    beam: float
    target: float
    threshold: float | None
    n_scored: int
    n_unaligned: int = 0
    error: str | None = None
    aligner_lost: bool = False

    @property
    def basis(self) -> str:
        """Where the threshold came from, or why there is none, in words."""
        if self.aligner_lost:
            return (
                "the threshold could not be calibrated because the aligner process was "
                f"lost ({self.error}), so every recovery at this beam is rejected"
            )
        if self.error is not None:
            return (
                f"the threshold could not be calibrated ({self.error}), so every recovery "
                "at this beam is rejected"
            )
        if self.threshold is None:
            plural = "" if self.n_scored == 1 else "s"
            return (
                f"the threshold could not be calibrated: {self.n_scored} first-pass "
                f"alignment{plural} realigned at beam {self.beam:.3g}, "
                f"{RETRY_ACCEPTANCE_MIN_SAMPLES} needed (RETRY_ACCEPTANCE_MIN_SAMPLES); "
                "pass a retry acceptance threshold, or set "
                "alignment.retry_acceptance_target to null to accept retries unchecked"
            )
        return (
            f"{_percent(self.target)} quantile of {self.n_scored} first-pass alignments, "
            f"of at most {RETRY_ACCEPTANCE_MAX_SAMPLES}, realigned at beam {self.beam:.3g}"
        )


class AlignmentRejectedError(RuntimeError):
    """A wider-beam retry aligned the utterance, and the acceptance check rejected it."""

    def __init__(self, message: str, outcome: RetryOutcome) -> None:
        super().__init__(message)
        self.outcome = outcome

    def __reduce__(self) -> tuple[type[AlignmentRejectedError], tuple[str, RetryOutcome]]:
        return (type(self), (str(self), self.outcome))


def _percent(target: float) -> str:
    return f"{target * 100:g}%"


def read_filler_words(filler_dict: Path | str | None) -> frozenset[str]:
    """The words whose frames the speech score leaves out.

    The filler dictionary's words, plus the sentence markers and ``<sil>``.
    An unreadable filler dictionary is the aligner's problem to report; here
    it leaves only the built-in fillers.
    """
    words = set(_BUILTIN_FILLER_WORDS)
    if filler_dict is not None:
        try:
            text = Path(filler_dict).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return frozenset(words)
        for line in text.splitlines():
            fields = line.split()
            if fields and not fields[0].startswith(";;"):
                words.add(_base_word(fields[0]))
    return frozenset(words)


def _base_word(word: str) -> str:
    """Strip a pronunciation variant suffix such as ``(2)``."""
    if word.endswith(")") and "(" in word:
        return word[: word.rindex("(")]
    return word


def speech_score(result: AlignmentResult, filler_words: frozenset[str]) -> float | None:
    """Mean path score per speech frame, in nats; ``None`` with no speech frames.

    Built from the word segments, which every alignment carries, and summed
    here rather than natively, so it needs neither phones nor states.
    """
    total = 0
    frames = 0
    for word in result.words:
        if _base_word(word.name) in filler_words:
            continue
        total += word.score
        frames += word.duration_frames
    if frames <= 0:
        return None
    return total * LOGS3_NATS / frames


def threshold_from_scores(scores: Sequence[float], target: float) -> float:
    """The ``target`` quantile of ``scores``, linearly interpolated."""
    return float(np.quantile(np.asarray(scores, dtype=np.float64), target))


def evenly_spaced(items: Sequence[T], limit: int) -> list[T]:
    """At most ``limit`` of ``items``, evenly spaced and in order."""
    if len(items) <= limit:
        return list(items)
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def calibration_from_scores(
    scores: Sequence[float | None],
    *,
    rung: int,
    factor: float,
    beam: float,
    target: float,
) -> RetryCalibration:
    """Calibrate a rung's threshold from realigned scores (``None``: did not align)."""
    scored = [s for s in scores if s is not None]
    unaligned = len(scores) - len(scored)
    threshold = (
        threshold_from_scores(scored, target)
        if len(scored) >= RETRY_ACCEPTANCE_MIN_SAMPLES
        else None
    )
    return RetryCalibration(
        rung=rung,
        factor=factor,
        beam=beam,
        target=target,
        threshold=threshold,
        n_scored=len(scored),
        n_unaligned=unaligned,
    )


def rejection_message(utterance_id: str | None, outcome: RetryOutcome) -> str:
    """Why a recovered alignment was rejected, naming score, threshold and source.

    Prefixed with the utterance id when one is given; a corpus's error table
    is already keyed by it.
    """
    head = (
        f"the retry at beam {outcome.beam:.3g} aligned it, but the acceptance check "
        "rejected the alignment"
    )
    if utterance_id is not None:
        head = f"{utterance_id}: {head}"
    if outcome.threshold is None:
        return f"{head}: {outcome.basis}"
    if outcome.score is None:
        return f"{head}: it has no speech frames to score"
    return (
        f"{head}: its speech score {outcome.score:.2f} nats per frame is below the "
        f"threshold {outcome.threshold:.2f} ({outcome.basis})"
    )
