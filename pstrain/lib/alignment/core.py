"""Core alignment dataclasses and the single-utterance convenience.

The alignment engine lives in :mod:`pstrain.lib.alignment.native` (CFFI
bindings to the sphinx3 aligner compiled into libpstrainc). This module
keeps the public :class:`AlignedSegment` / :class:`AlignmentResult`
shape and a thin :func:`align_utterance` wrapper that constructs an
:class:`~pstrain.lib.alignment.native.Aligner` for a one-shot call.

For corpus-scale work create one :class:`Aligner` and reuse it; see
:func:`pstrain.lib.alignment.batch.align_corpus`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pstrain.lib.retry_ladder import RetryBeamFactor

# Default pruning beam for the sphinx3 aligner. 1e-64 matches the
# upstream sphinx3_align CLI default; it is wide enough for cd-1g
# through cd-8g acoustic models on clean read speech.
DEFAULT_BEAM = 1e-64
# One retry at 1e-200 on the default beam. A retry-recovered alignment must
# pass the acceptance check (``pstrain.lib.alignment.acceptance``).
DEFAULT_RETRY_BEAM_FACTOR = 1e136
DEFAULT_RETRY_ACCEPTANCE_TARGET = 0.05

# Default retained for callers constructing segments independently of a result.
FRAME_SHIFT_SECONDS = 0.01


@dataclass
class AlignedSegment:
    """A single aligned segment (word, phone, or state).

    Attributes:
        name: Segment label (word, phone name, or state ID)
        start_frame: Starting frame number
        end_frame: Ending frame number (inclusive)
        score: Acoustic score for this segment
    """

    name: str
    start_frame: int
    end_frame: int
    score: int = 0

    @property
    def duration_frames(self) -> int:
        """Duration in frames."""
        return self.end_frame - self.start_frame + 1

    def start_time(self, frame_shift: float = FRAME_SHIFT_SECONDS) -> float:
        """Start time in seconds."""
        return self.start_frame * frame_shift

    def end_time(self, frame_shift: float = FRAME_SHIFT_SECONDS) -> float:
        """End time in seconds (end of segment)."""
        return (self.end_frame + 1) * frame_shift

    def duration_time(self, frame_shift: float = FRAME_SHIFT_SECONDS) -> float:
        """Duration in seconds."""
        return self.duration_frames * frame_shift


@dataclass
class RetryOutcome:
    """How a wider-beam retry produced an alignment, and how the check judged it.

    Attributes:
        rung: The ladder rung that aligned it, counting from 1.
        factor: That rung's retry beam factor.
        beam: The beam the rung searched at (nominal beam / factor).
        score: The alignment's speech score, in nats per speech frame: the
            summed scores of its non-filler words over their frames. ``None``
            when it has no speech frames.
        threshold: The acceptance threshold it was judged against, in the same
            units, or ``None`` when it was not judged.
        basis: Where the threshold came from, or why there was none.
    """

    rung: int
    factor: float
    beam: float
    score: float | None
    threshold: float | None = None
    basis: str = "not judged"


@dataclass
class AlignmentResult:
    """Complete alignment result for an utterance.

    Attributes:
        utterance_id: Utterance identifier
        words: Word-level segments
        phones: Phone-level segments
        states: State-level segments (optional)
        total_score: Total acoustic score
        n_frames: Total number of frames
        transcript: Original transcript
        frame_rate: Alignment frames per second, from the model record
        retry: ``None`` for a first-pass alignment; for one a wider-beam retry
            recovered, the rung, beam, score and acceptance threshold
    """

    utterance_id: str
    words: list[AlignedSegment]
    phones: list[AlignedSegment]
    states: list[AlignedSegment]
    total_score: int
    n_frames: int
    transcript: str = ""
    frame_rate: int = 100
    retry: RetryOutcome | None = None

    @property
    def frame_shift(self) -> float:
        """Seconds represented by one alignment frame."""
        return 1.0 / self.frame_rate

    def duration_time(self, frame_shift: float | None = None) -> float:
        """Total duration in seconds."""
        return self.n_frames * (self.frame_shift if frame_shift is None else frame_shift)


def align_utterance(
    audio_path: Path,
    transcript: str,
    model_dir: Path,
    dict_path: Path,
    filler_dict: Path | None = None,
    include_phones: bool = True,
    beam: float = DEFAULT_BEAM,
    retry_beam_factor: RetryBeamFactor = DEFAULT_RETRY_BEAM_FACTOR,
    failed_alignment: Literal["recover", "abort", "omit"] = "recover",
    verbatim_tokens: bool = False,
    retry_acceptance_target: float | None = DEFAULT_RETRY_ACCEPTANCE_TARGET,
    retry_acceptance_threshold: float | Sequence[float] | None = None,
) -> AlignmentResult:
    """Align a single utterance.

    Convenience wrapper around :class:`~pstrain.lib.alignment.native.Aligner`
    that constructs an aligner, runs one utterance, and tears down. For
    corpus-scale work prefer a long-lived ``Aligner`` (see
    :func:`pstrain.lib.alignment.batch.align_corpus`); the per-utterance cost
    of loading the acoustic model dominates everything else.

    Args:
        audio_path: WAV file (16 kHz, 16-bit, mono).
        transcript: Word transcript to align. ``<s>``/``</s>`` markers
            are tolerated and stripped.
        model_dir: Acoustic model directory containing ``mdef``,
            ``means``, ``variances``, ``mixture_weights``,
            ``transition_matrices``, and ``feat.params``.
        dict_path: Pronunciation dictionary.
        filler_dict: Filler / non-speech dictionary (optional).
        include_phones: Return phone-level segments in the result.
        beam: Viterbi pruning beam (default 1e-64, sphinx3_align default).
        retry_beam_factor: Factor for one wider-beam final-state retry, or an
            ascending sequence of factors tried in order until one succeeds.
        failed_alignment: ``"recover"`` retries final-state failures;
            ``"abort"`` and ``"omit"`` do not retry.
        verbatim_tokens: Honor explicit pronunciation variants exactly. The
            default collapses suffixes and considers every alternative.
        retry_acceptance_target: The retry acceptance check; ``None`` turns it
            off. With it on, a final-state failure is retried only when
            ``retry_acceptance_threshold`` is supplied, since a single
            utterance cannot calibrate one (see
            :meth:`~pstrain.lib.alignment.native.Aligner.calibrate_retry_acceptance`).
        retry_acceptance_threshold: The threshold per retry rung, in nats per
            speech frame at the rung's beam.

    Returns:
        :class:`AlignmentResult` with word- (and optionally phone-)
        level segments. A variant suffix such as ``reading(2)`` labels the
        dictionary pronunciation selected by Viterbi. With the default
        ``verbatim_tokens=False``, an input suffix does not constrain that
        selection. With the option enabled, it selects exactly that variant.

    Raises:
        FileNotFoundError: Audio file or model files are missing.
        RuntimeError: Alignment fails (final state not reached, etc.), or
            the acceptance check rejects what the retry recovered
            (:class:`~pstrain.lib.alignment.acceptance.AlignmentRejectedError`).
    """
    from pstrain.lib.alignment.native import Aligner

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with Aligner(
        model_dir,
        dict_path,
        filler_dict=filler_dict,
        beam=beam,
        retry_beam_factor=retry_beam_factor,
        failed_alignment=failed_alignment,
        retry_acceptance_target=retry_acceptance_target,
        retry_acceptance_threshold=retry_acceptance_threshold,
        include_phones=include_phones,
        verbatim_tokens=verbatim_tokens,
    ) as aligner:
        return aligner.align_audio(audio_path, transcript)
