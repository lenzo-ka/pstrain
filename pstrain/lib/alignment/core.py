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

# Default pruning beam for the sphinx3 aligner. 1e-64 matches the
# upstream sphinx3_align CLI default; it is wide enough for cd-1g
# through cd-8g acoustic models on clean read speech.
DEFAULT_BEAM = 1e-64

# Sphinx feature extraction uses a 10 ms frame shift. State-, phone-,
# and word-level segments report start/end positions in frames; this
# constant turns frame indices into seconds.
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
    """

    utterance_id: str
    words: list[AlignedSegment]
    phones: list[AlignedSegment]
    states: list[AlignedSegment]
    total_score: int
    n_frames: int
    transcript: str = ""

    def duration_time(self, frame_shift: float = FRAME_SHIFT_SECONDS) -> float:
        """Total duration in seconds."""
        return self.n_frames * frame_shift


def align_utterance(
    audio_path: Path,
    transcript: str,
    model_dir: Path,
    dict_path: Path,
    filler_dict: Path | None = None,
    include_phones: bool = True,
    beam: float = DEFAULT_BEAM,
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
            ``transition_matrices``, and ideally ``feat.params``.
        dict_path: Pronunciation dictionary.
        filler_dict: Filler / non-speech dictionary (optional).
        include_phones: Return phone-level segments in the result.
        beam: Viterbi pruning beam (default 1e-64, sphinx3_align default).

    Returns:
        :class:`AlignmentResult` with word- (and optionally phone-)
        level segments. A variant suffix such as ``reading(2)`` labels the
        dictionary pronunciation selected by Viterbi. A suffix in the input
        transcript does not constrain that selection and need not be preserved.

    Raises:
        FileNotFoundError: Audio file or model files are missing.
        RuntimeError: Alignment fails (final state not reached, etc.).
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
        include_phones=include_phones,
    ) as aligner:
        return aligner.align_audio(audio_path, transcript)
