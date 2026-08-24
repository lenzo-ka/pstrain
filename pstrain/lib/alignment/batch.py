"""Batch alignment for corpus processing.

Aligns an entire corpus of audio files to their transcripts with a
single long-lived :class:`Aligner`, so the acoustic model is loaded
exactly once per corpus pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from pstrain.lib.alignment.core import DEFAULT_BEAM, DEFAULT_RETRY_BEAM_FACTOR, AlignmentResult
from pstrain.lib.alignment.native import Aligner

logger = logging.getLogger(__name__)

# Cadence for "Progress: N/total" log lines during a long batch.
_PROGRESS_LOG_EVERY = 100
_ERROR_MESSAGE_LIMIT = 200


def _error_message(exc: Exception) -> str:
    """Keep the actionable end of a bounded native-worker diagnostic."""
    message = str(exc)
    if len(message) <= _ERROR_MESSAGE_LIMIT:
        return message
    return f"...{message[-(_ERROR_MESSAGE_LIMIT - 3) :]}"


@dataclass
class AlignmentJob:
    """Result of a batch alignment job.

    Attributes:
        model_dir: Path to acoustic model used
        n_utterances: Total utterances to align
        n_aligned: Successfully aligned utterances
        n_failed: Failed alignments
        results: Dict mapping utterance_id to AlignmentResult
        errors: Dict mapping utterance_id to error message
        timestamp: When the job was run
    """

    model_dir: Path
    n_utterances: int
    n_aligned: int
    n_failed: int
    results: dict[str, AlignmentResult] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        """Alignment success rate (0.0-1.0)."""
        if self.n_utterances == 0:
            return 0.0
        return self.n_aligned / self.n_utterances


def align_corpus(
    transcripts: dict[str, str],
    audio_dir: Path,
    model_dir: Path,
    dict_path: Path,
    filler_dict: Path | None = None,
    audio_ext: str = ".wav",
    include_phones: bool = True,
    beam: float = DEFAULT_BEAM,
    retry_beam_factor: float = DEFAULT_RETRY_BEAM_FACTOR,
    failed_alignment: Literal["recover", "abort", "omit"] = "recover",
    verbatim_tokens: bool = False,
) -> AlignmentJob:
    """Align an entire corpus.

    Loads the acoustic model once, keeps it resident for all utterances.

    Args:
        transcripts: Dict mapping utterance_id to transcript text.
        audio_dir: Directory containing audio files.
        model_dir: Path to acoustic model directory.
        dict_path: Path to pronunciation dictionary.
        filler_dict: Path to filler dictionary (optional).
        audio_ext: Audio file extension (default ``".wav"``).
        include_phones: Capture phone-level segmentation.
        beam: Viterbi pruning beam.
        retry_beam_factor: Factor for one wider-beam final-state retry.
        failed_alignment: Whether final-state failures are retried before being recorded.
        verbatim_tokens: Honor explicit pronunciation variants exactly.

    Returns:
        :class:`AlignmentJob` with all alignment results.

    Example:
        >>> transcripts = {"utt001": "hello world", "utt002": "goodbye"}
        >>> job = align_corpus(transcripts, audio_dir, model_dir, dict_path)
        >>> print(f"Aligned {job.n_aligned}/{job.n_utterances}")
    """
    audio_dir = Path(audio_dir)
    model_dir = Path(model_dir)
    dict_path = Path(dict_path)

    results: dict[str, AlignmentResult] = {}
    errors: dict[str, str] = {}
    n_aligned = 0
    n_failed = 0

    total = len(transcripts)
    if total == 0:
        return AlignmentJob(
            model_dir=model_dir,
            n_utterances=0,
            n_aligned=0,
            n_failed=0,
            results=results,
            errors=errors,
        )

    logger.info("Aligning %d utterances...", total)

    # If model_dir is missing required files Aligner raises before the
    # loop; surface that as a corpus-wide failure rather than per-utt.
    try:
        aligner = Aligner(
            model_dir,
            dict_path,
            filler_dict=filler_dict,
            beam=beam,
            retry_beam_factor=retry_beam_factor,
            failed_alignment=failed_alignment,
            include_phones=include_phones,
            verbatim_tokens=verbatim_tokens,
        )
    except (FileNotFoundError, RuntimeError) as e:
        logger.error("Failed to initialize aligner: %s", e)
        for utt_id in transcripts:
            errors[utt_id] = f"Aligner init failed: {e}"
        return AlignmentJob(
            model_dir=model_dir,
            n_utterances=total,
            n_aligned=0,
            n_failed=total,
            results=results,
            errors=errors,
        )

    try:
        for i, (utt_id, transcript) in enumerate(transcripts.items(), 1):
            audio_path = audio_dir / f"{utt_id}{audio_ext}"

            if i % _PROGRESS_LOG_EVERY == 0 or i == total:
                logger.info("  Progress: %d/%d (%.1f%%)", i, total, 100 * i / total)

            if not audio_path.exists():
                errors[utt_id] = f"Audio file not found: {audio_path}"
                n_failed += 1
                continue

            try:
                result = aligner.align_audio(audio_path, transcript, utterance_id=utt_id)
                results[utt_id] = result
                n_aligned += 1
            except Exception as e:
                message = _error_message(e)
                errors[utt_id] = message
                n_failed += 1
                logger.warning("Alignment failed for %s: %s", utt_id, message)
    finally:
        aligner.close()

    logger.info(
        "Alignment complete: %d/%d successful (%.1f%%)",
        n_aligned,
        total,
        100 * n_aligned / total,
    )

    return AlignmentJob(
        model_dir=model_dir,
        n_utterances=total,
        n_aligned=n_aligned,
        n_failed=n_failed,
        results=results,
        errors=errors,
    )


def load_transcripts(transcript_file: Path) -> dict[str, str]:
    """Load transcripts from a Sphinx-format transcription file.

    Format: <s> word word word </s> (utterance_id)

    Args:
        transcript_file: Path to transcription file

    Returns:
        Dict mapping utterance_id to transcript text (with sentence markers)
    """
    transcripts = {}

    with transcript_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse: <s> text </s> (utt_id)
            if "(" in line:
                paren_start = line.rfind("(")
                paren_end = line.rfind(")")
                if paren_start > 0 and paren_end > paren_start:
                    utt_id = line[paren_start + 1 : paren_end].strip()
                    transcript = line[:paren_start].strip()
                    transcripts[utt_id] = transcript

    return transcripts
