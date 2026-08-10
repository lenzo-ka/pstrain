"""WER (Word Error Rate) calculation using jiwer.

This module exposes all jiwer metrics:
- WER: Word Error Rate = (S + D + I) / N
- MER: Match Error Rate = (S + D + I) / (H + S + D + I)
- WIL: Word Information Lost = 1 - (H/N1) * (H/N2)
- WIP: Word Information Preserved = (H/N1) * (H/N2)
- CER: Character Error Rate (optional)

Where:
- S = Substitutions, D = Deletions, I = Insertions, H = Hits (correct)
- N = Reference words, N1 = H+S+D, N2 = H+S+I
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WERResult:
    """Result from WER calculation with all jiwer metrics.

    Attributes:
        wer: Word Error Rate (S+D+I)/N, can be >1.0 if many insertions
        mer: Match Error Rate (S+D+I)/(H+S+D+I), always 0.0-1.0
        wil: Word Information Lost, 0.0=perfect, 1.0=complete loss
        wip: Word Information Preserved, 1.0=perfect, 0.0=complete loss
        hits: Number of correct words (H)
        substitutions: Number of substitution errors (S)
        deletions: Number of deletion errors (D)
        insertions: Number of insertion errors (I)
        ref_words: Total words in reference (N = H+S+D)
        hyp_words: Total words in hypothesis (H+S+I)
        cer: Character Error Rate (optional, computed if requested)
        alignments: Word alignments (optional, for detailed analysis)
    """

    wer: float
    mer: float
    wil: float
    wip: float
    hits: int
    substitutions: int
    deletions: int
    insertions: int
    ref_words: int
    hyp_words: int
    cer: float | None = None
    alignments: list[tuple[str, str, str]] | None = None

    @property
    def errors(self) -> int:
        """Total number of errors (S + D + I)."""
        return self.substitutions + self.deletions + self.insertions

    @property
    def accuracy(self) -> float:
        """Word accuracy = 1 - WER, clamped to [0, 1]."""
        return max(0.0, min(1.0, 1.0 - self.wer))

    # Aliases for compatibility
    @property
    def total_words(self) -> int:
        """Alias for ref_words (reference word count)."""
        return self.ref_words

    @property
    def correct(self) -> int:
        """Alias for hits."""
        return self.hits

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "wer": self.wer,
            "mer": self.mer,
            "wil": self.wil,
            "wip": self.wip,
            "accuracy": self.accuracy,
            "hits": self.hits,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "errors": self.errors,
            "ref_words": self.ref_words,
            "hyp_words": self.hyp_words,
            "cer": self.cer,
        }


def calculate_wer(
    reference: str,
    hypothesis: str,
    compute_cer: bool = False,
    compute_alignments: bool = False,
) -> WERResult:
    """Calculate all WER metrics between reference and hypothesis.

    Uses jiwer library for accurate calculation of all metrics.

    Args:
        reference: Reference (ground truth) transcript
        hypothesis: Hypothesis (recognition result) transcript
        compute_cer: Also compute Character Error Rate
        compute_alignments: Store word alignments for analysis

    Returns:
        WERResult with all jiwer metrics

    Note:
        - WER can be > 1.0 if insertions exceed reference length
        - MER is always 0.0-1.0
        - Empty reference returns all zeros
    """
    from jiwer import process_words

    if not reference.strip():
        return WERResult(
            wer=0.0,
            mer=0.0,
            wil=0.0,
            wip=1.0,
            hits=0,
            substitutions=0,
            deletions=0,
            insertions=0,
            ref_words=0,
            hyp_words=0,
        )

    # Normalize: lowercase, collapse whitespace
    ref_norm = " ".join(reference.lower().split())
    hyp_norm = " ".join(hypothesis.lower().split())

    # Process words - returns all metrics
    output = process_words(ref_norm, hyp_norm)

    # Extract alignments if requested. jiwer's AlignmentChunk carries the
    # operation type plus start/end indices into the reference and hypothesis
    # word lists (not the words themselves), so slice the word lists to
    # recover the (type, ref_segment, hyp_segment) triples.
    alignments = None
    if compute_alignments and output.alignments:
        alignments = []
        for i, alignment in enumerate(output.alignments):
            ref_words = output.references[i]
            hyp_words = output.hypotheses[i]
            for chunk in alignment:
                ref_seg = " ".join(ref_words[chunk.ref_start_idx : chunk.ref_end_idx])
                hyp_seg = " ".join(hyp_words[chunk.hyp_start_idx : chunk.hyp_end_idx])
                alignments.append((chunk.type, ref_seg, hyp_seg))

    # Compute CER if requested
    cer = None
    if compute_cer:
        from jiwer import process_characters

        char_output = process_characters(ref_norm, hyp_norm)
        cer = char_output.cer

    return WERResult(
        wer=output.wer,
        mer=output.mer,
        wil=output.wil,
        wip=output.wip,
        hits=output.hits,
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
        ref_words=output.hits + output.substitutions + output.deletions,
        hyp_words=output.hits + output.substitutions + output.insertions,
        cer=cer,
        alignments=alignments,
    )


def aggregate_wer(results: list[WERResult]) -> WERResult:
    """Aggregate WER statistics across multiple utterances.

    Computes totals for counts and recalculates metrics from totals.

    Args:
        results: List of WERResult instances

    Returns:
        Aggregated WERResult with totals and overall metrics
    """
    if not results:
        return WERResult(
            wer=0.0,
            mer=0.0,
            wil=0.0,
            wip=1.0,
            hits=0,
            substitutions=0,
            deletions=0,
            insertions=0,
            ref_words=0,
            hyp_words=0,
        )

    # Sum raw counts
    total_hits = sum(r.hits for r in results)
    total_subs = sum(r.substitutions for r in results)
    total_dels = sum(r.deletions for r in results)
    total_ins = sum(r.insertions for r in results)
    total_ref = sum(r.ref_words for r in results)
    total_hyp = sum(r.hyp_words for r in results)

    # Recalculate metrics from totals
    total_errors = total_subs + total_dels + total_ins

    # WER = (S + D + I) / N
    wer = total_errors / total_ref if total_ref > 0 else 0.0

    # MER = (S + D + I) / (H + S + D + I)
    mer_denom = total_hits + total_subs + total_dels + total_ins
    mer = total_errors / mer_denom if mer_denom > 0 else 0.0

    # WIL = 1 - (H/N1) * (H/N2) where N1=ref_words, N2=hyp_words
    if total_ref > 0 and total_hyp > 0:
        wip = (total_hits / total_ref) * (total_hits / total_hyp)
        wil = 1.0 - wip
    else:
        wip = 0.0
        wil = 1.0

    # Average CER where available
    cer_values = [r.cer for r in results if r.cer is not None]
    avg_cer = sum(cer_values) / len(cer_values) if cer_values else None

    return WERResult(
        wer=wer,
        mer=mer,
        wil=wil,
        wip=wip,
        hits=total_hits,
        substitutions=total_subs,
        deletions=total_dels,
        insertions=total_ins,
        ref_words=total_ref,
        hyp_words=total_hyp,
        cer=avg_cer,
    )


def format_wer_summary(result: WERResult) -> str:
    """Format WER result as a human-readable summary.

    Args:
        result: WERResult to format

    Returns:
        Multi-line string with formatted metrics
    """
    lines = [
        f"Word Error Rate (WER):      {result.wer:.2%}",
        f"Match Error Rate (MER):     {result.mer:.2%}",
        f"Word Info Lost (WIL):       {result.wil:.3f}",
        f"Word Info Preserved (WIP):  {result.wip:.3f}",
        "",
        f"Reference words:  {result.ref_words}",
        f"Hypothesis words: {result.hyp_words}",
        "",
        f"Correct (hits):   {result.hits}",
        f"Substitutions:    {result.substitutions}",
        f"Deletions:        {result.deletions}",
        f"Insertions:       {result.insertions}",
        f"Total errors:     {result.errors}",
    ]

    if result.cer is not None:
        lines.insert(4, f"Character Error Rate (CER): {result.cer:.2%}")

    return "\n".join(lines)
