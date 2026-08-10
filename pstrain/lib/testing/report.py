"""Test report generation for model evaluation.

Generates comprehensive reports in JSON and text formats with all jiwer metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pstrain.lib.testing.test import TestResult


@dataclass
class TestReport:
    """Comprehensive test report with all metrics.

    Can be serialized to JSON or formatted as human-readable text.
    """

    result: TestResult
    title: str = "Model Test Report"
    description: str = ""
    corpus_name: str = ""
    test_set_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "description": self.description,
            "corpus_name": self.corpus_name,
            "test_set_name": self.test_set_name,
            "model": {
                "name": self.result.model_name,
                "dir": str(self.result.model_dir),
            },
            "metrics": {
                "wer": self.result.wer,
                "mer": self.result.mer,
                "wil": self.result.wil,
                "wip": self.result.wip,
                "cer": self.result.cer,
                "accuracy": self.result.accuracy,
            },
            "counts": {
                "utterances": self.result.n_utterances,
                "decoded": self.result.n_decoded,
                "ref_words": self.result.ref_words,
                "hyp_words": self.result.hyp_words,
                "hits": self.result.hits,
                "substitutions": self.result.substitutions,
                "deletions": self.result.deletions,
                "insertions": self.result.insertions,
                "errors": self.result.errors,
            },
            "timestamp": self.result.timestamp.isoformat(),
            "per_utterance": self.result.per_utterance,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to JSON string.

        Args:
            indent: JSON indentation level (default 2)

        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, output_path: Path) -> None:
        """Save report as JSON file.

        Args:
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json())

    def format_text(self, show_per_utterance: bool = False) -> str:
        """Format report as human-readable text.

        Args:
            show_per_utterance: Include per-utterance details

        Returns:
            Formatted text string
        """
        r = self.result
        lines = []

        # Header
        lines.append("=" * 70)
        lines.append(self.title)
        lines.append("=" * 70)
        lines.append("")

        if self.description:
            lines.append(self.description)
            lines.append("")

        # Model info
        lines.append(f"Model: {r.model_name}")
        lines.append(f"Path:  {r.model_dir}")
        if self.corpus_name:
            lines.append(f"Corpus: {self.corpus_name}")
        if self.test_set_name:
            lines.append(f"Test Set: {self.test_set_name}")
        lines.append(f"Date:  {r.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary metrics
        lines.append("-" * 40)
        lines.append("SUMMARY METRICS")
        lines.append("-" * 40)
        lines.append(f"Word Error Rate (WER):      {r.wer:7.2%}")
        lines.append(f"Match Error Rate (MER):     {r.mer:7.2%}")
        lines.append(f"Word Info Lost (WIL):       {r.wil:7.3f}")
        lines.append(f"Word Info Preserved (WIP):  {r.wip:7.3f}")
        if r.cer is not None:
            lines.append(f"Character Error Rate (CER): {r.cer:7.2%}")
        lines.append(f"Word Accuracy:              {r.accuracy:7.2%}")
        lines.append("")

        # Counts
        lines.append("-" * 40)
        lines.append("COUNTS")
        lines.append("-" * 40)
        lines.append(f"Test utterances:  {r.n_utterances:6d}")
        lines.append(f"Decoded:          {r.n_decoded:6d}")
        lines.append(f"Reference words:  {r.ref_words:6d}")
        lines.append(f"Hypothesis words: {r.hyp_words:6d}")
        lines.append("")
        lines.append(f"Correct (hits):   {r.hits:6d}")
        lines.append(f"Substitutions:    {r.substitutions:6d}")
        lines.append(f"Deletions:        {r.deletions:6d}")
        lines.append(f"Insertions:       {r.insertions:6d}")
        lines.append(f"Total errors:     {r.errors:6d}")
        lines.append("")

        # Per-utterance details
        if show_per_utterance and r.per_utterance:
            lines.append("-" * 40)
            lines.append("PER-UTTERANCE RESULTS")
            lines.append("-" * 40)

            # Sort by WER (worst first)
            sorted_utts = sorted(
                r.per_utterance.items(),
                key=lambda x: x[1].get("wer", 0),
                reverse=True,
            )

            for utt_id, utt_data in sorted_utts:
                wer = utt_data.get("wer", 0)
                ref = utt_data.get("reference", "")
                hyp = utt_data.get("hypothesis", "")

                lines.append("")
                lines.append(f"[{utt_id}] WER: {wer:.2%}")
                lines.append(f"  REF: {ref}")
                lines.append(f"  HYP: {hyp}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.format_text(show_per_utterance=False))

    def save_text(self, output_path: Path, show_per_utterance: bool = True) -> None:
        """Save report as text file.

        Args:
            output_path: Path to output file
            show_per_utterance: Include per-utterance details
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.format_text(show_per_utterance=show_per_utterance))


def create_report(
    result: TestResult,
    title: str | None = None,
    corpus_name: str = "",
    test_set_name: str = "",
) -> TestReport:
    """Create a test report from a TestResult.

    Args:
        result: TestResult from test_model()
        title: Report title (default: auto-generated)
        corpus_name: Name of the corpus tested on
        test_set_name: Name of the test set

    Returns:
        TestReport instance
    """
    if title is None:
        title = f"Test Report: {result.model_name}"

    return TestReport(
        result=result,
        title=title,
        corpus_name=corpus_name,
        test_set_name=test_set_name,
    )
