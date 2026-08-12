"""Model testing and evaluation utilities.

This module provides:
- WER (Word Error Rate) calculation using jiwer with all metrics
- PocketSphinx decoder wrapper for recognition
- Test result structures and report generation
"""

from pstrain.lib.testing.decoder import (
    Decoder,
    DecodingResult,
    check_pocketsphinx,
    pocketsphinx_version,
)
from pstrain.lib.testing.report import TestReport, create_report
from pstrain.lib.testing.test import TestResult, load_transcripts, test_model
from pstrain.lib.testing.wer import WERResult, aggregate_wer, calculate_wer, format_wer_summary

__all__ = [
    # WER calculation
    "WERResult",
    "calculate_wer",
    "aggregate_wer",
    "format_wer_summary",
    # Decoder
    "Decoder",
    "DecodingResult",
    "check_pocketsphinx",
    "pocketsphinx_version",
    # Testing
    "TestResult",
    "test_model",
    "load_transcripts",
    # Reports
    "TestReport",
    "create_report",
]
