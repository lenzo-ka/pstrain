"""Language model building step."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_build_lm(
    train_transcripts: Path,
    output_path: Path,
    max_order: int = 3,
    smoothing: str = "auto",
) -> Path:
    """Build an ARPA language model from training transcripts.

    Uses arpabo with auto mode (optimized Katz backoff) by default.

    Args:
        train_transcripts: Path to a simple or Sphinx-format training transcription file
        output_path: Path to write ARPA LM file
        max_order: N-gram order (default 3 for trigrams)
        smoothing: Smoothing method - "auto" (default), "good_turing", "kneser_ney"

    Returns:
        Path to created LM file
    """
    from pstrain.lib.lm import build_lm_from_file

    logger.info("Building %d-gram LM from %s", max_order, train_transcripts)
    logger.info("  Smoothing: %s", smoothing)
    logger.info("  Output: %s", output_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = build_lm_from_file(
        transcript_file=train_transcripts,
        output_path=output_path,
        max_order=max_order,
        smoothing=smoothing,
    )

    logger.info("LM built: %s", result)
    return result
