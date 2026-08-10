"""Language model building using arpabo.

Builds ARPA format n-gram language models from training transcripts.
Uses arpabo with auto mode (optimized Katz backoff) by default.
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_lm(
    transcripts: list[str] | dict[str, str],
    output_path: Path,
    max_order: int = 3,
    smoothing: str = "auto",
) -> Path:
    """Build an ARPA language model from transcripts using arpabo.

    Args:
        transcripts: List of transcript strings or dict mapping utt_id to transcript
        output_path: Path to write the ARPA LM file
        max_order: N-gram order (default 3 for trigrams)
        smoothing: Smoothing method - "auto" (default), "good_turing", "kneser_ney"

    Returns:
        Path to the created LM file

    Note:
        "auto" mode uses optimized Katz backoff, which works well for typical
        speech corpus sizes. For very small corpora, "good_turing" may be better.
    """
    from arpabo import ArpaBoLM

    output_path = Path(output_path)

    # Extract text from dict if needed
    texts = list(transcripts.values()) if isinstance(transcripts, dict) else list(transcripts)

    # Clean transcripts: remove sentence markers, preserve case to match dictionary
    clean_texts = []
    for text in texts:
        text = text.strip()
        # Remove Sphinx sentence markers
        if text.startswith("<s>"):
            text = text[3:]
        if text.endswith("</s>"):
            text = text[:-4]
        text = text.strip()
        if text:
            clean_texts.append(text)

    if not clean_texts:
        raise ValueError("No text content in transcripts")

    # Create corpus as file-like object (one sentence per line)
    corpus_text = "\n".join(clean_texts)
    corpus_file = StringIO(corpus_text)

    logger.info(
        "Building %d-gram LM from %d sentences (%d words), smoothing=%s",
        max_order,
        len(clean_texts),
        sum(len(t.split()) for t in clean_texts),
        smoothing,
    )

    # Build LM. ArpaBoLM is an untyped third-party class; treat the instance
    # as Any so its (unannotated) methods don't trip disallow_untyped_calls.
    lm: Any = ArpaBoLM(
        max_order=max_order,
        smoothing_method=smoothing,
        add_start=True,  # Add <s> and </s> markers
        unicode_norm=True,
    )
    lm.read_corpus(corpus_file)
    lm.compute()

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lm.write_file(str(output_path))

    logger.info("LM written to %s", output_path)
    return output_path


def build_lm_from_file(
    transcript_file: Path,
    output_path: Path,
    max_order: int = 3,
    smoothing: str = "auto",
) -> Path:
    """Build an ARPA LM from a Sphinx-format transcription file.

    Args:
        transcript_file: Path to transcription file (Sphinx format)
        output_path: Path to write ARPA LM
        max_order: N-gram order (default 3)
        smoothing: Smoothing method (default "auto")

    Returns:
        Path to created LM file
    """
    transcripts = load_transcripts(transcript_file)
    return build_lm(transcripts, output_path, max_order, smoothing)


def load_transcripts(transcript_file: Path) -> dict[str, str]:
    """Load transcripts from a Sphinx-format transcription file.

    Format: <s> word word word </s> (utterance_id)

    Args:
        transcript_file: Path to transcription file

    Returns:
        Dict mapping utterance_id to transcript text
    """
    transcripts = {}

    with transcript_file.open(encoding="utf-8", errors="replace") as f:
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
