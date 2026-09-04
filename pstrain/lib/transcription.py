"""Transcription file handling for pstrain projects.

Transcription = word-level text (what was said), not time-aligned.
Alignment = time-aligned phone/word boundaries (when each occurs).
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["parse_transcription_file", "get_fileids"]


def parse_transcription_file(transcription_path: Path) -> dict[str, str]:
    """Parse transcription file (word-level text, not time-aligned).

    Args:
        transcription_path: Path to transcription file

    Returns:
        Dict mapping fileid -> transcript text (words only)

    Raises:
        FileNotFoundError: If transcription file does not exist
        UnicodeDecodeError: If file is not UTF-8 encoded
        ValueError: If a nonempty line does not match a supported format

    Supports two formats:
    1. Simple: ``<fileid> <word1> <word2> ...``
    2. Sphinx: ``[<s>] <word1> <word2> [</s>] (<fileid>)``

    The Sphinx begin/end silence markers are optional, following the
    convention used by the arpabo language-model tools. A trailing
    ``(<fileid>)`` token selects this form; otherwise the first whitespace
    token is the file ID.
    """
    transcripts = {}
    with transcription_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # Format 2: trailing (fileid), with optional sentence markers.
            match = re.fullmatch(r"(.*?)\s+\(([^)]+)\)", line)
            if match:
                text = re.sub(r"^<s>(?:\s+|$)", "", match.group(1))
                text = re.sub(r"(?:^|\s+)</s>$", "", text)
                fileid = match.group(2)
                transcripts[fileid] = text
                continue

            # Format 1: Simple (fileid + words)
            parts = line.split(None, 1)
            if len(parts) == 2:
                fileid, text = parts
                transcripts[fileid] = text
                continue

            raise ValueError(
                f"Unrecognized transcription format at {transcription_path}:{line_number}: {line!r}"
            )

    return transcripts


def get_fileids(transcription_path: Path) -> list[str]:
    """Get list of fileids from transcription file.

    Args:
        transcription_path: Path to transcription file

    Returns:
        List of file IDs (utterance identifiers)

    Raises:
        FileNotFoundError: If transcription file does not exist
        UnicodeDecodeError: If file is not UTF-8 encoded
    """
    transcripts = parse_transcription_file(transcription_path)
    return list(transcripts.keys())
