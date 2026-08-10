"""Text normalization utilities for transcript processing.

Based on pocketsphinx lm.py normalization approach.
"""

from __future__ import annotations

import re
import unicodedata as ud
from pathlib import Path


def normalize_transcript(text: str) -> str:
    """Normalize transcript text for acoustic model training.

    Applies:
    - NFC Unicode normalization
    - Lowercase
    - Strip punctuation/symbols from token boundaries

    Args:
        text: Raw transcript text

    Returns:
        Normalized text suitable for dictionary lookup
    """
    # 1. NFC normalization (canonical decomposition + composition)
    text = ud.normalize("NFC", text)

    # 2. Lowercase
    text = text.lower()

    # 3. Strip punctuation/symbols from start/end of each token
    # Keep hyphens within words (e.g., "re-entered" stays intact)
    words = []
    for word in text.split():
        word = strip_boundary_punct(word)
        if word:  # Only keep non-empty tokens
            words.append(word)

    return " ".join(words)


def strip_boundary_punct(token: str) -> str:
    """Strip punctuation and symbols from token boundaries.

    Removes Unicode categories from start/end:
    - P: Punctuation (.,;:!? etc.)
    - S: Symbols ($%& etc.)
    - Z: Separators (spaces, etc.)

    Keeps internal punctuation (e.g., "re-entered", "o'clock")

    Args:
        token: Word token

    Returns:
        Token with boundary punctuation removed
    """
    # Strip from left
    while token and ud.category(token[0])[0] in {"P", "S", "Z"}:
        token = token[1:]

    # Strip from right
    while token and ud.category(token[-1])[0] in {"P", "S", "Z"}:
        token = token[:-1]

    return token


def normalize_transcript_file(
    input_path: Path,
    output_path: Path,
) -> int:
    """Normalize a Sphinx-format transcript file.

    Reads transcript in format: <s> text </s> (utt_id)
    Normalizes the text portion and writes to output.

    Args:
        input_path: Input transcript file
        output_path: Output normalized transcript file

    Returns:
        Number of utterances processed
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with (
        input_path.open(encoding="utf-8") as f_in,
        output_path.open("w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                f_out.write(line + "\n")
                continue

            # Parse: <s> text </s> (utt_id)
            match = re.match(r"<s>\s+(.*?)\s+</s>\s+\(([^)]+)\)", line)
            if match:
                text = match.group(1)
                utt_id = match.group(2)

                # Normalize text
                normalized = normalize_transcript(text)

                # Write normalized line
                f_out.write(f"<s> {normalized} </s> ({utt_id})\n")
                count += 1
            else:
                # Keep unrecognized lines as-is
                f_out.write(line + "\n")

    return count
