"""Filler dictionary for non-speech sounds.

Filler Dictionary Purpose
-------------------------
In Sphinx, filler words represent non-speech sounds and silence:
- <sil> - Inter-word silence (optional between words)
- <s>   - Sentence start boundary
- </s>  - Sentence end boundary
- +NOISE+, +BREATH+, +UH+, +UM+ - Non-speech sounds (optional)

How Fillers Are Used
--------------------
During training and testing, the filler dictionary enables:

1. **Optional Silence Insertion Between Words**
   The bw (Baum-Welch) binary can insert <sil> between any two words.
   This allows the model to handle natural pauses and word boundaries.
   The model learns when silence is likely (e.g., longer pauses after phrases).

2. **Sentence Boundary Modeling**
   <s> and </s> mark sentence boundaries, allowing the model to learn:
   - Different acoustic patterns at sentence start/end
   - Longer silence at sentence boundaries vs. within sentences
   - Proper sentence segmentation during decoding

3. **Acoustic Modeling**
   Filler phones are typically single-state (not 3-state like regular phones)
   because silence doesn't have internal temporal structure like vowels/consonants.

Critical Requirements
---------------------
1. **Silence Phone Must Match Main Dictionary**
   If your main dictionary uses "SIL" as the silence phone, the filler
   dictionary must also use "SIL". Mismatches will cause training/testing to fail.

2. **No Comments in Filler Dictionary**
   PocketSphinx fails to parse comment lines in filler dictionaries correctly.
   Always use get_standard_fillers_text() which produces comment-free output.
"""

from __future__ import annotations

from pathlib import Path


def get_standard_fillers_text(silence_phone: str = "SIL") -> str:
    """Get standard filler dictionary text with configurable silence phone.

    Note: NO COMMENTS - PocketSphinx fails to parse comment lines correctly.

    Args:
        silence_phone: Silence phone symbol (default: SIL)

    Returns:
        Filler dictionary text (no comments)
    """
    return f"""<sil> {silence_phone}
<s> {silence_phone}
</s> {silence_phone}
"""


def create_standard_filler_dict(output_path: Path, silence_phone: str = "SIL") -> None:
    """Create standard filler dictionary with configurable silence phone.

    Args:
        output_path: Path to write filler dictionary
        silence_phone: Silence phone symbol (default: SIL)
    """
    output_path.write_text(get_standard_fillers_text(silence_phone), encoding="utf-8")


def get_standard_fillers(silence_phone: str = "SIL") -> dict[str, list[str]]:
    """Get standard filler words and their phones.

    Args:
        silence_phone: Silence phone symbol (default: SIL)

    Returns:
        Dict mapping filler words to phone lists
    """
    return {
        "<s>": [silence_phone],
        "</s>": [silence_phone],
        "<sil>": [silence_phone],
    }


def get_filler_phones(silence_phone: str = "SIL") -> set[str]:
    """Get set of filler phones.

    Args:
        silence_phone: Silence phone symbol (default: SIL)

    Returns:
        Set of filler phone symbols
    """
    return {silence_phone}


# Legacy constant for backward compatibility
STANDARD_FILLERS = get_standard_fillers_text()
