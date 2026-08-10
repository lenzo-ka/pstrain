"""CMUDict/ARPABET-specific dictionary handling.

CMU ARPABET Stress Convention
-----------------------------
In ARPABET (used by CMUDict), lexical stress is indicated by a digit suffix on vowels:
- 0: No stress (reduced vowel, often schwa)
- 1: Primary stress
- 2: Secondary stress

Examples:
- HELLO: HH AH0 L OW1 (schwa unstressed, OW has primary stress)
- UNDERSTAND: AH2 N D ER0 S T AE1 N D (secondary on first, primary on last)

ARPABET Phoneset (39 phones for American English):
- Vowels: AA, AE, AH, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW
- Consonants: B, CH, D, DH, F, G, HH, JH, K, L, M, N, NG, P, R, S, SH, T, TH, V, W, Y, Z, ZH
- Special: SIL (silence)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Self

from st2.lib.dictionary.dictionary import Dictionary

# ARPABET vowels (can carry stress markers)
ARPABET_VOWELS = frozenset(
    [
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
    ]
)

# ARPABET consonants
ARPABET_CONSONANTS = frozenset(
    [
        "B",
        "CH",
        "D",
        "DH",
        "F",
        "G",
        "HH",
        "JH",
        "K",
        "L",
        "M",
        "N",
        "NG",
        "P",
        "R",
        "S",
        "SH",
        "T",
        "TH",
        "V",
        "W",
        "Y",
        "Z",
        "ZH",
    ]
)

# Full ARPABET phoneset (without stress markers)
ARPABET_PHONES = ARPABET_VOWELS | ARPABET_CONSONANTS | {"SIL"}


def strip_stress(phone: str) -> str:
    """Remove ARPABET stress marker (trailing digit) from phone.

    Args:
        phone: Phone possibly with stress marker (e.g., "AH0", "OW1")

    Returns:
        Phone without stress marker (e.g., "AH", "OW")

    Examples:
        strip_stress("AA1") -> "AA"
        strip_stress("AE0") -> "AE"
        strip_stress("HH") -> "HH" (no change, consonants don't have stress)
    """
    return re.sub(r"\d+$", "", phone)


def get_stress(phone: str) -> int | None:
    """Get ARPABET stress level from phone.

    Args:
        phone: Phone possibly with stress marker

    Returns:
        0 (no stress), 1 (primary), 2 (secondary), or None if no stress marker

    Examples:
        get_stress("AH0") -> 0
        get_stress("OW1") -> 1
        get_stress("AE2") -> 2
        get_stress("HH") -> None
    """
    match = re.search(r"(\d+)$", phone)
    if match:
        return int(match.group(1))
    return None


def is_vowel(phone: str) -> bool:
    """Check if phone is an ARPABET vowel (stress stripped).

    Args:
        phone: Phone with or without stress marker

    Returns:
        True if vowel
    """
    return strip_stress(phone) in ARPABET_VOWELS


class CMUDict(Dictionary):
    """CMUDict-style dictionary with ARPABET stress handling.

    Extends Dictionary with ARPABET-specific features:
    - Stress marker parsing and manipulation
    - Stress-stripped variants for training
    - Vowel/consonant classification
    """

    def strip_stress_from_entries(self) -> CMUDict:
        """Create a copy with stress markers removed from all phones.

        Returns:
            New CMUDict with stress-free pronunciations

        Note:
            This may create duplicate entries (e.g., "read" R IY D and R EH D
            both become R IY D and R EH D without stress). Duplicates are
            automatically deduplicated.
        """
        result = CMUDict()
        for word, phones in self._entries.items():
            phones_nostress = [strip_stress(p) for p in phones]
            result.add_entry(word, phones_nostress)
        return result

    def get_stressed_vowels(self, word: str) -> list[tuple[str, int]]:
        """Get vowels with their stress levels for a word.

        Args:
            word: Word to look up

        Returns:
            List of (vowel, stress_level) tuples
            stress_level is 0, 1, 2, or -1 if no stress marker

        Example:
            get_stressed_vowels("HELLO")
            -> [("AH", 0), ("OW", 1)]  # for HH AH0 L OW1
        """
        phones = self.get(word)
        if not phones:
            return []

        result = []
        for phone in phones:
            base = strip_stress(phone)
            if base in ARPABET_VOWELS:
                stress = get_stress(phone)
                result.append((base, stress if stress is not None else -1))
        return result

    def get_primary_stress_position(self, word: str) -> int | None:
        """Get the syllable position (0-indexed) of primary stress.

        Args:
            word: Word to look up

        Returns:
            Position of primary stress (0 = first vowel), or None if not found
        """
        stressed_vowels = self.get_stressed_vowels(word)
        for i, (_, stress) in enumerate(stressed_vowels):
            if stress == 1:
                return i
        return None

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Load CMUDict from file.

        Overrides parent to return CMUDict instance.
        """
        # Load using parent class logic
        base_dict = Dictionary.from_file(path)

        # Convert to CMUDict
        result = cls()
        result._entries = base_dict._entries
        result._variants = base_dict._variants
        return result


def strip_dictionary_stress(input_dict: Path, output_dict: Path) -> tuple[int, int]:
    """Strip stress from dictionary file.

    Args:
        input_dict: Input dictionary (with stress)
        output_dict: Output dictionary (without stress)

    Returns:
        Tuple of (entries_processed, unique_phones)
    """
    entries = []
    phoneset: set[str] = set()

    with input_dict.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Keep comments and empty lines
            if not line or line.startswith("#"):
                entries.append(line)
                continue

            # Process entry
            parts = line.split()
            if len(parts) < 2:
                entries.append(line)
                continue

            word = parts[0]
            phones = parts[1:]

            # Strip stress from phones
            phones_nostress = [strip_stress(p) for p in phones]
            phoneset.update(phones_nostress)

            # Reconstruct entry
            entries.append(f"{word} {' '.join(phones_nostress)}")

    # Write stress-less dictionary
    with output_dict.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry + "\n")

    n_entries = len([e for e in entries if e and not e.startswith("#")])
    return n_entries, len(phoneset)


def strip_phoneset_stress(input_phoneset: Path, output_phoneset: Path) -> int:
    """Strip stress from phoneset file.

    Args:
        input_phoneset: Input phoneset (with stress)
        output_phoneset: Output phoneset (without stress)

    Returns:
        Number of unique phones
    """
    phones: set[str] = set()

    with input_phoneset.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                phone_nostress = strip_stress(line)
                phones.add(phone_nostress)

    # Write stress-less phoneset
    with output_phoneset.open("w", encoding="utf-8") as f:
        # Write SIL first if present
        if "SIL" in phones:
            f.write("SIL\n")
            phones_to_write = phones - {"SIL"}
        else:
            phones_to_write = phones

        for phone in sorted(phones_to_write):
            f.write(phone + "\n")

    return len(phones)
