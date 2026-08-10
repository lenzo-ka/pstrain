"""Phoneset handling and phone mapping utilities.

A Phoneset defines the inventory of valid phones in an acoustic model.
Phone mapping functions convert between different phonetic representations.

Phone Mapping
-------------
Maps can be loaded from JSON or text files:

JSON format:
    {"AA": "ɑ", "AE": "æ", "_description": "ignored"}

Text format (-> delimiter):
    AA -> ɑ
    AE -> æ
    # Comments start with #

Mapping types:
- One-to-one: "AA" -> "ɑ" (simple substitution)
- Expansion: "CH" -> "t ʃ" (one phone expands to sequence, space-separated)
- Deletion: "X" -> "" (phone removed from output)

Mappings are directional (A -> B does not imply B -> A). For bidirectional
conversion, provide separate forward and reverse mapping files.

Note: True one-to-many (ambiguous) mappings where one source has multiple
possible targets are not supported. Each source phone maps to exactly one
target (which may be a sequence or empty).

Stress Removal via Maps
-----------------------
ARPABET stress can be stripped using a map:
    AA0 -> AA
    AA1 -> AA
    AA2 -> AA
    ...

This is more general than regex-based stripping and works with
any phoneset that uses numeric suffixes for stress/tone.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from pstrain.lib.dictionary import Dictionary


class Phoneset:
    """Phone inventory with validation and mapping capabilities.

    Supports:
    - Loading from file (one phone per line)
    - Extracting from dictionary
    - Validating dictionaries against phoneset
    - Phone mapping between phonesets
    """

    def __init__(self, phones: set[str]):
        """Initialize phoneset.

        Args:
            phones: Set of phone strings (UTF-8, case-sensitive)
        """
        self._phones = phones

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Load phoneset from file.

        File format:
        - One phone per line
        - Comments start with #
        - UTF-8 encoding, case-sensitive

        Args:
            path: Path to phoneset file

        Returns:
            Phoneset instance

        Raises:
            ValueError: If file has encoding or format errors
        """
        phones: set[str] = set()

        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    phones.add(line)

        except UnicodeDecodeError as e:
            raise ValueError(
                f"Phoneset file {path} is not valid UTF-8.\n"
                f"pstrain requires UTF-8 encoding for all text files.\n"
                f"Error: {e}"
            ) from e

        if not phones:
            raise ValueError(f"Phoneset file {path} contains no phones")

        return cls(phones)

    @classmethod
    def from_dictionary(
        cls, dictionary: Dictionary, include_sil: bool = True, silence_phone: str = "SIL"
    ) -> Self:
        """Extract phoneset from dictionary.

        Args:
            dictionary: Dictionary to extract phones from
            include_sil: If True, add silence phone
            silence_phone: Symbol for silence (default: "SIL")

        Returns:
            Phoneset instance
        """
        phones = dictionary.phonemes()
        if include_sil and silence_phone not in phones:
            phones.add(silence_phone)
        return cls(phones)

    def validate_dictionary(self, dictionary: Dictionary) -> tuple[bool, set[str]]:
        """Validate that dictionary phones are in phoneset.

        Args:
            dictionary: Dictionary to validate

        Returns:
            Tuple of (is_valid, missing_phones)
        """
        dict_phones = dictionary.phonemes()
        missing = dict_phones - self._phones
        return (len(missing) == 0, missing)

    def contains(self, phone: str) -> bool:
        """Check if phone is in phoneset."""
        return phone in self._phones

    def phones(self) -> set[str]:
        """Get all phones in phoneset."""
        return self._phones.copy()

    def has_sil(self, silence_phone: str = "SIL") -> bool:
        """Check if phoneset includes silence phone."""
        return silence_phone in self._phones

    def __len__(self) -> int:
        """Number of phones in phoneset."""
        return len(self._phones)

    def __repr__(self) -> str:
        """String representation."""
        silence_candidates = ["SIL", "sil", "_"]
        for candidate in silence_candidates:
            if self.has_sil(candidate):
                return f"Phoneset({len(self._phones)} phones, with {candidate})"
        return f"Phoneset({len(self._phones)} phones, no silence phone)"

    def to_file(self, path: Path, silence_phone: str = "SIL") -> None:
        """Save phoneset to file.

        Args:
            path: Output file path
            silence_phone: Silence phone to list first if present
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            # Write silence phone first if present
            if silence_phone in self._phones:
                f.write(f"{silence_phone}\n")
                phones_to_write = self._phones - {silence_phone}
            else:
                phones_to_write = self._phones

            # Filter junk (punctuation, numbers, whitespace)
            valid_phones = {p for p in phones_to_write if p and not re.search(r"[0-9,.\-#\s]", p)}

            for phone in sorted(valid_phones):
                f.write(phone + "\n")

    # =========================================================================
    # Phone mapping methods
    # =========================================================================

    def map_phone(
        self,
        phone: str,
        mapping: dict[str, str],
        passthrough_unmapped: bool = True,
    ) -> list[str]:
        """Map a single phone.

        Supports:
        - One-to-one: "AA" -> "ɑ" produces ["ɑ"]
        - Expansion: "CH" -> "t ʃ" produces ["t", "ʃ"] (splits on space)
        - Deletion: "X" -> "" produces []

        Args:
            phone: Phone to map
            mapping: Phone mapping dictionary
            passthrough_unmapped: If True, unmapped phones pass through

        Returns:
            List of mapped phones (may be empty for deletion, multiple for expansion)
        """
        if phone in mapping:
            mapped = mapping[phone]
            if not mapped:
                return []  # Deletion
            return mapped.split()  # Split for one-to-many

        return [phone] if passthrough_unmapped else []

    def map_pronunciation(
        self,
        phones: list[str],
        mapping: dict[str, str],
        passthrough_unmapped: bool = True,
    ) -> list[str]:
        """Map a pronunciation (list of phones).

        Args:
            phones: List of phones to map
            mapping: Phone mapping dictionary
            passthrough_unmapped: If True, unmapped phones pass through

        Returns:
            List of mapped phones (flattened)
        """
        result = []
        for phone in phones:
            result.extend(self.map_phone(phone, mapping, passthrough_unmapped))
        return result

    def create_mapped_phoneset(self, mapping: dict[str, str]) -> Phoneset:
        """Create new phoneset by mapping all phones.

        Args:
            mapping: Phone mapping dictionary

        Returns:
            New Phoneset with mapped phones
        """
        mapped_phones: set[str] = set()
        for phone in self._phones:
            mapped = self.map_phone(phone, mapping, passthrough_unmapped=True)
            mapped_phones.update(mapped)
        return Phoneset(mapped_phones)


# =============================================================================
# Phone map loading functions
# =============================================================================


def load_phone_map_json(map_file: Path) -> dict[str, str]:
    """Load phone mapping from JSON file.

    JSON format:
    {
      "_description": "Optional (ignored)",
      "PHONE1": "target1",
      "PHONE2": "target2"
    }

    Args:
        map_file: Path to JSON mapping file

    Returns:
        Dictionary mapping source phones to target phones
    """
    with map_file.open(encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_phone_map_text(map_file: Path) -> dict[str, str]:
    """Load phone mapping from text file.

    Format: SOURCE -> TARGET
    Example:
        AA -> ɑ
        # Comment

    Args:
        map_file: Path to mapping file

    Returns:
        Dictionary mapping source phones to target phones
    """
    phone_map: dict[str, str] = {}

    with map_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "->" not in line:
                continue

            parts = line.split("->")
            if len(parts) != 2:
                continue

            source = parts[0].strip()
            target = parts[1].strip()

            if source:
                phone_map[source] = target

    return phone_map


def reverse_phone_map(phone_map: dict[str, str]) -> dict[str, str]:
    """Attempt to create reverse mapping from a phone map.

    Mappings are directional by design. This utility tries to invert a
    mapping but may lose information:
    - Expansion mappings (A -> "x y") become multiple entries (x -> A, y -> A)
    - Multiple sources mapping to same target: only last one preserved
    - Deletions (A -> "") cannot be reversed

    For reliable bidirectional conversion, provide separate forward and
    reverse mapping files rather than relying on this function.

    Args:
        phone_map: Original phone mapping

    Returns:
        Best-effort reversed mapping (may be lossy)
    """
    reversed_map: dict[str, str] = {}
    for source, target in phone_map.items():
        if target:
            for t in target.split():
                reversed_map[t] = source
    return reversed_map


def create_stress_strip_map(vowels: set[str], stress_markers: str = "012") -> dict[str, str]:
    """Create a phone map that strips stress markers from vowels.

    This is a general approach that works with any phoneset using
    numeric suffixes for stress (like ARPABET) or tone markers.

    Args:
        vowels: Set of base vowel phones (without stress)
        stress_markers: Characters used as stress suffixes (default: "012")

    Returns:
        Phone map: {"AA0": "AA", "AA1": "AA", "AA2": "AA", ...}

    Example:
        vowels = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
                  "IH", "IY", "OW", "OY", "UH", "UW"}
        stress_map = create_stress_strip_map(vowels)
        # Result: {"AA0": "AA", "AA1": "AA", "AA2": "AA", "AE0": "AE", ...}
    """
    stress_map: dict[str, str] = {}
    for vowel in vowels:
        for marker in stress_markers:
            stress_map[f"{vowel}{marker}"] = vowel
    return stress_map
