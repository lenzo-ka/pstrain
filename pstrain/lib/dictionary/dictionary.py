"""Pronunciation dictionary handling.

This module handles CMU Sphinx-format pronunciation dictionaries.
The phoneset is not specified - dictionaries can use any phonetic alphabet
(ARPABET, IPA, X-SAMPA, etc.) as long as phones are whitespace-separated.

Pronunciation Variants
----------------------
Sphinx supports multiple pronunciations per word using variant notation:
    word        phone1 phone2 ...   # First/canonical pronunciation
    word(2)     phone3 phone4 ...   # Second variant
    word(3)     phone5 phone6 ...   # Third variant

Note: Variant numbering starts at 2 (no suffix means variant 1).

How Variants Are Used in Training
----------------------------------
During Baum-Welch training, the bw binary automatically:
1. Reads all pronunciation variants from the dictionary
2. For each word in the transcript, tries ALL variants during forward-backward pass
3. Computes posterior probabilities for each variant given the audio
4. Uses the best-scoring variant for that utterance
5. Accumulates statistics weighted by variant posterior probabilities

This means:
- The model automatically learns which variant fits each utterance
- You don't need to manually select variants
- Training data effectively "votes" on which variants are most common
- Rare variants still contribute if they match the audio
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Self


class Dictionary:
    """Pronunciation dictionary with Unicode and case-sensitive support.

    Supports:
    - UTF-8 encoding
    - Case-sensitive words (hello != Hello != HELLO)
    - Sphinx-style variants: word, word(2), word(3), ...
    - Multi-word entries: New_York, ice_cream
    - Any phoneset (ARPABET, IPA, X-SAMPA, custom)
    """

    def __init__(self) -> None:
        """Initialize empty dictionary."""
        # Map from word key (including variant suffix) to list of phonemes
        self._entries: dict[str, list[str]] = {}
        # Map from base word to list of variant keys
        self._variants: dict[str, list[str]] = {}

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Load dictionary from file.

        Args:
            path: Path to dictionary file (UTF-8 encoded)

        Returns:
            Dictionary instance

        Raises:
            ValueError: If file has encoding or format errors
        """
        dictionary = cls()

        try:
            with path.open(encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue

                    # CMUDict format: WORD<whitespace>P1 P2 P3 # optional comment
                    # Separator between word and pronunciation can be space(s) or tab(s)
                    # First split: separate word from remainder (maxsplit=1)
                    parts = line.split(None, 1)
                    if len(parts) < 2:
                        raise ValueError(
                            f"Line {line_num}: Invalid entry '{line}'\n"
                            f"Expected: word phone1 phone2 ..."
                        )

                    word = parts[0]
                    pronunciation = parts[1]

                    # Strip inline comments from pronunciation
                    # Common bug: forgetting this, getting "#" and comment words
                    # like "place" or "danish" in the pronunciation
                    if "#" in pronunciation:
                        pronunciation = pronunciation.split("#", 1)[0]

                    # Trim whitespace from pronunciation string
                    pronunciation = pronunciation.strip()
                    if not pronunciation:
                        raise ValueError(
                            f"Line {line_num}: No pronunciation for '{word}'\nLine was: {line}"
                        )

                    # Split to get individual phones
                    phonemes = pronunciation.split()

                    # Add entry
                    dictionary.add_entry(word, phonemes)

        except UnicodeDecodeError as e:
            raise ValueError(
                f"Dictionary file {path} is not valid UTF-8.\n"
                f"pstrain requires UTF-8 encoding for all text files.\n"
                f"Error: {e}"
            ) from e

        return dictionary

    def add_entry(self, word: str, phonemes: list[str]) -> None:
        """Add pronunciation entry to dictionary.

        Automatically handles duplicate pronunciations:
        - Same word + same pronunciation → Skip silently (true duplicate)
        - Same word + different pronunciation → Create variant (word(2), word(3), etc.)

        Args:
            word: Word (may include variant suffix like (2), (3))
            phonemes: List of phoneme strings

        Examples:
            READ  R EH D    # Stored as READ
            READ  R IY D    # Automatically stored as READ(2)
            READ  R EH D    # Skipped (duplicate of first)
        """
        # Parse base word and check for existing variants
        base_word, _ = self._parse_variant(word)

        # Get all existing variants for this base word
        existing_variants = self._variants.get(base_word, [])

        # Check if this exact pronunciation already exists
        phonemes_tuple = tuple(phonemes)
        for variant_key in existing_variants:
            if variant_key in self._entries:
                existing_phones = tuple(self._entries[variant_key])
                if existing_phones == phonemes_tuple:
                    # True duplicate - same word, same pronunciation, skip silently
                    return

        # Determine the correct variant number to use
        # Always assign sequential numbers: base word = 1, first variant = 2, etc.
        if existing_variants:
            # Find the highest existing variant number
            max_variant = 1
            for variant_key in existing_variants:
                _, var_num = self._parse_variant(variant_key)
                max_variant = max(max_variant, var_num)

            # This is a new pronunciation - assign next sequential number
            next_variant = max_variant + 1
            word = f"{base_word}({next_variant})" if next_variant > 1 else base_word
        else:
            # First occurrence of this word
            word = base_word  # Store without variant number

        # Store the entry
        self._entries[word] = phonemes

        # Track variants
        if base_word not in self._variants:
            self._variants[base_word] = []
        if word not in self._variants[base_word]:
            self._variants[base_word].append(word)

    @staticmethod
    def _parse_variant(word: str) -> tuple[str, int]:
        """Parse word into base word and variant number.

        Args:
            word: Word, possibly with variant suffix

        Returns:
            Tuple of (base_word, variant_number)
            variant_number is 1 for base word, 2+ for variants

        Examples:
            "hello" -> ("hello", 1)
            "hello(2)" -> ("hello", 2)
            "New_York" -> ("New_York", 1)
            "New_York(2)" -> ("New_York", 2)
        """
        # Match word(N) pattern
        match = re.match(r"^(.+)\((\d+)\)$", word)
        if match:
            base = match.group(1)
            num = int(match.group(2))
            return (base, num)
        return (word, 1)

    def get(self, word: str) -> list[str] | None:
        """Get pronunciation for exact word match.

        Args:
            word: Word to look up (case-sensitive, exact match)

        Returns:
            List of phonemes, or None if not found
        """
        return self._entries.get(word)

    def get_variants(self, base_word: str) -> list[list[str]]:
        """Get all pronunciation variants for a base word.

        Args:
            base_word: Base word (without variant suffix)

        Returns:
            List of pronunciations (each is a list of phonemes)
            Returns empty list if word not found

        Examples:
            get_variants("read") -> [["R", "IY", "D"], ["R", "EH", "D"]]
        """
        variant_keys = self._variants.get(base_word, [])
        return [self._entries[key] for key in variant_keys]

    def contains(self, word: str) -> bool:
        """Check if word exists in dictionary (exact match, case-sensitive)."""
        return word in self._entries

    def contains_base(self, base_word: str) -> bool:
        """Check if base word exists (any variant)."""
        return base_word in self._variants

    def words(self) -> list[str]:
        """Get all word keys (including variants)."""
        return list(self._entries.keys())

    def base_words(self) -> list[str]:
        """Get all base words (no variant suffixes)."""
        return list(self._variants.keys())

    def phonemes(self) -> set[str]:
        """Get set of all phonemes used in dictionary."""
        all_phonemes: set[str] = set()
        for phoneme_list in self._entries.values():
            all_phonemes.update(phoneme_list)
        return all_phonemes

    @property
    def pronunciations(self) -> dict[str, list[list[str]]]:
        """Get all pronunciations as a dictionary mapping base words to variant lists.

        Returns:
            Dict mapping base word to list of pronunciations
            Each pronunciation is a list of phonemes

        Examples:
            {"read": [["R", "IY", "D"], ["R", "EH", "D"]],
             "hello": [["HH", "AH", "L", "OW"]]}
        """
        result = {}
        for base_word, variant_keys in self._variants.items():
            result[base_word] = [self._entries[key] for key in variant_keys]
        return result

    def __len__(self) -> int:
        """Number of dictionary entries (including variants)."""
        return len(self._entries)

    def merge(self, other: Dictionary) -> None:
        """Merge another dictionary into this one.

        Args:
            other: Dictionary to merge in

        Note:
            Uses add_entry which automatically handles:
            - Deduplicating identical pronunciations
            - Creating variants for different pronunciations
            - Renumbering variants sequentially
        """
        for word, phones in other._entries.items():
            self.add_entry(word, phones)

    def filter_to_vocabulary(
        self, vocabulary: set[str], include_variants: bool = True
    ) -> Dictionary:
        """Create filtered dictionary containing only words in vocabulary.

        Args:
            vocabulary: Set of words to keep (case-sensitive)
            include_variants: If True (default), "word" in vocab gets all variants
                (word, word(2), word(3)). If False, only exact matches.

        Returns:
            New Dictionary containing matched words

        Examples:
            Given dict: read, read(2), hello, world
            vocab = {"read", "hello"}

            include_variants=True:  read, read(2), hello
            include_variants=False: read, hello (no read(2))

            vocab = {"read(2)"}
            include_variants=True:  read(2) only (exact variant specified)
            include_variants=False: read(2) only
        """
        filtered = Dictionary()

        for word in vocabulary:
            if include_variants:
                # Try as base word (gets all variants)
                base, _ = self._parse_variant(word)
                if base in self._variants:
                    for variant_key in self._variants[base]:
                        if variant_key in self._entries:
                            filtered.add_entry(variant_key, self._entries[variant_key])
                elif word in self._entries:
                    # Exact match fallback
                    filtered.add_entry(word, self._entries[word])
            else:
                # Exact match only
                if word in self._entries:
                    filtered.add_entry(word, self._entries[word])

        return filtered

    def save(self, path: Path) -> None:
        """Save dictionary to file.

        Args:
            path: Output file path

        Note:
            Writes in CMU Sphinx format:
            - One entry per line: word phone1 phone2 ...
            - Variants: word(2), word(3), etc.
            - UTF-8 encoding
        """
        with path.open("w", encoding="utf-8") as f:
            # Write entries sorted by base word, then by variant number
            for base_word in sorted(self._variants.keys()):
                for variant_key in self._variants[base_word]:
                    phones = self._entries[variant_key]
                    f.write(f"{variant_key}  {' '.join(phones)}\n")

    def __repr__(self) -> str:
        """String representation."""
        n_entries = len(self._entries)
        n_base = len(self._variants)
        n_phones = len(self.phonemes())
        return f"Dictionary({n_entries} entries, {n_base} base words, {n_phones} phonemes)"
