"""Dictionary handling for ST2 projects."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Self

__all__ = ["Dictionary"]


class Dictionary:
    """Pronunciation dictionary."""

    def __init__(self, entries: dict[str, list[str]]) -> None:
        """Initialize dictionary from entries.

        Args:
            entries: Dict mapping word -> list of phones
        """
        self._entries = entries

    @classmethod
    def from_file(cls, dict_path: Path) -> Self:
        """Load dictionary from file.

        Args:
            dict_path: Path to dictionary file

        Returns:
            Dictionary instance with loaded entries

        Raises:
            FileNotFoundError: If dictionary file does not exist
            UnicodeDecodeError: If file is not UTF-8 encoded

        Format:
            One entry per line: ``<word> <phone1> <phone2> ...``
            Comments start with ``#`` and are ignored
        """
        entries = {}
        with dict_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Remove inline comments
                if "#" in line:
                    line = line.split("#", 1)[0].strip()
                parts = line.split()
                if len(parts) < 2:
                    continue
                word = parts[0]
                phones = parts[1:]
                # Filter out comment marker as phone
                phones = [p for p in phones if p != "#"]
                if phones:
                    entries[word] = phones
        return cls(entries)

    def get_phones(self, word: str) -> list[str] | None:
        """Get phones for a word.

        Args:
            word: Word to look up

        Returns:
            List of phones for the word, or None if word not found
        """
        return self._entries.get(word)

    def has_word(self, word: str) -> bool:
        """Check if word is in dictionary.

        Args:
            word: Word to check

        Returns:
            True if word exists in dictionary, False otherwise
        """
        return word in self._entries

    def phonemes(self) -> set[str]:
        """Get all unique phonemes in dictionary.

        Returns:
            Set of all unique phone symbols used in the dictionary
        """
        phones = set()
        for phone_list in self._entries.values():
            phones.update(phone_list)
        return phones

    def words(self) -> set[str]:
        """Get all words in dictionary.

        Returns:
            Set of all words in the dictionary
        """
        return set(self._entries.keys())

    def base_words(self) -> set[str]:
        """Get words with pronunciation-variant suffixes removed."""
        return {re.sub(r"\(\d+\)$", "", word) for word in self._entries}

    def __len__(self) -> int:
        """Get number of entries in dictionary.

        Returns:
            Number of word entries
        """
        return len(self._entries)
