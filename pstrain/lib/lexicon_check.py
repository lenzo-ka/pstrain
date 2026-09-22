"""Check lexicon pronunciations against a phone inventory before a run.

A pronunciation that uses a phone the acoustic model does not define cannot
be loaded. The native lexicon reader names the phone on one line, names the
word on another, drops the entry, and carries on with a quietly smaller
lexicon; the end-of-load summary counts only what survived. The word then
resurfaces much later as an unresolvable transcript token, which reads like
an out-of-vocabulary problem rather than a phone-inventory one.

This module makes that condition visible where it can be fixed: before the
run, as one collected report that names every affected word, its
pronunciation, and the phones that are missing.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

from pstrain.lib.dictionary import Dictionary
from pstrain.lib.model import read_ci_phones
from pstrain.lib.phoneset import Phoneset

# Entries listed in full before the report summarizes the remainder.
DEFAULT_ENTRY_LIMIT = 20


@dataclass(frozen=True)
class UnsupportedPronunciation:
    """One pronunciation that uses phones outside the target inventory."""

    source: str
    word: str
    phones: tuple[str, ...]
    missing: tuple[str, ...]


@dataclass(frozen=True)
class UnsupportedPhoneReport:
    """Every pronunciation that a phone inventory cannot support.

    Attributes:
        inventory: Human-readable description of what was validated against.
        inventory_size: Number of phones in that inventory.
        entries: The offending pronunciations, ordered by source then word.
        unresolvable_words: Base words left with no usable pronunciation at
            all. A word that keeps a supported variant still aligns and is
            not listed here, though its dropped variant is still reported.
    """

    inventory: str
    inventory_size: int
    entries: tuple[UnsupportedPronunciation, ...] = ()
    unresolvable_words: frozenset[str] = frozenset()

    def __bool__(self) -> bool:
        """True when at least one pronunciation is unsupported."""
        return bool(self.entries)

    @property
    def missing_phones(self) -> tuple[str, ...]:
        """Every phone used by a pronunciation but absent from the inventory."""
        phones: set[str] = set()
        for entry in self.entries:
            phones.update(entry.missing)
        return tuple(sorted(phones))

    @property
    def words(self) -> tuple[str, ...]:
        """Dictionary keys of the affected entries, variant suffixes included."""
        return tuple(sorted({entry.word for entry in self.entries}))

    def format(self, limit: int = DEFAULT_ENTRY_LIMIT) -> str:
        """Render the collected report as a block of text."""
        if not self.entries:
            return f"All pronunciations use phones defined by {self.inventory}."

        missing = self.missing_phones
        lines = [
            "Pronunciations using phones the model does not define",
            f"  Inventory: {self.inventory} ({self.inventory_size} phones)",
            f"  Affected:  {len(self.entries)} pronunciations, "
            f"{len(self.words)} words, {len(missing)} undefined phones",
            f"  Undefined: {', '.join(missing)}",
        ]

        shown = self.entries[:limit]
        source_width = max(len(entry.source) for entry in shown)
        word_width = max(len(entry.word) for entry in shown)
        phone_width = max(len(" ".join(entry.phones)) for entry in shown)
        for entry in shown:
            pronunciation = " ".join(entry.phones)
            lines.append(
                f"    {entry.source:<{source_width}}  {entry.word:<{word_width}}  "
                f"{pronunciation:<{phone_width}}  undefined: {', '.join(entry.missing)}"
            )
        if len(self.entries) > limit:
            lines.append(f"    ... and {len(self.entries) - limit} more pronunciations")

        lines.extend(
            textwrap.wrap(
                "These pronunciations are dropped when the lexicon loads. Utterances "
                "containing a word left with no pronunciation fail as unresolvable "
                "transcript tokens, which reads like an out-of-vocabulary problem but "
                "is not one. Retrain the model with these phones, or map them onto "
                "phones the model defines.",
                width=88,
                initial_indent="  ",
                subsequent_indent="  ",
            )
        )
        return "\n".join(lines)


def _base_word(word: str) -> str:
    """Strip a Sphinx variant suffix, so ``READ(2)`` becomes ``READ``."""
    if word.endswith(")") and "(" in word:
        base, _, suffix = word.rpartition("(")
        if suffix[:-1].isdigit():
            return base
    return word


def unsupported_pronunciations(
    phoneset: Phoneset,
    dictionary: Dictionary,
    source: str,
) -> list[UnsupportedPronunciation]:
    """Find the pronunciations in one dictionary that the phoneset cannot support.

    Args:
        phoneset: Inventory to validate against.
        dictionary: Dictionary whose pronunciations are checked.
        source: Label for the dictionary, used in the report.

    Returns:
        One entry per offending pronunciation, ordered by word.
    """
    found: list[UnsupportedPronunciation] = []
    for word in sorted(dictionary.words()):
        phones = dictionary.get(word)
        if phones is None:
            continue
        missing = sorted({phone for phone in phones if not phoneset.contains(phone)})
        if missing:
            found.append(
                UnsupportedPronunciation(
                    source=source,
                    word=word,
                    phones=tuple(phones),
                    missing=tuple(missing),
                )
            )
    return found


def describe_unsupported(
    entries: list[UnsupportedPronunciation],
    limit: int = 5,
) -> str:
    """Summarize unsupported pronunciations on one line, naming the words.

    Args:
        entries: Offending pronunciations from one dictionary.
        limit: How many words to name before counting the rest.

    Returns:
        A one-line summary, or the empty string when nothing is unsupported.
    """
    if not entries:
        return ""
    missing = sorted({phone for entry in entries for phone in entry.missing})
    examples = "; ".join(f"{entry.word} [{' '.join(entry.phones)}]" for entry in entries[:limit])
    remainder = f"; and {len(entries) - limit} more" if len(entries) > limit else ""
    return (
        f"{len(missing)} undefined phones ({', '.join(missing)}) used by "
        f"{len(entries)} pronunciations: {examples}{remainder}"
    )


def check_lexicon_phones(
    phoneset: Phoneset,
    sources: dict[str, Dictionary],
    inventory: str,
) -> UnsupportedPhoneReport:
    """Validate several dictionaries against one phone inventory.

    Args:
        phoneset: Inventory to validate against.
        sources: Mapping of report label to dictionary.
        inventory: Human-readable description of the inventory.

    Returns:
        A collected :class:`UnsupportedPhoneReport`.
    """
    entries: list[UnsupportedPronunciation] = []
    dropped: dict[str, int] = {}
    for source, dictionary in sources.items():
        found = unsupported_pronunciations(phoneset, dictionary, source)
        entries.extend(found)
        for entry in found:
            base = _base_word(entry.word)
            dropped[base] = dropped.get(base, 0) + 1

    unresolvable = set()
    for base, n_dropped in dropped.items():
        n_total = sum(len(dictionary.get_variants(base)) for dictionary in sources.values())
        if n_dropped >= n_total:
            unresolvable.add(base)

    return UnsupportedPhoneReport(
        inventory=inventory,
        inventory_size=len(phoneset),
        entries=tuple(entries),
        unresolvable_words=frozenset(unresolvable),
    )


def check_model_lexicon(
    model_dir: str | Path,
    dict_path: str | Path,
    filler_dict: str | Path | None = None,
) -> UnsupportedPhoneReport:
    """Validate a dictionary against a trained model's phone inventory.

    The inventory comes from the model's ``mdef``: the context-independent
    phones the model was actually trained on. The filler dictionary, when
    supplied, is checked alongside the main one, because the native loader
    merges both against the same inventory.

    Args:
        model_dir: Acoustic model directory containing ``mdef``.
        dict_path: Pronunciation dictionary.
        filler_dict: Filler dictionary. Optional.

    Returns:
        A collected :class:`UnsupportedPhoneReport`.

    Raises:
        FileNotFoundError: If the model definition or a dictionary is missing.
        ValueError: If the model definition cannot be read.
    """
    model_dir = Path(model_dir)
    mdef_path = model_dir / "mdef"
    phoneset = Phoneset(set(read_ci_phones(mdef_path)))

    sources: dict[str, Dictionary] = {Path(dict_path).name: Dictionary.from_file(Path(dict_path))}
    if filler_dict is not None:
        filler_path = Path(filler_dict)
        sources[filler_path.name] = Dictionary.from_file(filler_path)

    return check_lexicon_phones(phoneset, sources, inventory=str(mdef_path))
