"""Check lexicon pronunciations against a phone inventory before a run.

A pronunciation that uses a phone the acoustic model does not define cannot
be loaded. Both loaders drop it, name the phone on one line and the word on
another, and carry on with a quietly smaller lexicon; the end-of-load summary
counts only what survived. The word then resurfaces much later, which reads
like an out-of-vocabulary problem rather than a phone-inventory one.

The two loaders do not agree on what happens next, so this module does not
state one rule for both:

* The alignment loader (``csrc/programs/sphinx3_align/dict.c``) drops the
  whole dictionary line, and then refuses to add a ``word(2)`` variant whose
  unsuffixed base is absent -- it ends the process. So a word whose unsuffixed
  pronunciation is dropped while an alternative survives stops alignment
  before the first utterance, including utterances that never use that word.
* The training loader (``csrc/libs/libcommon/lexicon.c``) has no such rule.
  It keeps the surviving alternative and carries on.

This module makes the condition visible where it can be fixed: before the
run, as one collected report that names every affected word, its
pronunciation, the phones that are missing, and which of the two outcomes
applies.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from pstrain.lib.dictionary import Dictionary
from pstrain.lib.model import read_ci_phones
from pstrain.lib.phoneset import Phoneset

# Entries listed in full before the report summarizes the remainder.
DEFAULT_ENTRY_LIMIT = 20

# Words named in full inside a summary line before it counts the rest.
_WORD_LIMIT = 10


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
            all. Utterances using them fail; other utterances are unaffected.
        fatal_words: Base words whose unsuffixed pronunciation was dropped
            while a suffixed alternative survived. The alignment loader ends
            the process rather than load such a word, so the whole run fails.
            Always empty for a check made against a phoneset rather than
            against a trained model, because the training loader permits it.
        missing_by_base: Every missing phone for a base word, collected
            across all of its dropped pronunciations.
    """

    inventory: str
    inventory_size: int
    entries: tuple[UnsupportedPronunciation, ...] = ()
    unresolvable_words: frozenset[str] = frozenset()
    fatal_words: frozenset[str] = frozenset()
    missing_by_base: dict[str, tuple[str, ...]] = field(default_factory=dict)

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
        headline = (
            "Alignment will not start: pronunciations use phones the model does not define"
            if self.fatal_words
            else "Pronunciations using phones the model does not define"
        )
        lines = [
            headline,
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

        for paragraph in self._explanations():
            lines.extend(
                textwrap.wrap(paragraph, width=88, initial_indent="  ", subsequent_indent="  ")
            )
        return "\n".join(lines)

    def _explanations(self) -> list[str]:
        """Say what each affected word actually does to a run."""
        paragraphs = []
        if self.fatal_words:
            paragraphs.append(
                f"The unsuffixed pronunciation of {_name_words(self.fatal_words)} was "
                "dropped while an alternative pronunciation survived. The aligner "
                "refuses to load an alternative whose base is absent and ends the run, "
                "so no utterance aligns -- including every utterance that does not use "
                "these words. Training does not share this rule: its loader keeps the "
                "surviving alternative and carries on."
            )
        if self.unresolvable_words:
            paragraphs.append(
                f"{_name_words(self.unresolvable_words)} kept no pronunciation at all. "
                "Utterances using these words fail as unresolvable transcript tokens, "
                "which reads like an out-of-vocabulary problem but is not one."
            )
        other = len(self.entries) - len(self.fatal_words) - len(self.unresolvable_words)
        if other > 0:
            paragraphs.append(
                "The remaining pronunciations are alternatives whose word keeps a "
                "usable pronunciation, so those words still resolve."
            )
        paragraphs.append(
            "Retrain the model with these phones, or map them onto phones the model defines."
        )
        return paragraphs


def _name_words(words: frozenset[str]) -> str:
    """List words for a summary line, counting the rest beyond a few."""
    ordered = sorted(words)
    shown = ", ".join(ordered[:_WORD_LIMIT])
    if len(ordered) > _WORD_LIMIT:
        return f"{shown} and {len(ordered) - _WORD_LIMIT} more"
    return shown


def base_word(word: str) -> str:
    """Strip a Sphinx variant suffix, so ``READ(2)`` becomes ``READ``."""
    if word.endswith(")") and "(" in word:
        base, _, suffix = word.rpartition("(")
        if suffix[:-1].isdigit():
            return base
    return word


def read_pronunciations(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    """Read a dictionary keeping each line's own spelling of its word.

    :class:`~pstrain.lib.dictionary.Dictionary` renumbers variant suffixes by
    order of appearance, so its keys are not the spellings in the file. The
    native loaders key on the spellings, and whether a word is written
    ``word`` or ``word(2)`` decides whether the aligner starts at all, so this
    check has to see the file as they do.

    Lines this skips are the ones the native alignment loader also skips:
    blank lines, comments, and a word with no pronunciation.

    Args:
        path: Dictionary file.

    Returns:
        ``(word, phones)`` in file order, with the word exactly as written.

    Raises:
        FileNotFoundError: If the dictionary does not exist.
        ValueError: If the file is not valid UTF-8.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"Dictionary file {path} is not valid UTF-8. "
            f"pstrain requires UTF-8 encoding for all text files. Error: {error}"
        ) from error

    pronunciations: list[tuple[str, tuple[str, ...]]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        word, remainder = parts
        phones = remainder.split("#", 1)[0].split()
        if not phones:
            continue
        pronunciations.append((word, tuple(phones)))
    return pronunciations


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


def _collect_missing_by_base(
    entries: list[UnsupportedPronunciation],
) -> dict[str, tuple[str, ...]]:
    """Gather every missing phone of a base word across all of its variants."""
    gathered: dict[str, set[str]] = {}
    for entry in entries:
        gathered.setdefault(base_word(entry.word), set()).update(entry.missing)
    return {base: tuple(sorted(phones)) for base, phones in gathered.items()}


def check_lexicon_phones(
    phoneset: Phoneset,
    sources: dict[str, Dictionary],
    inventory: str,
) -> UnsupportedPhoneReport:
    """Validate several dictionaries against one phone inventory.

    This is the training-loader view: a word that keeps any usable
    pronunciation still resolves, and no combination of drops ends the run.
    Use :func:`check_model_lexicon` for the alignment view, which has the
    stricter rule.

    Args:
        phoneset: Inventory to validate against.
        sources: Mapping of report label to dictionary.
        inventory: Human-readable description of the inventory.

    Returns:
        A collected :class:`UnsupportedPhoneReport` with no fatal words.
    """
    entries: list[UnsupportedPronunciation] = []
    dropped: dict[str, int] = {}
    for source, dictionary in sources.items():
        found = unsupported_pronunciations(phoneset, dictionary, source)
        entries.extend(found)
        for entry in found:
            base = base_word(entry.word)
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
        missing_by_base=_collect_missing_by_base(entries),
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
    reads both against the same inventory.

    This reproduces the alignment loader's rules, including the one that ends
    the run: a word whose unsuffixed pronunciation is dropped while a
    suffixed alternative survives is reported in ``fatal_words``.

    Args:
        model_dir: Acoustic model directory containing ``mdef``.
        dict_path: Pronunciation dictionary.
        filler_dict: Filler dictionary. Optional.

    Returns:
        A collected :class:`UnsupportedPhoneReport`.

    Raises:
        FileNotFoundError: If the model definition or a dictionary is missing.
        ValueError: If the model definition or a dictionary cannot be read.
    """
    model_dir = Path(model_dir)
    mdef_path = model_dir / "mdef"
    phoneset = Phoneset(set(read_ci_phones(mdef_path)))

    sources: list[tuple[str, Path]] = [("dictionary", Path(dict_path))]
    if filler_dict is not None:
        sources.append(("filler dictionary", Path(filler_dict)))

    entries: list[UnsupportedPronunciation] = []
    # Per base word, across every source: whether an unsuffixed spelling was
    # seen and kept or seen and dropped, and whether any spelling survived.
    base_kept: set[str] = set()
    base_dropped: set[str] = set()
    variant_kept: set[str] = set()
    any_kept: set[str] = set()

    for label, path in sources:
        for word, phones in read_pronunciations(path):
            base = base_word(word)
            unsuffixed = word == base
            missing = sorted({phone for phone in phones if not phoneset.contains(phone)})
            if missing:
                entries.append(
                    UnsupportedPronunciation(
                        source=label, word=word, phones=phones, missing=tuple(missing)
                    )
                )
                if unsuffixed:
                    base_dropped.add(base)
                continue
            any_kept.add(base)
            if unsuffixed:
                base_kept.add(base)
            else:
                variant_kept.add(base)

    affected = {base_word(entry.word) for entry in entries}
    # The aligner ends the process when it adds a surviving variant whose
    # unsuffixed base is absent. Attribute that to the phone inventory only
    # when the base really was present and really was dropped.
    fatal = {
        base
        for base in affected
        if base in base_dropped and base not in base_kept and base in variant_kept
    }
    unresolvable = {base for base in affected if base not in any_kept}

    return UnsupportedPhoneReport(
        inventory=str(mdef_path),
        inventory_size=len(phoneset),
        entries=tuple(entries),
        unresolvable_words=frozenset(unresolvable),
        fatal_words=frozenset(fatal),
        missing_by_base=_collect_missing_by_base(entries),
    )
