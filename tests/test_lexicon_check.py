"""Tests for the lexicon phone-inventory check (pstrain.lib.lexicon_check)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from pstrain.lib.alignment.batch import collect_phone_report, explain_failure
from pstrain.lib.dictionary import Dictionary
from pstrain.lib.lexicon_check import (
    check_lexicon_phones,
    check_model_lexicon,
    describe_unsupported,
    unsupported_pronunciations,
)
from pstrain.lib.model import read_ci_phones
from pstrain.lib.phoneset import Phoneset

_FIXTURES = Path(__file__).parent / "fixtures"
_MODEL = _FIXTURES / "multipron_final_state" / "model"
_DICT = _FIXTURES / "multipron_final_state" / "dictionary.dict"


def _dictionary_with(tmp_path: Path, extra_lines: str) -> Path:
    """Copy the fixture dictionary and append pronunciations to it."""
    target = tmp_path / "dictionary.dict"
    shutil.copy(_DICT, target)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(extra_lines)
    return target


class TestReadCiPhones:
    """The model's context-independent inventory is readable from Python."""

    def test_reads_the_fixture_inventory(self) -> None:
        phones = read_ci_phones(_MODEL / "mdef")
        assert len(phones) == 40
        assert phones[0] == "AA"
        assert "SIL" in phones
        assert all("-" not in phone for phone in phones)

    def test_missing_file_is_named(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Model definition not found"):
            read_ci_phones(tmp_path / "mdef")

    def test_wrong_version_is_rejected(self, tmp_path: Path) -> None:
        mdef = tmp_path / "mdef"
        mdef.write_text("0.2\n1 n_base\n")
        with pytest.raises(ValueError, match="declares version"):
            read_ci_phones(mdef)

    def test_missing_n_base_is_rejected(self, tmp_path: Path) -> None:
        mdef = tmp_path / "mdef"
        mdef.write_text("0.3\n0 n_tri\n")
        with pytest.raises(ValueError, match="no n_base declaration"):
            read_ci_phones(mdef)

    def test_truncated_records_are_rejected(self, tmp_path: Path) -> None:
        mdef = tmp_path / "mdef"
        mdef.write_text(
            "0.3\n2 n_base\n0 n_tri\n#\n   AA   -   - -    n/a    0    0    1    2    N\n"
        )
        with pytest.raises(ValueError, match="holds 1 model records"):
            read_ci_phones(mdef)


class TestUnsupportedPronunciations:
    """Offending words are named, not just their phones."""

    def test_clean_dictionary_reports_nothing(self) -> None:
        report = check_model_lexicon(_MODEL, _DICT)
        assert not report
        assert report.entries == ()
        assert report.missing_phones == ()
        assert "All pronunciations use phones defined by" in report.format()

    def test_names_the_word_the_pronunciation_and_the_phone(self, tmp_path: Path) -> None:
        dict_path = _dictionary_with(tmp_path, "boeuf B OE F\n")
        report = check_model_lexicon(_MODEL, dict_path)

        assert report
        assert report.words == ("boeuf",)
        assert report.missing_phones == ("OE",)
        entry = report.entries[0]
        assert entry.word == "boeuf"
        assert entry.phones == ("B", "OE", "F")
        assert entry.missing == ("OE",)
        assert entry.source == "dictionary.dict"

    def test_report_text_names_word_phones_and_inventory(self, tmp_path: Path) -> None:
        dict_path = _dictionary_with(tmp_path, "boeuf B OE F\noeil OE Y2\n")
        text = check_model_lexicon(_MODEL, dict_path).format()

        assert "Pronunciations using phones the model does not define" in text
        assert "40 phones" in text
        assert "Undefined: OE, Y2" in text
        assert "boeuf" in text
        assert "B OE F" in text
        assert "oeil" in text
        assert "undefined: OE, Y2" in text
        assert "out-of-vocabulary problem but" in text

    def test_long_lists_are_truncated_with_a_count(self, tmp_path: Path) -> None:
        extra = "".join(f"word{index} B OE F\n" for index in range(25))
        dict_path = _dictionary_with(tmp_path, extra)
        text = check_model_lexicon(_MODEL, dict_path).format(limit=5)

        assert "... and 20 more pronunciations" in text

    def test_filler_dictionary_is_checked_alongside(self, tmp_path: Path) -> None:
        filler = tmp_path / "filler.dict"
        filler.write_text("<cough> KQ\n", encoding="utf-8")
        report = check_model_lexicon(_MODEL, _DICT, filler)

        assert report.words == ("<cough>",)
        assert report.entries[0].source == "filler.dict"

    def test_a_word_keeping_a_supported_variant_is_not_unresolvable(self, tmp_path: Path) -> None:
        dict_path = _dictionary_with(tmp_path, "bear(2) B EH2 R\nboeuf B OE F\n")
        report = check_model_lexicon(_MODEL, dict_path)

        assert "bear(2)" in report.words
        assert report.unresolvable_words == frozenset({"boeuf"})

    def test_describe_unsupported_names_words_on_one_line(self) -> None:
        phoneset = Phoneset({"B", "F"})
        dictionary = Dictionary()
        dictionary.add_entry("boeuf", ["B", "OE", "F"])
        entries = unsupported_pronunciations(phoneset, dictionary, "dictionary")

        summary = describe_unsupported(entries)
        assert "boeuf [B OE F]" in summary
        assert "undefined phones (OE)" in summary
        assert describe_unsupported([]) == ""

    def test_check_lexicon_phones_accepts_a_plain_phoneset(self) -> None:
        phoneset = Phoneset({"B", "F"})
        dictionary = Dictionary()
        dictionary.add_entry("boeuf", ["B", "OE", "F"])
        report = check_lexicon_phones(phoneset, {"d.dict": dictionary}, inventory="phoneset.txt")

        assert report.inventory == "phoneset.txt"
        assert report.inventory_size == 2
        assert report.unresolvable_words == frozenset({"boeuf"})


class TestAlignmentPathReporting:
    """The alignment path collects the report and explains the later failure."""

    def test_collect_phone_report_finds_the_offending_word(self, tmp_path: Path) -> None:
        dict_path = _dictionary_with(tmp_path, "boeuf B OE F\n")
        report = collect_phone_report(_MODEL, dict_path)

        assert report is not None
        assert report.words == ("boeuf",)

    def test_collect_phone_report_survives_an_unreadable_model(self, tmp_path: Path) -> None:
        assert collect_phone_report(tmp_path / "absent-model", _DICT) is None

    def test_failure_message_names_the_phone_inventory_cause(self, tmp_path: Path) -> None:
        dict_path = _dictionary_with(tmp_path, "boeuf B OE F\n")
        report = collect_phone_report(_MODEL, dict_path)
        assert report is not None

        explained = explain_failure("boeuf not in dictionary", "<s> boeuf </s>", report)
        assert explained.startswith("boeuf not in dictionary")
        assert "no pronunciation survived the model's phone inventory for: boeuf (OE)" in explained

    def test_unrelated_failures_are_left_alone(self, tmp_path: Path) -> None:
        dict_path = _dictionary_with(tmp_path, "boeuf B OE F\n")
        report = collect_phone_report(_MODEL, dict_path)

        assert explain_failure("beam too narrow", "<s> bear </s>", report) == "beam too narrow"
        assert explain_failure("beam too narrow", "<s> boeuf </s>", None) == "beam too narrow"
