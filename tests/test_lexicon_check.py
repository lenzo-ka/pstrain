"""Tests for the lexicon phone-inventory check (pstrain.lib.lexicon_check)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest

from pstrain.lib.alignment.batch import collect_phone_report, explain_failure
from pstrain.lib.dictionary import Dictionary
from pstrain.lib.lexicon_check import (
    check_lexicon_phones,
    check_model_lexicon,
    describe_unsupported,
    read_pronunciations,
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

    def test_an_empty_inventory_is_rejected(self, tmp_path: Path) -> None:
        # An empty inventory would report every phone in use as undefined.
        mdef = tmp_path / "mdef"
        mdef.write_text("0.3\n0 n_base\n0 n_tri\n")
        with pytest.raises(ValueError, match="declares no base phones"):
            read_ci_phones(mdef)


class TestReadPronunciations:
    """The file's own spelling of each word is what the native loaders key on."""

    def test_variant_suffixes_are_kept_as_written(self, tmp_path: Path) -> None:
        # Dictionary renumbers these by order of appearance; this must not.
        path = tmp_path / "d.dict"
        path.write_text("boeuf(2) B OE F\nboeuf B AH F\n", encoding="utf-8")

        assert read_pronunciations(path) == [
            ("boeuf(2)", ("B", "OE", "F")),
            ("boeuf", ("B", "AH", "F")),
        ]

    def test_blank_comment_and_pronunciationless_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "d.dict"
        path.write_text("\n# a comment\nlonely\ngood B AH F # trailing\n", encoding="utf-8")

        assert read_pronunciations(path) == [("good", ("B", "AH", "F"))]


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
        assert entry.source == "dictionary"

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
        assert report.entries[0].source == "filler dictionary"

    def test_dictionaries_sharing_a_basename_are_both_reported(self, tmp_path: Path) -> None:
        # Keyed by basename, the second would have replaced the first.
        main_dir = tmp_path / "main"
        filler_dir = tmp_path / "filler"
        main_dir.mkdir()
        filler_dir.mkdir()
        (main_dir / "shared.dict").write_text("boeuf B OE F\n", encoding="utf-8")
        (filler_dir / "shared.dict").write_text("<cough> KQ\n", encoding="utf-8")

        report = check_model_lexicon(_MODEL, main_dir / "shared.dict", filler_dir / "shared.dict")

        assert report.words == ("<cough>", "boeuf")
        assert {entry.source for entry in report.entries} == {
            "dictionary",
            "filler dictionary",
        }

    def test_a_dropped_alternative_leaves_the_word_usable(self, tmp_path: Path) -> None:
        # The fixture supplies an unsuffixed "bear"; only the alternative goes.
        dict_path = _dictionary_with(tmp_path, "bear(2) B EH2 R\n")
        report = check_model_lexicon(_MODEL, dict_path)

        assert report.words == ("bear(2)",)
        assert report.unresolvable_words == frozenset()
        assert report.resolved_to_alternative == frozenset()
        assert "still resolve" in report.format()

    def test_a_dropped_base_resolves_to_its_surviving_alternative(self, tmp_path: Path) -> None:
        # Both loaders make "boeuf(2)" the pronunciation of "boeuf". The
        # dropped pronunciation is still reported, and the report says what
        # the word now resolves to rather than that the run will not start.
        dict_path = _dictionary_with(tmp_path, "boeuf B OE F\nboeuf(2) B AH F\n")
        report = check_model_lexicon(_MODEL, dict_path)

        assert report.words == ("boeuf",)
        assert report.resolved_to_alternative == frozenset({"boeuf"})
        assert report.unresolvable_words == frozenset()

        text = report.format()
        assert text.startswith("Pronunciations using phones the model does not define")
        assert "boeuf  B OE F  undefined: OE" in text
        # The paragraphs are wrapped, so compare against unwrapped text.
        prose = " ".join(text.split())
        assert "resolves to its first surviving alternative" in prose
        assert "in alignment as in multiple-pronunciation training" in prose
        assert "will not start" not in prose
        assert "still resolve" not in prose

    def test_an_alternative_read_before_its_dropped_base_is_not_resolved(
        self, tmp_path: Path
    ) -> None:
        # The aligner needs the base read before an alternative; an
        # alternative with no base yet fails whatever the phones, so this
        # check must not claim the word resolves.
        dict_path = _dictionary_with(tmp_path, "boeuf(2) B AH F\nboeuf B OE F\n")
        report = check_model_lexicon(_MODEL, dict_path)

        assert report.words == ("boeuf",)
        assert report.resolved_to_alternative == frozenset()
        assert report.unresolvable_words == frozenset()
        assert "resolves to" not in " ".join(report.format().split())

    def test_a_dropped_alternative_before_its_base_leaves_the_base(self, tmp_path: Path) -> None:
        # The native loader keys on the spelling in the file: the dropped line
        # is "boeuf(2)", and the unsuffixed "boeuf" loads. A check built on
        # Dictionary keys would renumber these and call the base dropped;
        # this is the guard against that.
        dict_path = _dictionary_with(tmp_path, "boeuf(2) B OE F\nboeuf B AH F\n")
        report = check_model_lexicon(_MODEL, dict_path)

        assert report.words == ("boeuf(2)",)
        assert report.resolved_to_alternative == frozenset()
        assert report.unresolvable_words == frozenset()

    def test_missing_phones_are_gathered_across_a_word_s_variants(self, tmp_path: Path) -> None:
        dict_path = _dictionary_with(tmp_path, "chien SH Y2\nchien(2) SH OE\n")
        report = check_model_lexicon(_MODEL, dict_path)

        assert report.missing_by_base["chien"] == ("OE", "Y2")

    def test_a_word_with_no_pronunciation_does_not_disable_the_check(self, tmp_path: Path) -> None:
        # The native loader ignores such a line; so must this, or one sloppy
        # line silently turns the whole check off.
        dict_path = _dictionary_with(tmp_path, "orphanword\nboeuf B OE F\n")
        report = check_model_lexicon(_MODEL, dict_path)

        assert report.words == ("boeuf",)

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

    def test_a_skipped_check_says_so(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Silently skipping turns the whole feature off with no trace.
        model = tmp_path / "model"
        model.mkdir()
        (model / "mdef").write_text("0.2\n1 n_base\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            assert collect_phone_report(model, _DICT) is None

        assert "Not checking the dictionary" in caplog.text
        assert "0.2" in caplog.text

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
