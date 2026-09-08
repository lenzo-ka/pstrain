from pathlib import Path

import pytest

from pstrain.lib.dictionary.cmudict import strip_dictionary_stress

ROOT = Path(__file__).parents[1]


def test_strip_dictionary_stress_merges_and_renumbers_variants(tmp_path: Path) -> None:
    source = tmp_path / "stressed.dict"
    output = tmp_path / "stripped.dict"
    source.write_text(
        "# fixture\n"
        "\n"
        "the DH AH0\n"
        "the(2) DH AH1\n"
        "the(3) DH IY0\n"
        "stay S T EY1\n"
        "stay(2) S T IY1\n"
        "x EH1 K S\n"
        "x(2) EH0 K S\n"
        "x(3) IH1 K S\n",
        encoding="utf-8",
    )

    entries, phones = strip_dictionary_stress(source, output)

    assert output.read_text(encoding="utf-8").splitlines() == [
        "stay S T EY",
        "stay(2) S T IY",
        "the DH AH",
        "the(2) DH IY",
        "x EH K S",
        "x(2) IH K S",
    ]
    assert entries == 6
    assert phones == 9


def _assert_matches_converter(dictionary: Path, regenerated: Path) -> None:
    strip_dictionary_stress(dictionary, regenerated)
    assert regenerated.read_bytes() == dictionary.read_bytes(), (
        "bundled Arctic dictionary differs from strip_dictionary_stress output; "
        "regenerate and commit it before changing the converter"
    )


def test_bundled_arctic_dictionary_matches_converter(tmp_path: Path) -> None:
    """Keep the committed artifact byte-identical to converter regeneration."""
    dictionary = ROOT / "benchmarks" / "arctic" / "data" / "cmu_arctic_slt.dict"
    _assert_matches_converter(dictionary, tmp_path / dictionary.name)


def test_arctic_dictionary_check_rejects_noncanonical_order(tmp_path: Path) -> None:
    dictionary = tmp_path / "out-of-order.dict"
    dictionary.write_text("can't K AE N T\ncan(2) K AH N\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="regenerate and commit"):
        _assert_matches_converter(dictionary, tmp_path / "regenerated.dict")
