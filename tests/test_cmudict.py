from pathlib import Path

from pstrain.lib.dictionary.cmudict import strip_dictionary_stress


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
