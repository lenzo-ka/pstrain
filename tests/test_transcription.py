"""Tests for accepted word-level transcription formats."""

from pathlib import Path

import pytest

from pstrain.lib.transcription import parse_transcription_file


@pytest.mark.parametrize(
    ("line", "fileid", "text"),
    [
        ("<s> SOME WORDS </s> (utt-marked)", "utt-marked", "SOME WORDS"),
        ("SOME WORDS (utt-unmarked)", "utt-unmarked", "SOME WORDS"),
        ("<s> SOME WORDS (utt-open)", "utt-open", "SOME WORDS"),
        ("SOME WORDS </s> (utt-close)", "utt-close", "SOME WORDS"),
        ("utt-simple SOME WORDS", "utt-simple", "SOME WORDS"),
        (
            "TEXT WITH (PARENTHESES) INSIDE (utt-parentheses)",
            "utt-parentheses",
            "TEXT WITH (PARENTHESES) INSIDE",
        ),
    ],
)
def test_parse_transcription_formats(tmp_path: Path, line: str, fileid: str, text: str) -> None:
    transcription = tmp_path / "transcription.txt"
    transcription.write_text(f"{line}\n", encoding="utf-8")

    assert parse_transcription_file(transcription) == {fileid: text}
