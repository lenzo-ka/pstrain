"""Persistent external train/test split contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from pstrain.lib.corpus.split import (
    SPLIT_FILENAMES,
    split_is_external,
    train_test_split,
    validate_external_split,
)


def _corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "all.transcription"
    source.write_text("a ALPHA\nb BRAVO\nc CHARLIE\n", encoding="utf-8")
    audio = tmp_path / "audio"
    audio.mkdir()
    for fileid in ("a", "b", "c"):
        (audio / f"{fileid}.wav").write_bytes(b"wav")
    split = tmp_path / "etc"
    split.mkdir()
    return source, audio, split


def _external_files(split: Path) -> None:
    (split / "train.fileids").write_text("c\na\n", encoding="utf-8")
    (split / "test.fileids").write_text("b\n", encoding="utf-8")
    (split / "train.transcription").write_text("c CHARLIE\na ALPHA\n", encoding="utf-8")
    (split / "test.transcription").write_text("b BRAVO\n", encoding="utf-8")


def test_external_split_is_preserved_exactly_and_in_order(tmp_path: Path) -> None:
    source, audio, split = _corpus(tmp_path)
    _external_files(split)
    before = {name: (split / name).read_bytes() for name in SPLIT_FILENAMES}

    assert split_is_external(split)
    result = validate_external_split(source, split, audio)

    assert (result.n_train, result.n_test) == (2, 1)
    assert {name: (split / name).read_bytes() for name in SPLIT_FILENAMES} == before
    assert (split / "test.decoder.transcription").read_text() == "<s> BRAVO </s> (b)\n"


def test_editing_generated_split_transfers_authority_to_files(tmp_path: Path) -> None:
    source, audio, split = _corpus(tmp_path)
    train_test_split(source, split, test_count=1)
    assert not split_is_external(split)

    train_id = (split / "train.fileids").read_text().splitlines()[0]
    (split / "train.fileids").write_text(f"{train_id}\n", encoding="utf-8")

    assert split_is_external(split)
    with pytest.raises(ValueError, match="order/membership"):
        validate_external_split(source, split, audio)


def test_moved_fileid_without_matching_transcription_fails_loudly(tmp_path: Path) -> None:
    source, audio, split = _corpus(tmp_path)
    _external_files(split)
    (split / "train.fileids").write_text("c\n", encoding="utf-8")
    (split / "test.fileids").write_text("b\na\n", encoding="utf-8")

    with pytest.raises(ValueError, match="order/membership"):
        validate_external_split(source, split, audio)


def test_external_split_rejects_missing_audio_without_dropping_id(tmp_path: Path) -> None:
    source, audio, split = _corpus(tmp_path)
    _external_files(split)
    (audio / "a.wav").unlink()

    with pytest.raises(ValueError, match="missing audio: a"):
        validate_external_split(source, split, audio)
    assert (split / "train.fileids").read_text() == "c\na\n"


def test_partial_split_remains_recoverable_by_automatic_generation(tmp_path: Path) -> None:
    _, _, split = _corpus(tmp_path)
    (split / "train.fileids").write_text("a\n")

    assert not split_is_external(split)


@pytest.mark.parametrize("second_text", ["WORD", "DIFFERENT"])
@pytest.mark.parametrize("existing_split", [False, True])
def test_generated_split_rejects_duplicate_ids_before_writing(
    tmp_path: Path, second_text: str, existing_split: bool
) -> None:
    source, _, split = _corpus(tmp_path)
    if existing_split:
        train_test_split(source, split, test_count=1)
    before = {path.name: path.read_bytes() for path in split.iterdir()}
    source.write_text(f"same WORD\nsame {second_text}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate fileid.*line 2: same"):
        train_test_split(source, split, test_count=1)

    assert {path.name: path.read_bytes() for path in split.iterdir()} == before


def test_external_partition_membership_uses_linear_hash_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Count actual key hashing rather than depending on machine timing."""
    from pstrain.lib.corpus import split as split_module

    count = 256
    source = tmp_path / "all.transcription"
    source.write_text("".join(f"u{i} WORD\n" for i in range(count)), encoding="utf-8")
    output = tmp_path / "split"
    train_test_split(source, output, test_count=1)
    hash_calls = 0

    class CountedID(str):
        def __hash__(self) -> int:
            nonlocal hash_calls
            hash_calls += 1
            return super().__hash__()

    read_fileids = split_module._read_fileids

    def counted_fileids(path: Path) -> list[str]:
        return [CountedID(value) for value in read_fileids(path)]

    monkeypatch.setattr(split_module, "_read_fileids", counted_fileids)
    # Audio validation happens after partition membership; missing audio avoids
    # constructing a filesystem corpus merely to measure ID membership work.
    with pytest.raises(ValueError, match="missing audio"):
        validate_external_split(source, output, tmp_path / "absent-audio")

    # Allow several linear membership/set passes. Rebuilding the full ID set
    # per source key instead hashes count**2 keys and exceeds this bound.
    max_hash_passes = 10
    assert hash_calls <= max_hash_passes * count
