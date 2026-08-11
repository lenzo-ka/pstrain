"""Train/test split for a corpus transcription.

Reads a Sphinx-format transcription (one line per utterance:
`fileid word word ...`), shuffles deterministically by seed, and writes
`{train,test}.{fileids,transcription}` to an output directory.

Used by both the CLI (`pstrain split`) and the pipeline runner (`split` task).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

# When neither train_ratio nor test_count is set, this fraction of the
# corpus goes to the training set. Matches the prior `pstrain split` CLI
# default.
DEFAULT_TRAIN_RATIO = 0.95

# Seed for the shuffle when callers don't override it. Picked once for
# reproducibility; the value itself is arbitrary.
DEFAULT_SEED = 42


@dataclass(frozen=True)
class SplitResult:
    """Files written by a train/test split."""

    train_fileids: Path
    test_fileids: Path
    train_transcription: Path
    test_transcription: Path
    test_decoder_transcription: Path
    n_train: int
    n_test: int


def train_test_split(
    transcription_file: Path,
    output_dir: Path,
    *,
    train_ratio: float | None = None,
    test_count: int | None = None,
    seed: int = DEFAULT_SEED,
) -> SplitResult:
    """Split a Sphinx-format transcription into train and test partitions.

    Exactly one of `train_ratio` or `test_count` may be set; if both are
    None, the default is `DEFAULT_TRAIN_RATIO` (95% train).

    Args:
        transcription_file: Path to the input transcription
            (e.g. `etc/all.transcription`).
        output_dir: Directory to write the four output files into; created
            if it doesn't exist.
        train_ratio: Fraction of utterances to put in the training set
            (e.g. 0.95). Mutually exclusive with `test_count`.
        test_count: Exact number of utterances to put in the test set.
            Mutually exclusive with `train_ratio`.
        seed: Random seed for the shuffle (default 42).

    Returns:
        A `SplitResult` with the four written paths and the train/test
        utterance counts.

    Raises:
        FileNotFoundError: If `transcription_file` doesn't exist.
        ValueError: If both `train_ratio` and `test_count` are set, or if
            the transcription is empty.
    """
    if train_ratio is not None and test_count is not None:
        raise ValueError("train_ratio and test_count are mutually exclusive")

    transcription_file = Path(transcription_file)
    if not transcription_file.exists():
        raise FileNotFoundError(f"transcription file not found: {transcription_file}")

    entries: list[tuple[str, str]] = []
    for raw in transcription_file.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) >= 2:
            entries.append((parts[0], parts[1]))
        else:
            entries.append((parts[0], ""))

    if not entries:
        raise ValueError(f"no entries in {transcription_file}")

    rng = random.Random(seed)
    shuffled = entries.copy()
    rng.shuffle(shuffled)

    if test_count is not None:
        split_idx = max(0, len(shuffled) - min(test_count, len(shuffled)))
    elif train_ratio is not None:
        split_idx = int(len(shuffled) * train_ratio)
    else:
        split_idx = int(len(shuffled) * DEFAULT_TRAIN_RATIO)

    train = sorted(shuffled[:split_idx], key=lambda e: e[0])
    test = sorted(shuffled[split_idx:], key=lambda e: e[0])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_fileids = output_dir / "train.fileids"
    test_fileids = output_dir / "test.fileids"
    train_transcription = output_dir / "train.transcription"
    test_transcription = output_dir / "test.transcription"
    test_decoder_transcription = output_dir / "test.decoder.transcription"

    _write_transcription(train_transcription, train)
    _write_transcription(test_transcription, test)
    _write_decoder_transcription(test_decoder_transcription, test)
    _write_fileids(train_fileids, train)
    _write_fileids(test_fileids, test)

    return SplitResult(
        train_fileids=train_fileids,
        test_fileids=test_fileids,
        train_transcription=train_transcription,
        test_transcription=test_transcription,
        test_decoder_transcription=test_decoder_transcription,
        n_train=len(train),
        n_test=len(test),
    )


def _write_transcription(path: Path, entries: list[tuple[str, str]]) -> None:
    with path.open("w") as f:
        for fileid, text in entries:
            f.write(f"{fileid} {text}\n")


def _write_fileids(path: Path, entries: list[tuple[str, str]]) -> None:
    with path.open("w") as f:
        for fileid, _ in entries:
            f.write(f"{fileid}\n")


def _write_decoder_transcription(path: Path, entries: list[tuple[str, str]]) -> None:
    with path.open("w") as f:
        for fileid, text in entries:
            f.write(f"<s> {text} </s> ({fileid})\n")
