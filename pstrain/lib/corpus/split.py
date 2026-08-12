"""Train/test split for a corpus transcription.

Reads a Sphinx-format transcription (one line per utterance:
`fileid word word ...`), shuffles deterministically by seed, and writes
`{train,test}.{fileids,transcription}` to an output directory.

Used by both the CLI (`pstrain split`) and the pipeline runner (`split` task).
"""

from __future__ import annotations

import hashlib
import json
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

SPLIT_FILENAMES = (
    "train.fileids",
    "test.fileids",
    "train.transcription",
    "test.transcription",
)
GENERATED_SPLIT_METADATA = ".split.generated.json"
VALIDATED_SPLIT = ".split.validated.json"


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
    _write_generated_metadata(output_dir)
    _write_validation(output_dir, "generated")

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


def split_is_external(output_dir: Path) -> bool:
    """Return whether a complete split is user-owned rather than generated.

    Generated files remain generated only while all four content hashes match
    the sidecar written by :func:`train_test_split`. Editing any file transfers
    authority to the persistent files on the next pipeline construction.
    """
    paths = [Path(output_dir) / name for name in SPLIT_FILENAMES]
    present = [path.exists() for path in paths]
    if not all(present):
        return False

    metadata_path = Path(output_dir) / GENERATED_SPLIT_METADATA
    if not metadata_path.exists():
        return True
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = metadata["sha256"]
        return any(expected[path.name] != _sha256(path) for path in paths)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return True


def validate_external_split(
    transcription_file: Path, output_dir: Path, audio_dir: Path
) -> SplitResult:
    """Validate and preserve an externally supplied Sphinx train/test split."""
    output_dir = Path(output_dir)
    source = _read_transcription(Path(transcription_file), "all.transcription")
    train_ids = _read_fileids(output_dir / "train.fileids")
    test_ids = _read_fileids(output_dir / "test.fileids")
    train = _read_transcription(output_dir / "train.transcription", "train.transcription")
    test = _read_transcription(output_dir / "test.transcription", "test.transcription")

    _require_same_order("train", train_ids, list(train))
    _require_same_order("test", test_ids, list(test))
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise ValueError(f"external split has train/test overlap: {_summarize(overlap)}")
    supplied = train_ids + test_ids
    unknown = [fileid for fileid in supplied if fileid not in source]
    missing = [fileid for fileid in source if fileid not in set(supplied)]
    if unknown or missing:
        details = []
        if unknown:
            details.append(f"unknown: {_summarize(unknown)}")
        if missing:
            details.append(f"missing: {_summarize(missing)}")
        raise ValueError(
            "external split does not exactly partition all.transcription; " + "; ".join(details)
        )
    for label, entries in (("train", train), ("test", test)):
        drift = [fileid for fileid, text in entries.items() if source[fileid] != text]
        if drift:
            raise ValueError(
                f"external {label}.transcription differs from all.transcription: {_summarize(drift)}"
            )
    missing_audio = [
        fileid for fileid in supplied if not (Path(audio_dir) / f"{fileid}.wav").is_file()
    ]
    if missing_audio:
        raise ValueError(f"external split references missing audio: {_summarize(missing_audio)}")

    test_entries = list(test.items())
    decoder = output_dir / "test.decoder.transcription"
    _write_decoder_transcription(decoder, test_entries)
    _write_validation(output_dir, "external")
    return SplitResult(
        train_fileids=output_dir / "train.fileids",
        test_fileids=output_dir / "test.fileids",
        train_transcription=output_dir / "train.transcription",
        test_transcription=output_dir / "test.transcription",
        test_decoder_transcription=decoder,
        n_train=len(train_ids),
        n_test=len(test_ids),
    )


def _read_fileids(path: Path) -> list[str]:
    values = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    _reject_duplicates(path.name, values)
    return values


def _read_transcription(path: Path, label: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.strip().split(None, 1)
        fileid, text = parts[0], parts[1] if len(parts) == 2 else ""
        if fileid in entries:
            raise ValueError(f"duplicate fileid in {label} at line {line_number}: {fileid}")
        entries[fileid] = text
    if not entries:
        raise ValueError(f"no entries in {path}")
    return entries


def _reject_duplicates(label: str, values: list[str]) -> None:
    seen: set[str] = set()
    duplicates = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"duplicate fileid in {label}: {_summarize(duplicates)}")


def _require_same_order(label: str, fileids: list[str], transcript_ids: list[str]) -> None:
    if fileids != transcript_ids:
        raise ValueError(
            f"external {label}.fileids order/membership does not exactly match "
            f"{label}.transcription"
        )


def _summarize(values: list[str], limit: int = 10) -> str:
    suffix = f" ... ({len(values)} total)" if len(values) > limit else ""
    return ", ".join(values[:limit]) + suffix


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_generated_metadata(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    (output_dir / GENERATED_SPLIT_METADATA).write_text(
        json.dumps(
            {
                "mode": "generated",
                "sha256": {name: _sha256(output_dir / name) for name in SPLIT_FILENAMES},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_validation(output_dir: Path, mode: str) -> None:
    output_dir = Path(output_dir)
    (output_dir / VALIDATED_SPLIT).write_text(
        json.dumps(
            {
                "mode": mode,
                "sha256": {name: _sha256(output_dir / name) for name in SPLIT_FILENAMES},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
