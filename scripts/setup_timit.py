#!/usr/bin/env python3
"""Prepare a reproducible TIMIT forced-alignment benchmark directory.

The canonical TRAIN/TEST directories are used, excluding the duplicated SA
sentences as required by TIMIT's ``doc/testset.txt``.  Existing complete split
files are authoritative and are validated, never regenerated.  The reduced
lexicon is content-addressed and is regenerated when its source or corpus
transcriptions change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SPLIT_NAMES = (
    "train.fileids",
    "test.fileids",
    "train.transcription",
    "test.transcription",
)
CORE_TEST_SPEAKERS = {
    "dab0", "wbt0", "elc0", "tas1", "wew0", "pas0", "jmp0", "lnt0", "pkt0",
    "lll0", "tls0", "jlm0", "bpm0", "klt0", "nlp0", "cmj0", "jdh0", "mgd0",
    "grt0", "njm0", "dhc0", "jln0", "pam0", "mld0",
}
TOKEN_EDGES = re.compile(r"^[^a-z0-9]+|[^a-z0-9'-]+$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="TIMIT root containing train/, test/, doc/")
    parser.add_argument("output", type=Path, help="Persistent benchmark data directory")
    args = parser.parse_args()
    corpus, output = args.corpus.resolve(), args.output.resolve()
    records = discover(corpus)
    output.mkdir(parents=True, exist_ok=True)
    split_state = prepare_split(records, output)
    lexicon_state = prepare_lexicon(corpus / "doc" / "timitdic.tbl", records, output)
    prepare_references(records, output)
    summary = {
        "policy": "TIMIT canonical TRAIN/TEST, SA1/SA2 excluded",
        "split_state": split_state,
        "lexicon_state": lexicon_state,
        "train_utterances": sum(r[0] == "train" for r in records),
        "test_utterances": sum(r[0] == "test" for r in records),
        "train_speakers": len({r[2] for r in records if r[0] == "train"}),
        "test_speakers": len({r[2] for r in records if r[0] == "test"}),
        "core_test_utterances": sum(
            r[0] == "test" and timit_speaker_code(r[2]) in CORE_TEST_SPEAKERS for r in records
        ),
    }
    (output / "preparation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def discover(corpus: Path) -> list[tuple[str, str, str, str, Path]]:
    """Return (partition, fileid, speaker, text, phn_path) records."""
    records = []
    for partition in ("train", "test"):
        root = corpus / partition
        if not root.is_dir():
            raise FileNotFoundError(f"missing TIMIT partition: {root}")
        for dialect in sorted(p for p in root.iterdir() if p.is_dir()):
            for speaker_dir in sorted(p for p in dialect.iterdir() if p.is_dir()):
                for wav in sorted(speaker_dir.glob("*.wav")):
                    if wav.stem.lower().startswith("sa"):
                        continue
                    text_path, phn_path = wav.with_suffix(".txt"), wav.with_suffix(".phn")
                    if not text_path.is_file() or not phn_path.is_file():
                        raise FileNotFoundError(f"missing annotation beside {wav}")
                    text = parse_text(text_path)
                    fileid = wav.relative_to(corpus).with_suffix("").as_posix()
                    records.append((partition, fileid, speaker_dir.name.lower(), text, phn_path))
    if not records:
        raise ValueError(f"no TIMIT utterances found under {corpus}")
    return records


def parse_text(path: Path) -> str:
    parts = path.read_text(encoding="utf-8").strip().split()
    if len(parts) < 3 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ValueError(f"unexpected TIMIT transcript: {path}")
    words = [normalize_word(token) for token in parts[2:]]
    return " ".join(word for word in words if word)


def normalize_word(word: str) -> str:
    """Match TIMIT orthography to dictionary keys without changing interiors."""
    return TOKEN_EDGES.sub("", word.lower()).split("~", 1)[0]


def timit_speaker_code(directory_name: str) -> str:
    """Remove TIMIT's leading M/F directory marker used outside testset.txt."""
    return directory_name[1:] if directory_name[:1] in {"m", "f"} else directory_name


def prepare_split(records: list[tuple[str, str, str, str, Path]], output: Path) -> str:
    present = [(output / name).exists() for name in SPLIT_NAMES]
    if any(present) and not all(present):
        missing = [name for name, exists in zip(SPLIT_NAMES, present, strict=True) if not exists]
        raise ValueError(f"partial persistent split; missing: {', '.join(missing)}")
    expected = {r[1]: r for r in records}
    if all(present):
        train_ids = read_ids(output / "train.fileids")
        test_ids = read_ids(output / "test.fileids")
        train_text = read_transcription(output / "train.transcription")
        test_text = read_transcription(output / "test.transcription")
        if train_ids != list(train_text) or test_ids != list(test_text):
            raise ValueError("persistent fileids and transcriptions differ in order/membership")
        supplied = train_ids + test_ids
        if len(supplied) != len(set(supplied)) or set(supplied) != set(expected):
            raise ValueError("persistent split must exactly partition eligible TIMIT utterances")
        drift = [i for i in supplied if (train_text | test_text)[i] != expected[i][3]]
        if drift:
            raise ValueError(f"persistent transcription text drift: {drift[:5]}")
        train_speakers = {expected[i][2] for i in train_ids}
        test_speakers = {expected[i][2] for i in test_ids}
        overlap = sorted(train_speakers & test_speakers)
        if overlap:
            raise ValueError(f"persistent split is not speaker-disjoint: {overlap[:5]}")
        return "honoured-existing"

    for partition in ("train", "test"):
        entries = [(r[1], r[3]) for r in records if r[0] == partition]
        (output / f"{partition}.fileids").write_text(
            "".join(f"{fileid}\n" for fileid, _ in entries), encoding="utf-8"
        )
        (output / f"{partition}.transcription").write_text(
            "".join(f"{fileid} {text}\n" for fileid, text in entries), encoding="utf-8"
        )
    core_ids = [
        r[1]
        for r in records
        if r[0] == "test" and timit_speaker_code(r[2]) in CORE_TEST_SPEAKERS
    ]
    (output / "core-test.fileids").write_text("".join(f"{i}\n" for i in core_ids))
    return "generated-canonical"


def prepare_lexicon(source: Path, records: list[tuple[str, str, str, str, Path]], output: Path) -> str:
    vocabulary = sorted({word for r in records for word in r[3].split()})
    fingerprint = hashlib.sha256(
        source.read_bytes() + b"\0" + "\n".join(vocabulary).encode()
    ).hexdigest()
    lexicon_path, metadata_path = output / "timit.reduced.dict", output / ".lexicon.json"
    if lexicon_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("input_sha256") == fingerprint:
                verify_coverage(vocabulary, lexicon_path)
                return "reused-current"
        except (ValueError, json.JSONDecodeError):
            pass

    entries: dict[str, list[str]] = {}
    for raw in source.read_text(encoding="utf-8").splitlines():
        # TIMIT documents four transcription-only corrections as commented
        # dictionary-shaped records near the top of timitdic.tbl.
        candidate = raw[2:] if raw.startswith("; ") else raw
        if not candidate or "  /" not in candidate:
            continue
        word, pronunciation = candidate.split("  /", 1)
        pronunciation = pronunciation.rstrip()
        if not pronunciation.endswith("/"):
            continue
        phones = re.sub(r"[0-9]", "", pronunciation.rstrip("/").strip())
        key = normalize_word(word)
        if key:
            entries.setdefault(key, []).append(phones)
    missing = [word for word in vocabulary if word not in entries]
    if missing:
        raise ValueError(f"TIMIT lexicon lacks {len(missing)} corpus words: {missing[:20]}")
    lines = []
    for word in vocabulary:
        for variant, phones in enumerate(entries[word]):
            key = word if variant == 0 else f"{word}({variant + 1})"
            lines.append(f"{key} {phones}\n")
    lexicon_path.write_text("".join(lines), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {"input_sha256": fingerprint, "vocabulary_words": len(vocabulary), "entries": len(lines)},
            indent=2,
        )
        + "\n"
    )
    verify_coverage(vocabulary, lexicon_path)
    return "regenerated"


def verify_coverage(vocabulary: list[str], lexicon: Path) -> None:
    words = {line.split()[0].split("(", 1)[0] for line in lexicon.read_text().splitlines() if line}
    missing = sorted(set(vocabulary) - words)
    if missing:
        raise ValueError(f"reduced lexicon coverage incomplete: {missing[:20]}")


def prepare_references(records: list[tuple[str, str, str, str, Path]], output: Path) -> None:
    lines = ["utterance_id\tstart_sample\tend_sample\tphone\n"]
    for _, fileid, _, _, path in records:
        for raw in path.read_text(encoding="utf-8").splitlines():
            start, end, phone = raw.split()
            lines.append(f"{fileid}\t{start}\t{end}\t{phone}\n")
    (output / "reference-phones.tsv").write_text("".join(lines), encoding="utf-8")


def read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicates in {path}")
    return ids


def read_transcription(path: Path) -> dict[str, str]:
    result = {}
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        fileid, text = raw.split(maxsplit=1)
        if fileid in result:
            raise ValueError(f"duplicate fileid in {path}: {fileid}")
        result[fileid] = text
    return result


if __name__ == "__main__":
    raise SystemExit(main())
