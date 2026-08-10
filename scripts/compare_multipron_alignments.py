#!/usr/bin/env python
"""Compare which pronunciation variants PocketSphinx alignment picks
when running against two acoustic models trained on the same data:
one with `multipron_training=True` (the default) and one with it
turned off.

For each test utterance we run forced alignment with both models via
the raw PocketSphinx `Decoder` (single-pass word-level via `seg()`).
Each segment's `.word` carries the variant suffix where applicable
(e.g. `reading(2)`). A difference at the same word index means the
two models disagree about which variant won the Viterbi search.

Usage:
  python scripts/compare_multipron_alignments.py MULTIPRON_MODEL_DIR LINEAR_MODEL_DIR

How to produce the two model dirs (cd-1g example):

  # 1. Build with multipron=True (the default)
  pstrain build cd-1g -j 4
  cp -r shared/models/cd-1g/default /tmp/cd-1g.multipron

  # 2. Set multipron_training=false in etc/configs.yaml, wipe affected
  #    model dirs, rebuild
  rm -rf shared/models/ci-1g shared/models/cd-*
  pstrain build cd-1g -j 4
  cp -r shared/models/cd-1g/default /tmp/cd-1g.linear

  # 3. Restore configs.yaml and run the comparison
  python scripts/compare_multipron_alignments.py /tmp/cd-1g.multipron /tmp/cd-1g.linear
"""

from __future__ import annotations

import argparse
import sys
import wave
from collections import Counter
from pathlib import Path

from pocketsphinx import Decoder

PROJECT_DIR = Path(__file__).resolve().parent.parent
DICT_PATH = PROJECT_DIR / "shared" / "dictionary.dict"
FILLER_DICT = PROJECT_DIR / "shared" / "filler.dict"
AUDIO_DIR = PROJECT_DIR / "audio"
TEST_TRANSCRIPTION = PROJECT_DIR / "experiments" / "default" / "etc" / "test.transcription"

# Wide beams; ci-1g has only 1 Gaussian/state so the default beams prune
# every path before reaching the final FSG state. cd-1g and above need
# less generous beams but these still work.
BEAMS = {"beam": 1e-100, "wbeam": 1e-80, "pbeam": 1e-90}


def load_test_transcripts(path: Path) -> list[tuple[str, str]]:
    """Return (utt_id, text) for each line of a Sphinx transcription."""
    out: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # `<s> w1 w2 ... </s> (utt_id)`
        if "(" not in line or not line.endswith(")"):
            continue
        utt_id = line[line.rfind("(") + 1 : -1].strip()
        text = line[: line.rfind("(")].strip()
        text = text.removeprefix("<s>").removesuffix("</s>").strip()
        out.append((utt_id, text))
    return out


def _read_audio(audio_path: Path) -> bytes:
    with wave.open(str(audio_path), "rb") as wf:
        return wf.readframes(wf.getnframes())


def align(model_dir: Path, utt_id: str, transcript: str) -> list[str]:
    """Return the list of content-word tokens (with variant suffix
    where applicable) from PocketSphinx forced alignment of
    `utt_id` against `model_dir`."""
    decoder = Decoder(
        hmm=str(model_dir),
        dict=str(DICT_PATH),
        fdict=str(FILLER_DICT) if FILLER_DICT.exists() else "",
        **BEAMS,
    )
    decoder.set_align_text(transcript)
    data = _read_audio(AUDIO_DIR / f"{utt_id}.wav")
    decoder.start_utt()
    decoder.process_raw(data, no_search=False, full_utt=True)
    decoder.end_utt()
    # Drop silence/filler tokens (anything starting with `<` is the
    # decoder's silence/sentence markers).
    return [s.word for s in decoder.seg() if not s.word.startswith("<")]


def base_word(token: str) -> str:
    """Strip the (N) variant suffix to get the base word."""
    if token.endswith(")") and "(" in token:
        i = token.rfind("(")
        suffix = token[i + 1 : -1]
        if suffix.isdigit():
            return token[:i]
    return token


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "multipron_model_dir",
        type=Path,
        help="Directory of the model trained with multipron_training=True",
    )
    parser.add_argument(
        "linear_model_dir",
        type=Path,
        help="Directory of the model trained with multipron_training=False",
    )
    args = parser.parse_args()

    multipron_dir = args.multipron_model_dir
    linear_dir = args.linear_model_dir

    for label, p in [("multipron", multipron_dir), ("linear", linear_dir)]:
        if not (p / "mdef").exists():
            print(f"Missing mdef in {label} model dir {p}", file=sys.stderr)
            return 1

    pairs = load_test_transcripts(TEST_TRANSCRIPTION)
    if not pairs:
        print(f"No test transcripts at {TEST_TRANSCRIPTION}", file=sys.stderr)
        return 1

    total_words = 0
    same_token = 0
    different_token = 0
    base_mismatch = 0  # should be zero if alignment is sound
    per_utt_diff = []
    examples: list[tuple[str, int, str, str]] = []
    variant_counts_multipron: Counter[str] = Counter()
    variant_counts_linear: Counter[str] = Counter()

    failed = []
    for i, (utt_id, transcript) in enumerate(pairs):
        try:
            multi = align(multipron_dir, utt_id, transcript)
            lin = align(linear_dir, utt_id, transcript)
        except Exception as exc:
            failed.append((utt_id, str(exc)))
            continue

        if len(multi) != len(lin):
            failed.append((utt_id, f"length mismatch multipron={len(multi)} linear={len(lin)}"))
            continue

        utt_diffs = 0
        for j, (a, b) in enumerate(zip(multi, lin, strict=False)):
            total_words += 1
            if a == b:
                same_token += 1
            else:
                different_token += 1
                utt_diffs += 1
                if base_word(a) != base_word(b):
                    base_mismatch += 1
                if len(examples) < 20:
                    examples.append((utt_id, j, a, b))
            # Track variant-suffix usage (only when (N) present).
            if a != base_word(a):
                variant_counts_multipron[a] += 1
            if b != base_word(b):
                variant_counts_linear[b] += 1
        if utt_diffs:
            per_utt_diff.append((utt_id, utt_diffs, len(multi)))

        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(pairs)} utterances processed", file=sys.stderr)

    print()
    print("=" * 60)
    print("Variant-selection comparison: multipron vs linear ci-1g")
    print("=" * 60)
    print(f"Utterances processed: {len(pairs) - len(failed)} / {len(pairs)}")
    if failed:
        print(f"Failed:               {len(failed)}")
        for utt_id, err in failed[:5]:
            print(f"  {utt_id}: {err}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    print()
    print(f"Word tokens compared: {total_words}")
    if total_words:
        print(f"  same variant:       {same_token}  " f"({same_token / total_words * 100:.1f}%)")
        print(
            f"  different variant:  {different_token}  "
            f"({different_token / total_words * 100:.1f}%)"
        )
        print(f"  base-word mismatch: {base_mismatch}  (should be 0)")
    print()
    print(f"Utterances with at least one disagreement: {len(per_utt_diff)}")
    if per_utt_diff:
        per_utt_diff.sort(key=lambda x: -x[1])
        print("  top by disagreement count:")
        for utt_id, n, total in per_utt_diff[:5]:
            print(f"    {utt_id}: {n}/{total} words")
    print()
    print("Sample disagreements (multipron | linear):")
    for utt_id, idx, a, b in examples[:15]:
        print(f"  {utt_id}[{idx}]:  {a:30s} | {b}")
    print()
    print("Variant tokens chosen (multipron mode):")
    for tok, c in variant_counts_multipron.most_common(10):
        print(f"  {tok:30s} {c}")
    print()
    print("Variant tokens chosen (linear mode):")
    for tok, c in variant_counts_linear.most_common(10):
        print(f"  {tok:30s} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
