"""Bounded diagnostics for PocketSphinx cross-utterance decoder state."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import multiprocessing
import os
import random
from pathlib import Path

from pstrain.benchmarks.arctic import load_transcripts
from pstrain.lib.testing.decoder import Decoder

PIN_ROOT = Path(
    os.environ.get(
        "PSTRAIN_DECODE_PROBE_ROOT",
        "/Volumes/experiments/pstrain-parity/pin-run/pstrain-6/.pstrain-benchmark/arctic",
    )
)
DATA = Path(__file__).parents[1] / "benchmarks" / "arctic" / "data"


def make_decoder(mode: str) -> Decoder:
    return Decoder(
        PIN_ROOT / mode / "shared/models/cd-8g" / mode,
        DATA / "cmu_arctic_slt.dict",
        PIN_ROOT / mode / "shared/filler.dict",
        DATA / "training-unigram.lm",
        beam=1e-80,
        wbeam=1e-40,
        pl_window=5,
        lw=10.0,
        wip=0.2,
        pbeam=1e-80,
        lpbeam=1e-80,
        lponlybeam=1e-80,
        fwdflatbeam=1e-80,
        fwdflatwbeam=1e-40,
    )


def audio_path(utterance: str) -> Path:
    if "/" in utterance:
        voice, local_id = utterance.split("/", 1)
    else:
        voice, local_id = "slt", utterance
    return PIN_ROOT / "corpora" / f"cmu_us_{voice}_arctic/wav" / f"{local_id}.wav"


def decode_order(mode: str, utterances: list[str], reset_each: bool = False) -> dict[str, str]:
    decoder = make_decoder(mode)
    hypotheses = {}
    for utterance in utterances:
        if reset_each:
            decoder._decoder.start_stream()
        result = decoder.decode_file(audio_path(utterance))
        if not result.success:
            raise RuntimeError(f"{utterance}: {result.error}")
        hypotheses[utterance] = result.hypothesis
    return hypotheses


def quantify(mode: str, corpus: str, sample_per_voice: int | None) -> dict[str, object]:
    refs = load_transcripts(DATA / f"{corpus}.transcription")
    canonical = list(refs)
    if sample_per_voice is not None:
        by_voice: dict[str, list[str]] = {}
        for utterance in canonical:
            by_voice.setdefault(utterance.split("/", 1)[0], []).append(utterance)
        rng = random.Random(7)
        selected = {
            utterance
            for voice in sorted(by_voice)
            for utterance in rng.sample(by_voice[voice], min(sample_per_voice, len(by_voice[voice])))
        }
        canonical = [utterance for utterance in canonical if utterance in selected]
    reverse = list(reversed(canonical))
    baseline = decode_order(mode, canonical)
    reversed_hyp = decode_order(mode, reverse)
    reset_hyp = decode_order(mode, canonical, reset_each=True)

    def differences(other: dict[str, str]) -> list[dict[str, str]]:
        return [
            {"utterance": utterance, "canonical": baseline[utterance], "other": other[utterance]}
            for utterance in canonical
            if baseline[utterance] != other[utterance]
        ]

    reverse_differences = differences(reversed_hyp)
    reset_differences = differences(reset_hyp)
    decoder = make_decoder(mode)._decoder
    return {
        "mode": mode,
        "corpus": corpus,
        "pin_root": str(PIN_ROOT.resolve()),
        "sample_per_voice": sample_per_voice,
        "seed": 7 if sample_per_voice is not None else None,
        "utterances": len(canonical),
        "decoder_config": {
            key: decoder.config[key]
            for key in ("remove_noise", "remove_dc", "dither", "cmn", "cmninit", "varnorm", "agc")
        },
        "canonical_vs_reverse_count": len(reverse_differences),
        "canonical_vs_reverse": reverse_differences,
        "canonical_vs_start_stream_each_count": len(reset_differences),
        "canonical_vs_start_stream_each": reset_differences,
    }


def import_provenance(_: None = None) -> dict[str, str]:
    import pocketsphinx
    import pstrain
    import pstrain.benchmarks.arctic

    return {
        "pstrain": str(Path(pstrain.__file__).resolve()),
        "arctic": str(Path(pstrain.benchmarks.arctic.__file__).resolve()),
        "pocketsphinx": str(Path(pocketsphinx.__file__).resolve()),
        "pocketsphinx_version": importlib.metadata.version("pocketsphinx"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance", action="store_true")
    parser.add_argument("--quantify", action="store_true")
    parser.add_argument("--mode", choices=("off", "on"), default="off")
    parser.add_argument("--corpus", choices=("slt55", "big"), default="slt55")
    parser.add_argument("--sample-per-voice", type=int)
    args = parser.parse_args()
    if args.provenance:
        context = multiprocessing.get_context("spawn")
        with context.Pool(1) as pool:
            worker = pool.map(import_provenance, [None])[0]
        print(json.dumps({"parent": import_provenance(), "spawned_worker": worker}, indent=2))
        return 0
    if args.quantify:
        print(json.dumps(quantify(args.mode, args.corpus, args.sample_per_voice), indent=2))
        return 0
    parser.error("a probe mode is required")


if __name__ == "__main__":
    raise SystemExit(main())
