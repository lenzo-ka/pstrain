#!/usr/bin/env python3
"""Measure optional-boundary BW SIL occupancy on one corpus."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.meta_path = [finder for finder in sys.meta_path if "_editable" not in type(finder).__module__]

from pstrain.lib import _pstrainc  # noqa: E402
from pstrain.lib.bw import BWConfig, BWTrainer  # noqa: E402
from pstrain.lib.features import read_sphinx_mfc  # noqa: E402


def transcripts(path: Path) -> dict[str, str]:
    result = {}
    for raw in path.read_text().splitlines():
        if "(" in raw:
            text, fileid = raw.rsplit("(", 1)
            result[fileid.rstrip(") ")] = text.strip()
        else:
            fileid, text = raw.split(maxsplit=1)
            result[fileid] = f"<s> {text} </s>"
    return result


def sil_senones(mdef: Path) -> list[int]:
    for line in mdef.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[:4] == ["SIL", "-", "-", "-"]:
            return [int(value) for value in fields[6:] if value != "N"]
    raise RuntimeError("SIL CI row missing from mdef")


def trailing_silence_seconds(path: Path) -> float:
    """Measure trailing silence in 20 ms RMS windows at -40 dB from peak."""
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2:
            raise RuntimeError(f"expected 16-bit PCM: {path}")
        rate = source.getframerate()
        channels = source.getnchannels()
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    samples = samples.astype(np.float64)
    peak = float(np.max(np.abs(samples), initial=0.0))
    if peak == 0.0:
        return len(samples) / rate
    window = max(1, round(rate * 0.02))
    threshold = peak * 10 ** (-40.0 / 20.0)
    silent_samples = 0
    for end in range(len(samples), 0, -window):
        chunk = samples[max(0, end - window) : end]
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        if rms > threshold:
            break
        silent_samples += len(chunk)
    return silent_samples / rate


def bw_pass(
    model: Path,
    dictionary: Path,
    filler: Path,
    features: Path,
    refs: dict[str, str],
    enabled: bool,
    multipron: bool,
    counts_path: Path,
) -> tuple[float, list[str], dict[str, float]]:
    config = BWConfig(
        pass2var=True,
        unobserved_gaussian_policy="zero",
        a_beam=1e-90,
        b_beam=1e-10,
        multipron=multipron,
        optional_boundary_silence=enabled,
    )
    trainer = BWTrainer(
        model / "mdef",
        model / "means",
        model / "variances",
        model / "mixture_weights",
        model / "transition_matrices",
        config=config,
    )
    trainer.set_dict(dictionary, filler)
    failed = []
    per_utterance = {}
    previous_occupancy = 0.0
    sil_ids = sil_senones(model / "mdef")
    for fileid, text in refs.items():
        mfcc = read_sphinx_mfc(features / f"{fileid}.mfc")
        if not trainer.process_utterance_mfcc(mfcc, text):
            failed.append(fileid)
        if not trainer.save_density_counts(counts_path):
            raise RuntimeError(f"could not save {counts_path}")
        counts = _pstrainc.read_dnom(str(counts_path))[0]
        occupancy = float(np.asarray(counts)[sil_ids].sum())
        per_utterance[fileid] = occupancy - previous_occupancy
        previous_occupancy = occupancy
    return previous_occupancy, failed, per_utterance


def summarize_mode(
    model: Path,
    dictionary: Path,
    filler: Path,
    features: Path,
    refs: dict[str, str],
    trailing_silence: dict[str, float],
    multipron: bool,
    output: Path,
) -> dict[str, object]:
    label = "multipron" if multipron else "single_pron"
    off, off_failed, off_per_utt = bw_pass(
        model,
        dictionary,
        filler,
        features,
        refs,
        False,
        multipron,
        output.with_suffix(f".{label}.off"),
    )
    on, on_failed, on_per_utt = bw_pass(
        model,
        dictionary,
        filler,
        features,
        refs,
        True,
        multipron,
        output.with_suffix(f".{label}.on"),
    )
    rows = []
    strata = {
        "no_measured_trailing_silence": {"utterances": 0, "sil_occupancy_delta_off_minus_on": 0.0},
        "measured_trailing_silence_present": {
            "utterances": 0,
            "sil_occupancy_delta_off_minus_on": 0.0,
        },
    }
    for fileid in refs:
        seconds = trailing_silence[fileid]
        delta = off_per_utt[fileid] - on_per_utt[fileid]
        stratum = (
            "measured_trailing_silence_present" if seconds > 0.0 else "no_measured_trailing_silence"
        )
        strata[stratum]["utterances"] += 1
        strata[stratum]["sil_occupancy_delta_off_minus_on"] += delta
        rows.append(
            {
                "fileid": fileid,
                "trailing_silence_seconds": seconds,
                "trailing_silence_present": seconds > 0.0,
                "sil_occupancy_off": off_per_utt[fileid],
                "sil_occupancy_on": on_per_utt[fileid],
                "sil_occupancy_delta_off_minus_on": delta,
            }
        )
    delta = off - on
    return {
        "multipron": multipron,
        "bw_sil_occupancy_off": off,
        "bw_sil_occupancy_on": on,
        "bw_sil_occupancy_delta_off_minus_on": delta,
        "bw_sil_occupancy_reduction_percent": 100.0 * delta / off if off else None,
        "bw_sil_occupancy_reduction_percent_denominator": "bw_sil_occupancy_off",
        "bw_failures_off": off_failed,
        "bw_failures_on": on_failed,
        "strata": strata,
        "per_utterance": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("transcription", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="shared/models/cd-1g/ladder4")
    parser.add_argument("--features", default="shared/features/ladder4")
    parser.add_argument("--audio", default="audio")
    args = parser.parse_args()
    project = args.project.resolve()
    refs = transcripts(args.transcription)
    model = project / args.model
    dictionary = project / "shared/dictionary.dict"
    filler = project / "shared/filler.dict"
    trailing_silence = {
        fileid: trailing_silence_seconds(project / args.audio / f"{fileid}.wav") for fileid in refs
    }
    payload = {
        "utterances": len(refs),
        "trailing_silence_measurement": {
            "window_seconds": 0.02,
            "rms_threshold_db_relative_to_peak": -40.0,
            "present_definition": "at least one trailing window at or below threshold",
        },
        "single_pron": summarize_mode(
            model,
            dictionary,
            filler,
            project / args.features,
            refs,
            trailing_silence,
            False,
            args.output,
        ),
        "multipron": summarize_mode(
            model,
            dictionary,
            filler,
            project / args.features,
            refs,
            trailing_silence,
            True,
            args.output,
        ),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
