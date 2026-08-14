#!/usr/bin/env python3
"""Measure optional-boundary alignment and BW SIL occupancy on one corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.meta_path = [finder for finder in sys.meta_path if "_editable" not in type(finder).__module__]

from pstrain.lib import _pstrainc
from pstrain.lib.alignment import Aligner
from pstrain.lib.bw import BWConfig, BWTrainer
from pstrain.lib.features import read_sphinx_mfc


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


def phone_signature(result: object) -> tuple[tuple[str, int, int], ...]:
    return tuple((seg.name, seg.start_frame, seg.end_frame) for seg in result.phones)


def align_pass(
    model: Path,
    dictionary: Path,
    filler: Path,
    audio: Path,
    refs: dict[str, str],
    enabled: bool,
) -> tuple[dict[str, tuple[tuple[str, int, int], ...]], dict[str, str], int]:
    aligned = {}
    failed = {}
    sil_frames = 0
    with Aligner(
        model,
        dictionary,
        filler_dict=filler,
        optional_boundary_silence=enabled,
        include_phones=True,
    ) as aligner:
        for fileid, text in refs.items():
            try:
                result = aligner.align_audio(audio / f"{fileid}.wav", text, fileid)
            except Exception as error:  # measurement records every failure verbatim
                failed[fileid] = str(error)
                continue
            aligned[fileid] = phone_signature(result)
            sil_frames += sum(
                seg.end_frame - seg.start_frame + 1 for seg in result.phones if seg.name == "SIL"
            )
    return aligned, failed, sil_frames


def sil_senones(mdef: Path) -> list[int]:
    for line in mdef.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[:4] == ["SIL", "-", "-", "-"]:
            return [int(value) for value in fields[6:] if value != "N"]
    raise RuntimeError("SIL CI row missing from mdef")


def bw_pass(
    model: Path,
    dictionary: Path,
    filler: Path,
    features: Path,
    refs: dict[str, str],
    enabled: bool,
    counts_path: Path,
) -> tuple[float, list[str]]:
    config = BWConfig(
        pass2var=True,
        unobserved_gaussian_policy="zero",
        a_beam=1e-90,
        b_beam=1e-10,
        multipron=False,
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
    for fileid, text in refs.items():
        mfcc = read_sphinx_mfc(features / f"{fileid}.mfc")
        if not trainer.process_utterance_mfcc(mfcc, text):
            failed.append(fileid)
    if not trainer.save_density_counts(counts_path):
        raise RuntimeError(f"could not save {counts_path}")
    counts = _pstrainc.read_dnom(str(counts_path))[0]
    occupancy = float(np.asarray(counts)[sil_senones(model / "mdef")].sum())
    return occupancy, failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("transcription", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="shared/models/cd-1g/ladder4")
    parser.add_argument("--features", default="shared/features/ladder4")
    args = parser.parse_args()
    project = args.project.resolve()
    refs = transcripts(args.transcription)
    model = project / args.model
    dictionary = project / "shared/dictionary.dict"
    filler = project / "shared/filler.dict"

    off_align, off_failed, off_sil = align_pass(
        model, dictionary, filler, project / "audio", refs, False
    )
    on_align, on_failed, on_sil = align_pass(
        model, dictionary, filler, project / "audio", refs, True
    )
    common = off_align.keys() & on_align.keys()
    off_bw, off_bw_failed = bw_pass(
        model,
        dictionary,
        filler,
        project / args.features,
        refs,
        False,
        args.output.with_suffix(".off"),
    )
    on_bw, on_bw_failed = bw_pass(
        model,
        dictionary,
        filler,
        project / args.features,
        refs,
        True,
        args.output.with_suffix(".on"),
    )
    payload = {
        "utterances": len(refs),
        "alignments_changed": sum(off_align[key] != on_align[key] for key in common),
        "align_off_failures": off_failed,
        "align_on_failures": on_failed,
        "align_regressions": sorted(on_failed.keys() - off_failed.keys()),
        "align_recoveries": sorted(off_failed.keys() - on_failed.keys()),
        "align_sil_frames_off": off_sil,
        "align_sil_frames_on": on_sil,
        "bw_sil_occupancy_off": off_bw,
        "bw_sil_occupancy_on": on_bw,
        "bw_failures_off": off_bw_failed,
        "bw_failures_on": on_bw_failed,
        "arctic_a0587_bw_succeeds_off": "arctic_a0587" not in off_bw_failed,
        "arctic_a0587_bw_succeeds_on": "arctic_a0587" not in on_bw_failed,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
