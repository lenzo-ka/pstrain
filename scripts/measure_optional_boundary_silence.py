#!/usr/bin/env python3
"""Measure optional-boundary BW SIL occupancy on one corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
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


ARMS = ("off", "final-only", "initial-only", "both")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bw_pass(
    model: Path,
    dictionary: Path,
    filler: Path,
    features: Path,
    refs: dict[str, str],
    arm: str,
    multipron: bool,
    counts_path: Path,
) -> tuple[float, list[str], dict[str, float]]:
    config = BWConfig(
        pass2var=True,
        unobserved_gaussian_policy="zero",
        a_beam=1e-90,
        b_beam=1e-10,
        multipron=multipron,
        optional_boundary_silence=arm != "off",
        optional_boundary_measurement_arm=arm,
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
    multipron: bool,
    output: Path,
) -> dict[str, object]:
    label = "multipron" if multipron else "single_pron"
    measurements = {}
    for arm in ARMS:
        occupancy, failed, per_utt = bw_pass(
            model,
            dictionary,
            filler,
            features,
            refs,
            arm,
            multipron,
            output.with_suffix(f".{label}.{arm}"),
        )
        measurements[arm] = {
            "bw_sil_occupancy": occupancy,
            "bw_failures": failed,
            "per_utterance": per_utt,
        }
    off = measurements["off"]
    rows = []
    for fileid in refs:
        row = {"mode": label, "fileid": fileid}
        for arm in ARMS:
            arm_data = measurements[arm]
            row[f"{arm}_failed"] = fileid in arm_data["bw_failures"]
            row[f"{arm}_sil_occupancy"] = arm_data["per_utterance"][fileid]
        rows.append(row)
    arms = {}
    off_failures = set(off["bw_failures"])
    for arm in ARMS:
        arm_data = measurements[arm]
        failures = set(arm_data["bw_failures"])
        delta = arm_data["bw_sil_occupancy"] - off["bw_sil_occupancy"]
        arms[arm] = {
            "arctic_a0587_recovers": "arctic_a0587" in off_failures
            and "arctic_a0587" not in failures,
            "bw_failure_count": len(failures),
            "bw_failures": sorted(failures),
            "recovered_failures_vs_off": sorted(off_failures - failures),
            "new_failures_vs_off": sorted(failures - off_failures),
            "bw_sil_occupancy": arm_data["bw_sil_occupancy"],
            "bw_sil_occupancy_delta_from_off": delta,
            "bw_sil_occupancy_delta_percent_of_off": (
                100.0 * delta / off["bw_sil_occupancy"] if off["bw_sil_occupancy"] else None
            ),
            "delta_denominator": "off.bw_sil_occupancy",
        }
    return {
        "multipron": multipron,
        "arms": arms,
        "per_utterance": rows,
    }


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
    features = project / args.features
    dictionary = project / "shared/dictionary.dict"
    filler = project / "shared/filler.dict"
    input_paths = {
        "transcription": args.transcription.resolve(),
        "model_mdef": (model / "mdef").resolve(),
        "model_means": (model / "means").resolve(),
        "model_variances": (model / "variances").resolve(),
        "model_mixture_weights": (model / "mixture_weights").resolve(),
        "model_transition_matrices": (model / "transition_matrices").resolve(),
        "dictionary": dictionary.resolve(),
        "filler_dictionary": filler.resolve(),
    }
    payload = {
        "utterances": len(refs),
        "inputs": {
            "project": str(project),
            "model": str(model.resolve()),
            "features": str(features.resolve()),
            "files": {
                label: {"path": str(path), "sha256": sha256(path)}
                for label, path in input_paths.items()
            },
        },
        "measurement_selector": "temporary internal four-arm split; not a shipped configuration surface",
        "single_pron": summarize_mode(
            model,
            dictionary,
            filler,
            features,
            refs,
            False,
            args.output,
        ),
        "multipron": summarize_mode(
            model,
            dictionary,
            filler,
            features,
            refs,
            True,
            args.output,
        ),
    }
    rows = payload["single_pron"]["per_utterance"] + payload["multipron"]["per_utterance"]
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload["per_utterance_csv"] = {"path": str(csv_path.resolve()), "sha256": sha256(csv_path)}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
