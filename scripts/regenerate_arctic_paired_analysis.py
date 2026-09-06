#!/usr/bin/env python3
"""Regenerate the checked-in paired analysis for the live Arctic pin cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pstrain.benchmarks.arctic import paired_delta_ci  # noqa: E402

DEFAULT_RECORD = ROOT / "evidence/arctic-pin/record.json"
DEFAULT_ORACLE = ROOT / "evidence/arctic-pin/oracle-sidecar.json"
DEFAULT_OUTPUT = ROOT / "evidence/arctic-pin/paired-analysis.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pairs(rows: list[list[Any]]) -> dict[str, dict[str, int]]:
    return {
        str(utterance): {"ref_words": int(ref_words), "errors": int(errors)}
        for utterance, ref_words, errors in rows
    }


def _resource_axis(record: dict[str, Any], oracle: dict[str, Any]) -> dict[str, str]:
    """Refuse to compare two arms that did not consume the same decode resources.

    A paired comparison across two lexicons or two language models measures the
    resource change as much as the model, so the arms are authenticated here
    rather than asserted in prose.
    """
    axis = {}
    for field in ("dictionary_sha256", "lm_sha256"):
        recorded = record["resources"][field]
        decoded = oracle["resources"][field]
        if recorded != decoded:
            raise SystemExit(
                f"paired analysis arms disagree on {field}: record={recorded}, oracle={decoded}"
            )
        axis[field] = recorded
    stated = oracle["resources"]["record_resources_match"]
    if stated != {"dictionary": True, "lm": True}:
        raise SystemExit(
            f"oracle sidecar does not claim a resource match with the record: {stated}"
        )
    return axis


def _engines(record: dict[str, Any], oracle: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Report each arm's engine so a residual engine difference stays visible."""
    fields = ("version", "git_describe", "native_library_sha256", "pocketsphinx_version")
    return {
        "oracle": {field: oracle["engine"][field] for field in fields},
        "pstrain": {field: record["engine"][field] for field in fields},
    }


def regenerate(record_path: Path, oracle_path: Path) -> str:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    resource_axis = _resource_axis(record, oracle)
    bootstrap = record["conditions"]["bootstrap"]
    comparisons = []
    for dataset in ("slt55", "big"):
        current = record["results"]["on"][dataset]
        upstream = oracle["results"]["on"][dataset]
        current_rows = _pairs(current["utterance_rows"])
        upstream_rows = _pairs(upstream["utterance_rows"])
        comparisons.append(
            {
                "dataset": dataset,
                "mode": "on",
                "oracle_errors": upstream["errors"],
                "oracle_wer": upstream["wer"],
                "paired_ci_95_pp": paired_delta_ci(
                    upstream["utterance_rows"],
                    current_rows,
                    speaker_stratified=dataset == "big",
                    resamples=int(bootstrap["resamples"]),
                    seed=int(bootstrap["seed"]),
                ),
                "per_utterance_error_rows_differ": sum(
                    current_rows[key]["errors"] != upstream_rows[key]["errors"]
                    for key in current_rows
                ),
                "pstrain_errors": current["errors"],
                "pstrain_minus_oracle_pp": 100.0
                * (current["errors"] - upstream["errors"])
                / current["ref_words"],
                "pstrain_wer": current["wer"],
                "ref_words": current["ref_words"],
            }
        )
    analysis = {
        "conditions": {
            "big_speaker_stratified": bool(bootstrap["big_speaker_stratified"]),
            "bootstrap_resamples": int(bootstrap["resamples"]),
            "bootstrap_seed": int(bootstrap["seed"]),
            "engines": _engines(record, oracle),
            "paired_delta_path": "pstrain.benchmarks.arctic.paired_delta_ci",
            "resource_axis": resource_axis,
        },
        "generated_from": {
            "oracle_sidecar": {
                "path": "oracle-sidecar.json",
                "sha256": _sha256(oracle_path),
            },
            "record": {"path": "record.json", "sha256": _sha256(record_path)},
        },
        "pstrain_vs_oracle": comparisons,
        "schema_version": 2,
    }
    return json.dumps(analysis, indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = regenerate(args.record, args.oracle)
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"stale generated artifact: {args.output}")
        return
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
