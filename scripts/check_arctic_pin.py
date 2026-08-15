#!/usr/bin/env python3
"""Check Arctic pin conditions or deliberately adopt fields it does not cover."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pstrain.benchmarks.arctic import (  # noqa: E402
    adopt_uncovered_conditions,
    authenticate_conditions,
    benchmark_conditions,
    resolved_configuration_provenance,
    validate_record,
)

DEFAULT_RECORD = ROOT / "docs/benchmarks/arctic-pin/record.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument(
        "--adopt-uncovered",
        action="store_true",
        help="add live-only condition fields; never change existing pins or benchmark results",
    )
    parser.add_argument(
        "--adopt-record",
        type=Path,
        metavar="CANDIDATE",
        help="adopt a freshly emitted record while preserving retired historical measurements",
    )
    args = parser.parse_args()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    if args.adopt_record:
        candidate = json.loads(args.adopt_record.read_text(encoding="utf-8"))
        validate_record(candidate)
        actual = benchmark_conditions()
        uncovered = authenticate_conditions(actual, candidate["conditions"])
        if uncovered:
            raise RuntimeError(f"candidate leaves live conditions uncovered: {uncovered}")
        stable_fields = (
            "wer",
            "errors",
            "ref_words",
            "utterances",
            "decoded",
            "oov_tokens",
            "known_skips",
            "utterance_rows",
            "configuration_provenance",
        )
        for dataset in ("slt55", "big"):
            old = record["results"]["off"][dataset]
            new = candidate["results"]["off"][dataset]
            for field in stable_fields:
                if new.get(field) != old.get(field):
                    raise RuntimeError(
                        f"retired off/{dataset} historical drift in {field}: "
                        f"recorded={old.get(field)!r}, candidate={new.get(field)!r}"
                    )
        temporary = args.record.with_suffix(args.record.suffix + ".tmp")
        temporary.write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(args.record)
        print("adopted fresh MULTIPRON-ONLY Arctic benchmark record")
        return 0
    validate_record(record)
    actual = benchmark_conditions()
    uncovered = authenticate_conditions(actual, record["conditions"])
    if args.adopt_uncovered:
        record["conditions"] = adopt_uncovered_conditions(actual, record["conditions"])
        pin_conditions = record["conditions"]["pin_conditions"]
        source_kinds = record["conditions"]["pin_condition_source_kinds"]
        for mode, datasets in record["results"].items():
            provenance = resolved_configuration_provenance(
                {
                    "profile": pin_conditions[mode],
                    "profile_name": mode,
                    "field_source_kinds": source_kinds[mode],
                }
            )
            for cell in datasets.values():
                cell["configuration_provenance"] = provenance
        validate_record(record)
        args.record.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"adopted {len(uncovered)} previously uncovered Arctic pin condition fields")
    elif uncovered:
        print("Arctic pin condition fields not covered by the record (non-fatal):")
        for path in uncovered:
            print(f"  {path}")
        print("adopt deliberately with: python scripts/check_arctic_pin.py --adopt-uncovered")
    else:
        print("Arctic pin conditions match; the record covers every live condition field")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
