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
    RECORD_BINDING_FIELDS,
    adopt_uncovered_conditions,
    authenticate_conditions,
    benchmark_conditions,
    bind_record,
    comparison_comparability,
    record_binding_sha256,
    require_committed_baseline,
    resolved_configuration_provenance,
    validate_record,
)

DEFAULT_RECORD = ROOT / "evidence/arctic-pin/record.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument(
        "--adopt-comparability",
        action="store_true",
        help="upgrade a schema-8 record with schema-owned per-cell comparability",
    )
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
    parser.add_argument("--allow-uncommitted-baseline", action="store_true")
    args = parser.parse_args()
    record_bytes = require_committed_baseline(
        args.record, allow_uncommitted=args.allow_uncommitted_baseline
    )
    record = json.loads(record_bytes)
    if args.adopt_comparability:
        if record.get("schema_version") != 8:
            raise RuntimeError("comparability adoption requires a schema-8 record")
        binding = record.get("record_binding")
        expected = {
            "algorithm": "sha256-canonical-json-v1",
            "fields": list(RECORD_BINDING_FIELDS),
            "sha256": record_binding_sha256(record),
        }
        if binding != expected:
            raise RuntimeError("record consistency digest mismatch before comparability adoption")
        record["schema_version"] = 9
        for mode, cells in record["results"].items():
            for cell in cells.values():
                cell["comparability"] = comparison_comparability(mode)
        bind_record(record)
        validate_record(record)
        args.record.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("adopted schema-owned Arctic comparability classifications")
        return 0
    if args.adopt_record:
        candidate = json.loads(args.adopt_record.read_text(encoding="utf-8"))
        validate_record(candidate)
        actual = benchmark_conditions()
        uncovered = authenticate_conditions(actual, candidate["conditions"])
        if uncovered:
            raise RuntimeError(f"candidate leaves live conditions uncovered: {uncovered}")
        for dataset in ("slt55", "big"):
            old = record["results"]["off"][dataset]
            new = candidate["results"]["off"][dataset]
            if new != old:
                raise RuntimeError(
                    f"retired off/{dataset} full-cell historical drift: "
                    "every serialized field must remain exactly equal"
                )
        if candidate["historical_provenance"] != record["historical_provenance"]:
            raise RuntimeError(
                "retired historical provenance drift: the engine, resource, and model "
                "identity recorded for the retired cells must remain exactly equal"
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
        for mode in pin_conditions:
            datasets = record["results"][mode]
            provenance = resolved_configuration_provenance(
                {
                    "profile": pin_conditions[mode],
                    "profile_name": mode,
                    "field_source_kinds": source_kinds[mode],
                }
            )
            for cell in datasets.values():
                cell["configuration_provenance"] = provenance
        bind_record(record)
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
