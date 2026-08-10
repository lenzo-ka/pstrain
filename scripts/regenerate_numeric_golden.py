#!/usr/bin/env python3
"""Regenerate tests/golden/numeric_bw.json on the current machine."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.numeric_harness import GOLDEN, write_golden


def main() -> None:
    """Build the fixed fixture and replace the golden JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=GOLDEN)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pstrain-numeric-golden-") as directory:
        write_golden(args.output, Path(directory) / "project")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
