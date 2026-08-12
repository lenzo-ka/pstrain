"""Bounded diagnostics for PocketSphinx cross-utterance decoder state."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import multiprocessing
from pathlib import Path


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
    args = parser.parse_args()
    if args.provenance:
        context = multiprocessing.get_context("spawn")
        with context.Pool(1) as pool:
            worker = pool.map(import_provenance, [None])[0]
        print(json.dumps({"parent": import_provenance(), "spawned_worker": worker}, indent=2))
        return 0
    parser.error("a probe mode is required")


if __name__ == "__main__":
    raise SystemExit(main())
