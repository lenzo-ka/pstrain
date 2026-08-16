#!/usr/bin/env python3
"""Reject pstrain imports that resolve outside this checkout."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = """
import json
try:
    import pstrain
except ModuleNotFoundError as error:
    if error.name != "pstrain":
        raise
    print(json.dumps({"status": "missing"}))
else:
    import pstrain.benchmarks.arctic as arctic
    print(json.dumps({
        "status": "resolved",
        "package_file": pstrain.__file__,
        "package_path": list(pstrain.__path__),
        "arctic_origin": arctic.__spec__.origin,
    }))
"""


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def main() -> int:
    # A neutral cwd prevents the checkout itself from making the bare import pass.
    with tempfile.TemporaryDirectory(prefix="pstrain-ambient-probe-") as directory:
        probe_path = Path(directory) / "probe.py"
        probe_path.write_text(PROBE, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(probe_path)],
            cwd=directory,
            check=False,
            capture_output=True,
            text=True,
        )
    if result.returncode:
        sys.stderr.write(result.stderr)
        raise SystemExit(
            "ambient pstrain import probe failed for a reason other than "
            "ModuleNotFoundError: pstrain"
        )

    try:
        outcome = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"ambient pstrain import probe returned invalid output: {result.stdout!r}"
        ) from error

    if outcome["status"] == "missing":
        print("ambient pstrain import: clean (ModuleNotFoundError)")
        return 0

    paths = [
        ("pstrain.__file__", outcome["package_file"]),
        *(("pstrain.__path__", path) for path in outcome["package_path"]),
        ("pstrain.benchmarks.arctic origin", outcome["arctic_origin"]),
    ]
    for label, raw_path in paths:
        resolved_path = Path(raw_path).resolve()
        if not _is_relative_to(resolved_path, ROOT):
            raise SystemExit(
                "ERROR: ambient pstrain import resolves outside this checkout\n"
                f"offending {label}: {resolved_path}\n"
                f"current checkout: {ROOT}"
            )
    print(f"ambient pstrain import: checkout-local ({Path(outcome['package_file']).resolve()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
