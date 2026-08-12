"""Run auditwheel with the PocketSphinx library bundled in its input wheel."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: repair_linux_wheel.py WHEEL DESTINATION")

    wheel = Path(sys.argv[1]).resolve(strict=True)
    destination = Path(sys.argv[2]).resolve()
    with zipfile.ZipFile(wheel) as archive:
        libraries = [
            name for name in archive.namelist() if Path(name).name.startswith("libpocketsphinx.so.")
        ]
        if len(libraries) != 1:
            raise RuntimeError(
                f"expected exactly one versioned PocketSphinx library in {wheel}, found {libraries}"
            )
        with tempfile.TemporaryDirectory(prefix="pstrain-auditwheel-") as temporary:
            lookup = Path(temporary)
            library = lookup / Path(libraries[0]).name
            library.write_bytes(archive.read(libraries[0]))
            environment = os.environ.copy()
            environment["AUDITWHEEL_LD_LIBRARY_PATH"] = str(lookup)
            subprocess.run(
                ["auditwheel", "repair", "-w", str(destination), str(wheel)],
                check=True,
                env=environment,
            )


if __name__ == "__main__":
    main()
