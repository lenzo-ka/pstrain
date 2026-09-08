"""Smoke-test an installed pstrain artifact from a neutral working directory."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from importlib.resources import files
from pathlib import Path

import pstrain
from pstrain.api import TUTORIAL_FILENAME
from pstrain.lib import _pstrainc
from pstrain.lib.testing.decoder import pocketsphinx_version

project_root = Path(__file__).parents[1].resolve()
package_path = Path(pstrain.__file__).resolve()
if package_path.is_relative_to(project_root):
    raise RuntimeError(f"import resolved to source tree: {package_path}")

_pstrainc.get_lib()
if not pocketsphinx_version().startswith("5.1.1+511126b4"):
    raise RuntimeError(f"unexpected linked PocketSphinx provenance: {pocketsphinx_version()}")
result = subprocess.run(
    [sys.executable, "-c", "import pstrain; print(pstrain.__version__)"],
    check=True,
    capture_output=True,
    text=True,
)
if pstrain.__version__ not in result.stdout:
    raise RuntimeError("installed version is not reported by a fresh interpreter")

with tempfile.TemporaryDirectory() as temporary_directory:
    tutorial_path = Path(temporary_directory) / TUTORIAL_FILENAME
    tutorial_result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pstrain.cli import main; raise SystemExit(main())",
            "tutorial",
            "--output",
            str(tutorial_path),
        ],
        capture_output=True,
        text=True,
    )
    if tutorial_result.returncode != 0:
        raise RuntimeError(
            "installed tutorial command failed:\n"
            f"stdout:\n{tutorial_result.stdout}\n"
            f"stderr:\n{tutorial_result.stderr}"
        )
    packaged_tutorial = files("pstrain.data").joinpath("notebooks", TUTORIAL_FILENAME)
    if tutorial_path.read_bytes() != packaged_tutorial.read_bytes():
        raise RuntimeError("tutorial command did not copy the packaged notebook")
