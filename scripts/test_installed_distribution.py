"""Smoke-test an installed pstrain artifact from a neutral working directory."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pstrain
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
