#!/usr/bin/env python3
"""Run pytest against this checkout despite a shared editable installation.

scikit-build-core editable finders are installed ahead of normal filesystem
lookup and may redirect ``pstrain`` to whichever worktree was installed last.
This runner removes only pstrain's editable finder and puts its own repository
root first.  The collection-time identity gate in ``tests/conftest.py`` remains
the authority that proves which Python package and native library were selected.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


def _prepare_checkout_imports() -> Path:
    root = Path(__file__).resolve().parent.parent
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if finder.__class__.__module__ != "_editable_skbc_pstrain"
    ]
    sys.path.insert(0, str(root))
    os.chdir(root)
    return root


_prepare_checkout_imports()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--exec"]:
        if len(sys.argv) != 3:
            raise SystemExit("usage: run_verified_tests.py --exec CODE")
        exec(sys.argv[2], {"__name__": "__main__"})
        raise SystemExit(0)

    import pstrain
    import pytest

    from pstrain.lib._cffi.core import _find_library

    result = pytest.main(sys.argv[1:])
    print(
        "verified subjects: "
        f"pstrain.__file__={Path(pstrain.__file__).resolve()}; "
        f"libpstrainc={_find_library().resolve()}"
    )
    raise SystemExit(result)
