from __future__ import annotations

import hashlib
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from pstrain.lib._cffi import core
from pstrain.lib._cffi.cdef import CDEF
from tests.clib import requires_c_library

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_init_rejects_library_abi_version_skew(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeFFI:
        def cdef(self, _declarations: str) -> None:
            pass

        def dlopen(self, _path: str) -> SimpleNamespace:
            return SimpleNamespace(pstrain_abi_version=lambda: 99)

    monkeypatch.setattr(core, "_ffi", None)
    monkeypatch.setattr(core, "_lib", None)
    monkeypatch.setattr(core, "FFI", FakeFFI)
    monkeypatch.setattr(core, "_find_library", lambda: core.Path("skewed-libpstrainc.so"))

    with pytest.raises(
        RuntimeError,
        match=(
            rf"Python expects ABI version {core.PSTRAIN_ABI_VERSION}, library reports 99\. "
            r"The native library at skewed-libpstrainc\.so is stale; "
            r"rebuild it with `make build-c`\."
        ),
    ):
        core._init()


def test_init_rejects_pre_handshake_library(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeFFI:
        def cdef(self, _declarations: str) -> None:
            pass

        def dlopen(self, _path: str) -> SimpleNamespace:
            return SimpleNamespace()

    monkeypatch.setattr(core, "_ffi", None)
    monkeypatch.setattr(core, "_lib", None)
    monkeypatch.setattr(core, "FFI", FakeFFI)
    monkeypatch.setattr(core, "_find_library", lambda: core.Path("old-libpstrainc.so"))

    with pytest.raises(
        RuntimeError,
        match=(
            rf"Python expects ABI version {core.PSTRAIN_ABI_VERSION}, but the library has no "
            r"ABI version handshake \(pre-handshake library\)\. "
            r"The native library at old-libpstrainc\.so is stale; "
            r"rebuild it with `make build-c`\."
        ),
    ):
        core._init()


def _patch_loader(monkeypatch: pytest.MonkeyPatch, fingerprint: bytes | None, name: str) -> None:
    """Load a stand-in library at the current ABI version with a chosen fingerprint."""

    class FakeFFI:
        def cdef(self, _declarations: str) -> None:
            pass

        def dlopen(self, _path: str) -> SimpleNamespace:
            symbols: dict[str, object] = {"pstrain_abi_version": lambda: core.PSTRAIN_ABI_VERSION}
            if fingerprint is not None:
                symbols["pstrain_interface_fingerprint"] = lambda: fingerprint
            return SimpleNamespace(**symbols)

        def string(self, value: bytes) -> bytes:
            return value

    monkeypatch.setattr(core, "_ffi", None)
    monkeypatch.setattr(core, "_lib", None)
    monkeypatch.setattr(core, "FFI", FakeFFI)
    monkeypatch.setattr(core, "_find_library", lambda: core.Path(name))


def _pre_fingerprint_message(path: str) -> str:
    return (
        r"libpstrainc interface fingerprint mismatch: Python expects interface "
        rf"fingerprint {core.interface_fingerprint()}, but the library has no interface "
        r"fingerprint \(pre-fingerprint library\)\. "
        rf"The native library at {re.escape(path)} is stale; rebuild it with `make build-c`\."
    )


def test_interface_fingerprint_is_sha256_of_cdef() -> None:
    expected = hashlib.sha256(CDEF.encode("utf-8")).hexdigest()
    assert core.interface_fingerprint() == expected
    assert core.interface_fingerprint(CDEF + "\nint added(void);") != expected


def test_init_rejects_interface_fingerprint_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_loader(monkeypatch, b"0" * 64, "drifted-libpstrainc.so")

    with pytest.raises(
        RuntimeError,
        match=(
            r"libpstrainc interface fingerprint mismatch: Python expects interface "
            rf"fingerprint {core.interface_fingerprint()}, library reports {'0' * 64}\. "
            r"The native library at drifted-libpstrainc\.so is stale; "
            r"rebuild it with `make build-c`\."
        ),
    ):
        core._init()
    assert core._ffi is None
    assert core._lib is None


def test_init_accepts_matching_interface_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    fingerprint = core.interface_fingerprint().encode("ascii")
    _patch_loader(monkeypatch, fingerprint, "fresh-libpstrainc.so")

    _ffi, lib = core._init()

    assert lib.pstrain_interface_fingerprint() == fingerprint


def test_init_rejects_library_without_interface_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_loader(monkeypatch, None, "prefingerprint-libpstrainc.so")

    with pytest.raises(
        RuntimeError, match=_pre_fingerprint_message("prefingerprint-libpstrainc.so")
    ):
        core._init()


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("cc") is None,
    reason="needs a POSIX C compiler to build a stub shared library",
)
def test_real_library_at_current_abi_without_fingerprint_is_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A library built before the fingerprint passes the ABI version check but
    lacks the fingerprint symbol; real cffi symbol lookup must reject it."""
    source = tmp_path / "stub.c"
    source.write_text(
        f"unsigned int pstrain_abi_version(void) {{ return {core.PSTRAIN_ABI_VERSION}; }}\n"
    )
    library = tmp_path / ("libstub.dylib" if sys.platform == "darwin" else "libstub.so")
    subprocess.run(["cc", "-shared", "-fPIC", "-o", str(library), str(source)], check=True)

    monkeypatch.setattr(core, "_ffi", None)
    monkeypatch.setattr(core, "_lib", None)
    monkeypatch.setattr(core, "_find_library", lambda: library)

    with pytest.raises(RuntimeError, match=_pre_fingerprint_message(str(library))):
        core._init()


def _load_generator() -> ModuleType:
    script = REPO_ROOT / "scripts" / "generate_cffi_exports.py"
    spec = importlib.util.spec_from_file_location("generate_cffi_exports", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generator_check_fails_when_cdef_changes_without_regenerating(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    generator = _load_generator()
    monkeypatch.setattr(sys, "argv", ["generate_cffi_exports.py", "--check"])
    assert generator.main() == 0

    # A comment-only edit leaves every export list unchanged, so only the
    # fingerprint header can catch it.
    monkeypatch.setattr(generator, "CDEF", CDEF + "\n// edited\n")
    assert generator.main() == 1
    assert "pstrain_interface_fingerprint.h" in capsys.readouterr().err


def test_checked_in_header_carries_current_fingerprint() -> None:
    header = (
        REPO_ROOT / "csrc" / "libs" / "libpstrain" / "pstrain_interface_fingerprint.h"
    ).read_text()
    assert f'#define PSTRAIN_INTERFACE_FINGERPRINT "{core.interface_fingerprint()}"' in header


@requires_c_library
def test_built_library_fingerprint_matches_python() -> None:
    from pstrain.lib._pstrainc import get_ffi, get_lib

    reported = get_ffi().string(get_lib().pstrain_interface_fingerprint()).decode("ascii")
    assert reported == core.interface_fingerprint()
