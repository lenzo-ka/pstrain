from __future__ import annotations

from types import SimpleNamespace

import pytest

from pstrain.lib._cffi import core


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
