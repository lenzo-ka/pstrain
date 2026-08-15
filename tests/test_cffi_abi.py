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
        match=r"Python expects ABI version 2, library reports 99",
    ):
        core._init()
