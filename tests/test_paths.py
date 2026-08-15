"""Tests for native artifact path discovery."""

from pathlib import Path

from pstrain.lib import paths


def test_find_lib_path_finds_windows_runtime_in_build_bin(tmp_path: Path, monkeypatch) -> None:
    """A development Windows DLL is discoverable in the runtime directory."""
    runtime = tmp_path / "build" / "bin" / "pstrainc.dll"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"dll")
    monkeypatch.delenv("PSTRAIN_LIB_PATH", raising=False)
    monkeypatch.setattr(paths, "_get_bundled_lib_dir", lambda: None)
    monkeypatch.setattr(paths, "_get_project_root", lambda: tmp_path)

    assert paths._find_lib_path() == runtime
