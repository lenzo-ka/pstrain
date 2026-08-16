"""Tests for native artifact path discovery."""

from pathlib import Path

from pstrain.lib import paths


def test_find_lib_path_finds_windows_runtime_in_build_bin(tmp_path: Path, monkeypatch) -> None:
    """A development Windows DLL is discoverable in the runtime directory."""
    runtime = tmp_path / "build" / "bin" / "pstrainc.dll"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"dll")
    monkeypatch.delenv("PSTRAIN_LIB_PATH", raising=False)
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setattr(paths, "_get_bundled_lib_dir", lambda: None)
    monkeypatch.setattr(paths, "_get_project_root", lambda: tmp_path)

    assert paths._find_lib_path() == runtime


def test_find_lib_path_finds_windows_runtime_in_release_subdirectory(
    tmp_path: Path, monkeypatch
) -> None:
    """A Visual Studio Release DLL is discoverable in its configuration directory."""
    runtime = tmp_path / "build" / "bin" / "Release" / "pstrainc.dll"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"dll")
    monkeypatch.delenv("PSTRAIN_LIB_PATH", raising=False)
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setattr(paths, "_get_bundled_lib_dir", lambda: None)
    monkeypatch.setattr(paths, "_get_project_root", lambda: tmp_path)

    assert paths._find_lib_path() == runtime


def test_find_lib_path_finds_windows_runtime_bundled_in_bin(tmp_path: Path, monkeypatch) -> None:
    """A wheel DLL is discoverable under its CMake RUNTIME destination."""
    bundled = tmp_path / "_lib"
    runtime = bundled / "bin" / "pstrainc.dll"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"dll")
    monkeypatch.delenv("PSTRAIN_LIB_PATH", raising=False)
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setattr(paths, "_get_bundled_lib_dir", lambda: bundled)
    monkeypatch.setattr(paths, "_get_project_root", lambda: None)

    assert paths._find_lib_path() == runtime


def test_find_lib_path_does_not_select_windows_runtime_on_posix(
    tmp_path: Path, monkeypatch
) -> None:
    """POSIX discovery never returns a planted Windows runtime."""
    runtime = tmp_path / "build" / "bin" / "pstrainc.dll"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"dll")
    monkeypatch.delenv("PSTRAIN_LIB_PATH", raising=False)
    monkeypatch.setattr(paths.sys, "platform", "darwin")
    monkeypatch.setattr(paths, "_get_bundled_lib_dir", lambda: None)
    monkeypatch.setattr(paths, "_get_project_root", lambda: tmp_path)

    assert paths._find_lib_path() is None


def test_find_lib_path_does_not_select_posix_library_on_windows(
    tmp_path: Path, monkeypatch
) -> None:
    """Windows discovery uses the DLL basename, not a planted POSIX library."""
    posix_library = tmp_path / "build" / "lib" / "libpstrainc.so"
    posix_library.parent.mkdir(parents=True)
    posix_library.write_bytes(b"shared library")
    monkeypatch.delenv("PSTRAIN_LIB_PATH", raising=False)
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setattr(paths, "_get_bundled_lib_dir", lambda: None)
    monkeypatch.setattr(paths, "_get_project_root", lambda: tmp_path)

    assert paths._find_lib_path() is None


def test_find_lib_path_keeps_platform_specific_posix_names(tmp_path: Path, monkeypatch) -> None:
    """Linux and macOS retain their existing shared-library basenames."""
    monkeypatch.delenv("PSTRAIN_LIB_PATH", raising=False)
    monkeypatch.setattr(paths, "_get_bundled_lib_dir", lambda: None)
    monkeypatch.setattr(paths, "_get_project_root", lambda: tmp_path)

    linux_library = tmp_path / "build" / "lib" / "libpstrainc.so"
    linux_library.parent.mkdir(parents=True)
    linux_library.write_bytes(b"shared library")
    monkeypatch.setattr(paths.sys, "platform", "linux")
    assert paths._find_lib_path() == linux_library

    linux_library.unlink()
    macos_library = tmp_path / "build" / "lib" / "libpstrainc.dylib"
    macos_library.write_bytes(b"shared library")
    monkeypatch.setattr(paths.sys, "platform", "darwin")
    assert paths._find_lib_path() == macos_library
