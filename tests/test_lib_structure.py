"""Test library structure and imports."""

from pathlib import Path
from tomllib import load

import pytest

from pstrain import __version__

# libpstrainc availability comes from the shared helper (real loader-based
# detection); see tests/clib.py.
from tests.clib import c_library_available as _lib_exists


def test_version() -> None:
    """Test that version is defined."""
    with (Path(__file__).parents[1] / "pyproject.toml").open("rb") as pyproject:
        assert __version__ == load(pyproject)["project"]["version"]


def test_lib_public_api() -> None:
    """Test that public API functions are available."""
    from pstrain.lib import (
        Profile,
        resolve_config,
        setup_project,
        validate_project,
    )

    assert Profile is not None
    assert callable(resolve_config)
    assert callable(setup_project)
    assert callable(validate_project)


def test_pstrainc_dunder_probe_does_not_load_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """Module metadata probes remain safe when libpstrainc is unavailable."""
    from pstrain.lib import _pstrainc

    def fail_if_loaded() -> None:
        raise AssertionError("dunder probe attempted to load libpstrainc")

    monkeypatch.setattr(_pstrainc, "get_lib", fail_if_loaded)

    with pytest.raises(AttributeError, match="__sphinx_mock__"):
        _ = _pstrainc.__sphinx_mock__


@pytest.mark.skipif(not _lib_exists(), reason="C library not built")
def test_pstrainc_bindings() -> None:
    """Test that C bindings work (requires built library)."""
    from pstrain.lib._pstrainc import get_ffi, get_lib

    ffi = get_ffi()
    lib = get_lib()

    assert ffi is not None
    assert lib is not None


@pytest.mark.skipif(not _lib_exists(), reason="C library not built")
def test_lib_singleton() -> None:
    """Test that get_lib returns the same instance."""
    from pstrain.lib._pstrainc import get_lib

    lib1 = get_lib()
    lib2 = get_lib()
    assert lib1 is lib2


@pytest.mark.skipif(not _lib_exists(), reason="C library not built")
def test_ffi_singleton() -> None:
    """Test that get_ffi returns the same instance."""
    from pstrain.lib._pstrainc import get_ffi

    ffi1 = get_ffi()
    ffi2 = get_ffi()
    assert ffi1 is ffi2


@pytest.mark.skipif(not _lib_exists(), reason="C library not built")
def test_library_has_functions() -> None:
    """Test that the library has expected functions."""
    from pstrain.lib._pstrainc import get_lib

    lib = get_lib()
    # Check for some key functions
    assert hasattr(lib, "pstrain_cffi_logmath_init")
    assert hasattr(lib, "pstrain_cffi_logmath_free")
    assert hasattr(lib, "pstrain_fe_create")
    assert hasattr(lib, "pstrain_bw_init")
    assert hasattr(lib, "s3gau_read")
    assert hasattr(lib, "s3mixw_read")


@pytest.mark.skipif(not _lib_exists(), reason="C library not built")
def test_ffi_can_create_types() -> None:
    """Test that FFI can create C types."""
    from pstrain.lib._pstrainc import get_ffi

    ffi = get_ffi()
    # Create some basic types
    int_ptr = ffi.new("int32 *", 42)
    assert int_ptr[0] == 42
    char_array = ffi.new("char[]", b"hello")
    assert ffi.string(char_array) == b"hello"
    float_val = ffi.new("float32 *", 3.14)
    assert abs(float_val[0] - 3.14) < 0.001
