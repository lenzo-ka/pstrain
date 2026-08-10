"""Tests for printp functionality."""

from pathlib import Path

import numpy as np
import pytest

# Check if library exists
# libpstrainc availability comes from the shared helper (real loader-based
# detection); see tests/clib.py.
from tests.clib import C_LIBRARY_AVAILABLE as _lib_exists


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestFormatGau:
    """Tests for format_gau function."""

    def test_basic_format(self, tmp_path: Path) -> None:
        """Test basic Gaussian formatting."""
        from pstrain.lib import _pstrainc
        from pstrain.lib.printp import format_gau

        # Create a simple Gaussian file
        gau_path = tmp_path / "means"
        n_mgau, n_feat, n_density, veclen = 2, 1, 4, 13
        gau = np.random.randn(n_mgau, n_feat, n_density, veclen).astype(np.float32)
        _pstrainc.write_gau(str(gau_path), gau)

        output = format_gau(gau_path)

        assert "param 2 1 4" in output
        assert "mgau 0" in output
        assert "mgau 1" in output
        assert "feat 0" in output
        assert "density" in output


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestFormatMixw:
    """Tests for format_mixw function."""

    def test_basic_format(self, tmp_path: Path) -> None:
        """Test basic mixture weight formatting."""
        from pstrain.lib import _pstrainc
        from pstrain.lib.printp import format_mixw

        # Create a simple mixw file
        mixw_path = tmp_path / "mixw"
        n_mixw, n_feat, n_density = 3, 1, 8
        mixw = np.random.rand(n_mixw, n_feat, n_density).astype(np.float32)
        # Normalize
        mixw = mixw / mixw.sum(axis=2, keepdims=True)
        _pstrainc.write_mixw(str(mixw_path), mixw)

        output = format_mixw(mixw_path)

        assert "mixw 3 1 8" in output
        assert "mixw [0 0]" in output
        assert "mixw [1 0]" in output
        assert "mixw [2 0]" in output


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestFormatTmat:
    """Tests for format_tmat function."""

    def test_basic_format(self, tmp_path: Path) -> None:
        """Test basic transition matrix formatting."""
        from pstrain.lib import _pstrainc
        from pstrain.lib.printp import format_tmat

        # Create a simple tmat file
        tmat_path = tmp_path / "tmat"
        n_tmat, n_state = 2, 4
        # tmat shape is (n_tmat, n_state-1, n_state) for write
        tmat = np.zeros((n_tmat, n_state - 1, n_state), dtype=np.float32)
        # Simple left-to-right topology
        for t in range(n_tmat):
            for i in range(n_state - 1):
                tmat[t, i, i] = 0.5  # self-loop
                tmat[t, i, i + 1] = 0.5  # forward
        _pstrainc.write_tmat(str(tmat_path), tmat)

        output = format_tmat(tmat_path)

        assert "tmat 2" in output
        assert "tmat [0]" in output
        assert "tmat [1]" in output


class TestPrintParamsShellout:
    """Tests for shell-out implementation."""

    def test_import(self) -> None:
        """Test that shell-out function can be imported."""
        from pstrain.lib.printp import print_params_shellout

        assert callable(print_params_shellout)

    def test_missing_binary(self, tmp_path: Path) -> None:
        """Test handling of missing binary."""
        import subprocess

        from pstrain.lib.printp import print_params_shellout

        with pytest.raises((subprocess.CalledProcessError, FileNotFoundError)):
            print_params_shellout(
                mixw_path=tmp_path / "fake.mixw",
                bin_path="/nonexistent/printp",
            )
