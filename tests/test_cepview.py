"""Tests for cepview functionality."""

from pathlib import Path

import numpy as np
import pytest

from st2.lib.cepview import view_cepstra


class TestViewCepstra:
    """Tests for view_cepstra function."""

    def test_basic_output(self, tmp_path: Path) -> None:
        """Test basic output format."""
        # Create a simple MFC file
        mfc_path = tmp_path / "test.mfc"
        n_frames = 10
        n_coeff = 13
        data = np.random.randn(n_frames, n_coeff).astype(np.float32)

        # Write MFC file
        with mfc_path.open("wb") as f:
            n_float = n_frames * n_coeff
            f.write(np.array([n_float], dtype=np.int32).tobytes())
            f.write(data.tobytes())

        output = view_cepstra(
            mfc_path,
            n_coeff=n_coeff,
            display_cols=5,
            start_frame=0,
            end_frame=3,
            show_header=True,
            show_frame_numbers=True,
        )

        lines = output.strip().split("\n")
        # Header + 3 data lines
        assert len(lines) == 4

    def test_no_header(self, tmp_path: Path) -> None:
        """Test output without header."""
        mfc_path = tmp_path / "test.mfc"
        n_frames = 5
        n_coeff = 13
        data = np.zeros((n_frames, n_coeff), dtype=np.float32)

        with mfc_path.open("wb") as f:
            n_float = n_frames * n_coeff
            f.write(np.array([n_float], dtype=np.int32).tobytes())
            f.write(data.tobytes())

        output = view_cepstra(
            mfc_path,
            n_coeff=n_coeff,
            display_cols=5,
            show_header=False,
            show_frame_numbers=False,
        )

        lines = output.strip().split("\n")
        # Just data lines, no header
        assert len(lines) == n_frames

    def test_frame_range(self, tmp_path: Path) -> None:
        """Test frame range selection."""
        mfc_path = tmp_path / "test.mfc"
        n_frames = 20
        n_coeff = 13
        data = np.arange(n_frames * n_coeff, dtype=np.float32).reshape(n_frames, n_coeff)

        with mfc_path.open("wb") as f:
            n_float = n_frames * n_coeff
            f.write(np.array([n_float], dtype=np.int32).tobytes())
            f.write(data.tobytes())

        output = view_cepstra(
            mfc_path,
            n_coeff=n_coeff,
            display_cols=3,
            start_frame=5,
            end_frame=10,
            show_header=False,
            show_frame_numbers=False,
        )

        lines = output.strip().split("\n")
        # 5 frames (5, 6, 7, 8, 9)
        assert len(lines) == 5

        # First value should be from frame 5
        first_val = float(lines[0].split()[0])
        assert abs(first_val - data[5, 0]) < 0.001


class TestViewCepstraShellout:
    """Tests for shell-out implementation."""

    def test_import(self) -> None:
        """Test that shell-out function can be imported."""
        from st2.lib.cepview import view_cepstra_shellout

        assert callable(view_cepstra_shellout)

    def test_missing_binary(self, tmp_path: Path) -> None:
        """Test handling of missing binary."""
        import subprocess

        from st2.lib.cepview import view_cepstra_shellout

        mfc_path = tmp_path / "test.mfc"
        mfc_path.write_bytes(b"\x00" * 100)

        with pytest.raises((subprocess.CalledProcessError, FileNotFoundError)):
            view_cepstra_shellout(
                mfc_path,
                bin_path="/nonexistent/sphinx_cepview",
            )
