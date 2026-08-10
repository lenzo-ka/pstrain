"""Tests for tied-state to codebook mapping."""

from pathlib import Path

import numpy as np
import pytest

from pstrain.lib import ts2cb
from pstrain.lib.ts2cb import TyingType

# Check if library exists
# libpstrainc availability comes from the shared helper (real loader-based
# detection); see tests/clib.py.
from tests.clib import C_LIBRARY_AVAILABLE as _lib_exists


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestCreateTs2cb:
    """Tests for creating ts2cb mappings."""

    def test_semi_continuous(self) -> None:
        """Test semi-continuous mapping (all states -> codebook 0)."""
        n_states = 100
        mapping, n_cb = ts2cb.create_ts2cb(n_states, TyingType.SEMI)

        assert len(mapping) == n_states
        assert n_cb == 1
        assert np.all(mapping == 0)

    def test_continuous(self) -> None:
        """Test continuous mapping (identity)."""
        n_states = 100
        mapping, n_cb = ts2cb.create_ts2cb(n_states, TyingType.CONT)

        assert len(mapping) == n_states
        assert n_cb == n_states
        assert np.array_equal(mapping, np.arange(n_states, dtype=np.uint32))

    def test_string_tying_type(self) -> None:
        """Test that string tying types work."""
        n_states = 50
        mapping, n_cb = ts2cb.create_ts2cb(n_states, "semi")

        assert len(mapping) == n_states
        assert n_cb == 1

    def test_ptm_requires_mdef(self) -> None:
        """Test that PTM tying requires mdef_path."""
        with pytest.raises(ValueError, match="mdef_path"):
            ts2cb.create_ts2cb(100, TyingType.PTM)

    def test_invalid_tying_type(self) -> None:
        """Test that invalid tying type raises error."""
        with pytest.raises(ValueError, match="invalid"):
            ts2cb.create_ts2cb(100, "invalid")


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestTs2cbIO:
    """Tests for ts2cb file I/O."""

    def test_roundtrip_continuous(self, tmp_path: Path) -> None:
        """Test roundtrip write/read for continuous mapping."""
        n_states = 100
        original, n_cb = ts2cb.create_ts2cb(n_states, TyingType.CONT)

        output_path = tmp_path / "test.ts2cb"
        ts2cb.write_ts2cb(output_path, original, n_cb)

        loaded, loaded_n_cb = ts2cb.read_ts2cb(output_path)

        assert np.array_equal(original, loaded)
        assert n_cb == loaded_n_cb

    def test_roundtrip_semi(self, tmp_path: Path) -> None:
        """Test roundtrip write/read for semi-continuous mapping."""
        n_states = 50
        original, n_cb = ts2cb.create_ts2cb(n_states, TyingType.SEMI)

        output_path = tmp_path / "test.ts2cb"
        ts2cb.write_ts2cb(output_path, original, n_cb)

        loaded, loaded_n_cb = ts2cb.read_ts2cb(output_path)

        assert np.array_equal(original, loaded)
        assert n_cb == loaded_n_cb

    def test_write_infers_n_cb(self, tmp_path: Path) -> None:
        """Test that write_ts2cb infers n_cb from data if not provided."""
        n_states = 100
        mapping = np.arange(n_states, dtype=np.uint32)

        output_path = tmp_path / "test.ts2cb"
        ts2cb.write_ts2cb(output_path, mapping)  # n_cb not specified

        loaded, loaded_n_cb = ts2cb.read_ts2cb(output_path)

        assert np.array_equal(mapping, loaded)
        assert loaded_n_cb == n_states

    def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        """Test that reading a non-existent file raises an error."""
        with pytest.raises(RuntimeError, match="Failed to read"):
            ts2cb.read_ts2cb(tmp_path / "nonexistent.ts2cb")


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestCreateAndWriteTs2cb:
    """Tests for the combined create and write function."""

    @pytest.fixture
    def mdef_file(self, tmp_path: Path) -> Path:
        """Create a minimal mdef file."""
        mdef = tmp_path / "test.mdef"
        mdef.write_text(
            """0.3
5 n_base
0 n_tri
15 n_state_map
15 n_tied_state
15 n_tied_ci_state
5 n_tied_tmat
AA - - - n/a 0 0 1 2 N
AE - - - n/a 1 3 4 5 N
SIL - - - filler 2 6 7 8 N
+NOISE+ - - - filler 3 9 10 11 N
+SPN+ - - - filler 4 12 13 14 N
"""
        )
        return mdef

    def test_create_and_write_continuous(self, mdef_file: Path, tmp_path: Path) -> None:
        """Test creating and writing a continuous ts2cb from mdef."""
        output_path = tmp_path / "output.ts2cb"

        mapping, n_cb = ts2cb.create_and_write_ts2cb(mdef_file, output_path, TyingType.CONT)

        assert output_path.exists()
        assert len(mapping) == 15  # n_tied_state from mdef
        assert n_cb == 15

        # Verify by reading back
        loaded, loaded_n_cb = ts2cb.read_ts2cb(output_path)
        assert np.array_equal(mapping, loaded)
        assert n_cb == loaded_n_cb

    def test_create_and_write_semi(self, mdef_file: Path, tmp_path: Path) -> None:
        """Test creating and writing a semi-continuous ts2cb from mdef."""
        output_path = tmp_path / "output.ts2cb"

        mapping, n_cb = ts2cb.create_and_write_ts2cb(mdef_file, output_path, TyingType.SEMI)

        assert output_path.exists()
        assert len(mapping) == 15
        assert n_cb == 1
        assert np.all(mapping == 0)
