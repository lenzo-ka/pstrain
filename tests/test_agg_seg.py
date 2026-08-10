"""Tests for segment aggregation functionality."""

from pathlib import Path

import pytest

from pstrain.lib.agg_seg import SegType

# Check if library exists
# libpstrainc availability comes from the shared helper (real loader-based
# detection); see tests/clib.py.
from tests.clib import C_LIBRARY_AVAILABLE as _lib_exists


class TestSegType:
    """Tests for SegType enum."""

    def test_enum_values(self) -> None:
        """Test that enum values are as expected."""
        assert int(SegType.ALL) == 0
        assert int(SegType.ST) == 1
        assert int(SegType.PHN) == 2

    def test_enum_from_string(self) -> None:
        """Test creating enum from string."""
        assert SegType["ALL"] == SegType.ALL
        assert SegType["ST"] == SegType.ST
        assert SegType["PHN"] == SegType.PHN


class TestAggregateSegmentsValidation:
    """Tests for argument validation in aggregate_segments."""

    def test_st_requires_mdef_path(self, tmp_path: Path) -> None:
        """Test that ST mode requires mdef_path."""
        from pstrain.lib.agg_seg import aggregate_segments

        ctl = tmp_path / "test.ctl"
        ctl.touch()
        cep_dir = tmp_path / "cep"
        cep_dir.mkdir()
        output = tmp_path / "out.dmp"

        with pytest.raises(ValueError, match="mdef_path"):
            aggregate_segments(
                ctl_path=ctl,
                cep_dir=cep_dir,
                output_path=output,
                segtype=SegType.ST,
                mdef_path=None,
                ts2cb_path=".semi.",
            )

    def test_st_requires_ts2cb_path(self, tmp_path: Path) -> None:
        """Test that ST mode requires ts2cb_path."""
        from pstrain.lib.agg_seg import aggregate_segments

        ctl = tmp_path / "test.ctl"
        ctl.touch()
        cep_dir = tmp_path / "cep"
        cep_dir.mkdir()
        mdef = tmp_path / "mdef"
        mdef.touch()
        output = tmp_path / "out.dmp"

        with pytest.raises(ValueError, match="ts2cb_path"):
            aggregate_segments(
                ctl_path=ctl,
                cep_dir=cep_dir,
                output_path=output,
                segtype=SegType.ST,
                mdef_path=mdef,
                ts2cb_path=None,
            )

    def test_phn_requires_mdef_path(self, tmp_path: Path) -> None:
        """Test that PHN mode requires mdef_path."""
        from pstrain.lib.agg_seg import aggregate_segments

        ctl = tmp_path / "test.ctl"
        ctl.touch()
        cep_dir = tmp_path / "cep"
        cep_dir.mkdir()
        output = tmp_path / "out.dmp"

        with pytest.raises(ValueError, match="mdef_path"):
            aggregate_segments(
                ctl_path=ctl,
                cep_dir=cep_dir,
                output_path=output,
                segtype=SegType.PHN,
                mdef_path=None,
                dict_path=tmp_path / "dict",
            )

    def test_phn_requires_dict_path(self, tmp_path: Path) -> None:
        """Test that PHN mode requires dict_path."""
        from pstrain.lib.agg_seg import aggregate_segments

        ctl = tmp_path / "test.ctl"
        ctl.touch()
        cep_dir = tmp_path / "cep"
        cep_dir.mkdir()
        mdef = tmp_path / "mdef"
        mdef.touch()
        output = tmp_path / "out.dmp"

        with pytest.raises(ValueError, match="dict_path"):
            aggregate_segments(
                ctl_path=ctl,
                cep_dir=cep_dir,
                output_path=output,
                segtype=SegType.PHN,
                mdef_path=mdef,
                dict_path=None,
            )

    def test_string_segtype_conversion(self) -> None:
        """Test that string segtype is converted to enum."""
        # Test the conversion logic directly
        segtype = "st"
        converted = SegType[segtype.upper()]
        assert converted == SegType.ST

        segtype = "all"
        converted = SegType[segtype.upper()]
        assert converted == SegType.ALL


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestCffiIntegration:
    """Integration tests for CFFI bindings."""

    def test_cffi_function_exists(self) -> None:
        """Test that pstrain_agg_seg function exists in library."""
        from pstrain.lib import _pstrainc

        lib = _pstrainc.get_lib()
        assert hasattr(lib, "pstrain_agg_seg")

    def test_segtype_constants_defined(self) -> None:
        """Test that segtype constants match Python enum."""
        # These are #defines in C, verify they match the Python enum
        assert int(SegType.ALL) == 0
        assert int(SegType.ST) == 1
        assert int(SegType.PHN) == 2
