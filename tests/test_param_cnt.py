"""Tests for parameter counting functionality."""

from pathlib import Path

import pytest

from pstrain.lib.native_worker import PstrainNativeError
from pstrain.lib.param_cnt import ParamType, count_params

# Check if library exists
# libpstrainc availability comes from the shared helper (real loader-based
# detection); see tests/clib.py.
from tests.clib import C_LIBRARY_AVAILABLE as _lib_exists


class TestParamType:
    """Tests for ParamType enum."""

    def test_enum_values(self) -> None:
        """Test that enum values match C constants."""
        assert ParamType.STATE.value == 0
        assert ParamType.CB.value == 1
        assert ParamType.PHONE.value == 2

    def test_enum_from_string(self) -> None:
        """Test creating enum from string."""
        assert ParamType["STATE"] == ParamType.STATE
        assert ParamType["CB"] == ParamType.CB
        assert ParamType["PHONE"] == ParamType.PHONE


class TestCountParamsValidation:
    """Tests for argument validation in count_params."""

    def test_state_requires_seg_dir(self, tmp_path: Path) -> None:
        """Test that STATE mode requires seg_dir."""
        mdef = tmp_path / "mdef"
        mdef.touch()
        dict_file = tmp_path / "dict"
        dict_file.touch()
        ctl = tmp_path / "test.ctl"
        ctl.touch()
        lsn = tmp_path / "test.lsn"
        lsn.touch()

        with pytest.raises(ValueError, match="seg_dir"):
            count_params(
                mdef_path=mdef,
                dict_path=dict_file,
                ctl_path=ctl,
                lsn_path=lsn,
                param_type=ParamType.STATE,
                seg_dir=None,
            )

    def test_cb_requires_seg_dir(self, tmp_path: Path) -> None:
        """Test that CB mode requires seg_dir."""
        mdef = tmp_path / "mdef"
        mdef.touch()
        dict_file = tmp_path / "dict"
        dict_file.touch()
        ctl = tmp_path / "test.ctl"
        ctl.touch()
        lsn = tmp_path / "test.lsn"
        lsn.touch()

        with pytest.raises(ValueError, match="seg_dir"):
            count_params(
                mdef_path=mdef,
                dict_path=dict_file,
                ctl_path=ctl,
                lsn_path=lsn,
                param_type=ParamType.CB,
                seg_dir=None,
                ts2cb_path=".cont.",
            )

    def test_cb_requires_ts2cb_path(self, tmp_path: Path) -> None:
        """Test that CB mode requires ts2cb_path."""
        mdef = tmp_path / "mdef"
        mdef.touch()
        dict_file = tmp_path / "dict"
        dict_file.touch()
        ctl = tmp_path / "test.ctl"
        ctl.touch()
        lsn = tmp_path / "test.lsn"
        lsn.touch()
        seg_dir = tmp_path / "seg"
        seg_dir.mkdir()

        with pytest.raises(ValueError, match="ts2cb_path"):
            count_params(
                mdef_path=mdef,
                dict_path=dict_file,
                ctl_path=ctl,
                lsn_path=lsn,
                param_type=ParamType.CB,
                seg_dir=seg_dir,
                ts2cb_path=None,
            )

    @pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
    def test_phone_does_not_require_seg_dir(self, tmp_path: Path) -> None:
        """Test PHONE counting through the C library without segmentations."""
        fixture = Path(__file__).parent / "fixtures" / "multipron_final_state"
        ctl = tmp_path / "test.ctl"
        ctl.write_text("test\n")
        lsn = tmp_path / "test.lsn"
        lsn.write_text("a\n")
        output = tmp_path / "phone.counts"

        count_params(
            mdef_path=fixture / "model" / "mdef",
            dict_path=fixture / "dictionary.dict",
            ctl_path=ctl,
            lsn_path=lsn,
            output_path=output,
            param_type=ParamType.PHONE,
            seg_dir=None,
        )

        assert output.read_text().splitlines()

    @pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
    def test_empty_control_file_is_unsuccessful(self, tmp_path: Path) -> None:
        """Test that corpus setup failures cross the CFFI boundary."""
        fixture = Path(__file__).parent / "fixtures" / "multipron_final_state"
        ctl = tmp_path / "empty.ctl"
        ctl.write_text("")
        lsn = tmp_path / "empty.lsn"
        lsn.write_text("")
        output = tmp_path / "phone.counts"

        with pytest.raises(PstrainNativeError):
            count_params(
                mdef_path=fixture / "model" / "mdef",
                dict_path=fixture / "dictionary.dict",
                ctl_path=ctl,
                lsn_path=lsn,
                output_path=output,
                param_type=ParamType.PHONE,
            )

        assert not output.exists() or output.read_text() == ""


class TestCountParamsStringConversion:
    """Tests for string-to-enum conversion."""

    def test_string_to_enum_state(self) -> None:
        """Test that 'state' string is converted to ParamType.STATE."""
        # Test the conversion logic
        param_type = "state"
        result = ParamType[param_type.upper()]
        assert result == ParamType.STATE

    def test_string_to_enum_cb(self) -> None:
        """Test that 'cb' string is converted to ParamType.CB."""
        param_type = "cb"
        result = ParamType[param_type.upper()]
        assert result == ParamType.CB

    def test_string_to_enum_phone(self) -> None:
        """Test that 'phone' string is converted to ParamType.PHONE."""
        param_type = "phone"
        result = ParamType[param_type.upper()]
        assert result == ParamType.PHONE


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestCffiIntegration:
    """Tests that CFFI functions exist and are callable."""

    def test_cffi_function_exists(self) -> None:
        """Test that pstrain_param_cnt function is declared in CFFI."""
        from pstrain.lib import _pstrainc

        lib = _pstrainc.get_lib()
        assert hasattr(lib, "pstrain_param_cnt")

    def test_cffi_constants_exist(self) -> None:
        """Test that PARAM_CNT_* constants are declared in CFFI."""
        from pstrain.lib import _pstrainc

        lib = _pstrainc.get_lib()
        assert lib.PARAM_CNT_STATE == 0
        assert lib.PARAM_CNT_CB == 1
        assert lib.PARAM_CNT_PHONE == 2
