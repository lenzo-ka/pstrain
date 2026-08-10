"""Tests for segment aggregation functionality."""

import os
import subprocess
import sys
import textwrap
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


_COUNT_PATH_SCRIPT = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    from pstrain.lib import mdef
    from pstrain.lib.agg_seg import SegType, aggregate_segments


    def main() -> None:
        base = Path(sys.argv[1])
        segtype = SegType[sys.argv[2].upper()]
        phones = base / "phones"
        phones.write_text("AA\\nSIL\\n")
        ci_mdef = base / "ci.mdef"
        mdef.generate_ci_mdef(phones, ci_mdef)
        dictionary = base / "dict"
        dictionary.write_text("HI\\tAA\\n")
        ctl = base / "train.ctl"
        ctl.write_text("")
        (base / "cep").mkdir()
        (base / "seg").mkdir()
        aggregate_segments(
            ctl_path=ctl,
            cep_dir=base / "cep",
            output_path=base / "out.dmp",
            segtype=segtype,
            mdef_path=ci_mdef,
            dict_path=dictionary if segtype is SegType.PHN else None,
            ts2cb_path=".semi." if segtype is SegType.ST else None,
            seg_dir=base / "seg",
            cnt_path=base / "counts.txt",
            index_path=base / "idx",
        )


    if __name__ == "__main__":
        main()
    """
)


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
@pytest.mark.parametrize("segtype", ["st", "phn"])
def test_count_path_returns_instead_of_exiting_under_library_build(
    tmp_path: Path, project_root: Path, segtype: str
) -> None:
    """agg_seg's count path no longer ends the process with a bare exit(0).

    ``cnt_st``/``cnt_phn`` wrote the count file and then called ``exit(0)`` on
    a normal first run -- a successful-looking status 0, no diagnostic, with
    the caller's work silently unfinished. Under ``PSTRAIN_LIBRARY_BUILD``
    those sites are compiled out, so control returns to ``agg_seg_run``.

    Run in a child process precisely because the old behaviour was to exit:
    an ``exit(0)`` inside pytest would end the test session looking green.
    The child's status is therefore the discriminator -- it must not be 0.

    The run still fails afterwards, for an unrelated reason outside this
    lane's scope: ``pstrain_agg_seg`` never supplies ``-segdmpdirs``, and
    ``segdmp_open_write`` walks that NULL list.
    """
    script = tmp_path / "count_path.py"
    script.write_text(_COUNT_PATH_SCRIPT)
    workdir = tmp_path / "work"
    workdir.mkdir()

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(project_root), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    result = subprocess.run(
        [sys.executable, str(script), str(workdir), segtype],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        env=env,
        timeout=300,
        check=False,
    )

    counts = workdir / "counts.txt"
    assert "not found; creating" in result.stderr, result.stderr[-2000:]
    assert counts.exists(), result.stderr[-2000:]
    assert counts.read_text() != ""
    # exit(0) at the count site would have produced a clean status 0 here.
    assert result.returncode != 0
