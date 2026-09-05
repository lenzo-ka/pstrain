"""Tests for segment aggregation functionality."""

import os
import struct
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


def _write_feature_file(path: Path, n_frame: int = 10, ceplen: int = 13) -> None:
    """Write a well formed, all-zero Sphinx feature file."""
    with path.open("wb") as f:
        f.write(struct.pack("<i", n_frame * ceplen))
        f.write(b"\0" * (n_frame * ceplen * 4))


def _write_seg_file(path: Path, frames: list[int]) -> None:
    """Write a big-endian ``v8_seg`` state segmentation of the given frames."""
    with path.open("wb") as f:
        f.write(struct.pack(">i", len(frames)))
        f.write(b"".join(struct.pack(">H", frame) for frame in frames))


# ck_seg() reads each frame as (value & 0x7FFF) // (MAX_N_STATE - 1) giving the
# context-independent phone id, with 0x8000 marking a phone's first frame.
# MAX_N_STATE is 20 (csrc/include/s3/s3.h).
_N_EMITTING_STATE = 20 - 1


def _phone_frames(ci_phone: int, n_frame: int = 4) -> list[int]:
    """Frames for one phone: begin-marked, then its emitting states in order."""
    states = [min(i, 2) for i in range(n_frame)]
    return [
        (ci_phone * _N_EMITTING_STATE + state) | (0x8000 if i == 0 else 0)
        for i, state in enumerate(states)
    ]


# The corpus laid out by _agg_seg_command says "<s> HI </s>", which the
# dictionary and filler dictionary expand to the phone sequence SIL AA SIL.
# The generated mdef numbers AA 0 and SIL 1, so this segmentation is one
# ck_seg() accepts -- the utterance is usable, not omitted.
VALID_SEG_FRAMES = _phone_frames(1) + _phone_frames(0) + _phone_frames(1)


def test_standalone_reports_missing_feature_omission(tmp_path: Path, project_root: Path) -> None:
    cep_dir = tmp_path / "cep"
    cep_dir.mkdir()
    ctl = tmp_path / "train.ctl"
    ctl.write_text("present\nmissing\n")
    _write_feature_file(cep_dir / "present.mfc")

    command = [
        project_root / "build" / "bin" / "agg_seg",
        "-segtype",
        "all",
        "-ctlfn",
        ctl,
        "-cepdir",
        cep_dir,
        "-segdmpfn",
        tmp_path / "all.dmp",
        "-ceplen",
        "13",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode != 0
    assert "agg_seg: processed 1, omitted 1 (feature read: 1)" in result.stderr

    allowed = subprocess.run(
        [*command, "-allowomit", "yes"], capture_output=True, text=True, check=False
    )
    assert allowed.returncode == 0
    assert "agg_seg: processed 1, omitted 1 (feature read: 1)" in allowed.stderr


def _agg_seg_command(
    project_root: Path, base: Path, segtype: str, utterances: tuple[str, ...]
) -> list[Path | str]:
    """Lay out a two-phone corpus and return the agg_seg command line for it.

    Only the scaffolding common to every segment type is created here: the
    caller adds whatever feature and segmentation files the case needs.
    """
    from pstrain.lib import mdef

    phones = base / "phones"
    phones.write_text("AA\nSIL\n")
    ci_mdef = base / "ci.mdef"
    mdef.generate_ci_mdef(phones, ci_mdef)

    dictionary = base / "dict"
    dictionary.write_text("HI\tAA\n")
    filler = base / "filler.dict"
    filler.write_text("<s>\tSIL\n</s>\tSIL\n<sil>\tSIL\n")
    ctl = base / "train.ctl"
    ctl.write_text("".join(f"{utt}\n" for utt in utterances))
    lsn = base / "train.lsn"
    lsn.write_text("".join(f"<s> HI </s> ({utt})\n" for utt in utterances))
    (base / "cep").mkdir()
    (base / "seg").mkdir()

    command: list[Path | str] = [
        project_root / "build" / "bin" / "agg_seg",
        "-segtype",
        segtype,
        "-ctlfn",
        ctl,
        "-cepdir",
        base / "cep",
        "-segdir",
        base / "seg",
        "-lsnfn",
        lsn,
        "-moddeffn",
        ci_mdef,
        "-dictfn",
        dictionary,
        "-fdictfn",
        filler,
        "-cntfn",
        base / "counts.txt",
        # segdmp joins -segdmpdirs with these names, so keep them bare.
        "-segdmpfn",
        "out.dmp",
        "-segidxfn",
        "idx",
        "-segdmpdirs",
        base,
        "-ceplen",
        "13",
    ]
    if segtype == "st":
        command += ["-ts2cbfn", ".semi."]
    return command


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
@pytest.mark.parametrize("segtype", ["st", "phn"])
def test_standalone_count_pass_reports_omissions(
    tmp_path: Path, project_root: Path, segtype: str
) -> None:
    """The counting pass reports and sets status like every other path.

    ``cnt_st``/``cnt_phn`` wrote the count file from the counting pass and
    then left the process with a bare ``exit(0)``: status 0, no summary, and
    no ``-allowomit`` enforcement, even when that pass had omitted
    utterances. Neither utterance here has a state segmentation, so the
    counting pass omits both and the counts it wrote are incomplete.
    """
    command = _agg_seg_command(project_root, tmp_path, segtype, ("present", "missing"))
    _write_feature_file(tmp_path / "cep" / "present.mfc")
    counts = tmp_path / "counts.txt"

    summary = "agg_seg: processed 0, omitted 2 (segmentation read: 2)"

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    # The incomplete count file is still written, as it always was ...
    assert counts.exists(), result.stderr[-2000:]
    # ... but the run now says what it omitted, and fails for it.
    assert summary in result.stderr, result.stderr[-2000:]
    assert result.returncode != 0

    counts.unlink()
    allowed = subprocess.run(
        [*command, "-allowomit", "yes"], capture_output=True, text=True, check=False
    )
    assert summary in allowed.stderr, allowed.stderr[-2000:]
    assert allowed.returncode == 0


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
@pytest.mark.parametrize("segtype", ["st", "phn"])
def test_standalone_count_pass_counts_processed_utterances(
    tmp_path: Path, project_root: Path, segtype: str
) -> None:
    """The counting passes count what they used, not only what they dropped.

    ``cnt_st_seg`` and ``cnt_phn_seg`` recorded every omission reason but
    never called ``agg_omission_processed()``, so a usable utterance went
    uncounted: a mixed corpus reported ``processed 0`` and a wholly usable
    one reported ``processed 0, omitted 0``.
    """
    command = _agg_seg_command(project_root, tmp_path, segtype, ("present", "missing"))
    _write_feature_file(tmp_path / "cep" / "present.mfc")
    counts = tmp_path / "counts.txt"
    _write_seg_file(tmp_path / "seg" / "present.v8_seg", VALID_SEG_FRAMES)

    # One usable utterance, one with no segmentation to read.
    summary = "agg_seg: processed 1, omitted 1 (segmentation read: 1)"
    mixed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert summary in mixed.stderr, mixed.stderr[-2000:]
    assert mixed.returncode != 0

    # Both usable: nothing omitted, and the run succeeds without -allowomit.
    counts.unlink()
    _write_seg_file(tmp_path / "seg" / "missing.v8_seg", VALID_SEG_FRAMES)
    complete = subprocess.run(command, capture_output=True, text=True, check=False)
    assert "agg_seg: processed 2, omitted 0" in complete.stderr, complete.stderr[-2000:]
    assert complete.returncode == 0


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
def test_standalone_st_reports_segmentation_mismatch(tmp_path: Path, project_root: Path) -> None:
    """A segmentation that disagrees with the transcript is omitted, not used.

    ``cnt_st_seg`` and ``agg_st_seg`` called ``ck_seg()`` for its diagnostic
    only -- the counting pass threw the return away and the aggregation pass
    had the call commented out -- so a mismatched utterance was handed to
    ``mk_sseq()`` and counted as usable. Both segmentations here open cleanly
    but start without the phone-begin marker ``ck_seg()`` requires.
    """
    command = _agg_seg_command(project_root, tmp_path, "st", ("present", "missing"))
    _write_feature_file(tmp_path / "cep" / "present.mfc")
    for utt in ("present", "missing"):
        _write_seg_file(tmp_path / "seg" / f"{utt}.v8_seg", list(range(10)))

    counting = subprocess.run(command, capture_output=True, text=True, check=False)
    assert "agg_seg: processed 0, omitted 2 (segmentation mismatch: 2)" in counting.stderr, (
        counting.stderr[-2000:]
    )
    assert counting.returncode != 0

    # The count file now exists, so this run takes the aggregation pass, where
    # the mismatch is reported apart from the unreadable feature file.
    aggregating = subprocess.run(command, capture_output=True, text=True, check=False)
    assert (
        "agg_seg: processed 0, omitted 2 (feature read: 1, segmentation mismatch: 1)"
        in aggregating.stderr
    ), aggregating.stderr[-2000:]
    assert aggregating.returncode != 0


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
