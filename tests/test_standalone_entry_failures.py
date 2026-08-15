"""Failure-status contracts for standalone C entry points."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pstrain.lib.commands import resolve_binary


def _run(binary: str, *args: str) -> subprocess.CompletedProcess[str]:
    executable = resolve_binary(binary)
    assert executable is not None
    return subprocess.run(
        [str(executable), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _param_cnt_args(tmp_path: Path, param_type: str) -> list[str]:
    fixture = Path(__file__).parent / "fixtures" / "multipron_final_state"
    ctl = tmp_path / "empty.ctl"
    ctl.write_text("")
    lsn = tmp_path / "empty.lsn"
    lsn.write_text("")
    segdir = tmp_path / "seg"
    segdir.mkdir()
    return [
        "-moddeffn",
        str(fixture / "model" / "mdef"),
        "-dictfn",
        str(fixture / "dictionary.dict"),
        "-ctlfn",
        str(ctl),
        "-lsnfn",
        str(lsn),
        "-segdir",
        str(segdir),
        "-paramtype",
        param_type,
    ]


def test_param_cnt_output_open_failure_is_unsuccessful(tmp_path: Path) -> None:
    output = tmp_path / "missing" / "counts"
    result = _run(
        "param_cnt",
        *_param_cnt_args(tmp_path, "phone"),
        "-outputfn",
        str(output),
    )

    assert result.returncode == 1, result.stderr
    assert f"Couldn't open {output} for writing" in result.stderr


def test_param_cnt_rejects_unknown_parameter_type(tmp_path: Path) -> None:
    result = _run("param_cnt", *_param_cnt_args(tmp_path, "unknown"))

    assert result.returncode == 1, result.stderr
    assert "Unknown parameter type 'unknown'; expected state, cb, or phone" in result.stderr


def test_param_cnt_cb_requires_ts2cb_mapping(tmp_path: Path) -> None:
    result = _run("param_cnt", *_param_cnt_args(tmp_path, "cb"))

    assert result.returncode == 1, result.stderr
    assert "CB parameter counting requires -ts2cbfn" in result.stderr


def test_mllr_transform_parse_failure_is_unsuccessful() -> None:
    result = _run("mllr_transform", "-not-a-real-option", "yes")

    assert result.returncode == 1, result.stderr
    assert "Errors initializing" in result.stderr
