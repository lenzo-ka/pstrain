"""Failure-status contracts for standalone C entry points."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

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
    ctl.write_text("test\n")
    lsn = tmp_path / "empty.lsn"
    lsn.write_text("a\n")
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


@pytest.mark.parametrize(
    ("contents", "name"),
    [
        ("   #comment\n", "indented-comment"),
        ("  \t  \n", "whitespace"),
        ("", "empty"),
    ],
)
def test_param_cnt_zero_utterance_control_is_unsuccessful(
    tmp_path: Path, contents: str, name: str
) -> None:
    ctl = tmp_path / f"{name}.ctl"
    args = _param_cnt_args(tmp_path, "phone")
    args[args.index("-ctlfn") + 1] = str(ctl)
    ctl.write_text(contents)
    output = tmp_path / "counts"
    result = _run(
        "param_cnt",
        *args,
        "-outputfn",
        str(output),
    )

    assert result.returncode != 0, result.stderr
    assert f"-ctlfn {ctl}" in result.stderr
    assert result.stdout == ""
    assert not output.exists()


@pytest.mark.parametrize("ctl_path", [Path("/dev/null")])
def test_param_cnt_non_file_control_is_unsuccessful(tmp_path: Path, ctl_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX control-file behavior")
    args = _param_cnt_args(tmp_path, "phone")
    args[args.index("-ctlfn") + 1] = str(ctl_path)
    output = tmp_path / "counts"

    result = _run("param_cnt", *args, "-outputfn", str(output))

    assert result.returncode != 0, result.stderr
    assert not output.exists()


def test_param_cnt_directory_control_is_unsuccessful(tmp_path: Path) -> None:
    ctl = tmp_path / "control-directory"
    ctl.mkdir()
    args = _param_cnt_args(tmp_path, "phone")
    args[args.index("-ctlfn") + 1] = str(ctl)
    output = tmp_path / "counts"

    result = _run("param_cnt", *args, "-outputfn", str(output))

    assert result.returncode != 0, result.stderr
    assert not output.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_param_cnt_unreadable_control_is_unsuccessful(tmp_path: Path) -> None:
    ctl = tmp_path / "unreadable.ctl"
    ctl.write_text("test\n")
    ctl.chmod(0)
    args = _param_cnt_args(tmp_path, "phone")
    args[args.index("-ctlfn") + 1] = str(ctl)
    output = tmp_path / "counts"

    try:
        result = _run("param_cnt", *args, "-outputfn", str(output))
    finally:
        ctl.chmod(0o600)

    assert result.returncode != 0, result.stderr
    assert not output.exists()


def test_param_cnt_malformed_second_control_line_is_unsuccessful(tmp_path: Path) -> None:
    ctl = tmp_path / "malformed.ctl"
    args = _param_cnt_args(tmp_path, "phone")
    args[args.index("-ctlfn") + 1] = str(ctl)
    ctl.write_text("test\ngarbage ???\n")
    output = tmp_path / "counts"

    result = _run("param_cnt", *args, "-outputfn", str(output))

    assert result.returncode != 0, result.stderr
    assert not output.exists()


def test_param_cnt_rejects_unknown_parameter_type(tmp_path: Path) -> None:
    result = _run("param_cnt", *_param_cnt_args(tmp_path, "unknown"))

    assert result.returncode == 1, result.stderr
    assert "Unknown parameter type 'unknown'; expected state, cb, or phone" in result.stderr


def test_param_cnt_cb_requires_ts2cb_mapping(tmp_path: Path) -> None:
    result = _run("param_cnt", *_param_cnt_args(tmp_path, "cb"))

    assert result.returncode == 1, result.stderr
    assert "CB parameter counting requires -ts2cbfn" in result.stderr
