"""View cepstral/MFCC feature files.

This module provides functionality to view/print MFCC feature files,
equivalent to the sphinx_cepview tool.

Two implementations:
1. Native Python (preferred) - using our existing MFC I/O
2. Shell-out to sphinx_cepview (for parity checking)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from st2.lib.features import read_sphinx_mfc

if TYPE_CHECKING:
    pass


def view_cepstra(
    mfc_path: Path,
    n_coeff: int = 13,
    display_cols: int = 10,
    start_frame: int = 0,
    end_frame: int | None = None,
    show_header: bool = True,
    show_frame_numbers: bool = True,
) -> str:
    """View cepstral features from an MFC file.

    Native Python implementation.

    Args:
        mfc_path: Path to the MFC file.
        n_coeff: Number of coefficients in the feature vector (for validation).
        display_cols: Number of columns to display per line.
        start_frame: Starting frame (0-based).
        end_frame: Ending frame (exclusive). None means all frames.
        show_header: Whether to show column headers.
        show_frame_numbers: Whether to show frame numbers.

    Returns:
        Formatted string representation of the features.
    """
    mfc = read_sphinx_mfc(mfc_path)
    n_frames, veclen = mfc.shape

    if veclen != n_coeff:
        # Just a warning, not an error - file may have different size
        pass

    if end_frame is None:
        end_frame = n_frames
    end_frame = min(end_frame, n_frames)

    display_cols = min(display_cols, veclen)

    lines = []

    # Header
    if show_header:
        header = f"{'frame#':>7}" if show_frame_numbers else ""
        for j in range(display_cols):
            header += f" c[{j:2d}]  "
        lines.append(header)

    # Data
    for i in range(start_frame, end_frame):
        line = f"{i:6d}:" if show_frame_numbers else ""
        for j in range(display_cols):
            line += f"{mfc[i, j]:7.3f} "
        lines.append(line)

    return "\n".join(lines)


def view_cepstra_shellout(
    mfc_path: Path,
    n_coeff: int = 13,
    display_cols: int = 10,
    start_frame: int = 0,
    end_frame: int | None = None,
    show_header: bool = True,
    show_frame_numbers: bool = True,
    bin_path: str | Path = "sphinx_cepview",
) -> str:
    """View cepstral features using sphinx_cepview binary.

    Shell-out implementation for parity checking.

    Args:
        mfc_path: Path to the MFC file.
        n_coeff: Number of coefficients in the feature vector.
        display_cols: Number of columns to display.
        start_frame: Starting frame (0-based).
        end_frame: Ending frame.
        show_header: Whether to show column headers.
        show_frame_numbers: Whether to show frame numbers.
        bin_path: Path to sphinx_cepview binary.

    Returns:
        Output from sphinx_cepview.

    Raises:
        subprocess.CalledProcessError: If sphinx_cepview fails.
        FileNotFoundError: If binary not found.
    """
    cmd = [
        str(bin_path),
        "-f",
        str(mfc_path),
        "-i",
        str(n_coeff),
        "-d",
        str(display_cols),
        "-b",
        str(start_frame),
    ]

    if end_frame is not None:
        cmd.extend(["-e", str(end_frame)])

    if show_header:
        cmd.extend(["-header", "1"])
    if show_frame_numbers:
        cmd.extend(["-describe", "1"])

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout


def check_parity(
    mfc_path: Path,
    n_coeff: int = 13,
    display_cols: int = 10,
    start_frame: int = 0,
    end_frame: int | None = None,
    bin_path: str | Path = "sphinx_cepview",
    tolerance: float = 1e-3,
) -> bool:
    """Check parity between Python and shell-out implementations.

    Args:
        mfc_path: Path to the MFC file.
        n_coeff: Number of coefficients.
        display_cols: Number of columns to display.
        start_frame: Starting frame.
        end_frame: Ending frame.
        bin_path: Path to sphinx_cepview binary.
        tolerance: Tolerance for floating point comparison.

    Returns:
        True if outputs match within tolerance.
    """
    import re

    py_output = view_cepstra(
        mfc_path,
        n_coeff=n_coeff,
        display_cols=display_cols,
        start_frame=start_frame,
        end_frame=end_frame,
        show_header=False,
        show_frame_numbers=False,
    )

    try:
        shell_output = view_cepstra_shellout(
            mfc_path,
            n_coeff=n_coeff,
            display_cols=display_cols,
            start_frame=start_frame,
            end_frame=end_frame,
            show_header=False,
            show_frame_numbers=False,
            bin_path=bin_path,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

    # Parse both outputs and compare numerically
    py_values = [float(x) for x in py_output.split()]
    shell_values = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", shell_output)]

    if len(py_values) != len(shell_values):
        return False

    return all(abs(pv - sv) <= tolerance for pv, sv in zip(py_values, shell_values, strict=False))
