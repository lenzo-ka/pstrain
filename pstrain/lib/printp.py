"""Print acoustic model parameters.

This module provides functionality to print model parameters in human-readable
format, equivalent to the printp tool.

Two implementations:
1. Native Python (preferred) - using our existing I/O functions
2. Shell-out to printp (for parity checking)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from pstrain.lib import _pstrainc

if TYPE_CHECKING:
    pass


def format_gau(
    gau_path: Path,
    sigfig: int = 6,
) -> str:
    """Format Gaussian parameters (means or variances) as text.

    Args:
        gau_path: Path to the Gaussian file.
        sigfig: Number of significant figures.

    Returns:
        Formatted string representation.
    """
    gau, n_mgau, n_feat, n_density, veclens = _pstrainc.read_gau(str(gau_path))

    lines = []
    lines.append(f"param {n_mgau} {n_feat} {n_density}")

    fmt = f"{{:.{sigfig - 1}e}} "

    for i in range(n_mgau):
        lines.append(f"mgau {i}")
        for j in range(n_feat):
            lines.append(f"feat {j}")
            veclen = veclens[j]
            for k in range(n_density):
                line = f"density {k:4d} "
                for m in range(veclen):
                    line += fmt.format(gau[i, j, k, m])
                lines.append(line)

    return "\n".join(lines)


def format_mixw(
    mixw_path: Path,
    sigfig: int = 6,
    normalize: bool = False,
) -> str:
    """Format mixture weights as text.

    Args:
        mixw_path: Path to the mixture weights file.
        sigfig: Number of significant figures.
        normalize: Whether to normalize weights.

    Returns:
        Formatted string representation.
    """
    mixw, n_mixw, n_feat, n_density = _pstrainc.read_mixw(str(mixw_path))

    lines = []
    lines.append(f"mixw {n_mixw} {n_feat} {n_density}")

    fmt = f"{{:.{sigfig - 1}e}} "

    for i in range(n_mixw):
        for j in range(n_feat):
            total = float(mixw[i, j, :].sum())
            lines.append(f"mixw [{i} {j}] {total:.6e}")

            norm = total if normalize and total > 0 else 1.0
            line = "\n\t"
            for k in range(n_density):
                if k > 0 and k % 8 == 0:
                    line += "\n\t"
                line += fmt.format(mixw[i, j, k] / norm)
            lines.append(line)

    return "\n".join(lines)


def format_tmat(
    tmat_path: Path,
    sigfig: int = 6,
    normalize: bool = False,
) -> str:
    """Format transition matrices as text.

    Args:
        tmat_path: Path to the transition matrix file.
        sigfig: Number of significant figures.
        normalize: Whether to normalize probabilities.

    Returns:
        Formatted string representation.
    """
    tmat, n_tmat, n_state = _pstrainc.read_tmat(str(tmat_path))

    lines = []
    lines.append(f"tmat {n_tmat} {n_state}")

    fmt = f" {{:.{sigfig - 1}e}}"

    for t in range(n_tmat):
        lines.append(f"tmat [{t}]")

        for i in range(n_state - 1):
            if normalize:
                total = float(tmat[t, i, :].sum())
                norm = total if total > 0 else 1.0
            else:
                norm = 1.0

            line = ""
            for j in range(n_state):
                val = tmat[t, i, j] / norm
                if val > 0:
                    line += fmt.format(val)
                else:
                    line += " " * 9
            lines.append(line)

    return "\n".join(lines)


def format_ts2cb(ts2cb_path: Path) -> str:
    """Format tied-state to codebook mapping as text.

    Args:
        ts2cb_path: Path to the ts2cb file.

    Returns:
        Formatted string representation.
    """
    from pstrain.lib.ts2cb import read_ts2cb

    ts2cb, n_cb = read_ts2cb(ts2cb_path)

    lines = []
    for i, cb in enumerate(ts2cb):
        lines.append(f"{i}: {cb}")

    return "\n".join(lines)


def print_params(
    mixw_path: Path | None = None,
    tmat_path: Path | None = None,
    gau_path: Path | None = None,
    ts2cb_path: Path | None = None,
    sigfig: int = 6,
    normalize: bool = False,
) -> str:
    """Print model parameters.

    Native Python implementation of printp.

    Args:
        mixw_path: Path to mixture weights file.
        tmat_path: Path to transition matrix file.
        gau_path: Path to Gaussian (mean/var) file.
        ts2cb_path: Path to ts2cb mapping file.
        sigfig: Number of significant figures.
        normalize: Whether to normalize probabilities.

    Returns:
        Formatted string representation of all specified parameters.
    """
    parts = []

    if mixw_path:
        parts.append(format_mixw(mixw_path, sigfig=sigfig, normalize=normalize))

    if tmat_path:
        parts.append(format_tmat(tmat_path, sigfig=sigfig, normalize=normalize))

    if gau_path:
        parts.append(format_gau(gau_path, sigfig=sigfig))

    if ts2cb_path:
        parts.append(format_ts2cb(ts2cb_path))

    return "\n\n".join(parts)


def print_params_shellout(
    mixw_path: Path | None = None,
    tmat_path: Path | None = None,
    gau_path: Path | None = None,
    ts2cb_path: Path | None = None,
    sigfig: int = 6,
    normalize: bool = False,
    bin_path: str | Path = "printp",
) -> str:
    """Print model parameters using printp binary.

    Shell-out implementation for parity checking.

    Args:
        mixw_path: Path to mixture weights file.
        tmat_path: Path to transition matrix file.
        gau_path: Path to Gaussian (mean/var) file.
        ts2cb_path: Path to ts2cb mapping file.
        sigfig: Number of significant figures.
        normalize: Whether to normalize.
        bin_path: Path to printp binary.

    Returns:
        Output from printp.

    Raises:
        subprocess.CalledProcessError: If printp fails.
        FileNotFoundError: If binary not found.
    """
    cmd = [str(bin_path), "-sigfig", str(sigfig)]

    if normalize:
        cmd.extend(["-norm", "1"])

    if mixw_path:
        cmd.extend(["-mixwfn", str(mixw_path)])
    if tmat_path:
        cmd.extend(["-tmatfn", str(tmat_path)])
    if gau_path:
        cmd.extend(["-gaufn", str(gau_path)])
    if ts2cb_path:
        cmd.extend(["-ts2cbfn", str(ts2cb_path)])

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout
