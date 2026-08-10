"""Flat model initialization.

Creates initial HMM parameters using CFFI bindings.
Wraps mk_flat (tmat/mixw) and init_gau (means/variances).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from pstrain.lib import _pstrainc
from pstrain.lib.mdef import generate_ci_mdef

__all__ = [
    "create_mdef",
    "create_topology_file",
    "create_transition_matrices",
    "create_mixture_weights",
    "normalize_gaussians",
    "init_gaussians",
    "init_flat_model",
]


def create_mdef(
    phones: list[str],
    n_state: int = 3,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Create model definition file using CFFI.

    Args:
        phones: List of phone names (including SIL and fillers)
        n_state: Number of emitting states per phone
        output_path: If provided, write mdef to this file

    Returns:
        Dict with model definition info
    """
    if output_path is None:
        raise ValueError("output_path is required")

    # Write phones to temp file for CFFI function
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(phones) + "\n")
        phone_list_path = Path(f.name)

    try:
        generate_ci_mdef(phone_list_path, output_path, n_state)
    finally:
        phone_list_path.unlink()

    # Parse mdef to get stats
    n_tied_state = len(phones) * n_state

    return {
        "n_phones": len(phones),
        "n_state": n_state,
        "n_tied_state": n_tied_state,
        "phones": phones,
    }


def create_topology_file(
    n_state: int = 3,
    output_path: Path | None = None,
) -> str:
    """Create topology file for left-to-right HMM.

    Args:
        n_state: Number of emitting states per phone
        output_path: If provided, write topology file

    Returns:
        Topology file content
    """
    # Topology file format:
    # 0.1  (version)
    # n_state+1  (total states including exit)
    # transition matrix rows (n_state rows, each with n_state+1 values)
    n_total = n_state + 1  # emitting states + exit

    lines = ["0.1", str(n_total)]

    # Left-to-right: self-loop (0.75) + forward (0.25) - matches SphinxTrain
    # SphinxTrain uses 3:1 ratio which normalizes to 0.75:0.25
    for i in range(n_state):
        row = ["0.0"] * n_total
        row[i] = "0.75"  # self-loop
        row[i + 1] = "0.25"  # forward
        lines.append(" ".join(row))

    content = "\n".join(lines) + "\n"

    if output_path:
        output_path.write_text(content)

    return content


def create_transition_matrices(
    mdef_path: Path,
    topo_path: Path,
    output_path: Path,
) -> None:
    """Create transition matrices using C library via CFFI.

    Args:
        mdef_path: Path to model definition file
        topo_path: Path to topology file
        output_path: Output path for transition matrices
    """
    lib = _pstrainc.get_lib()
    ret = lib.pstrain_flat_tmat(
        str(mdef_path).encode(),
        str(topo_path).encode(),
        str(output_path).encode(),
    )
    if ret != 0:
        raise RuntimeError(f"pstrain_flat_tmat failed with code {ret}")


def create_mixture_weights(
    n_tied_state: int,
    n_stream: int = 1,
    n_density: int = 1,
    output_path: Path | None = None,
) -> None:
    """Create uniform mixture weights using C library via CFFI.

    Args:
        n_tied_state: Number of tied states (senones)
        n_stream: Number of feature streams
        n_density: Number of Gaussians per state
        output_path: Output path for mixture weights
    """
    if output_path is None:
        raise ValueError("output_path is required")

    lib = _pstrainc.get_lib()
    ret = lib.pstrain_flat_mixw(
        n_tied_state,
        n_stream,
        n_density,
        str(output_path).encode(),
    )
    if ret != 0:
        raise RuntimeError(f"pstrain_flat_mixw failed with code {ret}")


def normalize_gaussians(
    accum_dir: Path,
    mean_path: Path | None = None,
    var_path: Path | None = None,
) -> None:
    """Normalize accumulated counts to get model parameters via CFFI.

    Args:
        accum_dir: Directory containing accumulator files from init_gaussians
        mean_path: Output path for means (None to skip)
        var_path: Output path for variances (None to skip)
    """
    lib = _pstrainc.get_lib()
    ret = lib.pstrain_norm_gau(
        str(accum_dir).encode(),
        _pstrainc.path_or_null(mean_path),
        _pstrainc.path_or_null(var_path),
    )
    if ret != 0:
        raise RuntimeError(f"pstrain_norm_gau failed with code {ret}")


def init_gaussians(
    ctl_path: Path,
    cep_dir: Path,
    accum_dir: Path,
    mdef_path: Path | None = None,
    dict_path: Path | None = None,
    filler_dict_path: Path | None = None,
    feat_type: str = "1s_c_d_dd",
    ceplen: int = 13,
    cep_ext: str = ".mfc",
    lsn_path: Path | None = None,
    seg_dir: Path | None = None,
    seg_ext: str | None = None,
    mean_path: Path | None = None,
) -> None:
    """Initialize Gaussian parameters from feature data via CFFI.

    First call (mean_path=None): computes means
    Second call (mean_path=computed_means): computes variances

    Args:
        ctl_path: Path to control file (list of utterances)
        cep_dir: Directory containing feature files
        accum_dir: Directory to write accumulator files
        mdef_path: Path to model definition (None for global mode)
        dict_path: Path to dictionary
        filler_dict_path: Path to filler dictionary
        feat_type: Feature type string
        ceplen: Cepstral dimension
        cep_ext: Feature file extension
        lsn_path: Path to transcription file
        seg_dir: Directory containing segmentation files
        seg_ext: Segmentation file extension
        mean_path: Path to means (for variance computation)
    """
    accum_dir = Path(accum_dir)
    accum_dir.mkdir(parents=True, exist_ok=True)

    lib = _pstrainc.get_lib()
    ffi = _pstrainc.get_ffi()
    # C code adds the dot, so strip it if present
    ext = cep_ext.lstrip(".")
    ret = lib.pstrain_init_gau(
        _pstrainc.path_or_null(mdef_path),
        _pstrainc.path_or_null(dict_path),
        _pstrainc.path_or_null(filler_dict_path),
        feat_type.encode(),
        ceplen,
        str(ctl_path).encode(),
        str(cep_dir).encode(),
        ext.encode(),
        _pstrainc.path_or_null(lsn_path),
        _pstrainc.path_or_null(seg_dir),
        seg_ext.encode() if seg_ext else ffi.NULL,
        str(accum_dir).encode(),
        _pstrainc.path_or_null(mean_path),
    )
    if ret != 0:
        raise RuntimeError(f"pstrain_init_gau failed with code {ret}")


def init_flat_model(
    phones: list[str],
    output_dir: Path,
    n_state: int = 3,
    n_density: int = 1,
    ctl_path: Path | None = None,
    cep_dir: Path | None = None,
    cep_ext: str = ".mfc",
    feat_type: str = "1s_c_d_dd",
    ceplen: int = 13,
) -> dict[str, Path]:
    """Initialize a complete flat model.

    If ctl_path and cep_dir are provided, computes means/variances from
    the feature data using init_gau. Otherwise creates placeholder files.

    Args:
        phones: List of phone names
        output_dir: Directory to write model files
        n_state: Number of emitting states per phone
        n_density: Number of Gaussians per state
        ctl_path: Control file for feature computation
        cep_dir: Feature directory for computation
        cep_ext: Feature file extension
        feat_type: Feature type string
        ceplen: Cepstral dimension

    Returns:
        Dict mapping file type to path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model definition
    mdef_info = create_mdef(phones, n_state, output_dir / "mdef")
    n_tied_state = mdef_info["n_tied_state"]

    # Create topology file
    topo_path = output_dir / "topo"
    create_topology_file(n_state, topo_path)

    # Create transition matrices via CFFI
    tmat_path = output_dir / "transition_matrices"
    create_transition_matrices(output_dir / "mdef", topo_path, tmat_path)

    # Create mixture weights via CFFI
    mixw_path = output_dir / "mixture_weights"
    create_mixture_weights(n_tied_state, n_stream=1, n_density=n_density, output_path=mixw_path)

    mean_path = output_dir / "means"
    var_path = output_dir / "variances"

    import numpy as np

    # Feature dimension
    n_feat = ceplen * 3  # cepstra + delta + delta-delta (for 1s_c_d_dd)

    # Initialize Gaussian parameters
    if ctl_path and cep_dir:
        # Compute global mean/variance from features
        accum_dir = output_dir / "accum"
        global_mean_path = output_dir / "global_mean"
        global_var_path = output_dir / "global_var"

        # First pass: accumulate for means
        init_gaussians(
            ctl_path=ctl_path,
            cep_dir=cep_dir,
            accum_dir=accum_dir,
            feat_type=feat_type,
            ceplen=ceplen,
            cep_ext=cep_ext,
        )

        # Normalize to get global mean
        normalize_gaussians(accum_dir, mean_path=global_mean_path)

        # Second pass: accumulate for variances (using computed means)
        accum_dir_var = output_dir / "accum_var"
        init_gaussians(
            ctl_path=ctl_path,
            cep_dir=cep_dir,
            accum_dir=accum_dir_var,
            feat_type=feat_type,
            ceplen=ceplen,
            cep_ext=cep_ext,
            mean_path=global_mean_path,
        )

        # Normalize to get global variance
        normalize_gaussians(accum_dir_var, var_path=global_var_path)

        # Read global mean/variance (1 codebook x 1 stream x 1 density x D)
        global_mean, _, _, _, _ = _pstrainc.read_gau(str(global_mean_path))
        global_var, _, _, _, _ = _pstrainc.read_gau(str(global_var_path))

        # Replicate to all tied states for flat start
        flat_means = np.tile(global_mean, (n_tied_state, 1, n_density, 1))
        flat_vars = np.tile(global_var, (n_tied_state, 1, n_density, 1))

        _pstrainc.write_gau(str(mean_path), flat_means)
        _pstrainc.write_gau(str(var_path), flat_vars)
    else:
        # Create placeholder files - uniform means and unit variances

        # Create placeholder means (zeros)
        placeholder_means = np.zeros((n_tied_state, 1, n_density, n_feat), dtype=np.float32)
        _pstrainc.write_gau(str(mean_path), placeholder_means)

        # Create placeholder variances (ones)
        placeholder_vars = np.ones((n_tied_state, 1, n_density, n_feat), dtype=np.float32)
        _pstrainc.write_gau(str(var_path), placeholder_vars)

    return {
        "mdef": output_dir / "mdef",
        "topo": topo_path,
        "transition_matrices": tmat_path,
        "mixture_weights": mixw_path,
        "means": mean_path,
        "variances": var_path,
    }
