"""Decision tree building for state tying.

Wraps the decision tree functionality via CFFI.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pstrain.lib import _pstrainc, native_worker

if TYPE_CHECKING:
    import numpy.typing as npt


def build_tree(
    mdef_path: Path,
    mixw_path: Path,
    pset_path: Path,
    output_path: Path,
    phone: str,
    state: int,
    mean_path: Path | None = None,
    var_path: Path | None = None,
    continuous: bool = True,
    ssplitmin: int = 1,
    ssplitmax: int = 5,
    ssplitthr: float = 8e-4,
    csplitmin: int = 1,
    csplitmax: int = 100,
    csplitthr: float = 8e-4,
    mwfloor: float = 1e-4,
    varfloor: float = 1e-5,
    cntthresh: float = 1e-5,
    state_weights: npt.NDArray[np.float32] | None = None,
    allphones: bool = False,
) -> None:
    """Build a decision tree for triphones of a given base phone.

    Args:
        mdef_path: Model definition file (with triphones).
        mixw_path: Mixture weights file.
        pset_path: Phone set (question) file.
        output_path: Output tree file path.
        phone: Base phone name (e.g., "AA").
        state: State index (0-based).
        mean_path: Means file (required for continuous).
        var_path: Variance file (required for continuous).
        continuous: True for continuous HMMs.
        ssplitmin: Minimum simple split count.
        ssplitmax: Maximum simple split count.
        ssplitthr: Simple split threshold.
        csplitmin: Minimum composite split count.
        csplitmax: Maximum composite split count.
        csplitthr: Composite split threshold.
        mwfloor: Mixture weight floor.
        varfloor: Variance floor.
        cntthresh: Count threshold for model inclusion.
        state_weights: State weights array (or None for uniform).
        allphones: Build for all phones at once.

    Raises:
        ValueError: If continuous=True but mean_path or var_path not provided.
        RuntimeError: If tree building fails.
    """
    if continuous and (mean_path is None or var_path is None):
        raise ValueError("Continuous mode requires mean_path and var_path")

    lib = _pstrainc.get_lib()
    ffi = _pstrainc.get_ffi()

    # Handle state weights
    if state_weights is not None:
        stwt = ffi.cast("float32 *", state_weights.ctypes.data)
        n_stwt = len(state_weights)
    else:
        stwt = ffi.NULL
        n_stwt = 0

    ret = lib.pstrain_build_tree(
        str(mdef_path).encode(),
        str(mixw_path).encode(),
        _pstrainc.path_or_null(mean_path),
        _pstrainc.path_or_null(var_path),
        str(pset_path).encode(),
        str(output_path).encode(),
        phone.encode(),
        state,
        1 if continuous else 0,
        ssplitmin,
        ssplitmax,
        ssplitthr,
        csplitmin,
        csplitmax,
        csplitthr,
        mwfloor,
        varfloor,
        cntthresh,
        stwt,
        n_stwt,
        1 if allphones else 0,
    )

    if ret != 0:
        raise RuntimeError(f"Failed to build tree: {output_path}")


def tie_states(
    input_mdef_path: Path,
    output_mdef_path: Path,
    tree_dir: Path,
    pset_path: Path,
    phone: str | None = None,
    allphones: bool = False,
) -> None:
    """Tie states using decision trees.

    Args:
        input_mdef_path: Input model definition file (untied).
        output_mdef_path: Output model definition file (tied).
        tree_dir: Directory containing decision tree files.
        pset_path: Phone set file.
        phone: Specific phone to tie (or None for all).
        allphones: Tie all phones.

    Raises:
        RuntimeError: If state tying fails.
    """
    lib = _pstrainc.get_lib()
    ffi = _pstrainc.get_ffi()

    ret = lib.pstrain_tie_states(
        str(input_mdef_path).encode(),
        str(output_mdef_path).encode(),
        str(tree_dir).encode(),
        str(pset_path).encode(),
        phone.encode() if phone else ffi.NULL,  # phone is str, not Path
        1 if allphones else 0,
    )

    if ret != 0:
        raise RuntimeError(f"Failed to tie states: {output_mdef_path}")


def make_quests(
    mdef_path: Path,
    mixw_path: Path,
    output_path: Path,
    mean_path: Path | None = None,
    var_path: Path | None = None,
    continuous: bool = True,
    npermute: int = 6,
    quests_per_state: int = 8,
    varfloor: float = 1e-8,
    niter: int = 0,
) -> None:
    """Generate phonetic questions by clustering CI distributions.

    Args:
        mdef_path: CI model definition file.
        mixw_path: Mixture weights file.
        output_path: Output question file.
        mean_path: Means file (for continuous).
        var_path: Variance file (for continuous).
        continuous: True for continuous HMMs.
        npermute: Number of permutations for clustering.
        quests_per_state: Questions per state.
        varfloor: Variance floor.
        niter: Number of iterations.

    This operation is routed through the persistent native worker, so a
    malformed input file cannot terminate the calling interpreter.

    Raises:
        ValueError: If continuous=True but mean_path or var_path not provided.
        PstrainNativeError: If question generation fails. ``PstrainNativeCrashError``
            when the native code died on a signal, ``PstrainNativeFatalError``
            when it reported a diagnosed failure.
    """
    if continuous and (mean_path is None or var_path is None):
        raise ValueError("Continuous mode requires mean_path and var_path")

    native_worker.call(
        "make_quests",
        (
            str(mdef_path),
            str(mixw_path),
            None if mean_path is None else str(mean_path),
            None if var_path is None else str(var_path),
            str(output_path),
            1 if continuous else 0,
            npermute,
            quests_per_state,
            varfloor,
            niter,
        ),
        tuple(path for path in (mdef_path, mixw_path, mean_path, var_path) if path),
    )


def prune_tree(
    mdef_path: Path,
    pset_path: Path,
    input_tree_dir: Path,
    output_tree_dir: Path,
    n_seno_target: int,
    min_occ: float = 0.0,
    allphones: bool = False,
) -> None:
    """Prune decision trees to a target number of senones.

    Removes bifurcations that resulted in minimum likelihood increase,
    pruning globally across all decision trees.

    Args:
        mdef_path: CI model definition file.
        pset_path: Phone set (question) file.
        input_tree_dir: Input tree directory.
        output_tree_dir: Output tree directory.
        n_seno_target: Target number of senones.
        min_occ: Prune nodes with fewer than this many observations.
        allphones: Prune all phones together as single tree.

    This operation is routed through the persistent native worker, so a
    malformed or truncated ``.dtree`` file cannot terminate the calling
    interpreter.

    Raises:
        PstrainNativeError: If tree pruning fails. ``PstrainNativeCrashError``
            when the native code died on a signal, ``PstrainNativeFatalError``
            when it reported a diagnosed failure.
    """
    # Create output directory if needed
    output_tree_dir = Path(output_tree_dir)
    output_tree_dir.mkdir(parents=True, exist_ok=True)

    native_worker.call(
        "prune_tree",
        (
            str(mdef_path),
            str(pset_path),
            str(input_tree_dir),
            str(output_tree_dir),
            n_seno_target,
            min_occ,
            1 if allphones else 0,
        ),
        (mdef_path, pset_path, input_tree_dir),
    )


def parse_questions(questions_path: Path) -> dict[str, list[str]]:
    """Parse a question file into a dictionary.

    Args:
        questions_path: Path to the question file.

    Returns:
        Dictionary mapping question names to lists of phones.
        Example: {"QUESTION0": ["AA", "AE", "AH"], "SILENCE": ["SIL"]}
    """
    questions: dict[str, list[str]] = {}

    with questions_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            name = parts[0]
            phones = parts[1:]
            questions[name] = phones

    return questions


def init_mixw(
    src_mdef_path: Path,
    src_mixw_path: Path,
    src_mean_path: Path,
    src_var_path: Path,
    src_tmat_path: Path,
    dest_mdef_path: Path,
    dest_mixw_path: Path,
    dest_mean_path: Path,
    dest_var_path: Path,
    dest_tmat_path: Path,
    continuous: bool = True,
) -> None:
    """Initialize tied CD model parameters from a CI model.

    Maps CI phone parameters to CD triphone tied states based on the
    model definitions. For each triphone in the destination mdef:
    - If an exact match exists in source, copy its parameters
    - If only base phone exists, use base phone parameters
    - Otherwise, initialize with uniform distribution

    Args:
        src_mdef_path: Source (CI) model definition file.
        src_mixw_path: Source mixture weights file.
        src_mean_path: Source means file.
        src_var_path: Source variances file.
        src_tmat_path: Source transition matrices file.
        dest_mdef_path: Destination (CD tied) model definition file.
        dest_mixw_path: Output mixture weights file.
        dest_mean_path: Output means file.
        dest_var_path: Output variances file.
        dest_tmat_path: Output transition matrices file.
        continuous: True for continuous HMMs.

    Raises:
        RuntimeError: If initialization fails.
    """
    lib = _pstrainc.get_lib()

    ret = lib.pstrain_init_mixw(
        str(src_mdef_path).encode(),
        str(src_mixw_path).encode(),
        str(src_mean_path).encode(),
        str(src_var_path).encode(),
        str(src_tmat_path).encode(),
        str(dest_mdef_path).encode(),
        str(dest_mixw_path).encode(),
        str(dest_mean_path).encode(),
        str(dest_var_path).encode(),
        str(dest_tmat_path).encode(),
        1 if continuous else 0,
    )

    if ret != 0:
        raise RuntimeError(f"Failed to initialize CD model: {dest_mdef_path}")
