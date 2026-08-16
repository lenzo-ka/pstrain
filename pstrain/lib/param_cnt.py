"""Parameter counting for training corpus.

Counts occurrences of states, codebooks, or phones in the training data.
This is useful for statistics and diagnostics during model building.
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path

from pstrain.lib import _pstrainc
from pstrain.lib.native_worker import contained


class ParamType(IntEnum):
    """Parameter type for counting."""

    STATE = 0  # Count tied state occurrences
    CB = 1  # Count codebook occurrences
    PHONE = 2  # Count phone occurrences


@contained
def count_params(
    mdef_path: Path,
    dict_path: Path,
    ctl_path: Path,
    lsn_path: Path,
    output_path: Path | None = None,
    param_type: ParamType | str = ParamType.STATE,
    fdict_path: Path | None = None,
    ts2cb_path: Path | str | None = None,
    seg_dir: Path | None = None,
    seg_ext: str = "v8_seg",
    n_skip: int = 0,
    run_len: int = -1,
    part: int = 0,
    n_part: int = 0,
) -> None:
    """Count parameter occurrences in the training corpus.

    Scans the training corpus and counts how many times each state,
    codebook, or phone appears based on transcripts and segmentation.

    Args:
        mdef_path: Model definition file.
        dict_path: Dictionary file.
        ctl_path: Control file listing utterances.
        lsn_path: Transcript file (one transcript per line, same order as ctl).
        output_path: Output file for counts (None for stdout).
        param_type: What to count (STATE, CB, or PHONE).
        fdict_path: Filler dictionary file (optional).
        ts2cb_path: Tied-state to codebook mapping (optional, can be ".semi.", ".cont.").
            Required for CB mode.
        seg_dir: Segmentation directory (required for STATE and CB modes).
        seg_ext: Segmentation file extension (default "v8_seg").
        n_skip: Number of utterances to skip.
        run_len: Number of utterances to process (-1 for all).
        part: Corpus part number (0 for none).
        n_part: Total corpus parts (0 for none).

    Raises:
        ValueError: If required arguments for param_type are missing.
        RuntimeError: If parameter counting fails.

    Output format:
        - STATE/CB mode: "<id> <count>" per line
        - PHONE mode: "<phone_name> <count>" per line
    """
    lib = _pstrainc.get_lib()

    # Convert string param_type to enum
    if isinstance(param_type, str):
        param_type = ParamType[param_type.upper()]

    # Validate required arguments based on param_type
    if param_type in (ParamType.STATE, ParamType.CB) and seg_dir is None:
        raise ValueError(f"param_type {param_type.name} requires seg_dir")
    if param_type == ParamType.CB and ts2cb_path is None:
        raise ValueError("param_type CB requires ts2cb_path")

    ret = lib.pstrain_param_cnt(
        str(mdef_path).encode(),
        str(dict_path).encode(),
        _pstrainc.path_or_null(fdict_path),
        str(ctl_path).encode(),
        str(lsn_path).encode(),
        (str(ts2cb_path).encode() if ts2cb_path is not None else _pstrainc.get_ffi().NULL),
        _pstrainc.path_or_null(seg_dir),
        seg_ext.encode() if seg_ext else b"v8_seg",
        _pstrainc.path_or_null(output_path),
        int(param_type),
        n_skip,
        run_len,
        part,
        n_part,
    )

    if ret != 0:
        raise RuntimeError(f"Parameter counting failed for {param_type.name}: -ctlfn {ctl_path}")
