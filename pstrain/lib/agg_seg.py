"""Segment aggregation for training.

Aggregates feature observations by state, phone, or all together.
This is used to pool training data for CD model initialization.
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path

from pstrain.lib import _pstrainc


class SegType(IntEnum):
    """Segment aggregation type."""

    ALL = 0  # All frames to one file
    ST = 1  # Aggregate by tied state
    PHN = 2  # Aggregate by phone


def aggregate_segments(
    ctl_path: Path,
    cep_dir: Path,
    output_path: Path,
    segtype: SegType | str = SegType.ST,
    mdef_path: Path | None = None,
    dict_path: Path | None = None,
    fdict_path: Path | None = None,
    cep_ext: str = "mfc",
    seg_dir: Path | None = None,
    seg_ext: str = "v8_seg",
    index_path: Path | None = None,
    ts2cb_path: str | Path | None = None,
    cnt_path: Path | None = None,
    feat_type: str = "1s_c_d_dd",
    ceplen: int = 13,
    stride: int = 1,
    cachesz: int = 200,
) -> None:
    """Aggregate feature segments from training corpus.

    This function aggregates feature vectors from a training corpus,
    grouping them by state, phone, or writing all to one file.

    Args:
        ctl_path: Control file listing utterances.
        cep_dir: Cepstrum directory.
        output_path: Output dump file path.
        segtype: Segment type (ALL, ST, PHN, or string). Default ST.
        mdef_path: Model definition file (required for ST/PHN modes).
        dict_path: Dictionary file (required for PHN mode).
        fdict_path: Filler dictionary file (optional).
        cep_ext: Cepstrum extension (default "mfc").
        seg_dir: Segmentation directory (for ST/PHN modes).
        seg_ext: Segmentation extension (default "v8_seg").
        index_path: Index file path (for ST/PHN modes).
        ts2cb_path: Tied-state to codebook mapping.
            Can be ".semi.", ".cont.", or a file path.
        cnt_path: Count file path (created if not exists for ST/PHN modes).
        feat_type: Feature type string (default "1s_c_d_dd").
        ceplen: Cepstrum length (default 13).
        stride: Take every stride-th frame (default 1).
        cachesz: Cache size in MB (default 200).

    Raises:
        ValueError: If required arguments for segtype are missing.
        RuntimeError: If segment aggregation fails.

    Example:
        >>> # Aggregate all frames to one file (for VQ codebook training)
        >>> aggregate_segments(
        ...     ctl_path=Path("train.ctl"),
        ...     cep_dir=Path("features"),
        ...     output_path=Path("all_frames.dmp"),
        ...     segtype=SegType.ALL,
        ...     stride=10,  # Take every 10th frame
        ... )
    """
    # Convert string segtype to enum
    if isinstance(segtype, str):
        segtype = SegType[segtype.upper()]

    # Validate arguments based on segtype
    if segtype == SegType.ST:
        if mdef_path is None:
            raise ValueError("segtype ST requires mdef_path")
        if ts2cb_path is None:
            raise ValueError("segtype ST requires ts2cb_path")
    elif segtype == SegType.PHN:
        if mdef_path is None:
            raise ValueError("segtype PHN requires mdef_path")
        if dict_path is None:
            raise ValueError("segtype PHN requires dict_path")

    lib = _pstrainc.get_lib()

    ret = lib.pstrain_agg_seg(
        _pstrainc.path_or_null(mdef_path),
        _pstrainc.path_or_null(dict_path),
        _pstrainc.path_or_null(fdict_path),
        str(ctl_path).encode(),
        str(cep_dir).encode(),
        cep_ext.encode() if cep_ext else b"mfc",
        _pstrainc.path_or_null(seg_dir),
        seg_ext.encode() if seg_ext else b"v8_seg",
        str(output_path).encode(),
        _pstrainc.path_or_null(index_path),
        (str(ts2cb_path).encode() if ts2cb_path is not None else _pstrainc.get_ffi().NULL),
        _pstrainc.path_or_null(cnt_path),
        int(segtype),
        feat_type.encode() if feat_type else b"1s_c_d_dd",
        ceplen,
        stride,
        cachesz,
    )

    if ret != 0:
        raise RuntimeError(f"Segment aggregation failed: {output_path}")
