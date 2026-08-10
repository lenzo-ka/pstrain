"""Model file I/O functions.

Python wrappers for S3 format file I/O operations.
These handle numpy array <-> C pointer conversion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pstrain.lib._cffi.core import _init

if TYPE_CHECKING:
    import numpy.typing as npt


def write_mixw(filename: str, mixw: npt.NDArray[np.float32]) -> int:
    """Write mixture weights to S3 format file.

    Args:
        filename: Output file path
        mixw: Mixture weights array of shape (n_mixw, n_feat, n_density)
              or (n_mixw, n_density) which will be reshaped

    Returns:
        0 on success, non-zero on error
    """
    ffi, lib = _init()

    # Ensure 3D: (n_mixw, n_feat, n_density)
    if mixw.ndim == 2:
        mixw = mixw.reshape(mixw.shape[0], 1, mixw.shape[1])

    mixw = np.ascontiguousarray(mixw, dtype=np.float32)
    n_mixw, n_feat, n_density = mixw.shape

    # Build pointer hierarchy for float32***
    # Level 1: array of float32** (one per mixw)
    mixw_level1 = ffi.new("float32**[]", n_mixw)
    # Level 2: arrays of float32* (one array per mixw, each with n_feat entries)
    mixw_level2 = [ffi.new("float32*[]", n_feat) for _ in range(n_mixw)]

    for i in range(n_mixw):
        mixw_level1[i] = mixw_level2[i]
        for j in range(n_feat):
            # Cast numpy buffer to float32*
            mixw_level2[i][j] = ffi.cast("float32*", ffi.from_buffer(mixw[i, j, :]))

    result: int = lib.s3mixw_write(filename.encode("utf-8"), mixw_level1, n_mixw, n_feat, n_density)
    return result


def write_tmat(filename: str, tmat: npt.NDArray[np.float32]) -> int:
    """Write transition matrices to S3 format file.

    Args:
        filename: Output file path
        tmat: Transition matrix array of shape (n_tmat, n_state, n_state)
              where n_state includes the exit state row.
              Only the first n_state-1 rows are written (exit state excluded).

    Returns:
        0 on success, non-zero on error
    """
    ffi, lib = _init()

    n_tmat, n_state, _ = tmat.shape

    # C function uses n_state-1 rows and writes from arr[0][0] contiguously.
    # We must provide a contiguous array with only n_state-1 rows per tmat.
    # Slice to exclude exit state row, then make contiguous.
    tmat_no_exit = np.ascontiguousarray(tmat[:, :-1, :], dtype=np.float32)
    n_rows = n_state - 1

    # Build pointer hierarchy for float32***
    tmat_level1 = ffi.new("float32**[]", n_tmat)
    tmat_level2 = [ffi.new("float32*[]", n_rows) for _ in range(n_tmat)]

    for t in range(n_tmat):
        tmat_level1[t] = tmat_level2[t]
        for i in range(n_rows):
            tmat_level2[t][i] = ffi.cast("float32*", ffi.from_buffer(tmat_no_exit[t, i, :]))

    result: int = lib.s3tmat_write(filename.encode("utf-8"), tmat_level1, n_tmat, n_state)
    return result


def write_gau(filename: str, gau: npt.NDArray[np.float32]) -> int:
    """Write Gaussian parameters (means or variances) to S3 format file.

    Args:
        filename: Output file path
        gau: Gaussian array of shape (n_mgau, n_feat, n_density, veclen)
             or (n_mgau, n_density, veclen) which will be reshaped

    Returns:
        0 on success, non-zero on error
    """
    ffi, lib = _init()

    # Ensure 4D: (n_mgau, n_feat, n_density, veclen)
    if gau.ndim == 3:
        gau = gau.reshape(gau.shape[0], 1, gau.shape[1], gau.shape[2])

    gau = np.ascontiguousarray(gau, dtype=np.float32)
    n_mgau, n_feat, n_density, veclen = gau.shape

    # Create veclen array
    veclen_arr = ffi.new("uint32[]", [veclen] * n_feat)

    # Create quadruple pointer: vector_t*** (float32****)
    # Actually s3gau_write takes vector_t*** which is float***(mgau x feat x density)
    # where each density is a vector of veclen floats
    gau_mgau = ffi.new("float32***[]", n_mgau)
    gau_feat = [ffi.new("float32**[]", n_feat) for _ in range(n_mgau)]
    gau_dens = [[ffi.new("float32*[]", n_density) for _ in range(n_feat)] for _ in range(n_mgau)]

    for m in range(n_mgau):
        gau_mgau[m] = gau_feat[m]
        for f in range(n_feat):
            gau_feat[m][f] = gau_dens[m][f]
            for d in range(n_density):
                gau_dens[m][f][d] = ffi.cast("float32*", ffi.from_buffer(gau[m, f, d, :]))

    result: int = lib.s3gau_write(
        filename.encode("utf-8"), gau_mgau, n_mgau, n_feat, n_density, veclen_arr
    )
    return result


def read_mixw(filename: str) -> tuple[npt.NDArray[np.float32], int, int, int]:
    """Read mixture weights from S3 format file.

    Args:
        filename: Input file path

    Returns:
        Tuple of (mixw_array, n_mixw, n_feat, n_density)
        mixw_array has shape (n_mixw, n_feat, n_density)

    Raises:
        RuntimeError: If file cannot be read
    """
    ffi, lib = _init()

    out_mixw = ffi.new("float32****")
    out_n_mixw = ffi.new("uint32*")
    out_n_feat = ffi.new("uint32*")
    out_n_density = ffi.new("uint32*")

    ret = lib.s3mixw_read(filename.encode("utf-8"), out_mixw, out_n_mixw, out_n_feat, out_n_density)
    if ret != 0:
        raise RuntimeError(f"Failed to read mixw from {filename}")

    n_mixw = out_n_mixw[0]
    n_feat = out_n_feat[0]
    n_density = out_n_density[0]

    # Convert C array to numpy
    mixw = np.zeros((n_mixw, n_feat, n_density), dtype=np.float32)
    for i in range(n_mixw):
        for j in range(n_feat):
            for k in range(n_density):
                mixw[i, j, k] = out_mixw[0][i][j][k]

    # Free C-allocated memory using proper 3d free
    lib.ckd_free_3d(out_mixw[0])

    return mixw, n_mixw, n_feat, n_density


def read_tmat(filename: str) -> tuple[npt.NDArray[np.float32], int, int]:
    """Read transition matrices from S3 format file.

    Args:
        filename: Input file path

    Returns:
        Tuple of (tmat_array, n_tmat, n_state)
        tmat_array has shape (n_tmat, n_state-1, n_state)

    Raises:
        RuntimeError: If file cannot be read
    """
    ffi, lib = _init()

    out_tmat = ffi.new("float32****")
    out_n_tmat = ffi.new("uint32*")
    out_n_state = ffi.new("uint32*")

    ret = lib.s3tmat_read(filename.encode("utf-8"), out_tmat, out_n_tmat, out_n_state)
    if ret != 0:
        raise RuntimeError(f"Failed to read tmat from {filename}")

    n_tmat = out_n_tmat[0]
    n_state = out_n_state[0]

    # tmat is (n_tmat, n_state-1, n_state) - n_state-1 rows because last is exit
    tmat = np.zeros((n_tmat, n_state - 1, n_state), dtype=np.float32)
    for t in range(n_tmat):
        for i in range(n_state - 1):
            for j in range(n_state):
                tmat[t, i, j] = out_tmat[0][t][i][j]

    # Free C-allocated memory using proper 3d free
    lib.ckd_free_3d(out_tmat[0])

    return tmat, n_tmat, n_state


def read_gau(filename: str) -> tuple[npt.NDArray[np.float32], int, int, int, list[int]]:
    """Read Gaussian parameters from S3 format file.

    Args:
        filename: Input file path

    Returns:
        Tuple of (gau_array, n_mgau, n_feat, n_density, veclen_list)
        gau_array has shape (n_mgau, n_feat, n_density, max_veclen)

    Raises:
        RuntimeError: If file cannot be read
    """
    ffi, lib = _init()

    out_gau = ffi.new("float32*****")
    out_n_mgau = ffi.new("uint32*")
    out_n_feat = ffi.new("uint32*")
    out_n_density = ffi.new("uint32*")
    out_veclen = ffi.new("uint32**")

    ret = lib.s3gau_read(
        filename.encode("utf-8"), out_gau, out_n_mgau, out_n_feat, out_n_density, out_veclen
    )
    if ret != 0:
        raise RuntimeError(f"Failed to read gau from {filename}")

    n_mgau = out_n_mgau[0]
    n_feat = out_n_feat[0]
    n_density = out_n_density[0]
    veclen = [out_veclen[0][f] for f in range(n_feat)]
    max_veclen = max(veclen)

    # Convert C array to numpy
    gau = np.zeros((n_mgau, n_feat, n_density, max_veclen), dtype=np.float32)
    for m in range(n_mgau):
        for f in range(n_feat):
            for d in range(n_density):
                for v in range(veclen[f]):
                    gau[m, f, d, v] = out_gau[0][m][f][d][v]

    # Free C-allocated memory (gauden_free_param pattern):
    # - p[0][0][0] is the raw data block
    # - p is the 3D pointer structure
    lib.ckd_free(out_gau[0][0][0][0])  # Free raw data
    lib.ckd_free_3d(out_gau[0])  # Free pointer structure
    lib.ckd_free(out_veclen[0])  # Free veclen array

    return gau, n_mgau, n_feat, n_density, veclen


def write_dnom(filename: str, dnom: npt.NDArray[np.float32]) -> int:
    """Write Gaussian density counts to S3 format file.

    These counts track how often each Gaussian is used during BW training.
    Used by inc_comp to decide which Gaussians to split.

    Args:
        filename: Output file path
        dnom: Density counts array of shape (n_cb, n_feat, n_density)
              or (n_cb, n_density) which will be reshaped

    Returns:
        0 on success, non-zero on error
    """
    ffi, lib = _init()

    # Ensure 3D: (n_cb, n_feat, n_density)
    if dnom.ndim == 2:
        dnom = dnom.reshape(dnom.shape[0], 1, dnom.shape[1])

    dnom = np.ascontiguousarray(dnom, dtype=np.float32)
    n_cb, n_feat, n_density = dnom.shape

    # Build pointer hierarchy for float32***
    dnom_level1 = ffi.new("float32**[]", n_cb)
    dnom_level2 = [ffi.new("float32*[]", n_feat) for _ in range(n_cb)]

    for i in range(n_cb):
        dnom_level1[i] = dnom_level2[i]
        for j in range(n_feat):
            dnom_level2[i][j] = ffi.cast("float32*", ffi.from_buffer(dnom[i, j, :]))

    result: int = lib.s3gaudnom_write(
        filename.encode("utf-8"), dnom_level1, n_cb, n_feat, n_density
    )
    return result


def read_dnom(filename: str) -> tuple[npt.NDArray[np.float32], int, int, int]:
    """Read Gaussian density counts from S3 format file.

    Args:
        filename: Input file path

    Returns:
        Tuple of (dnom_array, n_cb, n_feat, n_density)
        dnom_array has shape (n_cb, n_feat, n_density)

    Raises:
        RuntimeError: If file cannot be read
    """
    ffi, lib = _init()

    out_dnom = ffi.new("float32****")
    out_n_cb = ffi.new("uint32*")
    out_n_feat = ffi.new("uint32*")
    out_n_density = ffi.new("uint32*")

    ret = lib.s3gaudnom_read(
        filename.encode("utf-8"), out_dnom, out_n_cb, out_n_feat, out_n_density
    )
    if ret != 0:
        raise RuntimeError(f"Failed to read dnom from {filename}")

    n_cb = out_n_cb[0]
    n_feat = out_n_feat[0]
    n_density = out_n_density[0]

    # Convert C array to numpy
    dnom = np.zeros((n_cb, n_feat, n_density), dtype=np.float32)
    for i in range(n_cb):
        for j in range(n_feat):
            for k in range(n_density):
                dnom[i, j, k] = out_dnom[0][i][j][k]

    # Free C-allocated memory
    lib.ckd_free_3d(out_dnom[0])

    return dnom, n_cb, n_feat, n_density
