"""Gaussian splitting and k-means initialization.

Increases the number of Gaussian components per mixture by splitting
the most probable densities. Used to grow models from 1 -> N Gaussians.
Also provides k-means based initialization for Gaussians.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from pstrain.lib import _pstrainc
from pstrain.lib.native_worker import contained


@contained
def split_gaussians(
    in_mean_path: Path,
    in_var_path: Path,
    in_mixw_path: Path,
    dcount_path: Path,
    out_mean_path: Path,
    out_var_path: Path,
    out_mixw_path: Path,
    n_inc: int,
) -> None:
    """Split Gaussian components to increase model capacity.

    Takes existing model parameters and density counts from BW training,
    then splits the most probable Gaussians by:
    - Halving mixture weight: mixw_new = mixw_old / 2
    - Perturbing means: mean_a = mean + 0.2*std, mean_b = mean - 0.2*std
    - Keeping variance unchanged

    Args:
        in_mean_path: Input means file
        in_var_path: Input variances file
        in_mixw_path: Input mixture weights file
        dcount_path: Density counts file (from BW norm, typically gauden_counts)
        out_mean_path: Output means file (n_density + n_inc Gaussians)
        out_var_path: Output variances file
        out_mixw_path: Output mixture weights file
        n_inc: Number of components to add (typically doubles: n_inc = n_density)

    Raises:
        RuntimeError: If splitting fails
    """
    lib = _pstrainc.get_lib()
    ret = lib.pstrain_inc_comp(
        str(in_mean_path).encode(),
        str(in_var_path).encode(),
        str(in_mixw_path).encode(),
        str(dcount_path).encode(),
        str(out_mean_path).encode(),
        str(out_var_path).encode(),
        str(out_mixw_path).encode(),
        n_inc,
    )
    if ret != 0:
        raise RuntimeError(f"pstrain_inc_comp failed with code {ret}")


def double_gaussians(
    model_dir: Path,
    dcount_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Double the number of Gaussians in a model.

    Convenience function that reads current model, splits all Gaussians,
    and writes to output directory.

    Args:
        model_dir: Directory containing means, variances, mixture_weights
        dcount_path: Path to density counts file
        output_dir: Output directory (default: overwrites input)

    Returns:
        Dict mapping file type to path
    """
    output_dir = output_dir or model_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    in_mean = model_dir / "means"
    in_var = model_dir / "variances"
    in_mixw = model_dir / "mixture_weights"

    out_mean = output_dir / "means"
    out_var = output_dir / "variances"
    out_mixw = output_dir / "mixture_weights"

    # Read current n_density to double it
    _, _, _, n_density = _pstrainc.read_mixw(str(in_mixw))

    split_gaussians(
        in_mean_path=in_mean,
        in_var_path=in_var,
        in_mixw_path=in_mixw,
        dcount_path=dcount_path,
        out_mean_path=out_mean,
        out_var_path=out_var,
        out_mixw_path=out_mixw,
        n_inc=n_density,  # Double the Gaussians
    )

    return {
        "means": out_mean,
        "variances": out_var,
        "mixture_weights": out_mixw,
    }


@contained
def kmeans(
    observations: npt.NDArray[np.float32],
    n_clusters: int,
    max_iter: int = 100,
    min_ratio: float = 0.01,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint32], float]:
    """Run k-means clustering on observations.

    Args:
        observations: Feature array of shape (n_obs, veclen)
        n_clusters: Number of clusters (k)
        max_iter: Maximum iterations
        min_ratio: Convergence ratio threshold

    Returns:
        Tuple of (centroids, labels, squared_error):
        - centroids: Cluster centers of shape (n_clusters, veclen)
        - labels: Cluster assignments of shape (n_obs,)
        - squared_error: Final squared error

    Raises:
        RuntimeError: If clustering fails
    """
    ffi = _pstrainc.get_ffi()
    lib = _pstrainc.get_lib()

    observations = np.ascontiguousarray(observations, dtype=np.float32)
    n_obs, veclen = observations.shape

    # Allocate outputs
    centroids = np.zeros((n_clusters, veclen), dtype=np.float32)
    labels = np.zeros(n_obs, dtype=np.uint32)

    sqerr = lib.pstrain_kmeans(
        ffi.cast("float32*", ffi.from_buffer(observations)),
        n_obs,
        veclen,
        n_clusters,
        max_iter,
        min_ratio,
        ffi.cast("float32*", ffi.from_buffer(centroids)),
        ffi.cast("uint32*", ffi.from_buffer(labels)),
    )

    if sqerr < 0:
        raise RuntimeError(f"pstrain_kmeans failed with error {sqerr}")

    return centroids, labels, sqerr


@contained
def kmeans_init_gaussians(
    features: npt.NDArray[np.float32],
    n_density: int,
    max_iter: int = 100,
    min_ratio: float = 0.01,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Initialize Gaussian parameters using k-means clustering.

    Clusters features into n_density groups, then computes means,
    variances, and mixture weights from the cluster assignments.

    Args:
        features: Feature array of shape (n_frames, veclen)
        n_density: Number of Gaussians to initialize
        max_iter: Maximum k-means iterations
        min_ratio: Convergence threshold

    Returns:
        Tuple of (means, variances, weights):
        - means: Shape (n_density, veclen)
        - variances: Shape (n_density, veclen)
        - weights: Shape (n_density,)

    Raises:
        RuntimeError: If initialization fails
    """
    ffi = _pstrainc.get_ffi()
    lib = _pstrainc.get_lib()

    features = np.ascontiguousarray(features, dtype=np.float32)
    n_frames, veclen = features.shape

    # Allocate outputs
    means = np.zeros((n_density, veclen), dtype=np.float32)
    variances = np.zeros((n_density, veclen), dtype=np.float32)
    weights = np.zeros(n_density, dtype=np.float32)

    ret = lib.pstrain_kmeans_init(
        ffi.cast("float32*", ffi.from_buffer(features)),
        n_frames,
        veclen,
        n_density,
        max_iter,
        min_ratio,
        ffi.cast("float32*", ffi.from_buffer(means)),
        ffi.cast("float32*", ffi.from_buffer(variances)),
        ffi.cast("float32*", ffi.from_buffer(weights)),
    )

    if ret != 0:
        raise RuntimeError(f"pstrain_kmeans_init failed with code {ret}")

    return means, variances, weights
