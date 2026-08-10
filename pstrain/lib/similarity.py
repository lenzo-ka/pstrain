"""Acoustic similarity measures for HMM states and Gaussian distributions.

Compare the acoustic properties (Gaussian parameters) of HMM states
from two different models. Useful for analyzing model differences,
initialization quality, and training convergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class GaussianState:
    """Single Gaussian HMM state parameters."""

    mean: NDArray[np.float64]  # Feature vector mean (dimension D)
    variance: NDArray[np.float64]  # Diagonal variance (dimension D)
    weight: float = 1.0  # Mixture weight (for GMMs)

    def __post_init__(self) -> None:
        """Validate dimensions."""
        if self.mean.shape != self.variance.shape:
            raise ValueError(
                f"Mean and variance dimension mismatch: {self.mean.shape} vs {self.variance.shape}"
            )
        if len(self.mean.shape) != 1:
            raise ValueError(f"Mean must be 1D vector, got shape {self.mean.shape}")

    @property
    def dim(self) -> int:
        """Feature dimension."""
        return len(self.mean)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dict."""
        return {
            "mean": self.mean.tolist(),
            "variance": self.variance.tolist(),
            "weight": self.weight,
            "dim": self.dim,
        }


@dataclass
class Senone:
    """Tied HMM state with Gaussian mixture.

    A senone (tied state) contains one or more Gaussians that model
    the acoustic output distribution for that state. In Sphinx terminology:
    - CI models: one senone per phone state (n_phones * n_states)
    - CD models: senones are shared across triphone contexts via state tying

    Supports arbitrary numbers of Gaussians (1, 8, 16, 32, etc.).
    """

    gaussians: list[GaussianState]  # Mixture components

    def __post_init__(self) -> None:
        """Validate mixture."""
        if not self.gaussians:
            raise ValueError("Senone must have at least one Gaussian")

        # Check all Gaussians have same dimension
        dim = self.gaussians[0].dim
        for i, g in enumerate(self.gaussians[1:], 1):
            if g.dim != dim:
                raise ValueError(f"Gaussian {i} has dimension {g.dim}, expected {dim}")

        # Check weights sum to 1 (approximately)
        total_weight = sum(g.weight for g in self.gaussians)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Mixture weights sum to {total_weight}, expected 1.0")

    @property
    def dim(self) -> int:
        """Feature dimension."""
        return self.gaussians[0].dim

    @property
    def n_gaussians(self) -> int:
        """Number of Gaussians in mixture."""
        return len(self.gaussians)

    def as_single_gaussian(self) -> GaussianState:
        """Collapse to single Gaussian using weighted average.

        Computes:
            mean = sum_k w_k * μ_k
            variance = sum_k w_k * (σ_k^2 + μ_k^2) - mean^2

        Returns:
            Single Gaussian approximating the mixture
        """
        # Weighted mean
        mean = np.zeros(self.dim)
        for g in self.gaussians:
            mean += g.weight * g.mean

        # Weighted variance (law of total variance)
        variance = np.zeros(self.dim)
        for g in self.gaussians:
            # E[X^2] for each component
            variance += g.weight * (g.variance + g.mean**2)
        # Subtract (E[X])^2 to get Var[X]
        variance -= mean**2

        return GaussianState(mean=mean, variance=variance, weight=1.0)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dict."""
        return {
            "n_gaussians": self.n_gaussians,
            "dim": self.dim,
            "gaussians": [g.to_dict() for g in self.gaussians],
        }


def bhattacharyya_distance(
    state1: GaussianState,
    state2: GaussianState,
    variance_floor: float = 1e-6,
) -> float:
    """Compute Bhattacharyya distance between two Gaussian states.

    The Bhattacharyya distance measures the similarity between two
    probability distributions. For diagonal-covariance Gaussians:

        D_B = (1/8) * (μ1 - μ2)^T * Σ^-1 * (μ1 - μ2) + (1/2) * ln(det(Σ) / sqrt(det(Σ1) * det(Σ2)))

    where Σ = (Σ1 + Σ2) / 2

    Properties:
        - Range: [0, ∞)
        - 0 = identical distributions
        - Larger values = more different distributions
        - Considers both mean and variance differences

    Args:
        state1: First Gaussian state
        state2: Second Gaussian state
        variance_floor: Minimum variance value for numerical stability

    Returns:
        Bhattacharyya distance (0 = identical, higher = more different)

    Raises:
        ValueError: If states have different dimensions
    """
    if state1.dim != state2.dim:
        raise ValueError(f"State dimension mismatch: {state1.dim} vs {state2.dim}")

    # Apply variance floor for numerical stability
    var1 = np.maximum(state1.variance, variance_floor)
    var2 = np.maximum(state2.variance, variance_floor)

    # Mean difference
    mean_diff = state1.mean - state2.mean

    # Average covariance (diagonal)
    avg_var = (var1 + var2) / 2.0

    # Mahalanobis distance term: (1/8) * (μ1 - μ2)^T * Σ^-1 * (μ1 - μ2)
    mahal_term = 0.125 * np.sum((mean_diff**2) / avg_var)

    # Determinant term computed in log-space for numerical stability
    log_det_avg = np.sum(np.log(avg_var))
    log_det_1 = np.sum(np.log(var1))
    log_det_2 = np.sum(np.log(var2))

    det_term = 0.5 * (log_det_avg - 0.5 * (log_det_1 + log_det_2))

    result = mahal_term + det_term

    # Guard against numerical issues
    if not np.isfinite(result):
        return float(mahal_term)

    return float(result)


def kl_divergence(state1: GaussianState, state2: GaussianState) -> float:
    """Compute KL divergence from state1 to state2.

    KL(P||Q) measures how much information is lost when Q is used to
    approximate P. NOT symmetric.

    Args:
        state1: Reference distribution (P)
        state2: Approximating distribution (Q)

    Returns:
        KL divergence from state1 to state2
    """
    if state1.dim != state2.dim:
        raise ValueError(f"State dimension mismatch: {state1.dim} vs {state2.dim}")

    D = state1.dim
    mean_diff = state2.mean - state1.mean

    var2 = np.maximum(state2.variance, 1e-10)
    var1 = np.maximum(state1.variance, 1e-10)

    trace_term = np.sum(var1 / var2)
    mahal_term = np.sum((mean_diff**2) / var2)
    log_det_term = np.sum(np.log(var2)) - np.sum(np.log(var1))

    kl = 0.5 * (trace_term + mahal_term - D + log_det_term)

    return float(kl)


def symmetric_kl_divergence(state1: GaussianState, state2: GaussianState) -> float:
    """Compute symmetric KL divergence between two states.

    Symmetric KL = (KL(P||Q) + KL(Q||P)) / 2
    """
    kl_12 = kl_divergence(state1, state2)
    kl_21 = kl_divergence(state2, state1)
    return (kl_12 + kl_21) / 2.0


def mahalanobis_distance(state1: GaussianState, state2: GaussianState) -> float:
    """Compute Mahalanobis distance between state means.

    Uses the average variance of the two states as the covariance.
    """
    if state1.dim != state2.dim:
        raise ValueError(f"State dimension mismatch: {state1.dim} vs {state2.dim}")

    mean_diff = state1.mean - state2.mean
    avg_var = (state1.variance + state2.variance) / 2.0
    avg_var = np.maximum(avg_var, 1e-10)

    mahal_sq = np.sum((mean_diff**2) / avg_var)

    return float(math.sqrt(mahal_sq))


def euclidean_distance(state1: GaussianState, state2: GaussianState) -> float:
    """Compute Euclidean distance between state means."""
    if state1.dim != state2.dim:
        raise ValueError(f"State dimension mismatch: {state1.dim} vs {state2.dim}")

    return float(np.linalg.norm(state1.mean - state2.mean))


def cosine_similarity(state1: GaussianState, state2: GaussianState) -> float:
    """Compute cosine similarity between state means.

    Returns:
        Value in [-1, 1], where 1 = same direction, 0 = orthogonal
    """
    if state1.dim != state2.dim:
        raise ValueError(f"State dimension mismatch: {state1.dim} vs {state2.dim}")

    dot_product = np.dot(state1.mean, state2.mean)
    norm1 = np.linalg.norm(state1.mean)
    norm2 = np.linalg.norm(state2.mean)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def compare_states(
    state1: GaussianState,
    state2: GaussianState,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Compare two Gaussian states using multiple similarity metrics.

    Args:
        state1: First Gaussian state
        state2: Second Gaussian state
        metrics: List of metric names (default: all)
                 Options: 'bhattacharyya', 'kl', 'symmetric_kl',
                         'mahalanobis', 'euclidean', 'cosine'

    Returns:
        Dictionary mapping metric names to similarity values
    """
    if metrics is None:
        metrics = ["bhattacharyya", "kl", "symmetric_kl", "mahalanobis", "euclidean", "cosine"]

    results = {}

    for metric in metrics:
        if metric == "bhattacharyya":
            results[metric] = bhattacharyya_distance(state1, state2)
        elif metric == "kl":
            results[metric] = kl_divergence(state1, state2)
        elif metric == "symmetric_kl":
            results[metric] = symmetric_kl_divergence(state1, state2)
        elif metric == "mahalanobis":
            results[metric] = mahalanobis_distance(state1, state2)
        elif metric == "euclidean":
            results[metric] = euclidean_distance(state1, state2)
        elif metric == "cosine":
            results[metric] = cosine_similarity(state1, state2)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return results


def compare_senones(
    senone1: Senone,
    senone2: Senone,
    method: Literal["collapsed", "monte_carlo", "component_matching"] = "collapsed",
) -> dict[str, float]:
    """Compare two senones (tied states).

    Args:
        senone1: First senone
        senone2: Second senone
        method: Comparison method:
                - "collapsed": Collapse to single Gaussian and compare (fast)
                - "monte_carlo": Monte Carlo approximation (accurate, slower)
                - "component_matching": Match components by proximity (detailed)

    Returns:
        Dictionary with similarity metrics
    """
    if method == "collapsed":
        # Fast path for single Gaussians
        if senone1.n_gaussians == 1 and senone2.n_gaussians == 1:
            return compare_states(senone1.gaussians[0], senone2.gaussians[0])
        g1 = senone1.as_single_gaussian()
        g2 = senone2.as_single_gaussian()
        return compare_states(g1, g2)

    if method == "monte_carlo":
        n_samples = 10000
        kl_12 = _monte_carlo_kl(senone1, senone2, n_samples)
        kl_21 = _monte_carlo_kl(senone2, senone1, n_samples)
        return {
            "kl_mc": kl_12,
            "kl_mc_reverse": kl_21,
            "symmetric_kl_mc": (kl_12 + kl_21) / 2.0,
        }

    if method == "component_matching":
        return _compare_senone_components(senone1, senone2)

    raise ValueError(f"Unknown method: {method}")


def _monte_carlo_kl(senone_p: Senone, senone_q: Senone, n_samples: int) -> float:
    """Estimate KL(P||Q) via Monte Carlo sampling."""
    samples = _sample_senone(senone_p, n_samples)
    log_p = _senone_log_likelihood(samples, senone_p)
    log_q = _senone_log_likelihood(samples, senone_q)
    return float(np.mean(log_p - log_q))


def _sample_senone(senone: Senone, n_samples: int) -> NDArray[np.float64]:
    """Sample from a senone's Gaussian mixture."""
    samples = []
    weights = np.array([g.weight for g in senone.gaussians])
    components = np.random.choice(len(senone.gaussians), size=n_samples, p=weights)

    for k in range(len(senone.gaussians)):
        n_k = np.sum(components == k)
        if n_k > 0:
            g = senone.gaussians[k]
            samples_k = np.random.randn(n_k, senone.dim) * np.sqrt(g.variance) + g.mean
            samples.append(samples_k)

    return np.vstack(samples)


def _senone_log_likelihood(
    samples: NDArray[np.float64],
    senone: Senone,
    variance_floor: float = 1e-6,
) -> NDArray[np.float64]:
    """Compute log likelihood of samples under a senone's mixture."""
    n_samples = samples.shape[0]
    log_probs = np.full(n_samples, -np.inf)

    for g in senone.gaussians:
        var = np.maximum(g.variance, variance_floor)
        diff = samples - g.mean
        log_det = np.sum(np.log(2 * np.pi * var))
        mahal = np.sum((diff**2) / var, axis=1)
        log_component = -0.5 * (log_det + mahal)

        if g.weight > 0:
            log_probs = np.logaddexp(log_probs, np.log(g.weight) + log_component)

    return log_probs


def _compare_senone_components(senone1: Senone, senone2: Senone) -> dict[str, float]:
    """Compare senones by matching Gaussian components using Hungarian algorithm."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as err:
        raise ImportError("scipy required for component_matching method") from err

    n1 = senone1.n_gaussians
    n2 = senone2.n_gaussians

    cost_matrix = np.zeros((n1, n2))
    for i, g1 in enumerate(senone1.gaussians):
        for j, g2 in enumerate(senone2.gaussians):
            cost_matrix[i, j] = bhattacharyya_distance(g1, g2)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_distances = cost_matrix[row_ind, col_ind]

    return {
        "mean_matched_distance": float(np.mean(matched_distances)),
        "max_matched_distance": float(np.max(matched_distances)),
        "min_matched_distance": float(np.min(matched_distances)),
        "n_components_1": n1,
        "n_components_2": n2,
        "n_matched": len(row_ind),
    }


# Recommended metrics for different use cases
RECOMMENDED_METRICS = {
    "general": ["bhattacharyya", "symmetric_kl"],
    "mean_only": ["mahalanobis", "euclidean"],
    "direction": ["cosine"],
    "fast": ["euclidean"],
    "robust": ["bhattacharyya"],
}


__all__ = [
    "GaussianState",
    "Senone",
    "bhattacharyya_distance",
    "kl_divergence",
    "symmetric_kl_divergence",
    "mahalanobis_distance",
    "euclidean_distance",
    "cosine_similarity",
    "compare_states",
    "compare_senones",
    "RECOMMENDED_METRICS",
]
