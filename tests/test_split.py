"""Tests for Gaussian splitting (inc_comp) and k-means."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pstrain.lib import _pstrainc, split

# Check if library is available
# libpstrainc availability comes from the shared helper (real loader-based
# detection); see tests/clib.py.
from tests.clib import C_LIBRARY_AVAILABLE as _lib_exists


@pytest.fixture
def model_with_counts(tmp_path: Path) -> dict[str, Any]:
    """Create a simple model with synthetic density counts."""
    n_phones = 5
    n_state = 3
    n_tied_state = n_phones * n_state
    n_density = 1  # Start with 1 Gaussian
    n_feat = 1
    veclen = 13

    # Create model directory
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # Seeded generator: keep the synthetic model deterministic so the exact
    # split-perturbation assertions (mean +/- 0.2*std, rtol=1e-5) don't flake
    # when a split mean happens to land near zero under a different RNG state.
    rng = np.random.default_rng(20260703)

    # Create means (random but reasonable values)
    means = rng.standard_normal((n_tied_state, n_feat, n_density, veclen)).astype(np.float32)
    mean_path = model_dir / "means"
    assert _pstrainc.write_gau(str(mean_path), means) == 0

    # Create variances (must be positive)
    variances = (
        np.abs(rng.standard_normal((n_tied_state, n_feat, n_density, veclen)).astype(np.float32))
        + 0.1
    )
    var_path = model_dir / "variances"
    assert _pstrainc.write_gau(str(var_path), variances) == 0

    # Create uniform mixture weights
    mixw = np.ones((n_tied_state, n_feat, n_density), dtype=np.float32) / n_density
    mixw_path = model_dir / "mixture_weights"
    assert _pstrainc.write_mixw(str(mixw_path), mixw) == 0

    # Create density counts (simulate training observation counts)
    # Some states seen more than others
    dnom = rng.exponential(100, (n_tied_state, n_feat, n_density)).astype(np.float32)
    dnom_path = model_dir / "gauden_counts"
    assert _pstrainc.write_dnom(str(dnom_path), dnom) == 0

    return {
        "model_dir": model_dir,
        "means": means,
        "variances": variances,
        "mixw": mixw,
        "dnom": dnom,
        "n_tied_state": n_tied_state,
        "n_feat": n_feat,
        "n_density": n_density,
        "veclen": veclen,
    }


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestDnomIO:
    """Test density count read/write."""

    def test_dnom_roundtrip(self, tmp_path: Path) -> None:
        """Test that dnom can be written and read back."""
        dnom = np.random.exponential(100, (10, 1, 2)).astype(np.float32)
        path = tmp_path / "test_dnom"

        assert _pstrainc.write_dnom(str(path), dnom) == 0

        read_dnom, n_cb, n_feat, n_density = _pstrainc.read_dnom(str(path))

        assert n_cb == 10
        assert n_feat == 1
        assert n_density == 2
        np.testing.assert_allclose(read_dnom, dnom, rtol=1e-5)


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestSplitGaussians:
    """Test Gaussian splitting functionality."""

    def test_split_doubles_density(self, model_with_counts: dict[str, Any], tmp_path: Path) -> None:
        """Test that splitting doubles the number of Gaussians."""
        mc = model_with_counts
        output_dir = tmp_path / "split_output"
        output_dir.mkdir()

        split.split_gaussians(
            in_mean_path=mc["model_dir"] / "means",
            in_var_path=mc["model_dir"] / "variances",
            in_mixw_path=mc["model_dir"] / "mixture_weights",
            dcount_path=mc["model_dir"] / "gauden_counts",
            out_mean_path=output_dir / "means",
            out_var_path=output_dir / "variances",
            out_mixw_path=output_dir / "mixture_weights",
            n_inc=mc["n_density"],  # Double the Gaussians
        )

        # Read back and verify dimensions
        new_mixw, n_mixw, n_feat, n_density = _pstrainc.read_mixw_counts(
            str(output_dir / "mixture_weights")
        )

        assert n_mixw == mc["n_tied_state"]
        assert n_feat == mc["n_feat"]
        assert n_density == mc["n_density"] * 2  # Doubled!

    def test_split_preserves_total_weight(
        self, model_with_counts: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test that splitting preserves total mixture weight per state."""
        mc = model_with_counts
        output_dir = tmp_path / "split_output"
        output_dir.mkdir()

        split.split_gaussians(
            in_mean_path=mc["model_dir"] / "means",
            in_var_path=mc["model_dir"] / "variances",
            in_mixw_path=mc["model_dir"] / "mixture_weights",
            dcount_path=mc["model_dir"] / "gauden_counts",
            out_mean_path=output_dir / "means",
            out_var_path=output_dir / "variances",
            out_mixw_path=output_dir / "mixture_weights",
            n_inc=mc["n_density"],
        )

        # Original weights sum to 1 per state
        orig_sums = mc["mixw"].sum(axis=2)

        # New weights should also sum to 1 per state
        new_mixw, _, _, _ = _pstrainc.read_mixw_counts(str(output_dir / "mixture_weights"))
        new_sums = new_mixw.sum(axis=2)

        np.testing.assert_allclose(new_sums, orig_sums, rtol=1e-5)

    def test_split_perturbs_means(self, model_with_counts: dict[str, Any], tmp_path: Path) -> None:
        """Test that split means are perturbed by +/- 0.2*std."""
        mc = model_with_counts
        output_dir = tmp_path / "split_output"
        output_dir.mkdir()

        split.split_gaussians(
            in_mean_path=mc["model_dir"] / "means",
            in_var_path=mc["model_dir"] / "variances",
            in_mixw_path=mc["model_dir"] / "mixture_weights",
            dcount_path=mc["model_dir"] / "gauden_counts",
            out_mean_path=output_dir / "means",
            out_var_path=output_dir / "variances",
            out_mixw_path=output_dir / "mixture_weights",
            n_inc=mc["n_density"],
        )

        # Read new means
        new_means, n_mgau, n_feat, n_density, veclen = _pstrainc.read_gau(str(output_dir / "means"))

        # Original mean and variance
        orig_mean = mc["means"][:, :, 0, :]  # First (only) Gaussian
        orig_var = mc["variances"][:, :, 0, :]
        orig_std = np.sqrt(orig_var)

        # New means should be mean +/- 0.2*std
        mean_a = new_means[:, :, 0, : veclen[0]]
        mean_b = new_means[:, :, 1, : veclen[0]]

        expected_a = orig_mean + 0.2 * orig_std
        expected_b = orig_mean - 0.2 * orig_std

        np.testing.assert_allclose(mean_a, expected_a, rtol=1e-5)
        np.testing.assert_allclose(mean_b, expected_b, rtol=1e-5)

    def test_split_preserves_variance(
        self, model_with_counts: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test that variances are unchanged after splitting."""
        mc = model_with_counts
        output_dir = tmp_path / "split_output"
        output_dir.mkdir()

        split.split_gaussians(
            in_mean_path=mc["model_dir"] / "means",
            in_var_path=mc["model_dir"] / "variances",
            in_mixw_path=mc["model_dir"] / "mixture_weights",
            dcount_path=mc["model_dir"] / "gauden_counts",
            out_mean_path=output_dir / "means",
            out_var_path=output_dir / "variances",
            out_mixw_path=output_dir / "mixture_weights",
            n_inc=mc["n_density"],
        )

        # Read new variances
        new_vars, _, _, _, veclen = _pstrainc.read_gau(str(output_dir / "variances"))

        # Both new Gaussians should have same variance as original
        orig_var = mc["variances"][:, :, 0, :]
        new_var_a = new_vars[:, :, 0, : veclen[0]]
        new_var_b = new_vars[:, :, 1, : veclen[0]]

        np.testing.assert_allclose(new_var_a, orig_var, rtol=1e-5)
        np.testing.assert_allclose(new_var_b, orig_var, rtol=1e-5)


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestDoubleGaussians:
    """Test convenience function for doubling Gaussians."""

    def test_double_gaussians_inplace(self, model_with_counts: dict[str, Any]) -> None:
        """Test doubling Gaussians with in-place update."""
        mc = model_with_counts
        model_dir = mc["model_dir"]

        # Double in place
        result = split.double_gaussians(
            model_dir=model_dir,
            dcount_path=model_dir / "gauden_counts",
        )

        # Verify files updated
        new_mixw, _, _, n_density = _pstrainc.read_mixw_counts(str(result["mixture_weights"]))
        assert n_density == 2

    def test_double_gaussians_to_new_dir(
        self, model_with_counts: dict[str, Any], tmp_path: Path
    ) -> None:
        """Test doubling Gaussians to a new directory."""
        mc = model_with_counts
        output_dir = tmp_path / "doubled"

        result = split.double_gaussians(
            model_dir=mc["model_dir"],
            dcount_path=mc["model_dir"] / "gauden_counts",
            output_dir=output_dir,
        )

        # Verify new directory has files
        assert (output_dir / "means").exists()
        assert (output_dir / "variances").exists()
        assert (output_dir / "mixture_weights").exists()

        # Original still has 1 density
        orig_mixw, _, _, n_density = _pstrainc.read_mixw_counts(
            str(mc["model_dir"] / "mixture_weights")
        )
        assert n_density == 1

        # New has 2 densities
        new_mixw, _, _, n_density = _pstrainc.read_mixw_counts(str(result["mixture_weights"]))
        assert n_density == 2


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestSplitErrors:
    """Test error handling for split functions."""

    def test_split_invalid_path(self, tmp_path: Path) -> None:
        """Test that split_gaussians fails with invalid paths."""
        with pytest.raises(RuntimeError):
            split.split_gaussians(
                in_mean_path=tmp_path / "nonexistent",
                in_var_path=tmp_path / "nonexistent",
                in_mixw_path=tmp_path / "nonexistent",
                dcount_path=tmp_path / "nonexistent",
                out_mean_path=tmp_path / "out_mean",
                out_var_path=tmp_path / "out_var",
                out_mixw_path=tmp_path / "out_mixw",
                n_inc=1,
            )


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestKMeans:
    """Test k-means clustering functionality."""

    def test_kmeans_basic(self) -> None:
        """Test basic k-means clustering."""
        # Create synthetic data with 2 clear clusters
        np.random.seed(42)
        cluster1 = np.random.randn(50, 4).astype(np.float32) + np.array([5, 5, 5, 5])
        cluster2 = np.random.randn(50, 4).astype(np.float32) + np.array([-5, -5, -5, -5])
        observations = np.vstack([cluster1, cluster2])

        centroids, labels, sqerr = split.kmeans(observations, n_clusters=2)

        assert centroids.shape == (2, 4)
        assert labels.shape == (100,)
        assert sqerr >= 0

        # Each cluster should be near its center
        center_dists = [
            np.linalg.norm(centroids[0] - np.array([5, 5, 5, 5])),
            np.linalg.norm(centroids[1] - np.array([5, 5, 5, 5])),
        ]
        assert min(center_dists) < 2.0  # One centroid should be near [5,5,5,5]

    def test_kmeans_single_cluster(self) -> None:
        """Test k-means with single cluster."""
        np.random.seed(42)
        observations = np.random.randn(100, 3).astype(np.float32)

        centroids, labels, sqerr = split.kmeans(observations, n_clusters=1)

        assert centroids.shape == (1, 3)
        assert labels.shape == (100,)
        assert np.all(labels == 0)  # All assigned to cluster 0

    def test_kmeans_many_clusters(self) -> None:
        """Test k-means with more clusters."""
        np.random.seed(42)
        observations = np.random.randn(500, 8).astype(np.float32)

        centroids, labels, sqerr = split.kmeans(observations, n_clusters=5)

        assert centroids.shape == (5, 8)
        assert labels.shape == (500,)
        assert set(labels).issubset({0, 1, 2, 3, 4})


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestKMeansInit:
    """Test k-means based Gaussian initialization."""

    def test_kmeans_init_basic(self) -> None:
        """Test k-means initialization of Gaussians."""
        np.random.seed(42)
        # Create synthetic features with 2 clear clusters
        cluster1 = np.random.randn(100, 13).astype(np.float32) * 0.5 + 5
        cluster2 = np.random.randn(100, 13).astype(np.float32) * 0.5 - 5
        features = np.vstack([cluster1, cluster2])

        means, variances, weights = split.kmeans_init_gaussians(features, n_density=2)

        assert means.shape == (2, 13)
        assert variances.shape == (2, 13)
        assert weights.shape == (2,)

        # Weights should sum to 1
        np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-5)

        # Variances should be positive
        assert np.all(variances > 0)

    def test_kmeans_init_single_gaussian(self) -> None:
        """Test k-means init with single Gaussian (global mean/var)."""
        np.random.seed(42)
        features = np.random.randn(200, 10).astype(np.float32) * 2 + 3

        means, variances, weights = split.kmeans_init_gaussians(features, n_density=1)

        assert means.shape == (1, 10)
        assert variances.shape == (1, 10)
        assert weights.shape == (1,)

        # Single weight should be 1.0
        np.testing.assert_allclose(weights[0], 1.0, rtol=1e-5)

        # Mean should be near global mean
        global_mean = features.mean(axis=0)
        np.testing.assert_allclose(means[0], global_mean, rtol=0.2)

    def test_kmeans_init_weights_reflect_data(self) -> None:
        """Test that mixture weights reflect data distribution."""
        np.random.seed(42)
        # Unbalanced clusters: 150 vs 50
        cluster1 = np.random.randn(150, 5).astype(np.float32) + 10
        cluster2 = np.random.randn(50, 5).astype(np.float32) - 10
        features = np.vstack([cluster1, cluster2])

        means, variances, weights = split.kmeans_init_gaussians(features, n_density=2)

        # One weight should be ~0.75, other ~0.25
        assert max(weights) > 0.6
        assert min(weights) < 0.4

    def test_kmeans_init_variance_correctness(self) -> None:
        """Test that computed variances are reasonable."""
        np.random.seed(42)
        # Create data with known variance
        std = 2.0
        features = (np.random.randn(1000, 4) * std).astype(np.float32)

        means, variances, weights = split.kmeans_init_gaussians(features, n_density=1)

        # Variance should be close to std^2 = 4.0
        expected_var = std * std
        np.testing.assert_allclose(variances[0], expected_var, rtol=0.2)

    def test_kmeans_init_multiple_densities(self) -> None:
        """Test k-means init with 4 densities."""
        np.random.seed(42)
        features = np.random.randn(400, 13).astype(np.float32)

        means, variances, weights = split.kmeans_init_gaussians(features, n_density=4)

        assert means.shape == (4, 13)
        assert variances.shape == (4, 13)
        assert weights.shape == (4,)
        np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-5)
        assert np.all(variances >= 1e-4)  # Floor applied
