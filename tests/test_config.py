"""Tests for configuration system."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pstrain.lib.config.models import (
    FeatureConfig,
    PstrainConfig,
    TrainingConfig,
)


class TestFeatureConfig:
    """Tests for FeatureConfig model."""

    def test_default_values(self) -> None:
        """Test default feature settings."""
        cfg = FeatureConfig()
        assert cfg.num_ceps == 13
        assert cfg.num_filters == 40
        assert cfg.nfft == 512
        assert cfg.lower_freq == 130.0
        assert cfg.upper_freq == 6800.0
        assert cfg.preemphasis == pytest.approx(0.97)
        assert cfg.lifter == 22
        assert cfg.feature_type == "1s_c_d_dd"

    def test_custom_values(self) -> None:
        """Test custom feature settings."""
        cfg = FeatureConfig(num_ceps=26, nfft=256)
        assert cfg.num_ceps == 26
        assert cfg.nfft == 256

    def test_full_dimension_with_deltas(self) -> None:
        """Test full dimension calculation with deltas."""
        cfg = FeatureConfig(num_ceps=13, delta=True, delta_delta=True)
        # 13 * 3 = 39 for 1s_c_d_dd
        assert cfg.full_dimension == 39

    def test_sphinx_feat_type(self) -> None:
        """Test sphinx feature type property."""
        cfg = FeatureConfig(feature_type="1s_c_d_dd")
        assert cfg.sphinx_feat_type == "1s_c_d_dd"

    def test_num_streams_continuous(self) -> None:
        """Test num_streams for continuous features."""
        cfg = FeatureConfig(feature_type="1s_c_d_dd")
        assert cfg.num_streams == 1

    def test_num_streams_semicontinuous(self) -> None:
        """Test num_streams for semi-continuous features."""
        cfg = FeatureConfig(feature_type="s2_4x")
        assert cfg.num_streams == 4


class TestTrainingConfig:
    """Tests for TrainingConfig model."""

    def test_default_values(self) -> None:
        """Test default training settings."""
        cfg = TrainingConfig()
        assert cfg.n_states == 3
        assert cfg.ci is not None
        assert cfg.cd is not None

    def test_nested_ci_config(self) -> None:
        """Test nested CI training config."""
        cfg = TrainingConfig()
        assert cfg.ci.n_gaussians == 1
        assert cfg.ci.n_iterations == 10
        assert cfg.ci.min_iterations >= 1


class TestPstrainConfig:
    """Tests for main PstrainConfig model."""

    def test_default_values(self) -> None:
        """Test default config values."""
        cfg = PstrainConfig()
        assert isinstance(cfg.features, FeatureConfig)
        assert isinstance(cfg.training, TrainingConfig)

    def test_nested_access(self) -> None:
        """Test accessing nested settings."""
        cfg = PstrainConfig()
        assert cfg.features.num_ceps == 13
        assert cfg.training.n_states == 3

    def test_to_yaml_and_from_yaml(self) -> None:
        """Test YAML serialization roundtrip."""
        original = PstrainConfig(name="my_project")
        original.features.num_ceps = 26
        original.training.ci.n_gaussians = 4

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config_path = Path(f.name)

        try:
            original.to_yaml(config_path)
            assert config_path.exists()

            restored = PstrainConfig.from_yaml(config_path)
            assert restored.name == "my_project"
            assert restored.features.num_ceps == 26
            assert restored.training.ci.n_gaussians == 4
        finally:
            config_path.unlink(missing_ok=True)

    def test_from_yaml_nonexistent_raises(self) -> None:
        """Test loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PstrainConfig.from_yaml(Path("/nonexistent/config.yaml"))


class TestConfigValidation:
    """Tests for config validation."""

    def test_invalid_num_ceps(self) -> None:
        """Test that zero num_ceps is rejected."""
        with pytest.raises(ValueError):
            FeatureConfig(num_ceps=0)

    def test_invalid_nfft(self) -> None:
        """Test that too small nfft is rejected."""
        with pytest.raises(ValueError):
            FeatureConfig(nfft=32)  # min is 64

    def test_valid_config(self) -> None:
        """Test that valid config passes validation."""
        cfg = FeatureConfig(
            num_ceps=13,
            num_filters=40,
            nfft=512,
            lower_freq=130,
            upper_freq=6800,
        )
        assert cfg.num_ceps == 13


class TestConfigMerging:
    """Tests for config merging behavior."""

    def test_partial_update(self) -> None:
        """Test partial config updates."""
        cfg = PstrainConfig()
        cfg.features.num_ceps = 26
        assert cfg.features.num_ceps == 26
        # Other values should remain defaults
        assert cfg.features.num_filters == 40
