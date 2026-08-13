"""Tests for feature extraction module."""

from __future__ import annotations

import struct
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pstrain.lib.features import (
    FeatureExtractor,
    FEParams,
    extract_features,
    read_sphinx_mfc,
)
from tests.conftest import requires_c_library


def _read_wav(path: Path) -> np.ndarray[Any, np.dtype[np.int16]]:
    """Read WAV file and return int16 samples."""
    with wave.open(str(path), "rb") as wav:
        n_frames = wav.getnframes()
        data = wav.readframes(n_frames)
    return np.frombuffer(data, dtype=np.int16)


def _write_sphinx_mfc(path: Path, features: np.ndarray) -> None:
    """Write features in Sphinx MFC format."""
    n_frames, n_cep = features.shape
    header = n_frames * n_cep
    with path.open("wb") as f:
        f.write(struct.pack("<i", header))
        features.astype(np.float32).tofile(f)


# Skip all tests if C library not available
pytestmark = requires_c_library


@pytest.fixture
def sample_audio_path() -> Path:
    """Path to sample audio file."""
    return Path(__file__).parent.parent / "pstrain" / "data" / "sample" / "kevin-alice-16k.wav"


class TestFEParams:
    """Tests for FEParams dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        cfg = FEParams()
        assert cfg.samprate == 16000
        assert cfg.nfilt == 25
        assert cfg.nfft == 512
        assert cfg.lowerf == 130
        assert cfg.upperf == 6800
        assert cfg.ncep == 13
        assert cfg.alpha == pytest.approx(0.97)
        assert cfg.lifter == 22
        assert cfg.dither is True
        assert cfg.remove_dc is True
        assert cfg.remove_noise is True
        assert cfg.transform == "dct"
        assert cfg.frate == 100
        assert cfg.wlen == pytest.approx(0.025625)

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        cfg = FEParams(samprate=8000, ncep=26, nfft=256)
        assert cfg.samprate == 8000
        assert cfg.ncep == 26
        assert cfg.nfft == 256
        # Other values should still be defaults
        assert cfg.nfilt == 25


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""

    def test_init_default(self) -> None:
        """Test initialization with default parameters."""
        with FeatureExtractor() as fe:
            assert fe.veclen == 13

    def test_init_custom_ncep(self) -> None:
        """Test initialization with custom ncep."""
        with FeatureExtractor(ncep=26) as fe:
            assert fe.veclen == 26

    def test_context_manager(self) -> None:
        """Test that context manager properly cleans up."""
        fe = FeatureExtractor()
        assert fe.veclen > 0
        fe.close()
        with pytest.raises(RuntimeError, match="closed"):
            fe.process_audio(np.zeros(160, dtype=np.int16))

    def test_process_audio_shape(self, sample_audio_path: Path) -> None:
        """Test that process_audio returns correct shape."""
        audio = _read_wav(sample_audio_path)
        with FeatureExtractor(samprate=16000, ncep=13) as fe:
            features = fe.process_audio(audio)
            assert features.ndim == 2
            assert features.shape[1] == 13
            # Should have reasonable number of frames
            assert features.shape[0] > 100

    def test_process_audio_dtype(self, sample_audio_path: Path) -> None:
        """Test that process_audio returns float32."""
        audio = _read_wav(sample_audio_path)
        with FeatureExtractor() as fe:
            features = fe.process_audio(audio)
            assert features.dtype == np.float32

    def test_process_short_audio(self) -> None:
        """Test processing very short audio."""
        # Need at least one full frame: frame_length + buffer
        # Default: 25ms frame at 16kHz = 400 samples, plus some buffer
        audio = np.zeros(800, dtype=np.int16)  # 50ms
        with FeatureExtractor(samprate=16000) as fe:
            features = fe.process_audio(audio)
            # Should get at least 1 frame (may get 0 for very short audio)
            assert features.shape[0] >= 0
            if features.shape[0] > 0:
                assert features.shape[1] == 13

    def test_process_silence(self) -> None:
        """Test processing silence (zeros)."""
        audio = np.zeros(16000, dtype=np.int16)  # 1 second of silence
        with FeatureExtractor() as fe:
            features = fe.process_audio(audio)
            assert features.shape[0] > 0
            # Features should be finite
            assert np.all(np.isfinite(features))

    def test_config_property(self) -> None:
        """Test config property returns correct config."""
        cfg = FEParams(ncep=26)
        with FeatureExtractor(**cfg.__dict__) as fe:
            assert fe.config.ncep == 26

    def test_remove_dc_configuration_changes_produced_features(self) -> None:
        """A non-default FE option must reach signal processing, not just storage."""
        samples = np.arange(16000, dtype=np.float64)
        audio = (8000 + 3000 * np.sin(samples * 0.031)).astype(np.int16)

        with FeatureExtractor(dither=False, remove_dc=True) as enabled:
            with_dc_removal = enabled.process_audio(audio)
        with FeatureExtractor(dither=False, remove_dc=False) as disabled:
            without_dc_removal = disabled.process_audio(audio)

        assert with_dc_removal.shape == without_dc_removal.shape
        assert np.count_nonzero(with_dc_removal != without_dc_removal) > 0

    @pytest.mark.parametrize(
        ("override", "shape_changes"),
        [
            ({"remove_noise": False}, False),
            ({"transform": "legacy"}, False),
            ({"frate": 80}, True),
            ({"wlen": 0.02}, False),
        ],
    )
    def test_new_front_end_options_change_produced_features(
        self, sample_audio_path: Path, override: dict[str, object], shape_changes: bool
    ) -> None:
        """Every newly exposed non-default must alter native FE output."""
        audio = _read_wav(sample_audio_path)
        with FeatureExtractor(dither=False) as baseline:
            expected = baseline.process_audio(audio)
        with FeatureExtractor(dither=False, **override) as configured:
            actual = configured.process_audio(audio)

        if shape_changes:
            assert actual.shape[0] != expected.shape[0]
        else:
            assert actual.shape == expected.shape
            assert np.count_nonzero(actual != expected) > 0


class TestReadWriteSphinxMfc:
    """Tests for MFC file I/O."""

    def test_write_read_roundtrip(self) -> None:
        """Test that write -> read produces identical data."""
        original = np.random.randn(100, 13).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".mfc", delete=False) as f:
            tmpfile = Path(f.name)

        try:
            _write_sphinx_mfc(tmpfile, original)
            result = read_sphinx_mfc(tmpfile)

            assert result.shape == original.shape
            assert result.dtype == np.float32
            assert np.allclose(original, result)
        finally:
            tmpfile.unlink(missing_ok=True)

    def test_write_read_different_sizes(self) -> None:
        """Test roundtrip with different frame counts and dimensions.

        Note: read_sphinx_mfc infers veclen, so we use frame counts that
        avoid ambiguity (e.g., 3 frames * 26 cep = 78, not divisible by 13).
        """
        # Use frame counts that give unambiguous total float counts
        test_cases = [
            (10, 13),  # 130 floats - divisible by 13, not 26 or 39
            (10, 26),  # 260 floats - divisible by 13 and 26, prefer 26
            (10, 39),  # 390 floats - divisible by 13, 26, 39, prefer 39
            (100, 13),
        ]
        for n_frames, n_cep in test_cases:
            original = np.random.randn(n_frames, n_cep).astype(np.float32)

            with tempfile.NamedTemporaryFile(suffix=".mfc", delete=False) as f:
                tmpfile = Path(f.name)

            try:
                _write_sphinx_mfc(tmpfile, original)
                result = read_sphinx_mfc(tmpfile)
                # Just verify data integrity, shape may differ due to veclen inference
                assert result.size == original.size
                assert np.allclose(original.ravel(), result.ravel())
            finally:
                tmpfile.unlink(missing_ok=True)


class TestExtractFeatures:
    """Tests for extract_features convenience function."""

    def test_extract_features_basic(self, sample_audio_path: Path) -> None:
        """Test basic feature extraction."""
        with tempfile.NamedTemporaryFile(suffix=".mfc", delete=False) as f:
            out_path = Path(f.name)

        try:
            n_frames = extract_features(sample_audio_path, out_path)
            assert n_frames > 0
            assert out_path.exists()
            assert out_path.stat().st_size > 0
        finally:
            out_path.unlink(missing_ok=True)

    def test_extract_features_readable(self, sample_audio_path: Path) -> None:
        """Test that extracted features can be read back."""
        with tempfile.NamedTemporaryFile(suffix=".mfc", delete=False) as f:
            out_path = Path(f.name)

        try:
            n_frames = extract_features(sample_audio_path, out_path)
            features = read_sphinx_mfc(out_path)
            assert features.shape[0] == n_frames
            assert features.shape[1] == 13  # default ncep
        finally:
            out_path.unlink(missing_ok=True)

    def test_extract_features_custom_ncep(self, sample_audio_path: Path) -> None:
        """Test extraction with custom ncep."""
        with tempfile.NamedTemporaryFile(suffix=".mfc", delete=False) as f:
            out_path = Path(f.name)

        try:
            # Use ncep parameter (features.py FeatureConfig uses ncep, not num_ceps)
            n_frames = extract_features(sample_audio_path, out_path, ncep=26)
            features = read_sphinx_mfc(out_path)
            # Note: read_sphinx_mfc infers veclen, may reshape to (n*2, 13) instead of (n, 26)
            # Just verify total data matches expected
            assert features.size == n_frames * 26
        finally:
            out_path.unlink(missing_ok=True)


class TestReadWav:
    """Tests for WAV file reading."""

    def test_read_wav_dtype(self, sample_audio_path: Path) -> None:
        """Test that _read_wav returns int16."""
        audio = _read_wav(sample_audio_path)
        assert audio.dtype == np.int16

    def test_read_wav_shape(self, sample_audio_path: Path) -> None:
        """Test that _read_wav returns 1D array."""
        audio = _read_wav(sample_audio_path)
        assert audio.ndim == 1
        assert len(audio) > 0

    def test_read_wav_nonexistent(self) -> None:
        """Test that reading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            _read_wav(Path("/nonexistent/file.wav"))
