"""Feature extraction using CFFI bindings to sphinxbase.

This module provides Python wrappers around the sphinxbase front-end (fe)
functions for extracting MFCC features from audio files.

Example:
    from pstrain.lib.features import extract_features

    extract_features(
        "audio/utterance.wav",
        "features/utterance.mfc",
        samprate=16000,
        ncep=13,
    )
"""

from __future__ import annotations

import contextlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from pstrain.lib import _pstrainc, native_worker
from pstrain.lib.native_worker import contained

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "FEParams",
    "FeatureExtractor",
    "extract_features",
    "read_sphinx_mfc",
    "apply_cmn",
    "compute_deltas",
]


@dataclass
class FEParams:
    """Low-level front-end parameters for CFFI calls.

    These map directly to sphinxbase fe parameters. For high-level configuration,
    use pstrain.lib.config.FeatureConfig instead.
    """

    samprate: int = 16000
    nfilt: int = 25  # SphinxTrain wideband training default.
    nfft: int = 512
    lowerf: float = 130.0
    upperf: float = 6800.0
    ncep: int = 13
    alpha: float = 0.97
    lifter: int = 22
    dither: bool = True
    remove_dc: bool = True
    remove_noise: bool = True
    transform: str = "dct"
    frate: int = 100
    wlen: float = 0.025625

    @classmethod
    def from_config(cls, audio_config: Any, feature_config: Any) -> Self:
        """Create FEParams from high-level config objects.

        Args:
            audio_config: AudioConfig instance
            feature_config: FeatureConfig instance from config.models

        Returns:
            FEParams with values mapped from config
        """
        return cls(
            samprate=audio_config.sample_rate,
            nfilt=feature_config.num_filters,
            nfft=feature_config.nfft,
            lowerf=feature_config.lower_freq,
            upperf=feature_config.upper_freq,
            ncep=feature_config.num_ceps,
            alpha=feature_config.preemphasis,
            lifter=feature_config.lifter,
        )


class FeatureExtractor:
    """Extract acoustic features from audio using sphinxbase fe.

    This wraps the sphinxbase front-end (fe) library for MFCC extraction.
    Features are extracted in Sphinx format (13 static cepstra by default).

    Example:
        fe = FeatureExtractor(samprate=16000, ncep=13)
        features = fe.process_audio(audio_samples)
        fe.close()

    Or using context manager:
        with FeatureExtractor(samprate=16000) as fe:
            features = fe.process_audio(audio_samples)
    """

    def __init__(self, **config: Any) -> None:
        """Initialize feature extractor.

        Args:
            samprate: Sample rate in Hz (default: 16000)
            nfilt: Number of mel filters (default: 25 for SphinxTrain wideband)
            nfft: FFT size (default: 512)
            lowerf: Lower frequency bound (default: 130)
            upperf: Upper frequency bound (default: 6800)
            ncep: Number of cepstral coefficients (default: 13)
            alpha: Pre-emphasis coefficient (default: 0.97)
            lifter: Liftering coefficient (default: 22)
        """
        self._config = FEParams(**config)
        if not native_worker.in_worker():
            self._proxy = native_worker.NativeObjectProxy(__name__, "FeatureExtractor", (), config)
            self._veclen = int(self._proxy.call("get_veclen"))
            return
        self._ffi, self._lib = _pstrainc._init()
        self._fe: Any = None
        self._veclen = 0
        self._init_fe()

    def _init_fe(self) -> None:
        """Initialize the front-end with configuration."""
        cfg = self._config

        # Use pstrain_fe_create for simplified initialization (bypasses cmd_ln)
        self._fe = self._lib.pstrain_fe_create(
            float(cfg.samprate),
            cfg.nfilt,
            cfg.nfft,
            float(cfg.lowerf),
            float(cfg.upperf),
            cfg.ncep,
            float(cfg.alpha),
            cfg.lifter,
            cfg.dither,
            cfg.remove_dc,
            cfg.remove_noise,
            cfg.transform.encode(),
            cfg.frate,
            float(cfg.wlen),
        )
        if not self._fe:
            raise RuntimeError("Failed to initialize front-end")

        self._veclen = self._lib.pstrain_cffi_fe_get_output_size(self._fe)

    def close(self) -> None:
        """Release resources."""
        if hasattr(self, "_proxy"):
            self._proxy.close()
            return
        if self._fe:
            self._lib.pstrain_cffi_fe_free(self._fe)
            self._fe = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    @property
    def veclen(self) -> int:
        """Output vector length (number of cepstral coefficients)."""
        return self._veclen

    def get_veclen(self) -> int:
        """Worker-callable accessor for :attr:`veclen`."""
        return self.veclen

    @property
    def config(self) -> FEParams:
        """Feature extraction configuration."""
        return self._config

    def process_audio(self, audio: npt.NDArray[np.int16]) -> npt.NDArray[np.float32]:
        """Process audio samples and return features.

        Args:
            audio: Audio samples as int16 array

        Returns:
            Feature array of shape (n_frames, veclen)
        """
        if hasattr(self, "_proxy"):
            result = self._proxy.call("process_audio", audio)
            assert isinstance(result, np.ndarray)
            return result
        if self._fe is None:
            raise RuntimeError("FeatureExtractor is closed")

        audio = np.ascontiguousarray(audio, dtype=np.int16)
        n_samples = len(audio)

        # Estimate number of output frames (10ms frame shift typical)
        frame_shift = self._config.samprate // 100  # ~10ms
        max_frames = (n_samples // frame_shift) + 10  # buffer

        # Allocate output buffer
        feat_buf = np.zeros((max_frames, self._veclen), dtype=np.float32)

        # Create pointers
        audio_ptr = self._ffi.cast("int16*", self._ffi.from_buffer(audio))
        audio_ptr_ptr = self._ffi.new("int16 const**", audio_ptr)
        nsamp = self._ffi.new("size_t*", n_samples)

        # Allocate frame pointers for fe_process_frames
        feat_ptrs = self._ffi.new("float32*[]", max_frames)
        for i in range(max_frames):
            feat_ptrs[i] = self._ffi.cast("float32*", self._ffi.from_buffer(feat_buf[i]))

        nframes = self._ffi.new("int32*", max_frames)

        # Start utterance
        self._lib.pstrain_cffi_fe_start_stream(self._fe)
        self._lib.pstrain_cffi_fe_start_utt(self._fe)

        # Process all samples
        total_frames = 0
        while nsamp[0] > 0:
            nframes[0] = max_frames - total_frames
            self._lib.pstrain_cffi_fe_process_frames(
                self._fe, audio_ptr_ptr, nsamp, feat_ptrs + total_frames, nframes, self._ffi.NULL
            )
            total_frames += nframes[0]

        # End utterance - get any remaining frames
        if total_frames < max_frames:
            last_frame = self._ffi.new("float32[]", self._veclen)
            nframes[0] = 0
            self._lib.pstrain_cffi_fe_end_utt(self._fe, last_frame, nframes)
            if nframes[0] > 0:
                for i in range(self._veclen):
                    feat_buf[total_frames, i] = last_frame[i]
                total_frames += nframes[0]

        # Convert from mfcc_t to float32 (may be same type)
        # The fe_mfcc_to_float handles this
        result_ptrs = self._ffi.new("float32*[]", total_frames)
        for i in range(total_frames):
            result_ptrs[i] = self._ffi.cast("float32*", self._ffi.from_buffer(feat_buf[i]))
        self._lib.pstrain_cffi_fe_mfcc_to_float(self._fe, result_ptrs, result_ptrs, total_frames)

        return feat_buf[:total_frames].copy()


@contained
def extract_features(
    audio_path: str | Path,
    output_path: str | Path,
    *,
    fmt: str = "sphinx",
    **config: Any,
) -> int:
    """Extract features from audio file and write to output file.

    Args:
        audio_path: Path to input audio file (WAV format)
        output_path: Path to output feature file
        fmt: Output format ("sphinx" or "numpy")
        **config: Feature extraction parameters (samprate, ncep, etc.)

    Returns:
        Number of frames extracted
    """
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    # Read audio file
    audio = _read_wav(audio_path)

    # Extract features
    with FeatureExtractor(**config) as fe:
        features = fe.process_audio(audio)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "sphinx":
        _write_sphinx_mfc(features, output_path)
    elif fmt == "numpy":
        np.save(output_path, features)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    return len(features)


def _read_wav(path: Path) -> npt.NDArray[np.int16]:
    """Read WAV file and return samples as int16.

    Simple reader for standard 16-bit mono/stereo WAV files.
    """
    with path.open("rb") as f:
        # Read RIFF header
        riff = f.read(4)
        if riff != b"RIFF":
            raise ValueError(f"Not a WAV file: {path}")

        f.read(4)  # file size
        wave = f.read(4)
        if wave != b"WAVE":
            raise ValueError(f"Not a WAV file: {path}")

        # Find data chunk
        fmt_found = False
        channels = 1
        sampwidth = 2

        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                raise ValueError(f"No data chunk in WAV file: {path}")

            chunk_size = struct.unpack("<I", f.read(4))[0]

            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                audio_fmt, channels, samprate, _, _, sampwidth = struct.unpack(
                    "<HHIIHH", fmt_data[:16]
                )
                sampwidth //= 8  # bits to bytes
                fmt_found = True
            elif chunk_id == b"data":
                if not fmt_found:
                    raise ValueError(f"fmt chunk not found before data: {path}")
                data = f.read(chunk_size)
                break
            else:
                f.seek(chunk_size, 1)  # Skip unknown chunk

        # Convert to int16
        if sampwidth == 2:
            samples = np.frombuffer(data, dtype=np.int16)
        elif sampwidth == 1:
            # 8-bit unsigned to 16-bit signed
            samples = (np.frombuffer(data, dtype=np.uint8).astype(np.int16) - 128) * 256
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        # Convert stereo to mono by averaging
        if channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)
        elif channels > 2:
            raise ValueError(f"Unsupported number of channels: {channels}")

        return samples


def _write_sphinx_mfc(features: npt.NDArray[np.float32], path: Path) -> None:
    """Write features in Sphinx .mfc format.

    Format: 4-byte int32 header with total number of floats,
    followed by float32 data.
    """
    n_frames, veclen = features.shape
    n_floats = n_frames * veclen

    with path.open("wb") as f:
        # Header: total number of floats
        f.write(struct.pack("<i", n_floats))
        # Data: float32 features
        features.astype(np.float32).tofile(f)


def read_sphinx_mfc(path: Path) -> npt.NDArray[np.float32]:
    """Read features from Sphinx .mfc format.

    Args:
        path: Path to .mfc file

    Returns:
        Feature array of shape (n_frames, veclen)
    """
    with path.open("rb") as f:
        # Read header
        n_floats = struct.unpack("<i", f.read(4))[0]
        # Read data
        data = np.fromfile(f, dtype=np.float32)

    if len(data) != n_floats:
        raise ValueError(f"Expected {n_floats} floats, got {len(data)}")

    # Infer veclen (typically 13 for MFCCs)
    # Try common values
    for veclen in [13, 26, 39]:
        if n_floats % veclen == 0:
            return data.reshape(-1, veclen)

    raise ValueError(f"Cannot determine veclen for {n_floats} floats")


def apply_cmn(
    features: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Apply Cepstral Mean Normalization (CMN) to features.

    Batch CMN subtracts the mean of each coefficient across the utterance.
    This normalizes for channel/speaker variation.

    Args:
        features: Input features of shape (n_frames, veclen)

    Returns:
        CMN-normalized features of same shape
    """
    # Compute mean across frames for each coefficient
    mean = features.mean(axis=0, keepdims=True)
    # Subtract mean
    return (features - mean).astype(np.float32)


def compute_deltas(
    features: npt.NDArray[np.float32],
    window: int = 2,
) -> npt.NDArray[np.float32]:
    """Compute delta and delta-delta features.

    Sphinx-style regression delta computation using a window of frames.
    Returns concatenated [static; delta; delta-delta] features.

    Args:
        features: Input features of shape (n_frames, veclen)
        window: Delta window size (default: 2, uses frames [-2, -1, +1, +2])

    Returns:
        Feature array of shape (n_frames, veclen * 3) with static+delta+deltadelta
    """
    n_frames, veclen = features.shape

    # Compute regression weights
    denom = 2 * sum(i * i for i in range(1, window + 1))

    # Pad features for delta computation
    padded = np.pad(features, ((window, window), (0, 0)), mode="edge")

    # Compute deltas using linear regression
    deltas = np.zeros_like(features)
    for i in range(1, window + 1):
        deltas += i * (
            padded[window + i : window + i + n_frames] - padded[window - i : window - i + n_frames]
        )
    deltas /= denom

    # Compute delta-deltas from deltas
    padded_deltas = np.pad(deltas, ((window, window), (0, 0)), mode="edge")
    delta_deltas = np.zeros_like(features)
    for i in range(1, window + 1):
        delta_deltas += i * (
            padded_deltas[window + i : window + i + n_frames]
            - padded_deltas[window - i : window - i + n_frames]
        )
    delta_deltas /= denom

    # Concatenate: [static; delta; delta-delta]
    return np.hstack([features, deltas, delta_deltas]).astype(np.float32)
