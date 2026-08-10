"""Baum-Welch training via CFFI.

Thin wrapper around the C BW implementation via CFFI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np

from pstrain.lib import _pstrainc

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)

__all__ = ["BWConfig", "BWResult", "HMM", "BWTrainer"]


@dataclass
class BWConfig:
    """Configuration for Baum-Welch training."""

    # Beam probabilities closer to zero are wider (less pruning); larger values
    # impose a higher pruning threshold and are tighter in this engine.
    # SphinxTrain defaults: abeam=1e-90, bbeam=1e-10
    a_beam: float = 1e-90  # Forward beam (alpha beam)
    b_beam: float = 1e-10  # Backward beam (beta beam)
    topn: int = 1  # Number of Gaussian densities evaluated per codebook
    spthresh: float = 0.0  # State pruning threshold
    mixw_floor: float = 1e-8  # CI and CD-untied SphinxTrain -mwfloor
    tmat_floor: float = 1e-4  # Upstream bw -tpfloor default
    mean_reest: bool = True
    var_reest: bool = True
    mixw_reest: bool = True
    tmat_reest: bool = True
    pass2var: bool = True  # 2-pass variance to match SphinxTrain -2passvar yes
    multipron: bool = True  # Multi-pron training: build wide graphs that sum
    # posteriors across pronunciation variants. Set to False to fall back to
    # the legacy linear path that always uses the first listed variant per
    # word (bit-identical to SphinxTrain's default behavior).


@dataclass
class BWResult:
    """Result from BW training."""

    total_log_lik: float
    total_frames: int
    total_utts: int
    avg_log_prob: float


class HMM:
    """HMM parameters - uses CFFI for I/O."""

    def __init__(
        self,
        means: npt.NDArray[np.float32],
        variances: npt.NDArray[np.float32],
        mixw: npt.NDArray[np.float32],
        tmat: npt.NDArray[np.float32],
    ) -> None:
        self.means = means
        self.variances = variances
        self.mixw = mixw
        self.tmat = tmat

    @classmethod
    def load(cls, model_dir: Path) -> Self:
        """Load model from directory using CFFI."""
        model_dir = Path(model_dir)

        # Use CFFI to read model files
        means_raw, n_cb, n_density, veclen, _ = _pstrainc.read_gau(str(model_dir / "means"))
        variances_raw, _, _, _, _ = _pstrainc.read_gau(str(model_dir / "variances"))

        # Reshape to (n_cb, n_density, veclen)
        means = means_raw.reshape(n_cb, n_density, veclen)
        variances = variances_raw.reshape(n_cb, n_density, veclen)

        mixw_raw, n_mixw, n_feat_stream, n_density_mw = _pstrainc.read_mixw(
            str(model_dir / "mixture_weights")
        )
        mixw = mixw_raw.reshape(n_mixw, n_density_mw)

        tmat, n_tmat, n_state = _pstrainc.read_tmat(str(model_dir / "transition_matrices"))

        return cls(means, variances, mixw, tmat)

    def save(self, model_dir: Path) -> None:
        """Save model to directory using CFFI."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        _pstrainc.write_gau(str(model_dir / "means"), self.means)
        _pstrainc.write_gau(str(model_dir / "variances"), self.variances)
        _pstrainc.write_mixw(str(model_dir / "mixture_weights"), self.mixw)
        _pstrainc.write_tmat(str(model_dir / "transition_matrices"), self.tmat)


class BWTrainer:
    """Baum-Welch trainer using CFFI.

    A trainer owns one mutable native session and must not be shared across
    threads. In debug mode, the retry transaction asserts this contract.
    """

    def __init__(
        self,
        mdef_path: Path,
        means_path: Path,
        vars_path: Path,
        mixw_path: Path,
        tmat_path: Path,
        config: BWConfig | None = None,
    ) -> None:
        """Initialize BW trainer.

        Args:
            mdef_path: Path to model definition file
            means_path: Path to means file
            vars_path: Path to variances file
            mixw_path: Path to mixture weights file
            tmat_path: Path to transition matrices file
            config: Training configuration
        """
        self.config = config or BWConfig()
        self._ffi, self._lib = _pstrainc._init()

        # Create C config struct
        c_config = self._ffi.new("pstrain_bw_config_t *")
        c_config.a_beam = self.config.a_beam
        c_config.b_beam = self.config.b_beam
        c_config.topn = self.config.topn
        c_config.spthresh = self.config.spthresh
        c_config.mixw_floor = self.config.mixw_floor
        c_config.tmat_floor = self.config.tmat_floor
        c_config.mean_reest = 1 if self.config.mean_reest else 0
        c_config.var_reest = 1 if self.config.var_reest else 0
        c_config.mixw_reest = 1 if self.config.mixw_reest else 0
        c_config.tmat_reest = 1 if self.config.tmat_reest else 0
        c_config.pass2var = 1 if self.config.pass2var else 0

        # Initialize C context
        self._ctx = self._lib.pstrain_bw_init(
            str(mdef_path).encode(),
            str(means_path).encode(),
            str(vars_path).encode(),
            str(mixw_path).encode(),
            str(tmat_path).encode(),
            c_config,
        )

        if self._ctx == self._ffi.NULL:
            raise RuntimeError("Failed to initialize BW context")

        # Apply multipron setting (separate from the C config struct).
        if self._lib.pstrain_bw_set_multipron(self._ctx, 1 if self.config.multipron else 0) != 0:
            raise RuntimeError("Failed to set multipron flag")

        self._dict_set = False
        self._last_process_result = 0
        self._retry_transaction_active = False

    def __del__(self) -> None:
        """Clean up C context."""
        if hasattr(self, "_ctx") and self._ctx != self._ffi.NULL:
            self._lib.pstrain_bw_free(self._ctx)

    def set_dict(self, dict_path: Path | str, filler_dict_path: Path | str | None = None) -> None:
        """Set pronunciation dictionary for text-based processing.

        Args:
            dict_path: Path to pronunciation dictionary
            filler_dict_path: Path to filler dictionary (optional)

        Raises:
            RuntimeError: If setting dictionary fails
        """
        filler = str(filler_dict_path).encode() if filler_dict_path else self._ffi.NULL
        ret = self._lib.pstrain_bw_set_dict(
            self._ctx,
            str(dict_path).encode(),
            filler,
        )
        if ret != 0:
            raise RuntimeError(f"Failed to set dictionary: {dict_path}")
        self._dict_set = True

    def process_utterance_text(
        self,
        features: npt.NDArray[np.float32],
        transcript: str,
    ) -> bool:
        """Process a single utterance with text transcript.

        Args:
            features: Feature array (n_frames, feat_dim) - must be 39-dim
            transcript: Word transcript (space-separated words)

        Returns:
            True on success

        Raises:
            RuntimeError: If dictionary not set
        """
        if not self._dict_set:
            raise RuntimeError("Dictionary not set. Call set_dict() first.")

        n_frames = features.shape[0]

        # Ensure contiguous array
        features = np.ascontiguousarray(features, dtype=np.float32)

        ret = self._lib.pstrain_bw_process_utt_text(
            self._ctx,
            self._ffi.cast("float *", features.ctypes.data),
            n_frames,
            transcript.encode(),
        )

        return bool(ret == 0)

    def process_utterance_mfcc(
        self,
        mfcc: npt.NDArray[np.float32],
        transcript: str,
    ) -> bool:
        """Process utterance from raw MFCC features (13-dim).

        Uses C feat module to apply CMN and compute deltas, exactly like SphinxTrain.

        Args:
            mfcc: Raw MFCC features (n_frames, 13) - NOT preprocessed
            transcript: Word transcript (space-separated words)

        Returns:
            True on success

        Raises:
            RuntimeError: If dictionary not set
        """
        if not self._dict_set:
            raise RuntimeError("Dictionary not set. Call set_dict() first.")

        n_frames = mfcc.shape[0]

        # Validate dimensions
        if mfcc.shape[1] != 13:
            raise ValueError(f"Expected 13-dim MFCCs, got {mfcc.shape[1]}")

        # Ensure contiguous array
        mfcc = np.ascontiguousarray(mfcc, dtype=np.float32)

        ret = self._lib.pstrain_bw_process_utt_mfcc(
            self._ctx,
            self._ffi.cast("float *", mfcc.ctypes.data),
            n_frames,
            transcript.encode(),
        )
        self._last_process_result = ret
        return bool(ret == 0)

    @property
    def final_state_not_reached(self) -> bool:
        """Whether forward pruning lost the final state on the last MFCC update."""
        return self._last_process_result == -2

    def set_a_beam(self, a_beam: float) -> float:
        """Set the forward beam and return its previous value."""
        if a_beam <= 0:
            raise ValueError("a_beam must be positive")
        previous = float(self._lib.pstrain_bw_set_a_beam(self._ctx, a_beam))
        if previous <= 0:
            raise RuntimeError("Failed to set forward beam")
        return previous

    def process_utterance(
        self,
        features: npt.NDArray[np.float32],
        phone_ids: npt.NDArray[np.uint32],
    ) -> bool:
        """Process a single utterance.

        Args:
            features: Feature array (n_frames, feat_dim)
            phone_ids: Phone ID sequence

        Returns:
            True on success
        """
        n_frames = features.shape[0]
        n_phones = len(phone_ids)

        # Ensure contiguous arrays
        features = np.ascontiguousarray(features, dtype=np.float32)
        phone_ids = np.ascontiguousarray(phone_ids, dtype=np.uint32)

        ret = self._lib.pstrain_bw_process_utt(
            self._ctx,
            self._ffi.cast("float *", features.ctypes.data),
            n_frames,
            self._ffi.cast("uint32 *", phone_ids.ctypes.data),
            n_phones,
        )

        return bool(ret == 0)

    def normalize(self) -> bool:
        """Normalize accumulators and update model.

        Returns:
            True on success
        """
        return bool(self._lib.pstrain_bw_normalize(self._ctx) == 0)

    def save(
        self,
        means_path: Path,
        vars_path: Path,
        mixw_path: Path,
        tmat_path: Path,
    ) -> bool:
        """Save trained model.

        Returns:
            True on success
        """
        return bool(
            self._lib.pstrain_bw_save(
                self._ctx,
                str(means_path).encode(),
                str(vars_path).encode(),
                str(mixw_path).encode(),
                str(tmat_path).encode(),
            )
            == 0
        )

    def get_stats(self) -> BWResult:
        """Get training statistics."""
        total_log_lik = self._ffi.new("float64 *")
        total_frames = self._ffi.new("uint32 *")
        total_utts = self._ffi.new("uint32 *")

        self._lib.pstrain_bw_get_stats(self._ctx, total_log_lik, total_frames, total_utts)

        n_frames = total_frames[0]
        avg = total_log_lik[0] / n_frames if n_frames > 0 else 0.0

        return BWResult(
            total_log_lik=total_log_lik[0],
            total_frames=n_frames,
            total_utts=total_utts[0],
            avg_log_prob=avg,
        )

    def save_density_counts(self, counts_path: Path) -> bool:
        """Save density counts for Gaussian splitting.

        Args:
            counts_path: Output path for density counts (typically gauden_counts)

        Returns:
            True on success
        """
        return bool(self._lib.pstrain_bw_save_counts(self._ctx, str(counts_path).encode()) == 0)
