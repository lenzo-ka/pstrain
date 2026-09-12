"""Opt-in variance bounds relative to a fixed, matching one-density model."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pstrain.lib import _pstrainc
from pstrain.lib.bw import BW_FEATURE_LENGTH


def _read_variances(path: Path) -> tuple[npt.NDArray[np.float32], int, int]:
    values, codebooks, streams, densities, widths = _pstrainc.read_gau(str(path))
    if streams != 1 or widths != [BW_FEATURE_LENGTH] or codebooks < 1 or densities < 1:
        raise ValueError(
            f"Variance regularization requires nonempty single-stream {BW_FEATURE_LENGTH}-dimensional "
            f"Gaussians; {path} has {codebooks} codebooks, {streams} streams, "
            f"{densities} densities and widths {widths}"
        )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"Variances must be finite and nonnegative before regularization: {path}")
    return values, codebooks, densities


@dataclass(frozen=True)
class VarianceFloor:
    """A fixed lower bound; zero reference coordinates supply no positive floor."""

    reference_path: Path
    reference_sha256: str
    fraction: float
    lower_bound: npt.NDArray[np.float32]

    @classmethod
    def load(cls, reference_path: Path, fraction: float, initial_variances: Path) -> VarianceFloor:
        reference, codebooks, densities = _read_variances(reference_path)
        if densities != 1:
            raise ValueError(
                f"Variance floor reference must have exactly one density: {reference_path}"
            )
        candidate_codebooks = codebooks
        if initial_variances != reference_path:
            _, candidate_codebooks, _ = _read_variances(initial_variances)
        if candidate_codebooks != codebooks:
            raise ValueError(
                "Variance floor reference and model must have matching codebook counts"
            )
        with reference_path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256")
        # Compute before the final storage conversion, so a small fraction does
        # not underflow to zero before multiplying a large reference variance.
        bound = (reference.astype(np.float64) * fraction).astype(np.float32)
        bound.flags.writeable = False
        return cls(reference_path, digest.hexdigest(), fraction, bound)

    def apply(self, candidate_path: Path) -> int:
        """Validate and bound a staged variance file; return changed coordinate count."""
        values, codebooks, _ = _read_variances(candidate_path)
        if codebooks != self.lower_bound.shape[0]:
            raise ValueError(
                "Variance floor reference and candidate must have matching codebook counts"
            )
        changed = int(np.count_nonzero(values < self.lower_bound))
        if changed:
            np.maximum(values, self.lower_bound, out=values)
            if _pstrainc.write_gau(str(candidate_path), values) != 0:
                raise RuntimeError(f"Failed to write regularized variances: {candidate_path}")
        return changed

    def metadata(self) -> dict[str, object]:
        return {
            "reference": str(self.reference_path),
            "reference_sha256": self.reference_sha256,
            "fraction": self.fraction,
            "zero_floor_coordinates": int(np.count_nonzero(self.lower_bound == 0)),
        }


def load_variance_floor(
    reference_path: Path | None, fraction: float, initial_variances: Path
) -> VarianceFloor | None:
    """Resolve the explicit opt-in without reading files when disabled."""
    if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("variance_floor_fraction must be finite and between zero and one")
    if fraction == 0:
        return None
    if reference_path is None:
        raise ValueError(
            "variance_floor_reference is required when variance_floor_fraction is nonzero"
        )
    return VarianceFloor.load(Path(reference_path), fraction, initial_variances)
