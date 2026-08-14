"""Comparison utilities for debugging and verification.

Compare feature files, model parameters, and other artifacts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pstrain.lib._cffi import read_gau, read_mixw, read_tmat
from pstrain.lib.features import read_sphinx_mfc
from pstrain.lib.filetypes import FileType, detect_file_type
from pstrain.lib.model import MODEL_FILES_ALL, MODEL_FILES_REQUIRED
from pstrain.lib.similarity import (
    GaussianState,
    Senone,
    compare_senones,
)

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)


def format_json(data: Any, indent: int = 2, ensure_ascii: bool = False) -> str:
    """Format data as JSON with consistent settings."""
    return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)


__all__ = [
    "ArrayStats",
    "CompareResult",
    "ComponentCompare",
    "DimCompare",
    "ModelCompareResult",
    "array_stats",
    "compare_auto",
    "compare_features",
    "compare_gaussians",
    "compare_gaussians_detailed",
    "compare_senone_sets",
    "compare_mixw",
    "compare_models",
    "compare_tmat",
    "diagnose_gaussian_diff",
    "format_json",
    "load_senones",
    "load_senones_from_model",
    "print_gaussian_stats",
    "print_stats",
]


@dataclass
class ArrayStats:
    """Statistics for a single array."""

    shape: tuple[int, ...]
    min: float
    max: float
    mean: float
    std: float
    zeros_pct: float  # Percentage of zeros
    ones_pct: float  # Percentage of ones (for variances)

    def summary(self) -> str:
        """Return human-readable summary."""
        special = []
        if self.zeros_pct > 50:
            special.append(f"{self.zeros_pct:.0f}% zeros")
        if self.ones_pct > 50:
            special.append(f"{self.ones_pct:.0f}% ones")
        special_str = f" ({', '.join(special)})" if special else ""
        return f"range=[{self.min:.4f}, {self.max:.4f}], mean={self.mean:.4f}, std={self.std:.4f}{special_str}"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dict."""
        return {
            "shape": [int(x) for x in self.shape],
            "min": float(self.min),
            "max": float(self.max),
            "mean": float(self.mean),
            "std": float(self.std),
            "zeros_pct": float(self.zeros_pct),
            "ones_pct": float(self.ones_pct),
        }

    def to_json(self, indent: int = 2) -> str:
        """Return JSON string."""
        return format_json(self.to_dict(), indent=indent)


def array_stats(arr: npt.NDArray[np.floating]) -> ArrayStats:
    """Compute statistics for an array."""
    total = arr.size
    return ArrayStats(
        shape=arr.shape,
        min=float(arr.min()),
        max=float(arr.max()),
        mean=float(arr.mean()),
        std=float(arr.std()),
        zeros_pct=100.0 * np.sum(arr == 0) / total if total > 0 else 0,
        ones_pct=100.0 * np.sum(arr == 1) / total if total > 0 else 0,
    )


@dataclass
class CompareResult:
    """Result of comparing two arrays."""

    match: bool
    max_diff: float
    mean_diff: float
    shape_a: tuple[int, ...]
    shape_b: tuple[int, ...]
    max_diff_location: tuple[int, ...] | None = None
    value_a: float | None = None
    value_b: float | None = None
    stats_a: ArrayStats | None = None
    stats_b: ArrayStats | None = None

    def summary(self) -> str:
        """Return human-readable summary."""
        if self.shape_a != self.shape_b:
            return f"SHAPE MISMATCH: {self.shape_a} vs {self.shape_b}"
        if self.match:
            return f"MATCH (max_diff={self.max_diff:.2e})"
        # Format location as plain ints
        loc = tuple(int(x) for x in self.max_diff_location) if self.max_diff_location else None
        return (
            f"DIFFER: max={self.max_diff:.6f}, mean={self.mean_diff:.6f}, "
            f"at {loc}: {self.value_a:.6f} vs {self.value_b:.6f}"
        )

    def detailed_summary(self) -> str:
        """Return detailed summary with statistics."""
        lines = [self.summary()]
        if self.stats_a:
            lines.append(f"  A: {self.stats_a.summary()}")
        if self.stats_b:
            lines.append(f"  B: {self.stats_b.summary()}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dict."""
        return {
            "match": bool(self.match),
            "max_diff": float(self.max_diff),
            "mean_diff": float(self.mean_diff),
            "shape_a": [int(x) for x in self.shape_a],
            "shape_b": [int(x) for x in self.shape_b],
            "max_diff_location": [int(x) for x in self.max_diff_location]
            if self.max_diff_location
            else None,
            "value_a": float(self.value_a) if self.value_a is not None else None,
            "value_b": float(self.value_b) if self.value_b is not None else None,
            "stats_a": self.stats_a.to_dict() if self.stats_a else None,
            "stats_b": self.stats_b.to_dict() if self.stats_b else None,
        }

    def to_json(self, indent: int = 2) -> str:
        """Return JSON string."""
        return format_json(self.to_dict(), indent=indent)


def _compare_arrays(
    a: npt.NDArray[np.floating],
    b: npt.NDArray[np.floating],
    rtol: float = 1e-5,
    atol: float = 1e-8,
    include_stats: bool = True,
) -> CompareResult:
    """Compare two numpy arrays."""
    stats_a = array_stats(a) if include_stats else None
    stats_b = array_stats(b) if include_stats else None

    if a.shape != b.shape:
        return CompareResult(
            match=False,
            max_diff=float("inf"),
            mean_diff=float("inf"),
            shape_a=a.shape,
            shape_b=b.shape,
            stats_a=stats_a,
            stats_b=stats_b,
        )

    diff = np.abs(a - b)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    match = np.allclose(a, b, rtol=rtol, atol=atol)

    max_idx = np.unravel_index(np.argmax(diff), diff.shape)

    return CompareResult(
        match=match,
        max_diff=max_diff,
        mean_diff=mean_diff,
        shape_a=a.shape,
        shape_b=b.shape,
        max_diff_location=tuple(int(i) for i in max_idx),
        value_a=float(a[max_idx]),
        value_b=float(b[max_idx]),
        stats_a=stats_a,
        stats_b=stats_b,
    )


def compare_features(
    file_a: Path,
    file_b: Path,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> CompareResult:
    """Compare two feature files (.mfc format).

    Args:
        file_a: First feature file
        file_b: Second feature file
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        CompareResult with comparison details
    """
    feat_a = read_sphinx_mfc(file_a)
    feat_b = read_sphinx_mfc(file_b)
    return _compare_arrays(feat_a, feat_b, rtol, atol)


def compare_gaussians(
    file_a: Path,
    file_b: Path,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> CompareResult:
    """Compare two Gaussian parameter files (means or variances).

    Args:
        file_a: First Gaussian file
        file_b: Second Gaussian file
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        CompareResult with comparison details
    """
    gau_a, _, _, _, _ = read_gau(str(file_a))
    gau_b, _, _, _, _ = read_gau(str(file_b))
    return _compare_arrays(gau_a, gau_b, rtol, atol)


def compare_mixw(
    file_a: Path,
    file_b: Path,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> CompareResult:
    """Compare two mixture weight files.

    Args:
        file_a: First mixw file
        file_b: Second mixw file
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        CompareResult with comparison details
    """
    mixw_a, _, _, _ = read_mixw(str(file_a))
    mixw_b, _, _, _ = read_mixw(str(file_b))
    return _compare_arrays(mixw_a, mixw_b, rtol, atol)


@dataclass
class DimCompare:
    """Per-dimension comparison for Gaussian parameters."""

    dim: int
    name: str  # "static", "delta", "ddelta"
    mean_a: float
    mean_b: float
    diff: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "name": self.name,
            "mean_a": float(self.mean_a),
            "mean_b": float(self.mean_b),
            "diff": float(self.diff),
        }


def compare_gaussians_detailed(
    file_a: Path,
    file_b: Path,
) -> dict[str, Any]:
    """Compare two Gaussian files with per-dimension breakdown.

    Returns dict with overall comparison and per-dimension details.
    Useful for debugging flat model initialization.
    """
    gau_a, _, _, veclen_a, _ = read_gau(str(file_a))
    gau_b, _, _, veclen_b, _ = read_gau(str(file_b))

    result: dict[str, Any] = {
        "shape_a": list(gau_a.shape),
        "shape_b": list(gau_b.shape),
        "stats_a": array_stats(gau_a).to_dict(),
        "stats_b": array_stats(gau_b).to_dict(),
    }

    if gau_a.shape != gau_b.shape:
        result["match"] = False
        result["error"] = "Shape mismatch"
        return result

    # Per-dimension comparison (average across codebooks/densities)
    dim_means_a = gau_a.mean(axis=(0, 1, 2))
    dim_means_b = gau_b.mean(axis=(0, 1, 2))

    dims = []
    veclen = gau_a.shape[-1]
    for i in range(veclen):
        if veclen == 39:
            if i < 13:
                name = "static"
            elif i < 26:
                name = "delta"
            else:
                name = "ddelta"
        else:
            name = f"dim{i}"

        dims.append(
            DimCompare(
                dim=i,
                name=name,
                mean_a=dim_means_a[i],
                mean_b=dim_means_b[i],
                diff=abs(dim_means_a[i] - dim_means_b[i]),
            ).to_dict()
        )

    result["dimensions"] = dims
    result["max_dim_diff"] = max(d["diff"] for d in dims)
    result["match"] = result["max_dim_diff"] < 1e-5

    return result


def compare_tmat(
    file_a: Path,
    file_b: Path,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> CompareResult:
    """Compare two transition matrix files.

    Args:
        file_a: First tmat file
        file_b: Second tmat file
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        CompareResult with comparison details
    """
    tmat_a, _, _ = read_tmat(str(file_a))
    tmat_b, _, _ = read_tmat(str(file_b))
    return _compare_arrays(tmat_a, tmat_b, rtol, atol)


def diagnose_gaussian_diff(
    file_a: Path,
    file_b: Path,
    mdef_path: Path | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """Deep diagnostic comparison of Gaussian parameters.

    Provides detailed breakdown useful for debugging training divergence:
    - Relative differences (as percentage of magnitude)
    - Percentile distribution of differences
    - Per-codebook (phone/senone) breakdown
    - Identifies top divergent codebooks

    Args:
        file_a: First Gaussian file (means or variances)
        file_b: Second Gaussian file
        mdef_path: Optional mdef for phone names
        top_n: Number of top divergent codebooks to report

    Returns:
        Dict with diagnostic info
    """
    gau_a, n_cb, n_feat, n_den, veclen = read_gau(str(file_a))
    gau_b, _, _, _, _ = read_gau(str(file_b))

    if gau_a.shape != gau_b.shape:
        return {"error": f"Shape mismatch: {gau_a.shape} vs {gau_b.shape}"}

    # Absolute differences
    abs_diff = np.abs(gau_a - gau_b)

    # Relative differences (avoid division by zero)
    magnitude = np.maximum(np.abs(gau_a), np.abs(gau_b))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.where(magnitude > 1e-10, abs_diff / magnitude, 0)

    # Overall stats
    result: dict[str, Any] = {
        "shape": list(gau_a.shape),
        "n_codebooks": int(n_cb),
        "n_features": int(n_feat),
        "n_densities": int(n_den),
        "veclen": veclen[0] if veclen else 0,
        "abs_diff": {
            "max": float(abs_diff.max()),
            "mean": float(abs_diff.mean()),
            "percentiles": {
                "50": float(np.percentile(abs_diff, 50)),
                "90": float(np.percentile(abs_diff, 90)),
                "99": float(np.percentile(abs_diff, 99)),
            },
        },
        "rel_diff_pct": {
            "max": float(rel_diff.max() * 100),
            "mean": float(rel_diff.mean() * 100),
            "percentiles": {
                "50": float(np.percentile(rel_diff, 50) * 100),
                "90": float(np.percentile(rel_diff, 90) * 100),
                "99": float(np.percentile(rel_diff, 99) * 100),
            },
        },
    }

    # Per-codebook analysis
    cb_diffs = []
    for i in range(n_cb):
        cb_abs_diff = abs_diff[i].mean()
        cb_rel_diff = rel_diff[i].mean()
        cb_diffs.append(
            {
                "codebook": i,
                "abs_diff": float(cb_abs_diff),
                "rel_diff_pct": float(cb_rel_diff * 100),
                "max_abs_diff": float(abs_diff[i].max()),
            }
        )

    # Sort by absolute diff and get top divergent
    cb_diffs.sort(key=lambda x: x["abs_diff"], reverse=True)
    result["top_divergent_codebooks"] = cb_diffs[:top_n]

    # Count codebooks with significant divergence
    significant_threshold = abs_diff.mean() + 2 * abs_diff.std()
    n_significant = sum(1 for cb in cb_diffs if cb["abs_diff"] > significant_threshold)
    result["n_significantly_divergent"] = n_significant
    result["divergence_threshold"] = float(significant_threshold)

    return result


def print_stats(file_path: Path) -> None:
    """Print statistics for a single file.

    Detects file type and prints relevant statistics.

    Args:
        file_path: Path to file
    """
    file_type = detect_file_type(file_path)
    print(f"File: {file_path}")
    print(f"Type: {file_type.value}")

    if file_type == FileType.FEATURES:
        data = read_sphinx_mfc(file_path)
        stats = array_stats(data)
        print(f"Shape: {stats.shape} (frames x features)")
        print(f"Range: [{stats.min:.4f}, {stats.max:.4f}]")
        print(f"Mean: {stats.mean:.4f}, Std: {stats.std:.4f}")

    elif file_type in (FileType.MEANS, FileType.VARIANCES):
        data, n_cb, n_density, veclen, _ = read_gau(str(file_path))
        stats = array_stats(data)
        print(f"Shape: {n_cb} codebooks x {n_density} densities x {veclen} features")
        print(f"Range: [{stats.min:.4f}, {stats.max:.4f}]")
        print(f"Mean: {stats.mean:.4f}, Std: {stats.std:.4f}")
        if stats.zeros_pct > 50:
            print(f"WARNING: {stats.zeros_pct:.0f}% zeros (uninitialized?)")
        if stats.ones_pct > 50:
            print(f"WARNING: {stats.ones_pct:.0f}% ones (default variance?)")

    elif file_type == FileType.MIXTURE_WEIGHTS:
        data, n_mixw, n_feat, n_density = read_mixw(str(file_path))
        stats = array_stats(data)
        print(f"Shape: {n_mixw} states x {n_feat} features x {n_density} densities")
        print(f"Range: [{stats.min:.4f}, {stats.max:.4f}]")
        print(f"Mean: {stats.mean:.4f}, Std: {stats.std:.4f}")

    elif file_type == FileType.TRANSITION_MATRICES:
        data, n_tmat, n_state = read_tmat(str(file_path))
        stats = array_stats(data)
        print(f"Shape: {n_tmat} matrices x {n_state} states x {n_state} states")
        print(f"Range: [{stats.min:.4f}, {stats.max:.4f}]")

    else:
        print("(statistics not available for this file type)")


@dataclass
class ComponentCompare:
    """Comparison result for a single model component."""

    name: str
    exists_a: bool
    exists_b: bool
    result: CompareResult | None = None
    text_match: bool | None = None  # For text files

    @property
    def match(self) -> bool:
        """Return True if component matches."""
        if not self.exists_a and not self.exists_b:
            return True  # Both missing = match
        if self.exists_a != self.exists_b:
            return False  # One missing = no match
        if self.text_match is not None:
            return self.text_match
        if self.result is not None:
            return self.result.match
        return False

    def summary(self) -> str:
        """Return human-readable summary."""
        if not self.exists_a and not self.exists_b:
            return "not present in either"
        if not self.exists_a:
            return "MISSING in first model"
        if not self.exists_b:
            return "MISSING in second model"
        if self.text_match is not None:
            return "MATCH" if self.text_match else "DIFFER (text)"
        if self.result is not None:
            return self.result.summary()
        return "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dict."""
        return {
            "name": self.name,
            "exists_a": self.exists_a,
            "exists_b": self.exists_b,
            "match": self.match,
            "result": self.result.to_dict() if self.result else None,
            "text_match": self.text_match,
        }


@dataclass
class ModelCompareResult:
    """Result of comparing two complete models."""

    dir_a: Path
    dir_b: Path
    components: dict[str, ComponentCompare]
    topology_compatible: bool  # mdef + feat.params match

    @property
    def all_match(self) -> bool:
        """Return True if all components match."""
        return all(c.match for c in self.components.values())

    @property
    def critical_match(self) -> bool:
        """Return True if critical files (mdef, feat.params) match."""
        critical = ["mdef", "feat.params"]
        return all(
            self.components.get(c, ComponentCompare(c, False, False)).match
            for c in critical
            if c in self.components
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dict."""
        return {
            "dir_a": str(self.dir_a),
            "dir_b": str(self.dir_b),
            "components": {name: comp.to_dict() for name, comp in self.components.items()},
            "topology_compatible": self.topology_compatible,
            "all_match": self.all_match,
            "critical_match": self.critical_match,
        }

    def to_json(self, indent: int = 2) -> str:
        """Return JSON string."""
        return format_json(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = []

        # Group by status
        matching = []
        differing = []
        missing = []

        for name, comp in sorted(self.components.items()):
            if not comp.exists_a or not comp.exists_b:
                missing.append((name, comp))
            elif comp.match:
                matching.append((name, comp))
            else:
                differing.append((name, comp))

        # Overall status
        if self.all_match:
            lines.append("ALL MATCH")
        elif self.topology_compatible:
            lines.append("TOPOLOGY COMPATIBLE (parameters differ)")
        else:
            lines.append("TOPOLOGY MISMATCH (models incompatible)")

        # Critical files first
        lines.append("\nCritical files:")
        for name in ["mdef", "feat.params"]:
            if name in self.components:
                comp = self.components[name]
                status = "✓" if comp.match else "✗"
                lines.append(f"  {status} {name}: {comp.summary()}")

        # Parameters
        param_files = ["means", "variances", "mixture_weights", "transition_matrices", "sendump"]
        present_params = [n for n in param_files if n in self.components]
        if present_params:
            lines.append("\nParameters:")
            for name in present_params:
                comp = self.components[name]
                status = "✓" if comp.match else "✗"
                lines.append(f"  {status} {name}: {comp.summary()}")

        # Other files
        other = [n for n in self.components if n not in ["mdef", "feat.params"] + param_files]
        if other:
            lines.append("\nOther files:")
            for name in sorted(other):
                comp = self.components[name]
                status = "✓" if comp.match else "✗"
                lines.append(f"  {status} {name}: {comp.summary()}")

        return "\n".join(lines)


def _discover_model_files(model_dir: Path) -> set[str]:
    """Discover all model-related files in a directory."""
    known_files = set(MODEL_FILES_ALL) | {"README", "topo", "provenance.json"}
    found = set()
    for f in model_dir.iterdir():
        if f.is_file() and (f.name in known_files or f.suffix in (".mfc", ".dict")):
            found.add(f.name)
    return found


def _compare_text_file(file_a: Path, file_b: Path) -> bool:
    """Compare two text files, ignoring trailing whitespace."""
    try:
        text_a = file_a.read_text().rstrip()
        text_b = file_b.read_text().rstrip()
        return text_a == text_b
    except Exception:
        return False


def compare_models(
    dir_a: Path,
    dir_b: Path,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> ModelCompareResult:
    """Compare two complete model directories.

    Compares all model components:
    - Critical: mdef (topology), feat.params (feature config)
    - Parameters: means, variances, mixture_weights, transition_matrices, sendump
    - Other: noisedict, README, etc.

    Args:
        dir_a: First model directory
        dir_b: Second model directory
        rtol: Relative tolerance for numeric comparisons
        atol: Absolute tolerance for numeric comparisons

    Returns:
        ModelCompareResult with detailed comparison of all components
    """
    dir_a = Path(dir_a)
    dir_b = Path(dir_b)

    # Discover files in both directories
    files_a = _discover_model_files(dir_a)
    files_b = _discover_model_files(dir_b)
    all_files = files_a | files_b

    components: dict[str, ComponentCompare] = {}

    # Text files (exact match)
    text_files = {
        "mdef",
        "feat.params",
        "noisedict",
        "README",
        "topo",
        "provenance.json",
    }

    # Binary parameter files (numeric comparison)
    gau_files = {"means", "variances"}
    mixw_files = {"mixture_weights", "sendump"}
    tmat_files = {"transition_matrices"}

    for filename in all_files:
        exists_a = filename in files_a
        exists_b = filename in files_b

        if not exists_a or not exists_b:
            components[filename] = ComponentCompare(
                name=filename,
                exists_a=exists_a,
                exists_b=exists_b,
            )
            continue

        file_a = dir_a / filename
        file_b = dir_b / filename

        if filename in text_files:
            text_match = _compare_text_file(file_a, file_b)
            components[filename] = ComponentCompare(
                name=filename,
                exists_a=True,
                exists_b=True,
                text_match=text_match,
            )
        elif filename in gau_files:
            result = compare_gaussians(file_a, file_b, rtol, atol)
            components[filename] = ComponentCompare(
                name=filename,
                exists_a=True,
                exists_b=True,
                result=result,
            )
        elif filename in mixw_files:
            result = compare_mixw(file_a, file_b, rtol, atol)
            components[filename] = ComponentCompare(
                name=filename,
                exists_a=True,
                exists_b=True,
                result=result,
            )
        elif filename in tmat_files:
            result = compare_tmat(file_a, file_b, rtol, atol)
            components[filename] = ComponentCompare(
                name=filename,
                exists_a=True,
                exists_b=True,
                result=result,
            )
        else:
            # Unknown file - try text comparison
            text_match = _compare_text_file(file_a, file_b)
            components[filename] = ComponentCompare(
                name=filename,
                exists_a=True,
                exists_b=True,
                text_match=text_match,
            )

    # Determine topology compatibility
    mdef_comp = components.get("mdef")
    feat_comp = components.get("feat.params")
    topology_compatible = (mdef_comp is None or mdef_comp.match) and (
        feat_comp is None or feat_comp.match
    )

    return ModelCompareResult(
        dir_a=dir_a,
        dir_b=dir_b,
        components=components,
        topology_compatible=topology_compatible,
    )


def print_feature_stats(file_path: Path) -> None:
    """Print statistics for a feature file."""
    feat = read_sphinx_mfc(file_path)
    print(f"File: {file_path}")
    print(f"  Shape: {feat.shape}")
    print(f"  Range: [{feat.min():.4f}, {feat.max():.4f}]")
    print(f"  Mean: {feat.mean():.4f}")
    print(f"  Std: {feat.std():.4f}")
    print(f"  First frame: {feat[0, :5]}")
    print(f"  Last frame: {feat[-1, :5]}")


def print_gaussian_stats(file_path: Path, per_dim: bool = False) -> None:
    """Print statistics for a Gaussian parameter file.

    Args:
        file_path: Path to Gaussian file (means or variances)
        per_dim: If True, show per-dimension statistics
    """
    gau, n_cb, n_density, veclen, _ = read_gau(str(file_path))
    print(f"File: {file_path}")
    print(f"  Shape: {gau.shape}")
    print(f"  n_cb={n_cb}, n_density={n_density}, veclen={veclen}")
    print(f"  Range: [{gau.min():.4f}, {gau.max():.4f}]")
    print(f"  Mean: {gau.mean():.4f}")
    print(f"  Std: {gau.std():.4f}")

    if per_dim and veclen == 39:
        # Show per-dimension stats for 39-dim features (13 cep + 13 delta + 13 dd)
        # Average across all codebooks/densities
        dim_means = gau.mean(axis=(0, 1, 2))
        print("  Per-dimension means:")
        print(f"    Static (0-12):  [{dim_means[0]:.6f} ... {dim_means[12]:.6f}]")
        print(f"    Delta (13-25):  [{dim_means[13]:.6f} ... {dim_means[25]:.6f}]")
        print(f"    DDelta (26-38): [{dim_means[26]:.6f} ... {dim_means[38]:.6f}]")


def compare_auto(
    file_a: Path,
    file_b: Path,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> tuple[FileType, CompareResult | ModelCompareResult]:
    """Auto-detect file types and compare.

    Args:
        file_a: First file or directory
        file_b: Second file or directory
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Tuple of (detected_type, comparison_result)

    Raises:
        ValueError: If file types don't match or are unknown
    """
    type_a = detect_file_type(file_a)
    type_b = detect_file_type(file_b)

    if type_a == FileType.UNKNOWN:
        raise ValueError(f"Cannot detect type of: {file_a}")
    if type_b == FileType.UNKNOWN:
        raise ValueError(f"Cannot detect type of: {file_b}")
    if type_a != type_b:
        raise ValueError(f"Type mismatch: {file_a} is {type_a.value}, {file_b} is {type_b.value}")

    if type_a == FileType.FEATURES:
        return (type_a, compare_features(file_a, file_b, rtol, atol))
    if type_a in (FileType.MEANS, FileType.VARIANCES):
        return (type_a, compare_gaussians(file_a, file_b, rtol, atol))
    if type_a == FileType.MIXTURE_WEIGHTS:
        return (type_a, compare_mixw(file_a, file_b, rtol, atol))
    if type_a == FileType.TRANSITION_MATRICES:
        return (type_a, compare_tmat(file_a, file_b, rtol, atol))
    if type_a == FileType.MODEL:
        return (type_a, compare_models(file_a, file_b, rtol, atol))
    if type_a == FileType.MDEF:
        # Simple text comparison for mdef
        text_a = Path(file_a).read_text()
        text_b = Path(file_b).read_text()
        match = text_a == text_b
        return (
            type_a,
            CompareResult(
                match=match,
                max_diff=0.0 if match else float("inf"),
                mean_diff=0.0 if match else float("inf"),
                shape_a=(len(text_a),),
                shape_b=(len(text_b),),
            ),
        )
    raise ValueError(f"Comparison not supported for type: {type_a.value}")


def print_model_stats(file_path: Path) -> None:
    """Print statistics for any supported file type (extended version)."""
    file_type = detect_file_type(file_path)

    if file_type == FileType.FEATURES:
        print_feature_stats(file_path)
    elif file_type in (FileType.MEANS, FileType.VARIANCES):
        print_gaussian_stats(file_path)
    elif file_type == FileType.MODEL:
        print(f"Model directory: {file_path}")
        for component in MODEL_FILES_REQUIRED:
            component_path = file_path / component
            if component_path.exists():
                print(f"\n{component}:")
                if component in ("means", "variances"):
                    print_gaussian_stats(component_path)
                elif component == "mdef":
                    lines = component_path.read_text().strip().split("\n")
                    print(f"  {len(lines)} lines")
    else:
        print(f"Unknown file type ({file_type.value}): {file_path}")


# =============================================================================
# Gaussian/GMM loading from model files
# =============================================================================


def load_senones(
    means_path: Path,
    variances_path: Path,
    mixw_path: Path | None = None,
) -> list[Senone]:
    """Load senones (tied states) from model files.

    Each senone contains a Gaussian mixture. For CI models, there's
    one senone per phone state (n_phones * n_states_per_phone).

    Args:
        means_path: Path to means file
        variances_path: Path to variances file
        mixw_path: Optional path to mixture weights file

    Returns:
        List of Senone objects, one per tied state
    """
    # Read raw arrays
    # read_gau returns: (array, n_mgau, n_feat, n_density, veclen_list)
    # array shape: (n_mgau, n_feat, n_density, veclen)
    means, n_mgau, n_feat, n_density, veclen = read_gau(str(means_path))
    variances, _, _, _, _ = read_gau(str(variances_path))

    # Read mixture weights if provided
    if mixw_path and Path(mixw_path).exists():
        mixw, _, _, n_density_mw = read_mixw(str(mixw_path))
    else:
        # Uniform weights
        mixw = (np.ones((n_mgau, n_feat, n_density), dtype=np.float32) / n_density).astype(
            np.float32
        )

    # means/variances shape: (n_mgau, n_feat, n_density, veclen)
    # mixw shape: (n_mgau, n_feat, n_density)

    senones = []

    for cb_idx in range(n_mgau):
        # Get raw weights for this senone and normalize
        raw_weights = []
        for den_idx in range(n_density):
            w = float(mixw[cb_idx, 0, den_idx]) if cb_idx < mixw.shape[0] else 1.0
            raw_weights.append(w)

        # Normalize weights to sum to 1
        total = sum(raw_weights)
        weights = [w / total for w in raw_weights] if total > 0 else [1.0 / n_density] * n_density

        gaussians = []
        for den_idx in range(n_density):
            mean = means[cb_idx, 0, den_idx, :]  # stream 0
            var = variances[cb_idx, 0, den_idx, :]

            gaussians.append(
                GaussianState(
                    mean=mean.astype(np.float64),
                    variance=var.astype(np.float64),
                    weight=weights[den_idx],
                )
            )

        senones.append(Senone(gaussians=gaussians))

    return senones


def load_senones_from_model(model_dir: Path) -> list[Senone]:
    """Load all senones from a model directory.

    Args:
        model_dir: Path to acoustic model directory (means, variances, mixture_weights)

    Returns:
        List of Senone objects, one per tied state
    """
    model_dir = Path(model_dir)

    means_path = model_dir / "means"
    variances_path = model_dir / "variances"
    mixw_path = model_dir / "mixture_weights"

    if not means_path.exists():
        raise FileNotFoundError(f"Means not found: {means_path}")
    if not variances_path.exists():
        raise FileNotFoundError(f"Variances not found: {variances_path}")

    return load_senones(
        means_path,
        variances_path,
        mixw_path if mixw_path.exists() else None,
    )


def compare_senone_sets(
    senones_a: list[Senone],
    senones_b: list[Senone],
    method: Literal["collapsed", "monte_carlo", "component_matching"] = "collapsed",
) -> dict[str, Any]:
    """Compare two sets of senones (e.g., from two acoustic models).

    Args:
        senones_a: First list of senones
        senones_b: Second list of senones
        method: Comparison method for individual senones

    Returns:
        Dictionary with overall comparison statistics
    """
    if len(senones_a) != len(senones_b):
        return {
            "match": False,
            "error": f"Different number of senones: {len(senones_a)} vs {len(senones_b)}",
        }

    distances = []
    comparisons = []

    for senone_a, senone_b in zip(senones_a, senones_b, strict=False):
        result = compare_senones(senone_a, senone_b, method=method)
        comparisons.append(result)

        # Use Bhattacharyya distance as primary metric
        if "bhattacharyya" in result:
            distances.append(result["bhattacharyya"])
        elif "mean_matched_distance" in result:
            distances.append(result["mean_matched_distance"])

    return {
        "n_senones": len(senones_a),
        "method": method,
        "mean_distance": float(np.mean(distances)) if distances else 0.0,
        "max_distance": float(np.max(distances)) if distances else 0.0,
        "min_distance": float(np.min(distances)) if distances else 0.0,
        "std_distance": float(np.std(distances)) if distances else 0.0,
        "per_senone_comparisons": comparisons,
    }
