"""Lightweight target declarations shared by pipeline clients."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetSpec:
    name: str
    kind: str  # "ci", "cd", "trees", "lm", "test", "package", "features"
    description: str
    n_density: int = 0
    n_senones: int = 0
    default: bool = False


TARGETS: list[TargetSpec] = [
    TargetSpec("split", "split", "Partition the corpus into train/test sets"),
    TargetSpec("flat", "ci", "Initial flat (uniform) model", n_density=1),
    TargetSpec("ci-1g", "ci", "CI model with 1 Gaussian per state", n_density=1),
    TargetSpec("ci-2g", "ci", "CI model with 2 Gaussians per state", n_density=2),
    TargetSpec("ci-4g", "ci", "CI model with 4 Gaussians per state", n_density=4),
    TargetSpec("ci-8g", "ci", "CI model with 8 Gaussians per state", n_density=8),
    TargetSpec("cd-untied", "cd", "CD untied (per-triphone) model", n_density=1),
    TargetSpec("cd-1g", "cd", "CD tied model with 1 Gaussian", n_density=1, n_senones=200),
    TargetSpec("cd-2g", "cd", "CD tied model with 2 Gaussians", n_density=2, n_senones=200),
    TargetSpec("cd-4g", "cd", "CD tied model with 4 Gaussians", n_density=4, n_senones=200),
    TargetSpec(
        "cd-8g",
        "cd",
        "CD tied model with 8 Gaussians (default)",
        n_density=8,
        n_senones=200,
        default=True,
    ),
    TargetSpec("cd-16g", "cd", "CD tied model with 16 Gaussians", n_density=16, n_senones=200),
    TargetSpec("cd-32g", "cd", "CD tied model with 32 Gaussians", n_density=32, n_senones=200),
    TargetSpec("features", "features", "Extract MFCC features for all audio"),
    TargetSpec("lm", "lm", "3-gram language model from training transcripts"),
    TargetSpec("test-ci-8g", "test", "Decode test set with ci-8g and write WER report"),
    TargetSpec("test-cd-8g", "test", "Decode test set with cd-8g and write WER report"),
    TargetSpec("package-ci-8g", "package", "Package ci-8g for distribution"),
    TargetSpec("package-cd-8g", "package", "Package cd-8g for distribution"),
    TargetSpec("package-cd-32g", "package", "Package cd-32g for distribution"),
]

DEFAULT_TARGET = next(spec.name for spec in TARGETS if spec.default)
