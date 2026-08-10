"""Model type definitions and registry.

Model types represent different kinds of acoustic models (CI, CD, etc.).
Each model type has its own training process, directory structure, and parameters.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self


class ModelType(StrEnum):
    """Acoustic model types."""

    CI = "ci"  # Context-Independent (monophone)
    CD = "cd"  # Context-Dependent (triphone)

    def __str__(self) -> str:
        return self.value

    @property
    def display_name(self) -> str:
        """Human-readable name for the model type."""
        return {
            ModelType.CI: "Context-Independent",
            ModelType.CD: "Context-Dependent",
        }[self]

    @property
    def default_topn(self) -> int:
        """Default top-n Gaussians for this model type."""
        return {
            ModelType.CI: 1,
            ModelType.CD: 4,
        }[self]

    def get_model_dir(self, experiment_dir: str, config: str) -> str:
        """Get the model directory for this type and config.

        Args:
            experiment_dir: Experiment directory path
            config: Model configuration name

        Returns:
            Path to model directory: {experiment_dir}/models/{model_type}/{config}/model/
        """
        from pathlib import Path

        return str(Path(experiment_dir) / "models" / self.value / config / "model")

    def get_flat_dir(self, experiment_dir: str, config: str) -> str:
        """Get the flat model directory for this type and config."""
        return f"{self.get_model_dir(experiment_dir, config)}/flat"

    def get_hmm_dir(self, experiment_dir: str, config: str) -> str:
        """Get the trained HMM model directory for this type and config."""
        return f"{self.get_model_dir(experiment_dir, config)}/hmm"

    @classmethod
    def from_string(cls, value: str) -> Self:
        """Parse model type from string.

        Args:
            value: String representation (e.g., "ci", "CD", "context-independent")

        Returns:
            ModelType enum value

        Raises:
            ValueError: If value is not a valid model type
        """
        value_lower = value.lower()
        # Try direct match
        for model_type in cls:
            if model_type.value.lower() == value_lower:
                return model_type
        # Try aliases
        aliases = {
            "context-independent": cls.CI,
            "monophone": cls.CI,
            "context-dependent": cls.CD,
            "triphone": cls.CD,
        }
        if value_lower in aliases:
            return aliases[value_lower]
        raise ValueError(
            f"Unknown model type: {value}. Valid types: {', '.join(mt.value for mt in cls)}"
        )
