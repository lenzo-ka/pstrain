"""User-wide ST2 configuration (~/.st2/config.yaml).

Global defaults that apply to all projects unless overridden.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, Field


class JsonOutputConfig(BaseModel):
    """JSON output formatting options."""

    indent: int = Field(
        2,
        ge=0,
        description="JSON indentation level (0 for compact)",
    )
    ensure_ascii: bool = Field(
        False,
        description="Escape non-ASCII characters in output",
    )


class UserDefaults(BaseModel):
    """Default settings for new projects."""

    # Split
    train_split: float = Field(
        0.95,
        ge=0.0,
        le=1.0,
        description="Default train/test split ratio (0.95 = 95% train)",
    )

    # Audio
    sample_rate: int = Field(
        16000,
        description="Default audio sample rate in Hz",
    )

    # Features
    feature_type: str = Field(
        "1s_c_d_dd",
        description="Default feature type (1s_c_d_dd or s2_4x)",
    )

    # Training
    n_iterations: int = Field(
        10,
        ge=1,
        le=100,
        description="Default max iterations for training phases",
    )
    n_states: int = Field(
        3,
        ge=1,
        le=7,
        description="Default number of HMM states",
    )

    # Parallel
    n_jobs: int = Field(
        -1,
        description="Default number of parallel jobs (-1 = all cores minus 1)",
    )


class ST2UserConfig(BaseModel):
    """User-wide ST2 configuration.

    Stored in ~/.st2/config.yaml
    """

    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "st2",
        description="Cache directory for downloads, intermediate files",
    )

    defaults: UserDefaults = Field(
        default_factory=UserDefaults,
        description="Default settings for new projects",
    )

    json_output: JsonOutputConfig = Field(
        default_factory=JsonOutputConfig,
        description="JSON output formatting options",
    )

    @classmethod
    def get_config_dir(cls) -> Path:
        """Get path to user config directory."""
        config_dir = Path.home() / ".st2"
        return config_dir

    @classmethod
    def get_config_file(cls) -> Path:
        """Get path to user config file."""
        return cls.get_config_dir() / "config.yaml"

    @classmethod
    def load(cls, config_file: Path | None = None) -> Self:
        """Load user configuration.

        Args:
            config_file: Explicit config file (bypasses default location)

        Returns:
            ST2UserConfig instance
        """
        if config_file is None:
            config_file = cls.get_config_file()

        if config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls.model_validate(data)

        # Return defaults if no config exists
        return cls()

    def save(self, config_file: Path | None = None) -> None:
        """Save user configuration to file.

        Args:
            config_file: Explicit config file (bypasses default location)
        """
        if config_file is None:
            config_file = self.get_config_file()

        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Ensure cache directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Convert to dict with string paths
        data: dict[str, Any] = {}
        for key, value in self.model_dump().items():
            if isinstance(value, Path):
                data[key] = str(value)
            else:
                data[key] = value

        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Module-level cache
_user_config_cache: ST2UserConfig | None = None


def get_user_config(config_file: Path | None = None, force_reload: bool = False) -> ST2UserConfig:
    """Get user configuration instance with caching.

    Args:
        config_file: Path to config file (overrides default lookup)
        force_reload: Force reload from disk (bypass cache)

    Returns:
        ST2UserConfig instance
    """
    global _user_config_cache

    if force_reload or _user_config_cache is None or config_file is not None:
        _user_config_cache = ST2UserConfig.load(config_file)

    return _user_config_cache
