"""User-wide pstrain configuration (~/.pstrain/config.yaml).

Global defaults that apply to all projects unless overridden.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field

from pstrain.lib.config.models import SEMANTIC_BLOCKS, OverlayDocument


class JsonOutputConfig(BaseModel):
    """JSON output formatting options."""

    model_config = ConfigDict(extra="forbid")

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
    """Legacy preferences retained for API compatibility, not training overrides."""

    model_config = ConfigDict(extra="forbid")

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
        description="Default number of parallel jobs (-1 = CPU count minus 2)",
    )
    nice: int = Field(5, ge=0, description="Default POSIX worker niceness; 0 disables")


class PstrainUserConfig(OverlayDocument):
    """User-wide pstrain configuration.

    Stored in ~/.pstrain/config.yaml
    """

    # This document owns both semantic overrides and presentation preferences.
    # It remains mutable for callers that edit settings before save().
    model_config = ConfigDict(extra="forbid", frozen=False)

    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "pstrain",
        description="Cache directory for downloads, intermediate files",
    )

    defaults: UserDefaults = Field(
        default_factory=UserDefaults,
        description="Legacy preferences; use canonical semantic blocks for training overrides",
    )

    json_output: JsonOutputConfig = Field(
        default_factory=JsonOutputConfig,
        description="JSON output formatting options",
    )

    @classmethod
    def get_config_dir(cls) -> Path:
        """Get path to user config directory."""
        return Path.home() / ".pstrain"

    @classmethod
    def get_config_file(cls) -> Path:
        """Get path to user config file."""
        return Path(os.environ.get("PSTRAIN_USER_CONFIG", cls.get_config_dir() / "config.yaml"))

    @classmethod
    def load(cls, config_file: Path | None = None) -> Self:
        """Load user configuration.

        Args:
            config_file: Explicit config file (bypasses default location)

        Returns:
            PstrainUserConfig instance
        """
        if config_file is None:
            config_file = cls.get_config_file()

        if config_file.exists():
            from pstrain.lib.config.resolver import _load_yaml

            return cls.from_document(_load_yaml(config_file), config_file)

        # Return defaults if no config exists
        return cls()

    @classmethod
    def from_document(cls, data: dict[str, Any], path: Path) -> Self:
        """Read a versioned user document or migrate effective legacy fields.

        Legacy defaults remain available for callers inspecting old settings;
        they were never effective training overrides and do not become so here.
        """
        if "config_version" not in data:
            from pstrain.lib.config.resolver import _legacy_inactive_overlay

            data = {
                **_legacy_inactive_overlay(data, path),
                **{
                    key: data[key]
                    for key in ("cache_dir", "defaults", "json_output")
                    if key in data
                },
            }
        return cls.model_validate(data)

    def semantic_overlay(self) -> dict[str, Any]:
        """Return only explicitly present training/configuration blocks."""
        return self.model_dump(include=set(SEMANTIC_BLOCKS), exclude_none=True)

    def save(self, config_file: Path | None = None) -> None:
        """Atomically save semantic overrides and presentation preferences."""
        from pstrain.lib.config.resolver import _atomic_write

        config_file = config_file or self.get_config_file()
        # Validate the current mutable contents before touching the destination.
        document = type(self).model_validate(self.model_dump())
        config_file.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            config_file,
            yaml.safe_dump(document.model_dump(mode="json", exclude_none=True), sort_keys=False),
        )


# Module-level cache
_user_config_cache: PstrainUserConfig | None = None
_user_config_path: Path | None = None


def get_user_config(
    config_file: Path | None = None, force_reload: bool = False
) -> PstrainUserConfig:
    """Get user configuration instance with caching.

    Args:
        config_file: Path to config file (overrides default lookup)
        force_reload: Force reload from disk (bypass cache)

    Returns:
        PstrainUserConfig instance
    """
    global _user_config_cache, _user_config_path

    resolved_path = (config_file or PstrainUserConfig.get_config_file()).resolve()
    if (
        force_reload
        or _user_config_cache is None
        or config_file is not None
        or _user_config_path != resolved_path
    ):
        _user_config_cache = PstrainUserConfig.load(resolved_path)
        _user_config_path = resolved_path

    return _user_config_cache
