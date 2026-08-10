"""Three-tier configuration system: User → Project → Experiment.

Configuration Hierarchy:
1. User Config (~/.pstrain/config.yaml)
   - Global defaults for all projects
   - Keys: defaults.*, cache_dir

2. Project Config (project/etc/config.yaml)
   - Shared settings across all experiments in a project
   - Keys: audio.*, dictionary.*, corpus.*

3. Experiment Config (project/experiments/{name}/config.yaml)
   - Specific experimental settings
   - Keys: parallel.*, training.*, features.*, split.*

Merging: experiment > project > user (later overrides earlier)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pstrain.lib.config.models import PstrainConfig
from pstrain.lib.config.user import get_user_config


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dicts, with override taking precedence.

    Args:
        base: Base dictionary
        override: Override dictionary (values here win)

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class ConfigManager:
    """Manage three-tier configuration system."""

    PROJECT_CONFIG_FILE = "etc/config.yaml"
    EXPERIMENTS_DIR = "experiments"

    @classmethod
    def get_project_config_path(cls, project_dir: Path) -> Path:
        """Get path to project config file."""
        return project_dir / cls.PROJECT_CONFIG_FILE

    @classmethod
    def get_experiment_config_path(cls, project_dir: Path, experiment: str) -> Path:
        """Get path to experiment config file."""
        return project_dir / cls.EXPERIMENTS_DIR / experiment / "config.yaml"

    @classmethod
    def load_project_config(cls, project_dir: Path) -> dict[str, Any]:
        """Load project-level configuration.

        Args:
            project_dir: Project directory

        Returns:
            Project config dict (empty if file doesn't exist)
        """
        config_file = cls.get_project_config_path(project_dir)
        if config_file.exists():
            with config_file.open(encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    @classmethod
    def load_experiment_config(cls, project_dir: Path, experiment: str) -> dict[str, Any]:
        """Load experiment-level configuration.

        Args:
            project_dir: Project directory
            experiment: Experiment name

        Returns:
            Experiment config dict (empty if file doesn't exist)
        """
        config_file = cls.get_experiment_config_path(project_dir, experiment)
        if config_file.exists():
            with config_file.open(encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    @classmethod
    def merge_configs(
        cls,
        user_defaults: dict[str, Any],
        project_config: dict[str, Any],
        experiment_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge three-tier configs with proper precedence.

        Precedence: experiment > project > user

        Args:
            user_defaults: User-level defaults
            project_config: Project-level config
            experiment_config: Experiment-level config

        Returns:
            Merged configuration dict
        """
        # Start with user defaults, then apply project, then experiment
        merged: dict[str, Any] = {}
        merged = _deep_merge(merged, user_defaults)
        merged = _deep_merge(merged, project_config)
        return _deep_merge(merged, experiment_config)

    @classmethod
    def load_full_config(
        cls,
        project_dir: Path,
        experiment: str | None = None,
    ) -> PstrainConfig:
        """Load complete configuration with three-tier merging.

        Args:
            project_dir: Project directory
            experiment: Experiment name (if None, uses project config only)

        Returns:
            Complete PstrainConfig object bound to project
        """
        project_dir = Path(project_dir).resolve()

        # Load user defaults
        user_config = get_user_config()
        user_defaults = cls._user_config_to_dict(user_config)

        # Load project config
        project_config = cls.load_project_config(project_dir)

        # Load experiment config if specified
        experiment_config: dict[str, Any] = {}
        config_file: Path | None = None
        if experiment:
            experiment_config = cls.load_experiment_config(project_dir, experiment)
            config_file = cls.get_experiment_config_path(project_dir, experiment)

        # Merge
        merged = cls.merge_configs(user_defaults, project_config, experiment_config)

        # Create PstrainConfig from merged dict
        config = PstrainConfig.model_validate(merged)

        # Set experiment name if not in config
        if experiment and not config.name:
            config.name = experiment

        # Bind to project
        config.bind_to_project(project_dir, config_file)

        return config

    @classmethod
    def _user_config_to_dict(cls, user_config: Any) -> dict[str, Any]:
        """Convert user config defaults to PstrainConfig-compatible dict.

        Maps user default keys to PstrainConfig structure.

        Args:
            user_config: PstrainUserConfig instance

        Returns:
            Dict compatible with PstrainConfig structure
        """
        defaults = user_config.defaults

        return {
            "parallel": {
                "n_jobs": defaults.n_jobs,
            },
            "audio": {
                "sample_rate": defaults.sample_rate,
            },
            "features": {
                "feature_type": defaults.feature_type,
            },
            "training": {
                "n_states": defaults.n_states,
                "ci": {
                    "n_iterations": defaults.n_iterations,
                },
                "cd": {
                    "untied": {
                        "n_iterations": defaults.n_iterations,
                    },
                    "tied": {
                        "n_iterations": defaults.n_iterations,
                    },
                },
            },
            "split": {
                "train_ratio": defaults.train_split,
            },
        }

    @classmethod
    def save_project_config(cls, project_dir: Path, config: PstrainConfig) -> Path:
        """Save project-level configuration.

        Args:
            project_dir: Project directory
            config: Configuration to save

        Returns:
            Path to saved config file
        """
        config_file = cls.get_project_config_path(project_dir)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with config_file.open("w", encoding="utf-8") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)

        return config_file

    @classmethod
    def save_experiment_config(
        cls,
        project_dir: Path,
        experiment: str,
        config: PstrainConfig,
    ) -> Path:
        """Save experiment-level configuration.

        Args:
            project_dir: Project directory
            experiment: Experiment name
            config: Configuration to save

        Returns:
            Path to saved config file
        """
        config_file = cls.get_experiment_config_path(project_dir, experiment)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with config_file.open("w", encoding="utf-8") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)

        return config_file

    @classmethod
    def list_experiments(cls, project_dir: Path) -> list[str]:
        """List all experiments in a project.

        Args:
            project_dir: Project directory

        Returns:
            List of experiment names
        """
        experiments_dir = project_dir / cls.EXPERIMENTS_DIR
        if not experiments_dir.exists():
            return []

        experiments = []
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir():
                config_file = exp_dir / "config.yaml"
                if config_file.exists():
                    experiments.append(exp_dir.name)

        return sorted(experiments)

    @classmethod
    def create_experiment(
        cls,
        project_dir: Path,
        experiment: str,
        base_config: PstrainConfig | None = None,
    ) -> Path:
        """Create a new experiment configuration.

        Args:
            project_dir: Project directory
            experiment: Experiment name
            base_config: Base configuration (uses project config if None)

        Returns:
            Path to created config file

        Raises:
            FileExistsError: If experiment already exists
        """
        config_file = cls.get_experiment_config_path(project_dir, experiment)

        if config_file.exists():
            raise FileExistsError(f"Experiment already exists: {experiment}")

        if base_config is None:
            # Load project config as base
            base_config = cls.load_full_config(project_dir)

        base_config.name = experiment
        return cls.save_experiment_config(project_dir, experiment, base_config)
