"""Public API for configuration operations."""

from pathlib import Path
from typing import Any

from pstrain.lib.config import (
    CURRENT_CONFIG_VERSION,
    ResolvedConfig,
    generate_markdown_docs,
    generate_rst_docs,
    get_schema,
    list_parameters,
    migrate_project,
)
from pstrain.lib.config import list_profiles as _list_profiles
from pstrain.lib.config import resolve_config as _resolve_config
from pstrain.lib.config.user import get_user_config


def resolve_config(
    project_dir: Path | str,
    *,
    profile_name: str = "default",
    experiment: str = "default",
    cli_overrides: dict[str, Any] | None = None,
    user_config_path: Path | None = None,
) -> ResolvedConfig:
    """Resolve configuration through a concrete public-API call frame."""
    return _resolve_config(
        project_dir,
        profile_name=profile_name,
        experiment=experiment,
        cli_overrides=cli_overrides,
        user_config_path=user_config_path,
    )


def list_profiles(project_dir: Path) -> list[dict[str, Any]]:
    """List profiles through a concrete public-API call frame."""
    return _list_profiles(project_dir)


# These forwarders exist to put an API call frame on the stack. That is an
# implementation requirement and must not cost the published reference the
# documentation each re-exported name carried before.
resolve_config.__doc__ = _resolve_config.__doc__ or resolve_config.__doc__
list_profiles.__doc__ = _list_profiles.__doc__ or list_profiles.__doc__


__all__ = [
    "CURRENT_CONFIG_VERSION",
    "generate_markdown_docs",
    "generate_rst_docs",
    "get_schema",
    "get_user_config",
    "list_parameters",
    "list_profiles",
    "migrate_project",
    "resolve_config",
]
