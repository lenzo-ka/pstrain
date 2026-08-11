"""Canonical pstrain configuration API."""

from pstrain.lib.config.models import (
    CURRENT_CONFIG_VERSION,
    FeatureConfig,
    OverlayDocument,
    Profile,
    ProfilesDocument,
    RunnerConfig,
    SplitConfig,
    TrainingConfig,
    TrainingScheduleConfig,
)
from pstrain.lib.config.resolver import (
    CONSUMERS,
    FieldExplanation,
    ResolvedConfig,
    list_profiles,
    migrate_project,
    resolve_config,
)
from pstrain.lib.config.schema import (
    ParameterInfo,
    generate_markdown_docs,
    generate_rst_docs,
    get_parameter,
    get_schema,
    list_parameters,
)

DEFAULT_FEAT_PARAMS = FeatureConfig().model_dump()

__all__ = [
    "CURRENT_CONFIG_VERSION",
    "CONSUMERS",
    "DEFAULT_FEAT_PARAMS",
    "FeatureConfig",
    "FieldExplanation",
    "OverlayDocument",
    "ParameterInfo",
    "Profile",
    "ProfilesDocument",
    "ResolvedConfig",
    "RunnerConfig",
    "SplitConfig",
    "TrainingConfig",
    "TrainingScheduleConfig",
    "generate_markdown_docs",
    "generate_rst_docs",
    "get_parameter",
    "get_schema",
    "list_parameters",
    "list_profiles",
    "migrate_project",
    "resolve_config",
]
