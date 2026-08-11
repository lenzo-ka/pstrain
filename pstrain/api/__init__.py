"""pstrain public API.

This is the recommended entry point for using pstrain programmatically.
CLI and web clients should call into this API.

All public functions return JSON-serializable data structures.

Example::

    from pstrain.api import (
        # Project setup
        setup_project,
        validate_project,
        ValidationReport,
        # Configuration
        Profile,
        # Data structures
        Dictionary,
        Phoneset,
        # Models
        create_model,
        CIModel,
        CDModel,
        # Training steps
        run_step_ci_hmm,
    )
"""

from pstrain.api.steps import (
    run_step_cd_hmm_untied,
    run_step_ci_hmm,
    run_step_features,
    step_cd_hmm_untied,
    step_ci_hmm,
    step_features,
)

# Re-export lib API
from pstrain.lib import (
    CDModel,
    CIModel,
    # Data structures
    Dictionary,
    FeatureConfig,
    # Models
    Model,
    Phoneset,
    # Configuration
    Profile,
    TrainingConfig,
    create_model,
    get_fileids,
    get_model_class,
    parse_transcription_file,
    resolve_config,
    # Project setup
    setup_project,
    validate_project,
)
from pstrain.lib.validate import ValidationReport

__all__: list[str] = [
    # Project setup
    "setup_project",
    "validate_project",
    "ValidationReport",
    # Configuration
    "Profile",
    "FeatureConfig",
    "TrainingConfig",
    "resolve_config",
    # Data structures
    "Dictionary",
    "Phoneset",
    "get_fileids",
    "parse_transcription_file",
    # Models
    "Model",
    "CIModel",
    "CDModel",
    "create_model",
    "get_model_class",
    # Steps
    "step_features",
    "step_ci_hmm",
    "step_cd_hmm_untied",
    "run_step_features",
    "run_step_ci_hmm",
    "run_step_cd_hmm_untied",
]
