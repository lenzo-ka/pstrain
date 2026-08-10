"""pstrain library API.

This module exposes the core functionality. CLI and web are thin wrappers.
All public functions should return JSON-serializable data structures.

For most uses, prefer ``pstrain.api`` which re-exports everything from here
and adds step functions for training workflows.

Example::

    from pstrain.api import (
        setup_project,
        validate_project,
        create_model,
        PstrainConfig,
        ConfigManager,
        Dictionary,
        Phoneset,
    )

Low-level C bindings are available via::

    from pstrain.lib._pstrainc import get_ffi, get_lib
    lib = get_lib()  # Direct C function access
    ffi = get_ffi()  # cffi FFI instance
"""

# Don't import _pstrainc eagerly - it requires the C library to be built
# Import directly: from pstrain.lib._pstrainc import get_ffi, get_lib

# Project setup and validation
# Configuration
from pstrain.lib.compare import (
    CompareResult,
    ComponentCompare,
    ModelCompareResult,
    compare_auto,
    compare_features,
    compare_gaussians,
    compare_gaussians_detailed,
    compare_mixw,
    compare_models,
    compare_senone_sets,
    compare_tmat,
    load_senones,
    load_senones_from_model,
)
from pstrain.lib.config import (
    AudioConfig,
    ConfigManager,
    FeatureConfig,
    PstrainConfig,
    TrainingConfig,
    get_feature_dir_name,
    get_user_config,
)

# Data structures
from pstrain.lib.dictionary import Dictionary

# File type detection and comparison
from pstrain.lib.filetypes import (
    FileType,
    assert_file_type,
    describe_file,
    detect_file_type,
    validate_file_type,
)

# Model classes
from pstrain.lib.model import (
    CDModel,
    CIModel,
    Model,
    create_model,
    get_model_class,
)

# Path discovery
from pstrain.lib.paths import PstrainPaths, get_bin_dir, get_include_dir, get_lib_path, get_paths
from pstrain.lib.phoneset import Phoneset
from pstrain.lib.setup import setup_project

# Similarity metrics for Gaussian/senone comparison
from pstrain.lib.similarity import (
    GaussianState,
    Senone,
    bhattacharyya_distance,
    compare_senones,
    compare_states,
    cosine_similarity,
    euclidean_distance,
    kl_divergence,
    mahalanobis_distance,
    symmetric_kl_divergence,
)
from pstrain.lib.transcription import get_fileids, parse_transcription_file
from pstrain.lib.validate import ValidationReport, validate_project

__all__: list[str] = [
    # Project setup
    "setup_project",
    "validate_project",
    "ValidationReport",
    # Configuration
    "PstrainConfig",
    "AudioConfig",
    "FeatureConfig",
    "TrainingConfig",
    "ConfigManager",
    "get_user_config",
    "get_feature_dir_name",
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
    # Paths
    "PstrainPaths",
    "get_paths",
    "get_bin_dir",
    "get_lib_path",
    "get_include_dir",
    # File types
    "FileType",
    "detect_file_type",
    "describe_file",
    "validate_file_type",
    "assert_file_type",
    # Comparison
    "CompareResult",
    "ComponentCompare",
    "ModelCompareResult",
    "compare_auto",
    "compare_features",
    "compare_gaussians",
    "compare_gaussians_detailed",
    "compare_senone_sets",
    "compare_mixw",
    "compare_tmat",
    "compare_models",
    "load_senones",
    "load_senones_from_model",
    # Similarity metrics
    "GaussianState",
    "Senone",
    "bhattacharyya_distance",
    "compare_senones",
    "compare_states",
    "cosine_similarity",
    "euclidean_distance",
    "kl_divergence",
    "mahalanobis_distance",
    "symmetric_kl_divergence",
]
