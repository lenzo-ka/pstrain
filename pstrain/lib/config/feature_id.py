"""Feature configuration identification for caching and sharing.

Feature directories are named by a combination of:
1. Human-readable ID (sample rate, ceps, feature type)
2. Hash of ALL parameters (ensures uniqueness)

Example: shared/features/16khz_13cep_1s_c_d_dd_a1b2c3d4/
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pstrain.lib.config.models import AudioConfig, FeatureConfig


def get_feature_id(audio_config: AudioConfig, feature_config: FeatureConfig) -> str:
    """Generate human-readable identifier for feature extraction settings.

    This ID is used for caching and sharing features across experiments
    with identical feature extraction settings.

    Args:
        audio_config: Audio configuration
        feature_config: Feature extraction configuration

    Returns:
        Human-readable feature ID string

    Examples:
        16khz_13cep_1s_c_d_dd
        8khz_13cep_s2_4x
        16khz_20cep_1s_c_d_dd
    """
    sample_rate_khz = audio_config.sample_rate // 1000
    num_ceps = feature_config.num_ceps
    feature_type = feature_config.feature_type

    return f"{sample_rate_khz}khz_{num_ceps}cep_{feature_type}"


def get_feature_hash(audio_config: AudioConfig, feature_config: FeatureConfig) -> str:
    """Generate hash of all feature extraction parameters.

    More precise than feature_id - includes ALL parameters, not just major ones.
    Ensures that even minor parameter changes result in different feature directories.

    Args:
        audio_config: Audio configuration
        feature_config: Feature extraction configuration

    Returns:
        Short hash string (first 8 chars of SHA256)
    """
    # Serialize all relevant parameters as a sorted dict for determinism
    params = {
        "sample_rate": audio_config.sample_rate,
        "feature_type": feature_config.feature_type,
        "type": feature_config.type,
        "num_ceps": feature_config.num_ceps,
        "num_filters": feature_config.num_filters,
        "nfft": feature_config.nfft,
        "frame_length_ms": feature_config.frame_length_ms,
        "frame_shift_ms": feature_config.frame_shift_ms,
        "lower_freq": feature_config.lower_freq,
        "upper_freq": feature_config.upper_freq,
        "preemphasis": feature_config.preemphasis,
        "lifter": feature_config.lifter,
        "use_energy": feature_config.use_energy,
        "delta": feature_config.delta,
        "delta_delta": feature_config.delta_delta,
        "agc": feature_config.agc,
        "cmn": feature_config.cmn,
        "varnorm": feature_config.varnorm,
        "transform": feature_config.transform,
    }

    # Create deterministic JSON string
    param_str = json.dumps(params, sort_keys=True)
    hash_obj = hashlib.sha256(param_str.encode())
    return hash_obj.hexdigest()[:8]


def get_feature_dir_name(audio_config: AudioConfig, feature_config: FeatureConfig) -> str:
    """Get directory name for feature cache.

    Combines readable ID with hash for uniqueness. This ensures:
    1. Human can see what settings were used (from readable ID)
    2. Different parameter combinations never collide (from hash)

    Args:
        audio_config: Audio configuration
        feature_config: Feature extraction configuration

    Returns:
        Directory name like "16khz_13cep_1s_c_d_dd_a1b2c3d4"
    """
    readable_id = get_feature_id(audio_config, feature_config)
    hash_suffix = get_feature_hash(audio_config, feature_config)
    return f"{readable_id}_{hash_suffix}"


def get_full_feature_dim(feature_config: FeatureConfig) -> int:
    """Get full feature dimension including deltas.

    Args:
        feature_config: Feature extraction configuration

    Returns:
        Total feature dimension after delta computation:
        - For 1s_c_d_dd: num_ceps * 3 (base + delta + double-delta)
        - For s2_4x: num_ceps * 4 (4 streams)
    """
    if feature_config.feature_type == "s2_4x":
        return feature_config.num_ceps * 4
    # 1s_c_d_dd: base + delta + delta-delta
    return feature_config.num_ceps * 3
