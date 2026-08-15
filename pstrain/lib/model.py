"""Base model class and model implementations.

Each model type (CI, CD) is a class that knows about its own training process,
parameters, directory structure, and requirements.

Models provide metadata for the pipeline runner:
- File paths (inputs, outputs)
- Parameters (training settings)
- Dependencies (what stages must run first)

Complete-model validation follows native value ranges and enumerations conservatively,
but deliberately requires whole-token numeric spellings. Native command-line parsing
accepts numeric prefixes and silently truncates fractional spellings for integer options;
those lossy spellings are rejected here so a recorded front end cannot describe a value
different from the one the native parser actually used.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# Required while constructing and updating a model during training/BW.
MODEL_FILES_REQUIRED = ["mdef", "means", "variances", "mixture_weights", "transition_matrices"]

# A complete model consumed for decoding, alignment, packaging, or deployment
# additionally requires the training-time front-end record.
MODEL_FILES_COMPLETE_REQUIRED = [*MODEL_FILES_REQUIRED, "feat.params"]

# Optional acceleration and dictionary files for deployment/decoding.
MODEL_FILES_OPTIONAL = ["sendump", "noisedict"]

# All known model files
MODEL_FILES_ALL = MODEL_FILES_COMPLETE_REQUIRED + MODEL_FILES_OPTIONAL

# Every value serialized by ``pipeline.feat_params`` affects the training front end
# or pins an engine default on which training relies. Complete consumers require the
# entire record so no omitted value can fall through to a decoder-version default.
COMPLETE_MODEL_FEAT_PARAMS_REQUIRED = frozenset(
    {
        "-agc",
        "-alpha",
        "-cmn",
        "-cmninit",
        "-ceplen",
        "-dither",
        "-feat",
        "-frate",
        "-lifter",
        "-lowerf",
        "-ncep",
        "-nfft",
        "-nfilt",
        "-remove_dc",
        "-remove_noise",
        "-round_filters",
        "-samprate",
        "-transform",
        "-unit_area",
        "-upperf",
        "-varnorm",
        "-wlen",
    }
)

_BOOLEAN_FEAT_PARAMS = frozenset(
    {"-dither", "-remove_dc", "-remove_noise", "-unit_area", "-round_filters", "-varnorm"}
)
_POSITIVE_INTEGER_FEAT_PARAMS = frozenset({"-ceplen", "-ncep", "-nfilt", "-nfft", "-frate"})
_NONNEGATIVE_INTEGER_FEAT_PARAMS = frozenset({"-lifter"})
_NONNEGATIVE_FLOAT_FEAT_PARAMS = frozenset({"-lowerf"})
_POSITIVE_FLOAT_FEAT_PARAMS = frozenset({"-samprate", "-upperf", "-wlen"})


def _invalid_feat_param(feat_params: Path, name: str, value: str, requirement: str) -> ValueError:
    return ValueError(f"Invalid feat.params field {name}={value!r} in {feat_params}: {requirement}")


def _parse_finite_float(feat_params: Path, name: str, value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise _invalid_feat_param(feat_params, name, value, "must be a finite number") from None
    if not math.isfinite(parsed):
        raise _invalid_feat_param(feat_params, name, value, "must be a finite number")
    return parsed


def _validate_complete_feat_params(feat_params: Path, parsed: dict[str, str]) -> None:
    """Validate ranges/enums conservatively and numeric spellings strictly.

    Native parsing and feature-layout acceptance remain authoritative outside those
    checks. Numeric tokens are the deliberate exception: unlike native's prefix and
    truncating conversions, this validator requires the complete token to represent
    the recorded number.
    """
    integer_numbers: dict[str, int] = {}
    for name in _POSITIVE_INTEGER_FEAT_PARAMS | _NONNEGATIVE_INTEGER_FEAT_PARAMS:
        value = parsed[name]
        try:
            number = int(value)
        except ValueError:
            raise _invalid_feat_param(
                feat_params,
                name,
                value,
                "must use an exact integer spelling (native truncation is not accepted)",
            ) from None
        minimum = 0 if name in _NONNEGATIVE_INTEGER_FEAT_PARAMS else 1
        if number < minimum:
            raise _invalid_feat_param(feat_params, name, value, f"must be >= {minimum}")
        integer_numbers[name] = number

    numbers: dict[str, float | int] = dict(integer_numbers)

    for name in _POSITIVE_FLOAT_FEAT_PARAMS:
        value = parsed[name]
        float_number = _parse_finite_float(feat_params, name, value)
        if float_number <= 0:
            raise _invalid_feat_param(feat_params, name, value, "must be > 0")
        numbers[name] = float_number

    for name in _NONNEGATIVE_FLOAT_FEAT_PARAMS:
        value = parsed[name]
        float_number = _parse_finite_float(feat_params, name, value)
        if float_number < 0:
            raise _invalid_feat_param(feat_params, name, value, "must be >= 0")
        numbers[name] = float_number

    for name in _BOOLEAN_FEAT_PARAMS:
        if not parsed[name] or parsed[name][0] not in "ytYT1nfNF0":
            raise _invalid_feat_param(
                feat_params, name, parsed[name], "must begin with a native boolean alias"
            )

    enums = {
        "-transform": {"dct", "legacy", "htk"},
        "-cmn": {"none", "batch", "live", "current", "prior"},
    }
    for name, choices in enums.items():
        if parsed[name] not in choices:
            raise _invalid_feat_param(
                feat_params, name, parsed[name], "must be one of " + ", ".join(sorted(choices))
            )

    if parsed["-agc"] not in {"none", "max", "emax", "noise"}:
        raise _invalid_feat_param(
            feat_params, "-agc", parsed["-agc"], "must be one of emax, max, noise, none"
        )

    feature_type = parsed["-feat"]
    known_feature_type = feature_type in {"s2_4x", "s3_1x39", "1s_12c_12d_3p_12dd"} or any(
        feature_type.startswith(prefix)
        for prefix in (
            "1s_c_d_dd",
            "1s_c_d_ld_dd",
            "cep_dcep",
            "1s_c_d",
            "cep",
            "1s_c",
            "1s_3c",
            "1s_4c",
        )
    )
    generic_feature_type = re.fullmatch(r"[1-9]\d*(?:,[1-9]\d*)*(?::\d+)?", feature_type)
    if generic_feature_type:
        widths = feature_type.split(":", 1)[0]
        known_feature_type = sum(int(width) for width in widths.split(",")) == numbers["-ncep"]
    if not known_feature_type:
        raise _invalid_feat_param(
            feat_params, "-feat", feature_type, "must be a native-supported feature stream layout"
        )
    if feature_type in {"s2_4x", "s3_1x39", "1s_12c_12d_3p_12dd"} and numbers["-ncep"] != 13:
        raise _invalid_feat_param(feat_params, "-feat", feature_type, "requires -ncep 13")

    if numbers["-upperf"] <= numbers["-lowerf"]:
        raise _invalid_feat_param(
            feat_params, "-upperf", parsed["-upperf"], "must be greater than -lowerf"
        )
    if numbers["-ceplen"] != numbers["-ncep"]:
        raise _invalid_feat_param(
            feat_params,
            "-ceplen",
            parsed["-ceplen"],
            "must match -ncep so the native waveform and feature initializers agree",
        )
    if numbers["-upperf"] > numbers["-samprate"] / 2 + 1.0:
        raise _invalid_feat_param(
            feat_params, "-upperf", parsed["-upperf"], "must not exceed the Nyquist frequency"
        )
    if integer_numbers["-nfft"] & (integer_numbers["-nfft"] - 1):
        raise _invalid_feat_param(feat_params, "-nfft", parsed["-nfft"], "must be a power of two")
    if numbers["-frate"] > numbers["-samprate"]:
        raise _invalid_feat_param(
            feat_params, "-frate", parsed["-frate"], "must not exceed -samprate"
        )
    frame_shift = int(numbers["-samprate"] / numbers["-frate"] + 0.5)
    frame_size = int(numbers["-wlen"] * numbers["-samprate"] + 0.5)
    if frame_shift <= 1:
        raise _invalid_feat_param(
            feat_params, "-frate", parsed["-frate"], "must yield native frame_shift > 1"
        )
    if frame_size < frame_shift:
        raise _invalid_feat_param(
            feat_params, "-wlen", parsed["-wlen"], "must yield frame_size >= frame_shift"
        )
    if numbers["-nfft"] < numbers["-wlen"] * numbers["-samprate"]:
        raise _invalid_feat_param(
            feat_params, "-nfft", parsed["-nfft"], "must cover the analysis window"
        )


__all__ = [
    "MODEL_FILES_REQUIRED",
    "MODEL_FILES_COMPLETE_REQUIRED",
    "MODEL_FILES_OPTIONAL",
    "MODEL_FILES_ALL",
    "COMPLETE_MODEL_FEAT_PARAMS_REQUIRED",
    "Model",
    "CIModel",
    "CDModel",
    "create_model",
    "get_model_class",
    "read_complete_model_feat_params",
    "require_complete_model",
]


def read_complete_model_feat_params(model_dir: str | Path) -> dict[str, str]:
    """Require, validate, and return the complete pstrain front-end record.

    Range and enumeration rejection is a conservative subset of native rejection. Numeric
    spelling is deliberately stricter: the complete token must parse without native-style
    prefix acceptance or integer truncation. Native parsing remains authoritative otherwise.
    This does not prove compatibility between the record and the model's binary tensors.
    """
    model_dir = Path(model_dir)
    feat_params = model_dir / "feat.params"
    if not feat_params.is_file():
        raise FileNotFoundError(
            f"Missing feat.params ({feat_params}) from complete model directory {model_dir}. "
            "Without it, the decode-time front end is undefined and can silently differ "
            "from the training front end in feature shape and basis."
        )
    parsed: dict[str, str] = {}
    for line_number, raw_line in enumerate(feat_params.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not parts[0].startswith("-") or not parts[1].strip():
            raise ValueError(
                f"Malformed feat.params line {line_number} in {feat_params}: {raw_line!r}"
            )
        name, value = parts
        if name in parsed:
            raise ValueError(f"Duplicate feat.params field {name} in {feat_params}")
        parsed[name] = value.strip()

    missing = sorted(COMPLETE_MODEL_FEAT_PARAMS_REQUIRED - parsed.keys())
    if missing:
        if missing == ["-ceplen"] and "-ncep" in parsed:
            raise ValueError(
                f"feat.params ({feat_params}) is from the legacy pre-ceplen format. "
                f"Add the line '-ceplen {parsed['-ncep']}' (the value must equal -ncep) so "
                "the native waveform extractor and feature initializer use the same "
                "cepstral width. No other missing or misspelled fields are accepted."
            )
        raise ValueError(
            f"feat.params ({feat_params}) is missing required front-end fields: "
            + ", ".join(missing)
        )
    unexpected = sorted(parsed.keys() - COMPLETE_MODEL_FEAT_PARAMS_REQUIRED)
    if unexpected:
        raise ValueError(
            f"feat.params ({feat_params}) has unsupported front-end fields: "
            + ", ".join(unexpected)
        )
    _validate_complete_feat_params(feat_params, parsed)
    return parsed


def require_complete_model(model_dir: str | Path) -> Path:
    """Require and validate the complete pstrain front-end record."""
    model_dir = Path(model_dir)
    read_complete_model_feat_params(model_dir)
    return model_dir / "feat.params"


class Model(ABC):
    """Base class for acoustic models.

    Each model type (CI, CD) should inherit from this class and implement
    the abstract methods to define its specific behavior.
    """

    def __init__(self, config: str = "baseline") -> None:
        """Initialize model.

        Args:
            config: Model configuration name (e.g., "baseline", "1g", "lda")
        """
        self.config = config

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Model type identifier (e.g., "ci", "cd")."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for the model type."""
        pass

    @property
    @abstractmethod
    def default_topn(self) -> int:
        """Default top-n Gaussians for this model type."""
        pass

    def get_model_dir(self, experiment_dir: str | Path) -> Path:
        """Get the model directory for this model.

        Args:
            experiment_dir: Experiment directory path

        Returns:
            Path to model directory: {experiment_dir}/models/{model_type}/{config}/model/
        """
        experiment_dir = Path(experiment_dir)
        return experiment_dir / "models" / self.model_type / self.config / "model"

    def get_flat_dir(self, experiment_dir: str | Path) -> Path:
        """Get the flat model directory for this model."""
        return self.get_model_dir(experiment_dir) / "flat"

    def get_hmm_dir(self, experiment_dir: str | Path) -> Path:
        """Get the trained HMM model directory for this model."""
        return self.get_model_dir(experiment_dir) / "hmm"

    @abstractmethod
    def get_training_dependencies(self) -> list[str]:
        """Get list of dependencies required for training.

        Returns:
            List of dependency names (e.g., ["flat", "features", "dictionary", "split"])
        """
        pass

    @abstractmethod
    def get_default_training_params(self) -> dict[str, Any]:
        """Get default training parameters for this model type.

        Returns:
            Dictionary of parameter names to default values
        """
        pass

    def validate_training_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize training parameters.

        Args:
            params: Training parameters to validate

        Returns:
            Validated parameters with defaults filled in

        Raises:
            ValueError: If parameters are invalid
        """
        defaults = self.get_default_training_params()
        validated = defaults.copy()
        validated.update(params)
        return validated

    @classmethod
    @abstractmethod
    def from_string(cls, value: str) -> type[Model]:
        """Get model class from string identifier.

        Args:
            value: Model type string (e.g., "ci", "CD", "context-independent")

        Returns:
            Model class

        Raises:
            ValueError: If model type is unknown
        """
        pass


class CIModel(Model):
    """Context-Independent (monophone) acoustic model."""

    @property
    def model_type(self) -> str:
        return "ci"

    @property
    def display_name(self) -> str:
        return "Context-Independent"

    @property
    def default_topn(self) -> int:
        return 1

    def get_training_dependencies(self) -> list[str]:
        """Get list of dependencies required for CI model training.

        Returns:
            List of dependency names: ["flat", "features", "dictionary", "split"]
        """
        return ["flat", "features", "dictionary", "split"]

    def get_default_training_params(self) -> dict[str, Any]:
        """Get default training parameters for CI models.

        Returns:
            Dictionary of parameter names to default values for CI model training
        """
        return {
            "save_alignments": False,
            "gaussian_splitting": None,  # None = no splitting, or list like [1, 2, 4, 8]
            "n_iterations_after_split": 3,  # Iterations after each Gaussian split
        }

    @classmethod
    def from_string(cls, value: str) -> type[Model]:
        """Get model class from string identifier.

        Args:
            value: Model type string (e.g., "ci", "CD", "context-independent")

        Returns:
            Model class (CIModel)

        Raises:
            ValueError: If model type is unknown
        """
        value_lower = value.lower()
        if value_lower in ("ci", "context-independent", "monophone"):
            return cls
        raise ValueError(f"Unknown CI model type: {value}")


class CDModel(Model):
    """Context-Dependent (triphone) acoustic model."""

    @property
    def model_type(self) -> str:
        return "cd"

    @property
    def display_name(self) -> str:
        return "Context-Dependent"

    @property
    def default_topn(self) -> int:
        return 4

    def get_training_dependencies(self) -> list[str]:
        """Get list of dependencies required for CD model training.

        Returns:
            List of dependency names: ["ci", "features", "dictionary", "split"]
            Note: CD models depend on CI models, so "ci" is included
        """
        return ["ci", "features", "dictionary", "split"]

    def get_default_training_params(self) -> dict[str, Any]:
        """Get default training parameters for CD models.

        Returns:
            Dictionary of parameter names to default values for CD model training
        """
        return {
            "save_alignments": False,
            "gaussian_splitting": None,  # None = no splitting, or list like [1, 2, 4, 8]
            "n_iterations_after_split": 3,  # Iterations after each Gaussian split
        }

    @classmethod
    def from_string(cls, value: str) -> type[Model]:
        """Get model class from string identifier.

        Args:
            value: Model type string (e.g., "cd", "CD", "context-dependent")

        Returns:
            Model class (CDModel)

        Raises:
            ValueError: If model type is unknown
        """
        value_lower = value.lower()
        if value_lower in ("cd", "context-dependent", "triphone"):
            return cls
        raise ValueError(f"Unknown CD model type: {value}")


def get_model_class(model_type: str) -> type[Model]:
    """Get model class from model type string.

    Args:
        model_type: Model type identifier (e.g., "ci", "cd")

    Returns:
        Model class

    Raises:
        ValueError: If model type is unknown
    """
    value_lower = model_type.lower()

    # Try CI first
    try:
        return CIModel.from_string(value_lower)
    except ValueError:
        pass

    # Try CD
    try:
        return CDModel.from_string(value_lower)
    except ValueError:
        pass

    raise ValueError(
        f"Unknown model type: {model_type}. "
        f"Valid types: ci (Context-Independent), cd (Context-Dependent)"
    )


def create_model(model_type: str, config: str = "baseline") -> Model:
    """Create a model instance.

    Args:
        model_type: Model type identifier (e.g., "ci", "cd")
        config: Model configuration name (default: "baseline")

    Returns:
        Model instance of the specified type

    Raises:
        ValueError: If model type is unknown
    """
    model_class = get_model_class(model_type)
    return model_class(config=config)
