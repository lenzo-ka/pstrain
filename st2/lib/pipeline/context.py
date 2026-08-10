"""Pipeline context: per-run configuration and path conventions.

`PipelineContext` is the single object passed to task builders. It knows where
the project lives, which experiment + named config is active, and the
derived feature/training parameters.

Path conventions (mirroring the prior Snakefile):

    project/
      etc/configs.yaml                  # Named configurations
      audio/                            # Raw audio (input)
      shared/                           # Shared across experiments
        dictionary.dict
        phoneset.txt
        filler.dict (optional)
        features/{config_name}/         # Features for this config
        models/{target}/{config_name}/  # Acoustic models
        models/trees/{config_name}/     # Decision trees
        models/architecture/{config_name}/
      experiments/{experiment}/
        etc/                            # train.fileids, transcripts, ...
        lm/
        reports/
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FeatParams:
    """Acoustic front-end parameters.

    Defaults are SphinxTrain wideband: 16 kHz audio, 40-filter mel bank,
    DCT-transformed cepstra with 13 coefficients, batch CMN, no AGC,
    no variance normalization. All fields are emitted into per-model
    `feat.params` files so PocketSphinx and friends can match the
    training-time front-end at decode/align time.
    """

    samprate: int = 16000
    ncep: int = 13
    nfilt: int = 40
    nfft: int = 512
    lowerf: int = 130
    upperf: int = 6800
    # Pre-emphasis coefficient (`-alpha`). 0.97 is the engine default.
    alpha: float = 0.97
    feat_type: str = "1s_c_d_dd"
    # Cepstral lifter window (sphinx_fe `-lifter`). 22 = SphinxTrain default.
    lifter: int = 22
    # Linear-transform applied to filter bank outputs (`-transform`).
    transform: str = "dct"
    # Automatic gain control (`-agc`).
    agc: str = "none"
    # Cepstral mean normalization (`-cmn`). "batch" matches SphinxTrain.
    cmn: str = "batch"
    # Cepstral variance normalization (`-varnorm`).
    varnorm: str = "no"


@dataclass(frozen=True)
class TrainParams:
    n_state: int = 3
    n_senones: int = 200
    max_iterations: int = 10
    # Multi-pronunciation training: build wide utterance graphs that
    # sum Baum-Welch posteriors across pronunciation variants. On by
    # default; set to False to fall back to the legacy linear path that
    # always picks the first listed variant per word.
    multipron_training: bool = True


@dataclass(frozen=True)
class SplitParams:
    """Parameters for the train/test split task.

    Defaults match the `st2 split` CLI: 95% train, seed 42.
    """

    train_ratio: float | None = None
    test_count: int | None = None
    seed: int = 42


DEFAULT_CONFIGS: dict[str, dict[str, Any]] = {
    "default": {
        "description": "Default wideband configuration",
        "features": {
            "samprate": 16000,
            "ncep": 13,
            "nfilt": 40,
            "nfft": 512,
            "lowerf": 130,
            "upperf": 6800,
            "feat_type": "1s_c_d_dd",
        },
        "training": {
            "n_state": 3,
            "n_senones": 200,
            "max_iterations": 10,
        },
    },
    "wideband": {
        "description": "Wideband (16kHz) microphone speech",
        "features": {
            "samprate": 16000,
            "ncep": 13,
            "nfilt": 40,
            "nfft": 512,
            "lowerf": 130,
            "upperf": 6800,
            "feat_type": "1s_c_d_dd",
        },
        "training": {
            "n_state": 3,
            "n_senones": 200,
            "max_iterations": 10,
        },
    },
    "telephone": {
        "description": "Telephone (8kHz) narrowband speech",
        "features": {
            "samprate": 8000,
            "ncep": 13,
            "nfilt": 25,
            "nfft": 256,
            "lowerf": 200,
            "upperf": 3500,
            "feat_type": "1s_c_d_dd",
        },
        "training": {
            "n_state": 3,
            "n_senones": 200,
            "max_iterations": 10,
        },
    },
    "wideband_large": {
        "description": "Wideband with more senones (larger datasets)",
        "features": {
            "samprate": 16000,
            "ncep": 13,
            "nfilt": 40,
            "nfft": 512,
            "lowerf": 130,
            "upperf": 6800,
            "feat_type": "1s_c_d_dd",
        },
        "training": {
            "n_state": 3,
            "n_senones": 4000,
            "max_iterations": 10,
        },
    },
    "sphinxtrain": {
        "description": "Matched to SphinxTrain defaults for comparison",
        "features": {
            "samprate": 16000,
            "ncep": 13,
            "nfilt": 25,
            "nfft": 512,
            "lowerf": 130,
            "upperf": 6800,
            "alpha": 0.97,
            "lifter": 22,
            "feat_type": "1s_c_d_dd",
        },
        "training": {
            "n_state": 3,
            "n_senones": 200,
            "max_iterations": 10,
        },
    },
}


def _validate_params(
    profile: str,
    block: str,
    values: dict[str, Any],
    params_type: type[FeatParams] | type[TrainParams] | type[SplitParams],
) -> None:
    """Reject misspelled profile parameters with configuration context."""
    known = {item.name for item in fields(params_type)}
    unknown = sorted(set(values) - known)
    if unknown:
        parameter = block.removesuffix("s")
        raise ValueError(f"unknown {parameter} parameter {unknown[0]!r} in profile {profile!r}")


def load_configs(project_dir: Path) -> dict[str, dict[str, Any]]:
    """Load named configurations from `project_dir/etc/configs.yaml`,
    merged on top of the built-in defaults."""
    configs_file = project_dir / "etc" / "configs.yaml"
    if not configs_file.exists():
        return dict(DEFAULT_CONFIGS)
    with open(configs_file) as f:
        user_configs = yaml.safe_load(f) or {}
    merged = dict(DEFAULT_CONFIGS)
    merged.update(user_configs)
    return merged


@dataclass(frozen=True)
class PipelineContext:
    """Per-run configuration for the training pipeline."""

    project_dir: Path
    experiment: str = "default"
    config_name: str = "default"
    feat: FeatParams = field(default_factory=FeatParams)
    train: TrainParams = field(default_factory=TrainParams)
    split: SplitParams = field(default_factory=SplitParams)
    description: str = ""

    @classmethod
    def from_config(
        cls,
        project_dir: Path | str,
        *,
        experiment: str = "default",
        config_name: str = "default",
    ) -> PipelineContext:
        """Build a context by reading `project/etc/configs.yaml`."""
        project_dir = Path(project_dir).resolve()
        configs = load_configs(project_dir)
        if config_name not in configs:
            available = ", ".join(sorted(configs))
            raise ValueError(f"unknown config {config_name!r}; available: {available}")
        cfg = configs[config_name]
        feature_values = cfg.get("features", {})
        training_values = cfg.get("training", {})
        split_values = cfg.get("split", {})
        _validate_params(config_name, "features", feature_values, FeatParams)
        _validate_params(config_name, "training", training_values, TrainParams)
        _validate_params(config_name, "split", split_values, SplitParams)
        return cls(
            project_dir=project_dir,
            experiment=experiment,
            config_name=config_name,
            description=cfg.get("description", ""),
            feat=FeatParams(**feature_values),
            train=TrainParams(**training_values),
            split=SplitParams(**split_values),
        )

    @property
    def shared_dir(self) -> Path:
        return self.project_dir / "shared"

    @property
    def audio_dir(self) -> Path:
        return self.project_dir / "audio"

    @property
    def experiment_dir(self) -> Path:
        return self.project_dir / "experiments" / self.experiment

    @property
    def etc_dir(self) -> Path:
        return self.experiment_dir / "etc"

    @property
    def features_dir(self) -> Path:
        return self.shared_dir / "features" / self.config_name

    @property
    def models_dir(self) -> Path:
        return self.shared_dir / "models"

    def model_dir(self, target: str) -> Path:
        """Directory for an acoustic model output, e.g. `cd-8g`."""
        return self.models_dir / target / self.config_name

    def model_files(self, target: str) -> list[Path]:
        """Standard set of files that constitute a trained model directory."""
        d = self.model_dir(target)
        return [
            d / "mdef",
            d / "means",
            d / "variances",
            d / "mixture_weights",
            d / "transition_matrices",
            d / "feat.params",
        ]

    @property
    def trees_dir(self) -> Path:
        return self.models_dir / "trees" / self.config_name

    @property
    def architecture_dir(self) -> Path:
        return self.models_dir / "architecture" / self.config_name

    @property
    def lm_dir(self) -> Path:
        return self.experiment_dir / "lm"

    @property
    def reports_dir(self) -> Path:
        return self.experiment_dir / "reports"

    @property
    def dist_dir(self) -> Path:
        return self.project_dir / "dist" / "models"

    @property
    def filler_dict(self) -> Path | None:
        """Optional filler dictionary; None if not present."""
        p = self.shared_dir / "filler.dict"
        return p if p.exists() else None

    @property
    def all_transcription(self) -> Path:
        """Master (pre-split) transcription, input to the `split` task."""
        return self.project_dir / "etc" / "all.transcription"

    def read_fileids(self, split: str) -> list[str]:
        """Read a fileid list (e.g. 'train', 'test', 'dev').

        Returns an empty list if the file doesn't exist (planning before
        corpus setup is allowed; tasks that need fileids will fail at run
        time with a clearer message).
        """
        path = self.etc_dir / f"{split}.fileids"
        if not path.exists():
            return []
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

    def all_fileids(self) -> list[str]:
        """Train + test fileids (post-split). Use `audio_fileids()` if you
        need the full corpus before split has run."""
        return self.read_fileids("train") + self.read_fileids("test")

    def audio_fileids(self, extension: str = ".wav") -> list[str]:
        """All audio fileids, recursively derived from `audio/**/*<ext>`.

        Returns sorted POSIX-style paths relative to the audio directory,
        without the extension. This is the canonical set of files that
        feature extraction operates on and is independent of whether the
        train/test split has run yet.
        """
        if not self.audio_dir.exists():
            return []
        return sorted(
            p.relative_to(self.audio_dir).with_suffix("").as_posix()
            for p in self.audio_dir.rglob(f"*{extension}")
        )
