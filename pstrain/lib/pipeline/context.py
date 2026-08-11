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

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from functools import cache
from pathlib import Path
from typing import Any, Self

import yaml

from pstrain import __version__
from pstrain.lib.paths import get_lib_path


@cache
def _sha256_file(path: Path, size: int, mtime_ns: int) -> str:
    """Hash each observed native-library version once per Python process."""
    del size, mtime_ns  # They form the cache key and detect in-place rebuilds.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_library_identity() -> dict[str, str]:
    """Return path-independent content identity for the library used by CFFI."""
    lib_path = get_lib_path()
    if lib_path is None:
        return {"state": "absent"}
    resolved = lib_path.resolve()
    stat = resolved.stat()
    return {"sha256": _sha256_file(resolved, stat.st_size, stat.st_mtime_ns)}


@dataclass(frozen=True)
class FeatParams:
    """Acoustic front-end parameters.

    Defaults are SphinxTrain wideband: 16 kHz audio, 25-filter mel bank,
    DCT-transformed cepstra with 13 coefficients, batch CMN, no AGC,
    no variance normalization. All fields are emitted into per-model
    `feat.params` files so PocketSphinx and friends can match the
    training-time front-end at decode/align time.
    """

    samprate: int = 16000
    ncep: int = 13
    nfilt: int = 25
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
class TrainingSchedule:
    """Convergence controller for one family of Baum-Welch stages."""

    max_iterations: int = 10
    min_iterations: int = 1
    convergence_ratio: float = 0.001


@dataclass(frozen=True)
class TrainParams:
    n_state: int = 3
    n_senones: int = 200
    a_beam: float = 1e-90
    b_beam: float = 1e-10
    # A7c matched the upstream signed absolute likelihood-delta decision,
    # while measurements retained 0.001 rather than upstream's literal 0.1.
    # CI and tied stages keep that controller and upstream's ten-pass cap.
    ci: TrainingSchedule = field(default_factory=TrainingSchedule)
    tied: TrainingSchedule = field(default_factory=TrainingSchedule)
    # SphinxTrain scripts/30.cd_hmm_untied/norm_and_launchbw.pl uses the same
    # converge-with-cap controller. The preserved SLT oracle ended at pass 6;
    # cap this stage there so a stricter pstrain threshold cannot run to 10.
    untied: TrainingSchedule = field(default_factory=lambda: TrainingSchedule(max_iterations=6))
    # Warn on every skipped update; fail a stage above five percent.
    max_skip_fraction: float = 0.05
    # Retry forward-final-state pruning failures once at a beam this many
    # times wider (1e-90 / 1e10 = 1e-100).
    retry_beam_factor: float = 1e10
    # Tuned by SphinxTrain scripts/40.buildtrees/buildtree.pl for 3-state HMMs.
    tree_state_weights: tuple[float, ...] = (1.0, 0.05, 0.0)
    tree_ssplitmax: int = 7
    tree_ssplitthr: float = 0.0
    tree_csplitmax: int = 2000
    tree_csplitthr: float = 0.0
    tree_mwfloor: float = 1e-8
    question_npermute: int = 12
    question_quests_per_state: int = 20
    question_niter: int = 1
    # Multi-pronunciation training: build wide utterance graphs that
    # sum Baum-Welch posteriors across pronunciation variants. On by
    # default; set to False to fall back to the legacy linear path that
    # always picks the first listed variant per word.
    multipron_training: bool = True
    # Untied model inventory is independent of graph training. Keep the M4b
    # all-dictionary policy by default; PP3g experiments opt into the exact
    # transcript-reachable graph domain.
    untied_inventory: str = "all-triphone"


@dataclass(frozen=True)
class SplitParams:
    """Parameters for the train/test split task.

    Defaults match the `pstrain split` CLI: 95% train, seed 42.
    """

    train_ratio: float | None = None
    test_count: int | None = None
    seed: int = 42


@dataclass(frozen=True)
class RunnerParams:
    """Local process-allocation policy for pipeline fan-outs."""

    jobs: int | None = None
    nice: int = 5


DEFAULT_CONFIGS: dict[str, dict[str, Any]] = {
    "default": {
        "description": "Default wideband configuration",
        "features": {
            "samprate": 16000,
            "ncep": 13,
            "nfilt": 25,
            "nfft": 512,
            "lowerf": 130,
            "upperf": 6800,
            "feat_type": "1s_c_d_dd",
        },
        "training": {
            "n_state": 3,
            "n_senones": 200,
            "ci": {"max_iterations": 10, "min_iterations": 1, "convergence_ratio": 0.001},
            "untied": {"max_iterations": 6, "min_iterations": 1, "convergence_ratio": 0.001},
            "tied": {"max_iterations": 10, "min_iterations": 1, "convergence_ratio": 0.001},
        },
    },
    "wideband": {
        "description": "Wideband (16kHz) microphone speech",
        "features": {
            "samprate": 16000,
            "ncep": 13,
            "nfilt": 25,
            "nfft": 512,
            "lowerf": 130,
            "upperf": 6800,
            "feat_type": "1s_c_d_dd",
        },
        "training": {
            "n_state": 3,
            "n_senones": 200,
            "ci": {"max_iterations": 10, "min_iterations": 1, "convergence_ratio": 0.001},
            "untied": {"max_iterations": 6, "min_iterations": 1, "convergence_ratio": 0.001},
            "tied": {"max_iterations": 10, "min_iterations": 1, "convergence_ratio": 0.001},
        },
    },
    "telephone": {
        "description": "Telephone (8kHz) narrowband speech",
        "features": {
            "samprate": 8000,
            "ncep": 13,
            "nfilt": 15,
            "nfft": 256,
            "lowerf": 200,
            "upperf": 3500,
            "feat_type": "1s_c_d_dd",
        },
        "training": {
            "n_state": 3,
            "n_senones": 200,
            "ci": {"max_iterations": 10, "min_iterations": 1, "convergence_ratio": 0.001},
            "untied": {"max_iterations": 6, "min_iterations": 1, "convergence_ratio": 0.001},
            "tied": {"max_iterations": 10, "min_iterations": 1, "convergence_ratio": 0.001},
        },
    },
    "wideband_large": {
        "description": "Wideband with more senones (larger datasets)",
        "features": {
            "samprate": 16000,
            "ncep": 13,
            "nfilt": 25,
            "nfft": 512,
            "lowerf": 130,
            "upperf": 6800,
            "feat_type": "1s_c_d_dd",
        },
        "training": {
            "n_state": 3,
            "n_senones": 4000,
            "ci": {"max_iterations": 10, "min_iterations": 1, "convergence_ratio": 0.001},
            "untied": {"max_iterations": 6, "min_iterations": 1, "convergence_ratio": 0.001},
            "tied": {"max_iterations": 10, "min_iterations": 1, "convergence_ratio": 0.001},
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
            "ci": {"max_iterations": 10, "min_iterations": 1, "convergence_ratio": 0.001},
            "untied": {"max_iterations": 6, "min_iterations": 1, "convergence_ratio": 0.001},
            "tied": {"max_iterations": 10, "min_iterations": 1, "convergence_ratio": 0.001},
        },
    },
}


def _validate_params(
    profile: str,
    block: str,
    values: dict[str, Any],
    params_type: (
        type[FeatParams]
        | type[TrainingSchedule]
        | type[TrainParams]
        | type[SplitParams]
        | type[RunnerParams]
    ),
) -> None:
    """Reject misspelled profile parameters with configuration context."""
    known = {item.name for item in fields(params_type)}
    unknown = sorted(set(values) - known)
    if unknown:
        parameter = block.removesuffix("s")
        raise ValueError(f"unknown {parameter} parameter {unknown[0]!r} in profile {profile!r}")


def _coerce_dataclass_values(values: dict[str, Any], params_type: type[Any]) -> dict[str, Any]:
    """Coerce YAML scalars to the runtime types of dataclass defaults."""
    defaults = params_type()
    coerced = dict(values)
    for item in fields(params_type):
        if item.name not in coerced:
            continue
        default = getattr(defaults, item.name)
        value = coerced[item.name]
        if isinstance(default, TrainingSchedule):
            if not isinstance(value, dict):
                raise ValueError(f"{item.name} schedule must be a mapping")
            _validate_params("training", item.name, value, TrainingSchedule)
            coerced[item.name] = TrainingSchedule(
                **_coerce_dataclass_values(value, TrainingSchedule)
            )
        elif isinstance(default, tuple):
            coerced[item.name] = tuple(value)
        elif isinstance(default, float):
            coerced[item.name] = float(value)
        elif isinstance(default, int) and not isinstance(default, bool):
            coerced[item.name] = int(value)
    return coerced


def load_configs(project_dir: Path) -> dict[str, dict[str, Any]]:
    """Load named configurations from `project_dir/etc/configs.yaml`,
    merged on top of the built-in defaults."""
    configs_file = project_dir / "etc" / "configs.yaml"
    if not configs_file.exists():
        return dict(DEFAULT_CONFIGS)
    with configs_file.open() as f:
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
    runner: RunnerParams = field(default_factory=RunnerParams)
    description: str = ""

    @classmethod
    def from_config(
        cls,
        project_dir: Path | str,
        *,
        experiment: str = "default",
        config_name: str = "default",
    ) -> Self:
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
        runner_values = cfg.get("runner", {})
        _validate_params(config_name, "features", feature_values, FeatParams)
        _validate_params(config_name, "training", training_values, TrainParams)
        _validate_params(config_name, "split", split_values, SplitParams)
        _validate_params(config_name, "runner", runner_values, RunnerParams)
        nice = runner_values.get("nice", 5)
        if not isinstance(nice, int) or nice < 0:
            raise ValueError(f"config {config_name!r} runner.nice must be a non-negative integer")
        configured_jobs = runner_values.get("jobs")
        if configured_jobs is not None and (
            not isinstance(configured_jobs, int) or configured_jobs < 1
        ):
            raise ValueError(f"config {config_name!r} runner.jobs must be a positive integer")
        untied_inventory = training_values.get("untied_inventory", "all-triphone")
        if untied_inventory not in {"all-triphone", "transcript-reachable", "linear"}:
            raise ValueError(
                f"config {config_name!r} training.untied_inventory must be "
                "all-triphone, transcript-reachable, or linear"
            )
        return cls(
            project_dir=project_dir,
            experiment=experiment,
            config_name=config_name,
            description=cfg.get("description", ""),
            feat=FeatParams(**_coerce_dataclass_values(feature_values, FeatParams)),
            train=TrainParams(**_coerce_dataclass_values(training_values, TrainParams)),
            split=SplitParams(**_coerce_dataclass_values(split_values, SplitParams)),
            runner=RunnerParams(**runner_values),
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
            d / "provenance.json",
        ]

    def provenance_payload(self, stage: str) -> dict[str, Any]:
        """Canonical effective configuration governing a pipeline stage."""
        payload: dict[str, Any] = {
            "stage": stage,
            "tool_version": __version__,
            "native_library": _native_library_identity(),
        }
        if stage == "features":
            payload["features"] = asdict(self.feat)
        elif stage == "split":
            payload["split"] = asdict(self.split)
        elif stage == "training":
            payload.update(
                features=asdict(self.feat),
                training=asdict(self.train),
                split=asdict(self.split),
            )
        else:
            raise ValueError(f"unknown provenance stage: {stage!r}")
        return payload

    def provenance_path(self, stage: str) -> Path:
        """Content-addressed path for a stage's effective configuration."""
        canonical = json.dumps(
            self.provenance_payload(stage),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        fingerprint = hashlib.sha256(canonical).hexdigest()
        return (
            self.project_dir
            / ".pstrain"
            / "provenance"
            / self.experiment
            / self.config_name
            / f"{stage}-{fingerprint}.json"
        )

    def provenance_document(self, stage: str) -> dict[str, Any]:
        """Serializable provenance, including its effective-config fingerprint."""
        payload = json.loads(json.dumps(self.provenance_payload(stage), allow_nan=False))
        fingerprint = self.provenance_path(stage).stem.removeprefix(f"{stage}-")
        document = {"fingerprint": fingerprint, **payload}
        lib_path = get_lib_path()
        if lib_path is not None:
            document["native_library"] = {
                **payload["native_library"],
                "path": str(lib_path.resolve()),
            }
        return document

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
        with path.open() as f:
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
