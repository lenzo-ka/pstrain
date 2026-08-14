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
import platform
import socket
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Self, cast

from pstrain import __version__
from pstrain.lib.commands import PSTRAIN_BINARIES, resolve_binary
from pstrain.lib.config.models import Profile, TrainingScheduleConfig
from pstrain.lib.config.resolver import ResolvedConfig, resolve_config
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
    return {
        "sha256": _sha256_file(resolved, stat.st_size, stat.st_mtime_ns),
        "fp_contract_declared": _fp_contract_policy(),
    }


def _fp_contract_policy() -> str:
    """Return the contraction policy declared by the loaded native build."""
    from pstrain.lib._cffi.core import get_ffi, get_lib

    return cast(str, get_ffi().string(get_lib().pstrain_fp_contract_policy()).decode("ascii"))


def _native_program_identities(
    stage: str, *, include_paths: bool = False
) -> dict[str, dict[str, str]]:
    """Identify the resolved standalone programs that can govern a stage."""
    names = {
        "features": ("sphinx_fe",),
        "split": (),
        "training": tuple(PSTRAIN_BINARIES),
    }[stage]
    identities: dict[str, dict[str, str]] = {}
    for name in names:
        path = resolve_binary(name)
        if path is None:
            identities[name] = {"state": "absent"}
            continue
        stat = path.stat()
        identity = {"sha256": _sha256_file(path, stat.st_size, stat.st_mtime_ns)}
        if include_paths:
            identity["path"] = str(path)
        identities[name] = identity
    return identities


@dataclass(frozen=True)
class FeatParams:
    """Acoustic front-end parameters.

    Defaults are SphinxTrain wideband: 16 kHz audio, 25-filter mel bank,
    DCT-transformed cepstra with 13 coefficients, batch CMN, no AGC,
    no variance normalization. All fields are emitted into per-model
    `feat.params` files so PocketSphinx and friends can match the
    training-time front-end at decode/align time.
    """

    samprate: int = field(default_factory=lambda: Profile().features.samprate)
    ncep: int = field(default_factory=lambda: Profile().features.ncep)
    nfilt: int = field(default_factory=lambda: Profile().features.nfilt)
    nfft: int = field(default_factory=lambda: Profile().features.nfft)
    lowerf: int = field(default_factory=lambda: Profile().features.lowerf)
    upperf: int = field(default_factory=lambda: Profile().features.upperf)
    # Pre-emphasis coefficient (`-alpha`). 0.97 is the engine default.
    alpha: float = field(default_factory=lambda: Profile().features.alpha)
    dither: bool = field(default_factory=lambda: Profile().features.dither)
    remove_dc: bool = field(default_factory=lambda: Profile().features.remove_dc)
    remove_noise: bool = field(default_factory=lambda: Profile().features.remove_noise)
    frate: int = field(default_factory=lambda: Profile().features.frate)
    wlen: float = field(default_factory=lambda: Profile().features.wlen)
    feat_type: str = field(default_factory=lambda: Profile().features.feat_type)
    # Cepstral lifter window (sphinx_fe `-lifter`). 22 = SphinxTrain default.
    lifter: int = field(default_factory=lambda: Profile().features.lifter)
    # Linear-transform applied to filter bank outputs (`-transform`).
    transform: str = field(default_factory=lambda: Profile().features.transform)
    # Automatic gain control (`-agc`).
    agc: str = field(default_factory=lambda: Profile().features.agc)
    # Cepstral mean normalization (`-cmn`). "batch" matches SphinxTrain.
    cmn: str = field(default_factory=lambda: Profile().features.cmn)
    cmninit: str = field(default_factory=lambda: Profile().features.cmninit)
    # Cepstral variance normalization (`-varnorm`).
    varnorm: str = field(default_factory=lambda: Profile().features.varnorm)


@dataclass(frozen=True)
class TrainingSchedule:
    """Convergence controller for one family of Baum-Welch stages."""

    max_iterations: int = field(default_factory=lambda: TrainingScheduleConfig().max_iterations)
    min_iterations: int = field(default_factory=lambda: TrainingScheduleConfig().min_iterations)
    convergence_ratio: float = field(
        default_factory=lambda: TrainingScheduleConfig().convergence_ratio
    )


@dataclass(frozen=True)
class TrainParams:
    n_state: int = field(default_factory=lambda: Profile().training.n_state)
    n_senones: int = field(default_factory=lambda: Profile().training.n_senones)
    a_beam: float = field(default_factory=lambda: Profile().training.a_beam)
    b_beam: float = field(default_factory=lambda: Profile().training.b_beam)
    # A7c matched the upstream signed absolute likelihood-delta decision,
    # while measurements retained 0.001 rather than upstream's literal 0.1.
    # CI and tied stages keep that controller and upstream's ten-pass cap.
    ci: TrainingSchedule = field(
        default_factory=lambda: TrainingSchedule(**Profile().training.ci.model_dump())
    )
    tied: TrainingSchedule = field(
        default_factory=lambda: TrainingSchedule(**Profile().training.tied.model_dump())
    )
    # SphinxTrain scripts/30.cd_hmm_untied/norm_and_launchbw.pl uses the same
    # converge-with-cap controller. The preserved SLT oracle ended at pass 6;
    # cap this stage there so a stricter pstrain threshold cannot run to 10.
    untied: TrainingSchedule = field(
        default_factory=lambda: TrainingSchedule(**Profile().training.untied.model_dump())
    )
    # Warn on every skipped update; fail a stage above five percent.
    max_skip_fraction: float = field(default_factory=lambda: Profile().training.max_skip_fraction)
    # Retry forward-final-state pruning failures once at a beam this many
    # times wider (1e-90 / 1e10 = 1e-100).
    retry_beam_factor: float = field(default_factory=lambda: Profile().training.retry_beam_factor)
    arctic_a0302_zero_codebook_band: tuple[int, int] | None = field(
        default_factory=lambda: Profile().training.arctic_a0302_zero_codebook_band
    )
    accept_arctic_a0587_known_skip: bool = field(
        default_factory=lambda: Profile().training.accept_arctic_a0587_known_skip
    )
    optional_boundary_silence: bool = field(
        default_factory=lambda: Profile().training.optional_boundary_silence
    )
    # Tuned by SphinxTrain scripts/40.buildtrees/buildtree.pl for 3-state HMMs.
    tree_state_weights: tuple[float, ...] = field(
        default_factory=lambda: Profile().training.tree_state_weights
    )
    tree_rotate_state_weights: bool = field(
        default_factory=lambda: Profile().training.tree_rotate_state_weights
    )
    tree_directional_questions: bool = field(
        default_factory=lambda: Profile().training.tree_directional_questions
    )
    tree_ssplitmax: int = field(default_factory=lambda: Profile().training.tree_ssplitmax)
    tree_ssplitthr: float = field(default_factory=lambda: Profile().training.tree_ssplitthr)
    tree_csplitmax: int = field(default_factory=lambda: Profile().training.tree_csplitmax)
    tree_csplitthr: float = field(default_factory=lambda: Profile().training.tree_csplitthr)
    tree_mwfloor: float = field(default_factory=lambda: Profile().training.tree_mwfloor)
    tree_intermediate_dumps: bool = field(
        default_factory=lambda: Profile().training.tree_intermediate_dumps
    )
    question_npermute: int = field(default_factory=lambda: Profile().training.question_npermute)
    question_quests_per_state: int = field(
        default_factory=lambda: Profile().training.question_quests_per_state
    )
    question_niter: int = field(default_factory=lambda: Profile().training.question_niter)
    # Multi-pronunciation training: build wide utterance graphs that
    # sum Baum-Welch posteriors across pronunciation variants. On by
    # default; set to False to fall back to the legacy linear path that
    # always picks the first listed variant per word.
    multipron_training: bool = field(default_factory=lambda: Profile().training.multipron_training)
    # PipelineContext resolves an omitted config value by training mode:
    # all-dictionary for multipron and occurrence-based for linear training.
    untied_inventory: str = field(default_factory=lambda: Profile().training.untied_inventory)
    # Experimental parity instrument: stage -> pass (or "*") -> utterance IDs.
    exclusion_schedule: dict[str, dict[int | str, list[str]]] = field(
        default_factory=lambda: Profile().training.exclusion_schedule
    )


@dataclass(frozen=True)
class SplitParams:
    """Parameters for the train/test split task.

    Defaults match the `pstrain split` CLI: 95% train, seed 42.
    """

    train_ratio: float | None = field(default_factory=lambda: Profile().split.train_ratio)
    test_count: int | None = field(default_factory=lambda: Profile().split.test_count)
    seed: int = field(default_factory=lambda: Profile().split.seed)


@dataclass(frozen=True)
class RunnerParams:
    """Local process-allocation policy for pipeline fan-outs."""

    jobs: int | None = field(default_factory=lambda: Profile().runner.jobs)
    nice: int = field(default_factory=lambda: Profile().runner.nice)


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


def load_configs(project_dir: Path) -> dict[str, dict[str, Any]]:
    """Compatibility view of named profiles through the canonical reader."""
    from pstrain.lib.config.resolver import _profile_documents

    profiles, _, _, _ = _profile_documents(project_dir.resolve())
    return profiles


def _runtime_values(profile: Profile) -> tuple[FeatParams, TrainParams, SplitParams, RunnerParams]:
    values = profile.model_dump(mode="python")
    training = values["training"]
    for stage in ("ci", "untied", "tied"):
        training[stage] = TrainingSchedule(**training[stage])
    return (
        FeatParams(**values["features"]),
        TrainParams(**training),
        SplitParams(**values["split"]),
        RunnerParams(**values["runner"]),
    )


@dataclass(frozen=True)
class PipelineContext:
    """Per-run configuration for the training pipeline."""

    project_dir: Path
    experiment: str = "default"
    config_name: str = "default"
    feat: FeatParams = field(default_factory=lambda: _runtime_values(Profile())[0])
    train: TrainParams = field(default_factory=lambda: _runtime_values(Profile())[1])
    split: SplitParams = field(default_factory=lambda: _runtime_values(Profile())[2])
    runner: RunnerParams = field(default_factory=lambda: _runtime_values(Profile())[3])
    description: str = ""
    resolved_config: ResolvedConfig | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_config(
        cls,
        project_dir: Path | str,
        *,
        experiment: str = "default",
        config_name: str = "default",
        cli_overrides: dict[str, Any] | None = None,
    ) -> Self:
        """Build a context by reading `project/etc/configs.yaml`."""
        project_dir = Path(project_dir).resolve()
        resolved = resolve_config(
            project_dir,
            profile_name=config_name,
            experiment=experiment,
            cli_overrides=cli_overrides,
        )
        feat, train, split, runner = _runtime_values(resolved.profile)
        return cls(
            project_dir=project_dir,
            experiment=experiment,
            config_name=config_name,
            description=resolved.profile.description,
            feat=feat,
            train=train,
            split=split,
            runner=runner,
            resolved_config=resolved,
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
            "native_programs": _native_program_identities(stage),
            "config_version": self.resolved_config.config_version if self.resolved_config else 1,
        }
        if self.resolved_config:
            payload["config_sources"] = {
                path: item.winner.source_kind for path, item in self.resolved_config.fields.items()
            }
        if stage == "features":
            payload["features"] = asdict(self.feat)
        elif stage == "split":
            payload["split"] = asdict(self.split)
        elif stage == "training":
            requested_bw_jobs = self.runner.jobs or 1
            effective_bw_shards = (
                1 if requested_bw_jobs > 1 and self.train.multipron_training else requested_bw_jobs
            )
            payload.update(
                features=asdict(self.feat),
                training=asdict(self.train),
                split=asdict(self.split),
                execution={
                    "host": socket.gethostname(),
                    "architecture": platform.machine(),
                    "requested_jobs": requested_bw_jobs,
                    "bw_shard_count": effective_bw_shards,
                },
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
        document["native_programs"] = _native_program_identities(stage, include_paths=True)
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
