"""Configuration models for pstrain using Pydantic.

Three-tier configuration:
1. User (~/.pstrain/config.yaml) - Global defaults
2. Project (project/etc/config.yaml) - Shared across experiments
3. Experiment (project/experiments/{name}/config.yaml) - Experiment-specific

All feature extraction parameters live in config. No separate feat.params file
is needed as input - sphinx_fe reads from config via CLI args.
feat.params is generated as OUTPUT for decoder compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, field_validator


class ParallelConfig(BaseModel):
    """Parallel execution configuration."""

    n_jobs: int = Field(
        -1,
        description="Number of parallel jobs: 1=serial, -1=all cores minus 1, N=specific count",
    )
    show_progress: bool = Field(
        True,
        description="Show progress bars during parallel execution",
    )

    @field_validator("n_jobs")
    @classmethod
    def validate_n_jobs(cls, v: int) -> int:
        """Validate n_jobs value."""
        if v < -1 or v == 0:
            raise ValueError("n_jobs must be -1 (all cores), 1 (serial), or positive")
        return v


class AudioConfig(BaseModel):
    """Audio input configuration."""

    sample_rate: int = Field(16000, gt=0, description="Audio sample rate in Hz")
    format: Literal["wav", "raw", "sphere"] = Field("wav", description="Audio format")

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        """Warn about unusual sample rates."""
        common_rates = {8000, 11025, 16000, 22050, 44100, 48000}
        if v not in common_rates:
            import warnings

            warnings.warn(f"Unusual sample rate: {v} Hz", stacklevel=2)
        return v


class FeatureConfig(BaseModel):
    """Feature extraction configuration.

    These parameters control sphinx_fe. They are passed via CLI args,
    and also written to feat.params for decoder compatibility.
    """

    # Feature type
    type: Literal["mfcc"] = Field("mfcc", description="Feature type")
    feature_type: Literal["1s_c_d_dd", "s2_4x"] = Field(
        "1s_c_d_dd",
        description="Sphinx feature stream type (1s_c_d_dd=continuous, s2_4x=semi-continuous)",
    )

    # MFCC parameters
    num_ceps: int = Field(13, ge=1, le=50, description="Number of cepstral coefficients")
    num_filters: int = Field(
        25,
        ge=1,
        le=100,
        description="Number of mel filters (25 SphinxTrain wideband; 15 for 8 kHz telephone)",
    )
    nfft: int = Field(512, ge=64, le=4096, description="FFT size")

    # Frame parameters
    frame_length_ms: float = Field(25.0, gt=0, description="Frame length in milliseconds")
    frame_shift_ms: float = Field(10.0, gt=0, description="Frame shift in milliseconds")

    # Frequency range
    lower_freq: float = Field(
        130.0, ge=0, description="Lower frequency cutoff (Hz) - 130 wideband, 200 telephone"
    )
    upper_freq: float = Field(
        6800.0, ge=0, description="Upper frequency cutoff (Hz) - 6800 wideband, 3500 telephone"
    )

    # Processing
    preemphasis: float = Field(0.97, ge=0.0, le=1.0, description="Preemphasis coefficient")
    lifter: int = Field(22, ge=0, description="Liftering parameter (0=no liftering)")
    use_energy: bool = Field(True, description="Include energy feature")

    # Deltas
    delta: bool = Field(True, description="Compute delta features")
    delta_delta: bool = Field(True, description="Compute delta-delta features")

    # Normalization
    agc: Literal["none", "max"] = Field("none", description="Automatic Gain Control")
    cmn: Literal["batch", "current", "none"] = Field(
        "batch", description="Cepstral Mean Normalization"
    )
    varnorm: bool = Field(False, description="Variance normalization")
    transform: Literal["dct", "legacy"] = Field("dct", description="Transform type")

    @property
    def sphinx_feat_type(self) -> str:
        """Get Sphinx feature type string for binaries."""
        return self.feature_type

    @property
    def num_streams(self) -> int:
        """Get number of feature streams."""
        return 4 if self.feature_type == "s2_4x" else 1

    @property
    def model_type(self) -> str:
        """Get corresponding HMM model type (.cont., .semi.)."""
        return ".semi." if self.feature_type == "s2_4x" else ".cont."

    @property
    def full_dimension(self) -> int:
        """Get full feature dimension including deltas."""
        if self.feature_type == "s2_4x":
            return self.num_ceps * 4
        # 1s_c_d_dd: base + delta + delta-delta
        return self.num_ceps * 3

    def to_sphinx_fe_args(self) -> list[str]:
        """Convert to sphinx_fe command-line arguments."""
        return [
            "-samprate",
            str(self.num_ceps),  # Will use audio.sample_rate
            "-nfilt",
            str(self.num_filters),
            "-nfft",
            str(self.nfft),
            "-lowerf",
            str(self.lower_freq),
            "-upperf",
            str(self.upper_freq),
            "-ncep",
            str(self.num_ceps),
            "-alpha",
            str(self.preemphasis),
            "-lifter",
            str(self.lifter),
            "-transform",
            self.transform,
        ]


class CITrainingConfig(BaseModel):
    """CI (context-independent, monophone) training configuration."""

    n_gaussians: int = Field(1, ge=1, description="Initial number of Gaussians per state")
    n_iterations: int = Field(10, ge=1, le=100, description="Maximum training iterations")
    convergence_threshold: float = Field(
        0.001, gt=0, description="Convergence threshold (fractional log-likelihood improvement)"
    )
    min_iterations: int = Field(
        1, ge=1, description="Minimum iterations before checking convergence"
    )

    # Beams
    abeam: float = Field(1e-90, gt=0, description="Alpha beam for BW forward pass")
    bbeam: float = Field(1e-10, gt=0, description="Beta beam for BW backward pass")

    # Floors
    varfloor: float = Field(1e-4, gt=0, description="Variance floor")
    mixw_floor: float = Field(1e-8, gt=0, description="Mixture weight floor")

    # Gaussian selection
    topn: int = Field(1, ge=1, description="Number of top Gaussians to use in BW")


class CDUntiedConfig(BaseModel):
    """CD-Untied (untied triphone) training configuration."""

    n_gaussians: int = Field(1, ge=1, description="Number of Gaussians for untied models")
    n_iterations: int = Field(10, ge=1, le=100, description="Maximum training iterations")
    convergence_threshold: float = Field(0.001, gt=0, description="Convergence threshold")
    min_iterations: int = Field(1, ge=1, description="Minimum iterations")

    abeam: float = Field(1e-90, gt=0, description="Alpha beam")
    bbeam: float = Field(1e-10, gt=0, description="Beta beam")
    varfloor: float = Field(1e-4, gt=0, description="Variance floor")
    mixw_floor: float = Field(1e-5, gt=0, description="Mixture weight floor")
    tmat_floor: float = Field(1e-5, gt=0, description="Transition probability floor")
    topn: int = Field(8, ge=1, description="Number of top Gaussians")


class CDTiedConfig(BaseModel):
    """CD-Tied (tied triphone) training configuration."""

    n_gaussians: int = Field(8, ge=1, description="Number of Gaussians per state")
    n_senones: int = Field(200, ge=10, description="Target number of senones (tied states)")
    n_iterations: int = Field(10, ge=1, le=100, description="Maximum training iterations")
    convergence_threshold: float = Field(0.001, gt=0, description="Convergence threshold")
    min_iterations: int = Field(1, ge=1, description="Minimum iterations")

    abeam: float = Field(1e-90, gt=0, description="Alpha beam")
    bbeam: float = Field(1e-10, gt=0, description="Beta beam")
    varfloor: float = Field(1e-4, gt=0, description="Variance floor")
    mixw_floor: float = Field(1e-5, gt=0, description="Mixture weight floor")
    tmat_floor: float = Field(1e-5, gt=0, description="Transition probability floor")
    topn: int = Field(8, ge=1, description="Number of top Gaussians")


class CDConfig(BaseModel):
    """CD (context-dependent, triphone) training configuration."""

    untied: CDUntiedConfig = Field(
        default_factory=lambda: CDUntiedConfig(),
        description="Untied triphone training configuration",
    )
    tied: CDTiedConfig = Field(
        default_factory=lambda: CDTiedConfig(),
        description="Tied triphone training configuration",
    )


class GaussianIncrementConfig(BaseModel):
    """Gaussian increment (splitting) configuration."""

    enabled: bool = Field(False, description="Enable Gaussian splitting")
    schedule: list[int] = Field(
        default=[1, 2, 4, 8],
        description="Gaussian splitting schedule (powers of 2)",
    )
    n_iterations_after_split: int = Field(
        10, ge=1, description="Re-training iterations after split"
    )

    @field_validator("schedule")
    @classmethod
    def validate_schedule(cls, v: list[int]) -> list[int]:
        """Ensure schedule is monotonically increasing powers of 2."""
        if not v:
            raise ValueError("Schedule cannot be empty")
        if not all(x > 0 and (x & (x - 1)) == 0 for x in v):
            raise ValueError("Schedule must contain only powers of 2")
        if v != sorted(v):
            raise ValueError("Schedule must be monotonically increasing")
        return v


class DecisionTreeConfig(BaseModel):
    """Decision tree clustering configuration."""

    questions_file: Path | None = Field(
        None, description="Path to questions file (auto-generate if None)"
    )
    min_observations: int = Field(100, ge=1, description="Minimum observations per leaf node")
    max_depth: int = Field(50, ge=1, description="Maximum tree depth")


class TrainingConfig(BaseModel):
    """Complete training configuration."""

    n_states: int = Field(
        3, ge=1, le=7, description="Number of emitting states per HMM (3 or 5 typical)"
    )

    ci: CITrainingConfig = Field(
        default_factory=lambda: CITrainingConfig(),
        description="Context-independent (monophone) training",
    )
    cd: CDConfig = Field(
        default_factory=lambda: CDConfig(),
        description="Context-dependent (triphone) training",
    )
    tree: DecisionTreeConfig = Field(
        default_factory=lambda: DecisionTreeConfig(),
        description="Decision tree configuration",
    )
    gaussian_increment: GaussianIncrementConfig = Field(
        default_factory=lambda: GaussianIncrementConfig(),
        description="Gaussian splitting configuration",
    )


class DictionaryConfig(BaseModel):
    """Dictionary handling configuration."""

    main_dict: str = Field(
        "shared/dictionary.dict",
        description="Main dictionary file (relative to project)",
    )
    filler_dict: str = Field(
        "shared/filler.dict",
        description="Filler dictionary (sentence boundaries)",
    )
    phoneset: str = Field(
        "shared/phoneset.txt",
        description="Phoneset file",
    )
    case_sensitive: bool = Field(
        False,
        description="Case-sensitive word lookup",
    )
    silence_phone: str = Field(
        "SIL",
        description="Silence phone symbol",
    )


class CorpusConfig(BaseModel):
    """Corpus location configuration."""

    audio_dir: str | None = Field(
        None,
        description="Audio directory (if None, uses project_dir/audio/)",
    )
    transcript_file: str | None = Field(
        None,
        description="Transcript file (if None, uses etc/all.transcription)",
    )


class SplitConfig(BaseModel):
    """Train/test split configuration."""

    train_ratio: float = Field(
        0.9, ge=0.0, le=1.0, description="Fraction for training (0.9 = 90% train, 10% test)"
    )
    seed: int | None = Field(
        None, description="Random seed for reproducible splits (None = random)"
    )


class PstrainConfig(BaseModel):
    """Top-level pstrain configuration.

    Combines all configuration sections. Can be bound to a project directory
    to enable computed path properties.
    """

    # Experiment metadata
    name: str | None = Field(None, description="Experiment name")
    description: str | None = Field(None, description="Experiment description")

    # Core configuration sections
    parallel: ParallelConfig = Field(default_factory=lambda: ParallelConfig())
    audio: AudioConfig = Field(default_factory=lambda: AudioConfig())
    features: FeatureConfig = Field(default_factory=lambda: FeatureConfig())
    training: TrainingConfig = Field(default_factory=lambda: TrainingConfig())
    dictionary: DictionaryConfig = Field(default_factory=lambda: DictionaryConfig())
    corpus: CorpusConfig = Field(default_factory=lambda: CorpusConfig())
    split: SplitConfig = Field(default_factory=lambda: SplitConfig())

    # Runtime binding (not serialized)
    _project_dir: Path | None = None
    _config_file: Path | None = None

    model_config = {"arbitrary_types_allowed": True}

    def bind_to_project(self, project_dir: Path, config_file: Path | None = None) -> Self:
        """Bind config to a project directory, enabling path properties.

        Args:
            project_dir: Project root directory
            config_file: Optional config file path

        Returns:
            Self (for chaining)
        """
        self._project_dir = Path(project_dir).resolve()
        self._config_file = Path(config_file).resolve() if config_file else None
        return self

    @property
    def project_dir(self) -> Path:
        """Get project directory (requires bind_to_project() first)."""
        if self._project_dir is None:
            raise RuntimeError("Config not bound to project. Call bind_to_project() first.")
        return self._project_dir

    @property
    def etc_dir(self) -> Path:
        """Get etc directory."""
        return self.project_dir / "etc"

    @property
    def shared_dir(self) -> Path:
        """Get shared directory."""
        return self.project_dir / "shared"

    @property
    def audio_dir(self) -> Path:
        """Get audio directory."""
        if self.corpus.audio_dir:
            audio_path = Path(self.corpus.audio_dir)
            return audio_path if audio_path.is_absolute() else self.project_dir / audio_path
        return self.project_dir / "audio"

    @property
    def dictionary_path(self) -> Path:
        """Get main dictionary path."""
        return self.project_dir / self.dictionary.main_dict

    @property
    def filler_dict_path(self) -> Path:
        """Get filler dictionary path."""
        return self.project_dir / self.dictionary.filler_dict

    @property
    def phoneset_path(self) -> Path:
        """Get phoneset path."""
        return self.project_dir / self.dictionary.phoneset

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (for serialization)."""
        return self.model_dump(exclude={"_project_dir", "_config_file"})

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        """Load configuration from YAML file."""
        import yaml

        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        import yaml

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def default(cls) -> Self:
        """Get default configuration."""
        return cls()
