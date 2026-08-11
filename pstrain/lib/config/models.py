"""Canonical, versioned configuration schema for pstrain.

This module is the only place semantic configuration fields and their defaults
are declared.  Runtime dataclasses are projections of :class:`Profile`.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CURRENT_CONFIG_VERSION: Literal[1] = 1


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FeatureConfig(StrictModel):
    """Acoustic front-end parameters."""

    samprate: Annotated[int, Field(gt=0, description="Audio sample rate in Hz")] = 16000
    ncep: Annotated[int, Field(gt=0, description="Number of cepstral coefficients")] = 13
    nfilt: Annotated[int, Field(gt=0, description="Number of mel filters")] = 25
    nfft: Annotated[int, Field(gt=0, description="FFT size")] = 512
    lowerf: Annotated[int, Field(ge=0, description="Lower filter-bank frequency in Hz")] = 130
    upperf: Annotated[int, Field(gt=0, description="Upper filter-bank frequency in Hz")] = 6800
    alpha: Annotated[float, Field(description="Pre-emphasis coefficient")] = 0.97
    feat_type: Annotated[str, Field(description="Sphinx feature stream type")] = "1s_c_d_dd"
    lifter: Annotated[int, Field(ge=0, description="Cepstral lifter window")] = 22
    transform: Annotated[str, Field(description="Filter-bank transform")] = "dct"
    agc: Annotated[str, Field(description="Automatic gain-control mode")] = "none"
    cmn: Annotated[str, Field(description="Cepstral mean-normalization mode")] = "batch"
    varnorm: Annotated[str, Field(description="Cepstral variance-normalization mode")] = "no"

    @model_validator(mode="after")
    def validate_band(self) -> FeatureConfig:
        if self.upperf <= self.lowerf:
            raise ValueError("upperf must be greater than lowerf")
        return self


class TrainingScheduleConfig(StrictModel):
    """Convergence controller for one Baum-Welch stage family."""

    max_iterations: Annotated[int, Field(ge=1, description="Maximum training passes")] = 10
    min_iterations: Annotated[int, Field(ge=1, description="Minimum training passes")] = 1
    convergence_ratio: Annotated[
        float, Field(gt=0, description="Absolute likelihood-delta convergence threshold")
    ] = 0.001

    @model_validator(mode="after")
    def validate_iterations(self) -> TrainingScheduleConfig:
        if self.min_iterations > self.max_iterations:
            raise ValueError("min_iterations must not exceed max_iterations")
        return self


class TrainingConfig(StrictModel):
    """Acoustic-model training parameters."""

    n_state: Annotated[int, Field(ge=1, description="Emitting states per HMM")] = 3
    n_senones: Annotated[int, Field(ge=1, description="Target tied-state count")] = 200
    a_beam: Annotated[float, Field(gt=0, description="Forward alignment beam")] = 1e-90
    b_beam: Annotated[float, Field(gt=0, description="Backward alignment beam")] = 1e-10
    ci: TrainingScheduleConfig = Field(default_factory=TrainingScheduleConfig)
    tied: TrainingScheduleConfig = Field(default_factory=TrainingScheduleConfig)
    untied: TrainingScheduleConfig = Field(
        default_factory=lambda: TrainingScheduleConfig(max_iterations=6)
    )
    max_skip_fraction: Annotated[
        float, Field(ge=0, le=1, description="Maximum skipped-update fraction")
    ] = 0.05
    retry_beam_factor: Annotated[
        float, Field(gt=0, description="Beam widening factor for one retry")
    ] = 1e10
    tree_state_weights: Annotated[
        tuple[float, ...], Field(min_length=1, description="Decision-tree state weights")
    ] = (1.0, 0.05, 0.0)
    tree_ssplitmax: Annotated[int, Field(ge=0, description="Maximum state splits")] = 7
    tree_ssplitthr: Annotated[float, Field(ge=0, description="State split threshold")] = 0.0
    tree_csplitmax: Annotated[int, Field(ge=0, description="Maximum phone-context splits")] = 2000
    tree_csplitthr: Annotated[float, Field(ge=0, description="Phone-context split threshold")] = 0.0
    tree_mwfloor: Annotated[float, Field(gt=0, description="Tree mixture-weight floor")] = 1e-8
    question_npermute: Annotated[int, Field(ge=1, description="Question permutations")] = 12
    question_quests_per_state: Annotated[
        int, Field(ge=1, description="Questions generated per state")
    ] = 20
    question_niter: Annotated[int, Field(ge=1, description="Question generation iterations")] = 1
    multipron_training: Annotated[
        bool, Field(description="Sum posteriors over pronunciation variants")
    ] = True
    untied_inventory: Annotated[
        Literal["all-triphone", "transcript-reachable", "linear"],
        Field(description="Untied-model phone inventory policy"),
    ] = "all-triphone"
    exclusion_schedule: Annotated[
        dict[str, dict[int | str, list[str]]],
        Field(description="Experimental stage/pass utterance exclusions"),
    ] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def select_inventory_default(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("multipron_training") is False:
            data = dict(data)
            data.setdefault("untied_inventory", "linear")
        return data

    @field_validator("exclusion_schedule")
    @classmethod
    def validate_exclusions(
        cls, value: dict[str, dict[int | str, list[str]]]
    ) -> dict[str, dict[int | str, list[str]]]:
        stages = {
            "ci-1g",
            "ci-2g",
            "ci-4g",
            "ci-8g",
            "cd-untied",
            "cd-1g",
            "cd-2g",
            "cd-4g",
            "cd-8g",
            "cd-16g",
            "cd-32g",
        }
        for stage, passes in value.items():
            if stage not in stages:
                raise ValueError(f"unknown BW stage {stage!r}")
            for selector, utterances in passes.items():
                if selector != "*" and (isinstance(selector, bool) or int(selector) < 1):
                    raise ValueError("pass selectors must be positive integers or '*'")
                if not all(utterances):
                    raise ValueError("utterance IDs must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_training(self) -> TrainingConfig:
        if not self.multipron_training and self.untied_inventory == "transcript-reachable":
            raise ValueError(
                "training.untied_inventory 'transcript-reachable' requires "
                "training.multipron_training: true; linear mode's equivalent is the "
                "'linear' policy"
            )
        return self


class SplitConfig(StrictModel):
    """Train/test split parameters."""

    train_ratio: Annotated[float | None, Field(gt=0, lt=1, description="Training fraction")] = None
    test_count: Annotated[
        int | None,
        Field(ge=0, description="Fixed test utterance count; zero disables an additional holdout"),
    ] = None
    seed: Annotated[int, Field(description="Deterministic split seed")] = 42

    @model_validator(mode="after")
    def validate_choice(self) -> SplitConfig:
        if self.train_ratio is not None and self.test_count is not None:
            raise ValueError("train_ratio and test_count are mutually exclusive")
        return self


class RunnerConfig(StrictModel):
    """Local pipeline execution policy."""

    jobs: Annotated[int | None, Field(ge=1, description="Parallel workers; null means auto")] = None
    nice: Annotated[int, Field(ge=0, description="Worker niceness increment")] = 5


class Profile(StrictModel):
    """One complete named model-training profile."""

    description: Annotated[str, Field(description="Human-readable profile purpose")] = ""
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)


class ProfileDefinition(StrictModel):
    """On-disk profile; inheritance is allowed only through ``extends``."""

    extends: str | None = Field(None, description="Profile to deep-merge before validation")
    description: str | None = None
    features: dict[str, Any] | None = None
    training: dict[str, Any] | None = None
    split: dict[str, Any] | None = None
    runner: dict[str, Any] | None = None


class ProfilesDocument(StrictModel):
    """Canonical ``etc/configs.yaml`` document."""

    config_version: Literal[1] = CURRENT_CONFIG_VERSION
    profiles: dict[str, ProfileDefinition]


class OverlayDocument(StrictModel):
    """Canonical user, project, or experiment field overlay."""

    config_version: Literal[1] = CURRENT_CONFIG_VERSION
    profile: str | None = None
    features: dict[str, Any] | None = None
    training: dict[str, Any] | None = None
    split: dict[str, Any] | None = None
    runner: dict[str, Any] | None = None


SEMANTIC_BLOCKS = ("features", "training", "split", "runner")


def default_profile() -> Profile:
    """Return the schema-default profile."""
    return Profile()
