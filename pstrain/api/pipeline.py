"""Public API for the training pipeline driver."""

from pathlib import Path
from typing import Any, Self

from pstrain.lib.pipeline import Pipeline, UnknownTargetError
from pstrain.lib.pipeline import PipelineContext as _PipelineContext
from pstrain.lib.pipeline.context import DEFAULT_CONFIGS
from pstrain.lib.pipeline.tasks import DEFAULT_TARGET, TARGETS
from pstrain.lib.pipeline.tasks import build_pipeline as _build_pipeline


class PipelineContext(_PipelineContext):
    """Public pipeline context with an observable API construction route."""

    @classmethod
    def from_config(
        cls,
        project_dir: Path | str,
        *,
        experiment: str = "default",
        config_name: str = "default",
        cli_overrides: dict[str, Any] | None = None,
    ) -> Self:
        """Build a context through a concrete public-API call frame."""
        return super().from_config(
            project_dir,
            experiment=experiment,
            config_name=config_name,
            cli_overrides=cli_overrides,
        )


def build_pipeline(ctx: PipelineContext) -> Pipeline:
    """Build the training graph through a concrete public-API call frame."""
    return _build_pipeline(ctx)


def run_pipeline(
    pipeline: Pipeline,
    target: str | Path,
    *,
    dry_run: bool = False,
    force: bool = False,
    jobs: int | None = None,
    verbose: bool = False,
) -> int:
    """Run a training graph through a concrete public-API call frame."""
    return pipeline.run(
        target,
        dry_run=dry_run,
        force=force,
        jobs=jobs,
        verbose=verbose,
    )


__all__ = [
    "PipelineContext",
    "UnknownTargetError",
    "build_pipeline",
    "run_pipeline",
    "TARGETS",
    "DEFAULT_TARGET",
    "DEFAULT_CONFIGS",
]
