"""pstrain training pipeline.

A small Python-native task runner that replaces Snakemake for the pstrain training
workflow. Tasks declare their inputs and outputs as file paths; the pipeline
walks the dependency graph, checks mtime-based staleness, and runs only what's
needed.

Design notes
------------
* `Task` is a passive dataclass: name, inputs, outputs, callable, metadata.
  Tasks know nothing about the runner.
* `Pipeline` resolves dependencies by matching one task's outputs against
  another's inputs (the same model Snakemake uses).
* `PipelineContext` holds the per-run config: project dir, experiment, config
  name, and the feature/training parameters derived from `etc/configs.yaml`.
* Tasks for the pstrain training workflow live in `pstrain.lib.pipeline.tasks`.

The runner intentionally does not support cluster execution, conda envs, or
content-hash staleness. Add those only if a concrete need shows up.
"""

from pstrain.lib.pipeline.context import PipelineContext, load_configs
from pstrain.lib.pipeline.runner import (
    Pipeline,
    Task,
    TaskFailure,
    UnknownTargetError,
)

__all__ = [
    "Pipeline",
    "PipelineContext",
    "Task",
    "TaskFailure",
    "UnknownTargetError",
    "load_configs",
]
