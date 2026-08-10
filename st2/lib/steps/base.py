"""Base class for training steps.

Steps are the atomic units of the training pipeline. Each step:
1. Declares its input and output file paths so it can be used standalone.
2. Can be executed directly via CLI or library.
3. Emits shell commands in dry-run mode.

This abstraction is for *single-step debugging and incremental execution*
(e.g. `st2 step ci_hmm`). The full multi-step pipeline lives in
`st2.lib.pipeline`, which has its own task definitions.

The base class provides:
- Inputs/outputs declarations (for dry-run and dependency reasoning).
- Unified execution (dry-run emits shell, normal executes).
- CLI argument handling.
- Common path resolution.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StepContext:
    """Context for step execution.

    Similar to CLI's CommandContext but for library-level steps.
    Supports dry-run mode where actions emit shell instead of executing.
    """

    project_dir: Path
    experiment: str = "default"
    config: str = "baseline"
    dry_run: bool = False
    _header_emitted: bool = field(default=False, repr=False)

    def _emit_header(self) -> None:
        """Emit script header."""
        if self.dry_run and not self._header_emitted:
            print("#!/usr/bin/env bash")
            print("set -euo pipefail")
            print()
            self._header_emitted = True

    def _emit(self, cmd: str) -> None:
        """Emit a shell command."""
        if self.dry_run:
            self._emit_header()
            print(cmd)

    # Action methods - emit shell OR execute

    def mkdir(self, path: Path, parents: bool = True) -> None:
        """Create directory."""
        if self.dry_run:
            flags = "-p " if parents else ""
            self._emit(f"mkdir {flags}{shlex.quote(str(path))}")
        else:
            path.mkdir(parents=parents, exist_ok=True)

    def copy(self, src: Path, dst: Path) -> None:
        """Copy file."""
        if self.dry_run:
            self._emit(f"cp {shlex.quote(str(src))} {shlex.quote(str(dst))}")
        else:
            shutil.copy(src, dst)

    def symlink(self, src: Path, dst: Path) -> None:
        """Create symlink."""
        if self.dry_run:
            self._emit(f"ln -s {shlex.quote(str(src))} {shlex.quote(str(dst))}")
        else:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)

    def run_cmd(self, program: str, *args: str, **kwargs: Any) -> int:
        """Run a command."""
        cmd_parts = [program]
        for arg in args:
            cmd_parts.append(str(arg))
        for key, value in kwargs.items():
            key_str = key.replace("_", "-")
            if value is True:
                cmd_parts.append(f"--{key_str}")
            elif value is not False and value is not None:
                cmd_parts.extend([f"--{key_str}", str(value)])

        if self.dry_run:
            self._emit(" ".join(shlex.quote(p) for p in cmd_parts))
            return 0
        result = subprocess.run(cmd_parts)
        return result.returncode

    def comment(self, text: str) -> None:
        """Add a comment."""
        if self.dry_run:
            self._emit_header()
            for line in text.split("\n"):
                print(f"# {line}")

    def blank(self) -> None:
        """Add blank line."""
        if self.dry_run:
            self._emit_header()
            print()

    def log(self, message: str) -> None:
        """Log a message."""
        if self.dry_run:
            self.comment(message)
        else:
            print(message)

    def log_comment(self, text: str) -> None:
        """Alias for comment."""
        self.comment(text)

    # NOTE: sphinx_fe, bw, mk_flat methods removed - use library functions:
    # - st2.lib.features.extract_features() for feature extraction
    # - st2.lib.flat.init_flat_model() for flat model creation
    # - st2.lib.bw (TODO) for Baum-Welch training

    # Path resolution helpers

    @property
    def experiment_dir(self) -> Path:
        """Get experiment directory."""
        return self.project_dir / "experiments" / self.experiment

    @property
    def shared_dir(self) -> Path:
        """Get shared directory."""
        return self.project_dir / "shared"

    @property
    def etc_dir(self) -> Path:
        """Get etc directory."""
        return self.project_dir / "etc"

    def model_dir(self, model_type: str) -> Path:
        """Get model directory for a model type and config."""
        return self.experiment_dir / "models" / model_type / self.config / "model"

    def flat_dir(self, model_type: str = "ci") -> Path:
        """Get flat model directory."""
        return self.model_dir(model_type) / "flat"

    def hmm_dir(self, model_type: str = "ci") -> Path:
        """Get HMM model directory."""
        return self.model_dir(model_type) / "hmm"


@dataclass
class StepDefinition:
    """Declarative rule definition for a step (inputs/outputs/params/script)."""

    name: str
    description: str
    inputs: list[str]
    outputs: list[str]
    params: dict[str, Any]
    script: str


class Step(ABC):
    """Base class for training steps.

    Subclasses must implement:
    - name, description, script (class attributes)
    - get_inputs(), get_outputs() - return file paths
    - execute() - perform the step
    """

    # Subclasses must define these
    name: str = ""
    description: str = ""
    script: str = ""  # Program to run (e.g., "bw", "sphinx_fe")

    # Default parameters (subclasses can override)
    default_params: dict[str, Any] = {}

    @abstractmethod
    def get_inputs(self, ctx: StepContext) -> list[Path]:
        """Get input file paths.

        Args:
            ctx: Step context with project/experiment info

        Returns:
            List of input file paths (must exist before step runs)
        """

    @abstractmethod
    def get_outputs(self, ctx: StepContext) -> list[Path]:
        """Get output file paths.

        Args:
            ctx: Step context with project/experiment info

        Returns:
            List of output file paths (created by this step)
        """

    def get_params(self, ctx: StepContext, **overrides: Any) -> dict[str, Any]:
        """Get step parameters.

        Args:
            ctx: Step context
            **overrides: Parameter overrides

        Returns:
            Merged parameters
        """
        params = self.default_params.copy()
        params.update(overrides)
        return params

    def get_definition(self, ctx: StepContext, **params: Any) -> StepDefinition:
        """Get this step's declarative definition.

        Args:
            ctx: Step context
            **params: Parameter overrides

        Returns:
            StepDefinition with inputs, outputs, params, and script.
        """
        return StepDefinition(
            name=self.name,
            description=self.description,
            inputs=[str(p) for p in self.get_inputs(ctx)],
            outputs=[str(p) for p in self.get_outputs(ctx)],
            params=self.get_params(ctx, **params),
            script=self.script,
        )

    def to_dict(self, ctx: StepContext, **params: Any) -> dict[str, Any]:
        """Get this step's definition as a dict.

        Args:
            ctx: Step context
            **params: Parameter overrides

        Returns:
            Dict with rule definition
        """
        defn = self.get_definition(ctx, **params)
        return {
            "name": defn.name,
            "description": defn.description,
            "inputs": defn.inputs,
            "outputs": defn.outputs,
            "params": defn.params,
            "script": defn.script,
        }

    @abstractmethod
    def execute(self, ctx: StepContext, **params: Any) -> int:
        """Execute the step.

        This is the single code path - ctx methods emit shell in dry-run,
        execute otherwise.

        Args:
            ctx: Step context (with dry_run support)
            **params: Step parameters

        Returns:
            Exit code (0 for success)
        """

    def run(
        self,
        project_dir: Path | str,
        experiment: str = "default",
        config: str = "baseline",
        dry_run: bool = False,
        **params: Any,
    ) -> int:
        """Run the step (convenience wrapper).

        Args:
            project_dir: Project directory
            experiment: Experiment name
            config: Model configuration name
            dry_run: If True, emit shell commands instead of executing
            **params: Step parameters

        Returns:
            Exit code
        """
        ctx = StepContext(
            project_dir=Path(project_dir),
            experiment=experiment,
            config=config,
            dry_run=dry_run,
        )
        return self.execute(ctx, **params)

    # CLI support

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add step-specific arguments to parser.

        Override in subclasses to add custom arguments.
        """
        parser.add_argument(
            "--project-dir",
            type=str,
            default=".",
            help="Project directory (default: current directory)",
        )
        parser.add_argument(
            "--experiment",
            type=str,
            default="default",
            help="Experiment name (default: default)",
        )
        parser.add_argument(
            "--config",
            type=str,
            default="baseline",
            help="Model configuration name (default: baseline)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be done without executing",
        )

    def get_params_from_args(self, args: argparse.Namespace) -> dict[str, Any]:
        """Extract step parameters from parsed args.

        Override in subclasses to handle custom arguments.
        """
        return {}

    def main(self, args: list[str] | None = None) -> int:
        """CLI entry point.

        Args:
            args: Command line arguments (defaults to sys.argv[1:])

        Returns:
            Exit code
        """
        parser = argparse.ArgumentParser(description=f"{self.name}: {self.description}")
        self.add_arguments(parser)
        parsed = parser.parse_args(args)

        return self.run(
            project_dir=parsed.project_dir,
            experiment=parsed.experiment,
            config=parsed.config,
            dry_run=parsed.dry_run,
            **self.get_params_from_args(parsed),
        )
