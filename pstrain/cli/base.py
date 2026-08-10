"""Base classes for CLI commands."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self


def format_json(data: Any, indent: int = 2, sort_keys: bool = False) -> str:
    """Format data as JSON with consistent settings."""
    return json.dumps(data, indent=indent, ensure_ascii=False, sort_keys=sort_keys)


def add_dry_run_argument(parser: argparse.ArgumentParser) -> None:
    """Add --dry-run argument to a parser."""
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )


def add_json_argument(parser: argparse.ArgumentParser) -> None:
    """Add --json argument and formatting options to a parser."""
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--json-indent",
        type=int,
        default=2,
        metavar="N",
        help="JSON indentation level (default: 2, use 0 for compact)",
    )
    parser.add_argument(
        "--json-ascii",
        action="store_true",
        help="Escape non-ASCII characters in JSON output",
    )


# =============================================================================
# Actions - can emit shell OR execute
# =============================================================================


class Action(ABC):
    """Base class for actions that can emit shell or execute."""

    @abstractmethod
    def to_shell(self) -> str:
        """Return shell command string."""

    @abstractmethod
    def execute(self) -> None:
        """Execute the action."""

    def run(self, dry_run: bool) -> str | None:
        """Run action: emit shell if dry_run, else execute.

        Returns shell command if dry_run, None otherwise.
        """
        if dry_run:
            return self.to_shell()
        self.execute()
        return None


@dataclass
class MkdirAction(Action):
    """Create directory."""

    path: Path
    parents: bool = True

    def to_shell(self) -> str:
        flags = "-p " if self.parents else ""
        return f"mkdir {flags}{shlex.quote(str(self.path))}"

    def execute(self) -> None:
        self.path.mkdir(parents=self.parents, exist_ok=True)


@dataclass
class CopyAction(Action):
    """Copy file."""

    src: Path
    dst: Path

    def to_shell(self) -> str:
        return f"cp {shlex.quote(str(self.src))} {shlex.quote(str(self.dst))}"

    def execute(self) -> None:
        shutil.copy(self.src, self.dst)


@dataclass
class CopyTreeAction(Action):
    """Copy directory tree."""

    src: Path
    dst: Path

    def to_shell(self) -> str:
        return f"cp -r {shlex.quote(str(self.src))} {shlex.quote(str(self.dst))}"

    def execute(self) -> None:
        shutil.copytree(self.src, self.dst, dirs_exist_ok=True)


@dataclass
class SymlinkAction(Action):
    """Create symlink."""

    src: Path
    dst: Path

    def to_shell(self) -> str:
        return f"ln -s {shlex.quote(str(self.src))} {shlex.quote(str(self.dst))}"

    def execute(self) -> None:
        if self.dst.exists() or self.dst.is_symlink():
            self.dst.unlink()
        self.dst.symlink_to(self.src)


@dataclass
class RemoveAction(Action):
    """Remove file or directory."""

    path: Path
    recursive: bool = False
    force: bool = False

    def to_shell(self) -> str:
        flags = ""
        if self.recursive:
            flags += "r"
        if self.force:
            flags += "f"
        if flags:
            flags = f"-{flags} "
        return f"rm {flags}{shlex.quote(str(self.path))}"

    def execute(self) -> None:
        if self.path.is_dir() and self.recursive:
            shutil.rmtree(self.path, ignore_errors=self.force)
        elif self.path.exists():
            self.path.unlink(missing_ok=self.force)


@dataclass
class WriteFileAction(Action):
    """Write content to file."""

    path: Path
    content: str

    def to_shell(self) -> str:
        # Use heredoc for multi-line, echo for single-line
        if "\n" in self.content:
            return f"cat > {shlex.quote(str(self.path))} << 'EOF'\n{self.content}\nEOF"
        return f"echo {shlex.quote(self.content)} > {shlex.quote(str(self.path))}"

    def execute(self) -> None:
        self.path.write_text(self.content, encoding="utf-8")


@dataclass
class ShellCommandAction(Action):
    """Run arbitrary shell command."""

    program: str
    args: tuple[str, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    cwd: Path | None = None

    def to_shell(self) -> str:
        parts = [self.program]
        for arg in self.args:
            parts.append(shlex.quote(str(arg)))
        for key, value in self.kwargs.items():
            key_str = key.replace("_", "-")
            if value is True:
                parts.append(f"--{key_str}")
            elif value is not False and value is not None:
                parts.append(f"--{key_str}")
                parts.append(shlex.quote(str(value)))
        cmd = " ".join(parts)
        if self.cwd:
            return f"(cd {shlex.quote(str(self.cwd))} && {cmd})"
        return cmd

    def execute(self) -> None:
        cmd = [self.program]
        for arg in self.args:
            cmd.append(str(arg))
        for key, value in self.kwargs.items():
            key_str = key.replace("_", "-")
            if value is True:
                cmd.append(f"--{key_str}")
            elif value is not False and value is not None:
                cmd.extend([f"--{key_str}", str(value)])
        subprocess.run(cmd, check=True, cwd=self.cwd)


@dataclass
class CommentAction(Action):
    """A comment (no-op for execution)."""

    text: str

    def to_shell(self) -> str:
        lines = self.text.split("\n")
        return "\n".join(f"# {line}" for line in lines)

    def execute(self) -> None:
        pass  # Comments do nothing when executed


# =============================================================================
# pstrain Actions - emit shell commands
# =============================================================================


@dataclass
class PstrainAction(Action):
    """Base class for pstrain C program actions.

    These actions emit shell commands. Execute methods use CFFI where available.
    For now, execute() raises NotImplementedError - use library functions directly.
    """

    def _get_shell_cmd(self) -> list[str]:
        """Return shell command args. Override in subclass."""
        raise NotImplementedError

    def to_shell(self) -> str:
        return shlex.join(self._get_shell_cmd())

    def execute(self) -> None:
        """Execute using CFFI. Override in subclass or use library directly."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.execute() not implemented. "
            "Use pstrain.lib functions directly for CFFI execution."
        )


@dataclass
class FeatureExtractAction(PstrainAction):
    """Extract features from audio file."""

    input_file: Path
    output_file: Path
    samprate: int = 16000
    nfilt: int = 25
    ncep: int = 13

    def _get_shell_cmd(self) -> list[str]:
        return [
            "sphinx_fe",
            "-i",
            str(self.input_file),
            "-o",
            str(self.output_file),
            "-samprate",
            str(self.samprate),
            "-nfilt",
            str(self.nfilt),
            "-ncep",
            str(self.ncep),
        ]

    def execute(self) -> None:
        from pstrain.lib.features import extract_features

        extract_features(self.input_file, self.output_file)


@dataclass
class BaumWelchAction(PstrainAction):
    """Run Baum-Welch training iteration."""

    mdef: Path
    mean: Path
    var: Path
    mixw: Path
    tmat: Path
    ctl: Path
    cepdir: Path
    dictfn: Path
    lsnfn: Path | None = None
    accumdir: Path | None = None
    cepext: str = ".mfc"

    def _get_shell_cmd(self) -> list[str]:
        cmd = [
            "bw",
            "-moddeffn",
            str(self.mdef),
            "-meanfn",
            str(self.mean),
            "-varfn",
            str(self.var),
            "-mixwfn",
            str(self.mixw),
            "-tmatfn",
            str(self.tmat),
            "-ctlfn",
            str(self.ctl),
            "-cepdir",
            str(self.cepdir),
            "-cepext",
            self.cepext,
            "-dictfn",
            str(self.dictfn),
        ]
        if self.lsnfn:
            cmd.extend(["-lsnfn", str(self.lsnfn)])
        if self.accumdir:
            cmd.extend(["-accumdir", str(self.accumdir)])
        return cmd


@dataclass
class NormAction(PstrainAction):
    """Normalize BW accumulators."""

    accumdir: Path
    meanfn: Path
    varfn: Path
    mixwfn: Path
    tmatfn: Path

    def _get_shell_cmd(self) -> list[str]:
        return [
            "norm",
            "-accumdir",
            str(self.accumdir),
            "-meanfn",
            str(self.meanfn),
            "-varfn",
            str(self.varfn),
            "-mixwfn",
            str(self.mixwfn),
            "-tmatfn",
            str(self.tmatfn),
        ]


@dataclass
class SplitGaussiansAction(PstrainAction):
    """Split Gaussian components."""

    inmeanfn: Path
    invarfn: Path
    inmixwfn: Path
    outmeanfn: Path
    outvarfn: Path
    outmixwfn: Path

    def _get_shell_cmd(self) -> list[str]:
        return [
            "inc_comp",
            "-inmeanfn",
            str(self.inmeanfn),
            "-invarfn",
            str(self.invarfn),
            "-inmixwfn",
            str(self.inmixwfn),
            "-outmeanfn",
            str(self.outmeanfn),
            "-outvarfn",
            str(self.outvarfn),
            "-outmixwfn",
            str(self.outmixwfn),
        ]


@dataclass
class MakeQuestsAction(PstrainAction):
    """Generate phonetic questions for decision trees."""

    moddeffn: Path
    meanfn: Path
    varfn: Path
    mixwfn: Path
    questsfn: Path

    def _get_shell_cmd(self) -> list[str]:
        return [
            "make_quests",
            "-moddeffn",
            str(self.moddeffn),
            "-meanfn",
            str(self.meanfn),
            "-varfn",
            str(self.varfn),
            "-mixwfn",
            str(self.mixwfn),
            "-questsfn",
            str(self.questsfn),
        ]


@dataclass
class BuildTreeAction(PstrainAction):
    """Build decision tree for state tying."""

    moddeffn: Path
    meanfn: Path
    varfn: Path
    mixwfn: Path
    psetfn: Path
    treefn: Path
    phone: str
    state: int

    def _get_shell_cmd(self) -> list[str]:
        return [
            "bldtree",
            "-moddeffn",
            str(self.moddeffn),
            "-meanfn",
            str(self.meanfn),
            "-varfn",
            str(self.varfn),
            "-mixwfn",
            str(self.mixwfn),
            "-psetfn",
            str(self.psetfn),
            "-treefn",
            str(self.treefn),
            "-phone",
            self.phone,
            "-state",
            str(self.state),
        ]


@dataclass
class TieStatesAction(PstrainAction):
    """Tie states using decision trees."""

    imoddeffn: Path
    omoddeffn: Path
    treedir: Path
    psetfn: Path

    def _get_shell_cmd(self) -> list[str]:
        return [
            "tiestate",
            "-imoddeffn",
            str(self.imoddeffn),
            "-omoddeffn",
            str(self.omoddeffn),
            "-treedir",
            str(self.treedir),
            "-psetfn",
            str(self.psetfn),
        ]


@dataclass
class AggSegAction(PstrainAction):
    """Aggregate segmentation statistics."""

    segdir: Path
    moddeffn: Path
    ctlfn: Path

    def _get_shell_cmd(self) -> list[str]:
        return [
            "agg_seg",
            "-segdir",
            str(self.segdir),
            "-moddeffn",
            str(self.moddeffn),
            "-ctlfn",
            str(self.ctlfn),
        ]


# =============================================================================
# Command Context
# =============================================================================


@dataclass
class CommandContext:
    """Context passed to command execution.

    Provides unified methods that either emit shell or execute directly.
    """

    args: argparse.Namespace
    dry_run: bool = False
    verbose: bool = False
    json_output: bool = False
    json_indent: int = 2
    json_ascii: bool = False
    _header_emitted: bool = False

    def format_json(self, data: Any) -> str:
        """Format data as JSON using context settings."""
        indent = self.json_indent if self.json_indent > 0 else None
        return json.dumps(data, indent=indent, ensure_ascii=self.json_ascii)

    @property
    def project_dir(self) -> Path:
        """Get resolved project directory."""
        if hasattr(self.args, "project_dir") and self.args.project_dir:
            return Path(self.args.project_dir).resolve()
        return Path.cwd()

    @property
    def experiment(self) -> str:
        """Get experiment name."""
        if hasattr(self.args, "experiment") and self.args.experiment:
            return str(self.args.experiment)
        return "default"

    @property
    def config_name(self) -> str:
        """Get config name."""
        if hasattr(self.args, "config") and self.args.config:
            return str(self.args.config)
        return "baseline"

    def _emit_header(self) -> None:
        """Emit script header if not already done."""
        if self.dry_run and not self._header_emitted:
            print("#!/usr/bin/env bash")
            print("# Generated by: pstrain --dry-run")
            print("# Run this script to execute the commands")
            print("set -euo pipefail")
            print()
            self._header_emitted = True

    def _run_action(self, action: Action) -> None:
        """Run an action (emit or execute based on dry_run)."""
        if self.dry_run:
            self._emit_header()
            print(action.to_shell())
        else:
            action.execute()

    # =========================================================================
    # Unified action methods - emit shell OR execute
    # =========================================================================

    def mkdir(self, path: Path | str, parents: bool = True) -> None:
        """Create directory."""
        self._run_action(MkdirAction(Path(path), parents))

    def copy(self, src: Path | str, dst: Path | str) -> None:
        """Copy file."""
        self._run_action(CopyAction(Path(src), Path(dst)))

    def copy_tree(self, src: Path | str, dst: Path | str) -> None:
        """Copy directory tree."""
        self._run_action(CopyTreeAction(Path(src), Path(dst)))

    def symlink(self, src: Path | str, dst: Path | str) -> None:
        """Create symlink."""
        self._run_action(SymlinkAction(Path(src), Path(dst)))

    def remove(self, path: Path | str, recursive: bool = False, force: bool = False) -> None:
        """Remove file or directory."""
        self._run_action(RemoveAction(Path(path), recursive, force))

    def write_file(self, path: Path | str, content: str) -> None:
        """Write content to file."""
        self._run_action(WriteFileAction(Path(path), content))

    def run_cmd(self, program: str, *args: str, cwd: Path | None = None, **kwargs: Any) -> None:
        """Run a shell command."""
        self._run_action(ShellCommandAction(program, args, kwargs, cwd))

    def comment(self, text: str) -> None:
        """Add a comment (no-op when executing)."""
        self._run_action(CommentAction(text))

    def blank(self) -> None:
        """Add a blank line in dry-run output."""
        if self.dry_run:
            self._emit_header()
            print()

    # =========================================================================
    # pstrain methods - emit shell commands OR execute CFFI
    # =========================================================================

    def extract_features(
        self,
        input_file: Path,
        output_file: Path,
        samprate: int = 16000,
        nfilt: int = 25,
        ncep: int = 13,
    ) -> None:
        """Extract features from audio file."""
        self._run_action(
            FeatureExtractAction(
                input_file=Path(input_file),
                output_file=Path(output_file),
                samprate=samprate,
                nfilt=nfilt,
                ncep=ncep,
            )
        )

    def baum_welch(
        self,
        mdef: Path,
        mean: Path,
        var: Path,
        mixw: Path,
        tmat: Path,
        ctl: Path,
        cepdir: Path,
        dictfn: Path,
        lsnfn: Path | None = None,
        accumdir: Path | None = None,
        cepext: str = ".mfc",
    ) -> None:
        """Run Baum-Welch training."""
        self._run_action(
            BaumWelchAction(
                mdef=Path(mdef),
                mean=Path(mean),
                var=Path(var),
                mixw=Path(mixw),
                tmat=Path(tmat),
                ctl=Path(ctl),
                cepdir=Path(cepdir),
                dictfn=Path(dictfn),
                lsnfn=Path(lsnfn) if lsnfn else None,
                accumdir=Path(accumdir) if accumdir else None,
                cepext=cepext,
            )
        )

    def normalize(
        self,
        accumdir: Path,
        meanfn: Path,
        varfn: Path,
        mixwfn: Path,
        tmatfn: Path,
    ) -> None:
        """Normalize BW accumulators."""
        self._run_action(
            NormAction(
                accumdir=Path(accumdir),
                meanfn=Path(meanfn),
                varfn=Path(varfn),
                mixwfn=Path(mixwfn),
                tmatfn=Path(tmatfn),
            )
        )

    def split_gaussians(
        self,
        inmeanfn: Path,
        invarfn: Path,
        inmixwfn: Path,
        outmeanfn: Path,
        outvarfn: Path,
        outmixwfn: Path,
    ) -> None:
        """Split Gaussian components."""
        self._run_action(
            SplitGaussiansAction(
                inmeanfn=Path(inmeanfn),
                invarfn=Path(invarfn),
                inmixwfn=Path(inmixwfn),
                outmeanfn=Path(outmeanfn),
                outvarfn=Path(outvarfn),
                outmixwfn=Path(outmixwfn),
            )
        )

    def make_quests(
        self,
        moddeffn: Path,
        meanfn: Path,
        varfn: Path,
        mixwfn: Path,
        questsfn: Path,
    ) -> None:
        """Generate phonetic questions."""
        self._run_action(
            MakeQuestsAction(
                moddeffn=Path(moddeffn),
                meanfn=Path(meanfn),
                varfn=Path(varfn),
                mixwfn=Path(mixwfn),
                questsfn=Path(questsfn),
            )
        )

    def build_tree(
        self,
        moddeffn: Path,
        meanfn: Path,
        varfn: Path,
        mixwfn: Path,
        psetfn: Path,
        treefn: Path,
        phone: str,
        state: int,
    ) -> None:
        """Build decision tree."""
        self._run_action(
            BuildTreeAction(
                moddeffn=Path(moddeffn),
                meanfn=Path(meanfn),
                varfn=Path(varfn),
                mixwfn=Path(mixwfn),
                psetfn=Path(psetfn),
                treefn=Path(treefn),
                phone=phone,
                state=state,
            )
        )

    def tie_states(
        self,
        imoddeffn: Path,
        omoddeffn: Path,
        treedir: Path,
        psetfn: Path,
    ) -> None:
        """Tie states using decision trees."""
        self._run_action(
            TieStatesAction(
                imoddeffn=Path(imoddeffn),
                omoddeffn=Path(omoddeffn),
                treedir=Path(treedir),
                psetfn=Path(psetfn),
            )
        )

    def agg_seg(
        self,
        segdir: Path,
        moddeffn: Path,
        ctlfn: Path,
    ) -> None:
        """Aggregate segmentation statistics."""
        self._run_action(
            AggSegAction(
                segdir=Path(segdir),
                moddeffn=Path(moddeffn),
                ctlfn=Path(ctlfn),
            )
        )

    # =========================================================================
    # Convenience methods
    # =========================================================================

    def pstrain(self, *args: str, **kwargs: Any) -> None:
        """Run pstrain CLI command (for nested pstrain calls in dry-run)."""
        self.run_cmd("pstrain", *args, **kwargs)

    # =========================================================================
    # Logging (comments in dry-run, print otherwise)
    # =========================================================================

    def log(self, message: str) -> None:
        """Log a message."""
        if self.dry_run:
            self._emit_header()
            print(f"# {message}")
        else:
            print(message)

    def log_action(self, action: str, target: str) -> None:
        """Log an action being performed."""
        self.log(f"{action}: {target}")

    def log_comment(self, text: str) -> None:
        """Alias for comment."""
        self.comment(text)

    def emit_blank(self) -> None:
        """Alias for blank."""
        self.blank()

    def error(self, message: str) -> None:
        """Log an error message."""
        print(f"Error: {message}", file=sys.stderr)


# =============================================================================
# Command Result
# =============================================================================


@dataclass
class CommandResult:
    """Result of command execution."""

    success: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    exit_code: int = 0

    @classmethod
    def ok(cls, message: str = "", data: dict[str, Any] | None = None) -> Self:
        """Create successful result."""
        return cls(success=True, message=message, data=data or {}, exit_code=0)

    @classmethod
    def fail(cls, message: str, exit_code: int = 1) -> Self:
        """Create failure result."""
        return cls(success=False, message=message, exit_code=exit_code)


# =============================================================================
# Command base classes
# =============================================================================


class Command(ABC):
    """Base class for CLI commands."""

    name: str = ""
    help: str = ""
    description: str = ""

    needs_project_dir: bool = True
    needs_experiment: bool = False
    needs_config: bool = False
    supports_dry_run: bool = True
    supports_json_output: bool = False

    def register(self, subparsers: Any) -> argparse.ArgumentParser:
        """Register command with argument parser."""
        parser: argparse.ArgumentParser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.description or self.help,
        )

        if self.needs_project_dir:
            parser.add_argument(
                "--project-dir",
                type=str,
                help="Project directory (default: current directory)",
            )

        if self.needs_experiment:
            parser.add_argument(
                "--experiment",
                type=str,
                default="default",
                help="Experiment name (default: default)",
            )

        if self.needs_config:
            parser.add_argument(
                "--config",
                type=str,
                default="baseline",
                help="Model configuration name (default: baseline)",
            )

        if self.supports_dry_run:
            add_dry_run_argument(parser)

        if self.supports_json_output:
            add_json_argument(parser)

        self.add_arguments(parser)
        parser.set_defaults(command_instance=self)
        return parser

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command-specific arguments."""

    @abstractmethod
    def execute(self, ctx: CommandContext) -> CommandResult:
        """Execute the command."""

    def run(self, args: argparse.Namespace) -> int:
        """Run command with error handling."""
        from pstrain.lib.config.user import get_user_config

        # Load user config for defaults
        user_config = get_user_config()
        json_cfg = user_config.json_output

        # CLI args override config file defaults
        json_indent = getattr(args, "json_indent", None)
        json_ascii = getattr(args, "json_ascii", None)

        ctx = CommandContext(
            args=args,
            dry_run=getattr(args, "dry_run", False),
            verbose=getattr(args, "verbose", False),
            json_output=getattr(args, "json", False),
            json_indent=json_indent if json_indent is not None else json_cfg.indent,
            json_ascii=json_ascii if json_ascii is not None else json_cfg.ensure_ascii,
        )

        try:
            result = self.execute(ctx)

            if result.message:
                if result.success:
                    ctx.log(result.message)
                else:
                    ctx.error(result.message)

            return result.exit_code

        except FileNotFoundError as e:
            ctx.error(f"File not found: {e}")
            return 1
        except PermissionError as e:
            ctx.error(f"Permission denied: {e}")
            return 1
        except subprocess.CalledProcessError as e:
            ctx.error(f"Command failed: {e}")
            return e.returncode
        except Exception as e:
            ctx.error(str(e))
            return 1


class ProjectCommand(Command):
    """Command that operates on a project directory."""

    needs_project_dir = True
    needs_experiment = True
    needs_config = False

    def get_config(self, ctx: CommandContext) -> Any:
        """Load project configuration."""
        from pstrain.api import ConfigManager

        return ConfigManager.load_full_config(ctx.project_dir, ctx.experiment)


class ModelCommand(ProjectCommand):
    """Command that operates on models within a project."""

    needs_config = True
    default_model_type: str = "ci"

    def add_model_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add common model arguments."""
        parser.add_argument(
            "--model-type",
            type=str,
            default=self.default_model_type,
            help=f"Model type (default: {self.default_model_type}). Options: ci, cd",
        )

    def get_model(self, ctx: CommandContext) -> Any:
        """Create model instance from args."""
        from pstrain.api import create_model

        model_type = getattr(ctx.args, "model_type", self.default_model_type)
        config = ctx.config_name
        return create_model(model_type, config=config)


def execute_command(args: argparse.Namespace) -> int:
    """Execute command from parsed args (used by main CLI)."""
    if hasattr(args, "command_instance"):
        return int(args.command_instance.run(args))
    if hasattr(args, "func"):
        result = args.func(args)
        return int(result) if result is not None else 0
    return 0
