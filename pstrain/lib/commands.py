"""pstrain command registry and shell-out support.

Provides shell-out capabilities for pstrain C programs,
with dry-run support for debugging and CI.

Design principle: CFFI is preferred, shell-out is fallback for debugging.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pstrain.lib.paths import get_bin_dir

logger = logging.getLogger(__name__)


def _binary_names(name: str) -> tuple[str, ...]:
    """Return the native executable filenames accepted on this platform."""
    if sys.platform == "win32" and not name.lower().endswith(".exe"):
        return (f"{name}.exe", name)
    return (name,)


def _reject_comma_path(path: Path, flag: str) -> None:
    """Reject paths unrepresentable by the core ARG_STRING_LIST parser."""
    if "," in str(path):
        raise ValueError(
            f"{flag} paths cannot contain ',' because the core string-list parser "
            f"has no escaping: {path!r}"
        )


__all__ = [
    "Command",
    "CommandBuilder",
    "PSTRAIN_BINARIES",
    "find_binary",
    "resolve_binary",
]


@dataclass
class Command:
    """A shell command with arguments."""

    binary: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path | None = None

    def to_shell(self) -> str:
        """Format as shell command string."""
        parts = [self.binary, *self.args]
        cmd = shlex.join(parts)
        if self.env:
            env_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env.items())
            cmd = f"{env_str} {cmd}"
        if self.cwd:
            cmd = f"cd {shlex.quote(str(self.cwd))} && {cmd}"
        return cmd

    def run(self, check: bool = True) -> subprocess.CompletedProcess[str]:
        """Execute the command."""
        import os

        full_env = os.environ.copy()
        full_env.update(self.env)

        logger.info("Running: %s", self.to_shell())
        return subprocess.run(
            [self.binary, *self.args],
            cwd=self.cwd,
            env=full_env,
            capture_output=True,
            text=True,
            check=check,
        )


# Registry of pstrain C programs
# Maps operation name -> binary name
PSTRAIN_BINARIES = {
    # Feature extraction
    "sphinx_fe": "sphinx_fe",
    # Flat initialization
    "mk_flat": "mk_flat",
    "init_gau": "init_gau",
    # BW training
    "bw": "bw",
    "norm": "norm",
    # Gaussian splitting
    "inc_comp": "inc_comp",
    "kmeans_init": "kmeans_init",
    # Mdef generation
    "mk_mdef_gen": "mk_mdef_gen",
    # Decision trees
    "make_quests": "make_quests",
    "bldtree": "bldtree",
    "prunetree": "prunetree",
    "tiestate": "tiestate",
    # Model operations
    "mk_ts2cb": "mk_ts2cb",
    "agg_seg": "agg_seg",
    "param_cnt": "param_cnt",
    # Adaptation
    "map_adapt": "map_adapt",
    "mllr_solve": "mllr_solve",
    "mllr_transform": "mllr_transform",
    # Utilities
    "sphinx_cepview": "sphinx_cepview",
    "printp": "printp",
    "delint": "delint",
    "kdtree": "kdtree",
    # Alignment
    "sphinx3_align": "sphinx3_align",
}


def find_binary(name: str, search_paths: list[Path] | None = None) -> Path | None:
    """Find a pstrain binary.

    Args:
        name: Binary name.
        search_paths: Additional paths to search.

    Returns:
        Path to binary if found, None otherwise.
    """
    import shutil

    # Check if it's in PATH
    which = shutil.which(name)
    if which:
        return Path(which)

    # Check search paths
    if search_paths:
        for p in search_paths:
            for filename in _binary_names(name):
                candidate = p / filename
                if candidate.exists() and candidate.is_file():
                    return candidate

    return None


def resolve_binary(name: str, bin_dir: Path | None = None) -> Path | None:
    """Resolve a program exactly as :class:`CommandBuilder` will execute it."""
    if bin_dir is None:
        bin_dir = get_bin_dir()
    if bin_dir is not None:
        for filename in _binary_names(name):
            candidate = bin_dir / filename
            if candidate.is_file():
                return candidate.resolve()
    return find_binary(name)


@dataclass
class CommandBuilder:
    """Builder for pstrain C program commands.

    Args:
        bin_dir: Directory containing pstrain binaries. If None, auto-discovers
            using get_bin_dir() which checks PSTRAIN_BIN_DIR env, build/, libexec/.
        dry_run: If True, commands are printed but not executed.
    """

    bin_dir: Path | None = field(default_factory=lambda: get_bin_dir())
    dry_run: bool = False
    _commands: list[Command] = field(default_factory=list)

    def _get_binary(self, name: str) -> str:
        """Get full path to binary."""
        resolved = resolve_binary(name, self.bin_dir)
        if resolved is not None:
            return str(resolved)
        # Fall back to PATH lookup
        return name

    def add(self, cmd: Command) -> None:
        """Add a command to the queue."""
        self._commands.append(cmd)

    def sphinx_fe(
        self,
        input_file: Path,
        output_file: Path,
        samprate: int = 16000,
        nfilt: int = 25,
        nfft: int = 512,
        lowerf: float = 130.0,
        upperf: float = 6800.0,
        ncep: int = 13,
        remove_dc: bool = True,
        dither: bool = True,
        **kwargs: Any,
    ) -> Command:
        """Build sphinx_fe command."""
        args = [
            "-i",
            str(input_file),
            "-o",
            str(output_file),
            "-samprate",
            str(samprate),
            "-nfilt",
            str(nfilt),
            "-nfft",
            str(nfft),
            "-lowerf",
            str(lowerf),
            "-upperf",
            str(upperf),
            "-ncep",
            str(ncep),
            "-remove_dc",
            "yes" if remove_dc else "no",
            "-dither",
            "yes" if dither else "no",
        ]
        for k, v in kwargs.items():
            args.extend([f"-{k}", str(v)])

        cmd = Command(self._get_binary("sphinx_fe"), args)
        self.add(cmd)
        return cmd

    def bw(
        self,
        mdef: Path,
        mean: Path,
        var: Path,
        mixw: Path,
        tmat: Path,
        ctl: Path,
        lsn: Path | None = None,
        dictfn: Path | None = None,
        fdictfn: Path | None = None,
        cepdir: Path | None = None,
        cepext: str = ".mfc",
        feat: str = "1s_c_d_dd",
        ceplen: int = 13,
        accumdir: Path | None = None,
        **kwargs: Any,
    ) -> Command:
        """Build bw command."""
        args = [
            "-moddeffn",
            str(mdef),
            "-meanfn",
            str(mean),
            "-varfn",
            str(var),
            "-mixwfn",
            str(mixw),
            "-tmatfn",
            str(tmat),
            "-ctlfn",
            str(ctl),
            "-feat",
            feat,
            "-ceplen",
            str(ceplen),
        ]
        if lsn:
            args.extend(["-lsnfn", str(lsn)])
        if dictfn:
            args.extend(["-dictfn", str(dictfn)])
        if fdictfn:
            args.extend(["-fdictfn", str(fdictfn)])
        if cepdir:
            args.extend(["-cepdir", str(cepdir)])
        args.extend(["-cepext", cepext])
        if accumdir:
            args.extend(["-accumdir", str(accumdir)])
        for k, v in kwargs.items():
            args.extend([f"-{k}", str(v)])

        cmd = Command(self._get_binary("bw"), args)
        self.add(cmd)
        return cmd

    def norm(
        self,
        accumdir: Path,
        meanfn: Path,
        varfn: Path,
        mixwfn: Path,
        tmatfn: Path,
        **kwargs: Any,
    ) -> Command:
        """Build norm command."""
        _reject_comma_path(accumdir, "norm -accumdir")
        args = [
            "-accumdir",
            str(accumdir),
            "-meanfn",
            str(meanfn),
            "-varfn",
            str(varfn),
            "-mixwfn",
            str(mixwfn),
            "-tmatfn",
            str(tmatfn),
        ]
        for k, v in kwargs.items():
            args.extend([f"-{k}", str(v)])

        cmd = Command(self._get_binary("norm"), args)
        self.add(cmd)
        return cmd

    def mk_mdef_gen(
        self,
        phnlist: Path,
        output: Path,
        dictfn: Path | None = None,
        fdictfn: Path | None = None,
        n_state: int = 3,
        **kwargs: Any,
    ) -> Command:
        """Build mk_mdef_gen command."""
        args = [
            "-phnlstfn",
            str(phnlist),
            "-ocimdef",
            str(output),
            "-n_state_pm",
            str(n_state),
        ]
        if dictfn:
            args.extend(["-dictfn", str(dictfn)])
        if fdictfn:
            args.extend(["-fdictfn", str(fdictfn)])
        for k, v in kwargs.items():
            args.extend([f"-{k}", str(v)])

        cmd = Command(self._get_binary("mk_mdef_gen"), args)
        self.add(cmd)
        return cmd

    def make_quests(
        self,
        moddeffn: Path,
        meanfn: Path,
        varfn: Path,
        mixwfn: Path,
        questsfn: Path,
        **kwargs: Any,
    ) -> Command:
        """Build make_quests command."""
        args = [
            "-moddeffn",
            str(moddeffn),
            "-meanfn",
            str(meanfn),
            "-varfn",
            str(varfn),
            "-mixwfn",
            str(mixwfn),
            # The make_quests binary's flag is -questfn (singular); -questsfn
            # is rejected as an unknown argument.
            "-questfn",
            str(questsfn),
        ]
        for k, v in kwargs.items():
            args.extend([f"-{k}", str(v)])

        cmd = Command(self._get_binary("make_quests"), args)
        self.add(cmd)
        return cmd

    def bldtree(
        self,
        moddeffn: Path,
        meanfn: Path,
        varfn: Path,
        mixwfn: Path,
        psetfn: Path,
        treefn: Path,
        phone: str,
        state: int,
        **kwargs: Any,
    ) -> Command:
        """Build bldtree command."""
        args = [
            "-moddeffn",
            str(moddeffn),
            "-meanfn",
            str(meanfn),
            "-varfn",
            str(varfn),
            "-mixwfn",
            str(mixwfn),
            "-psetfn",
            str(psetfn),
            "-treefn",
            str(treefn),
            "-phone",
            phone,
            "-state",
            str(state),
        ]
        for k, v in kwargs.items():
            args.extend([f"-{k}", str(v)])

        cmd = Command(self._get_binary("bldtree"), args)
        self.add(cmd)
        return cmd

    def tiestate(
        self,
        imoddeffn: Path,
        omoddeffn: Path,
        treedir: Path,
        psetfn: Path,
        **kwargs: Any,
    ) -> Command:
        """Build tiestate command."""
        args = [
            "-imoddeffn",
            str(imoddeffn),
            "-omoddeffn",
            str(omoddeffn),
            "-treedir",
            str(treedir),
            "-psetfn",
            str(psetfn),
        ]
        for k, v in kwargs.items():
            args.extend([f"-{k}", str(v)])

        cmd = Command(self._get_binary("tiestate"), args)
        self.add(cmd)
        return cmd

    def prunetree(
        self,
        itreefn: Path,
        otreefn: Path,
        psetfn: Path,
        **kwargs: Any,
    ) -> Command:
        """Build prunetree command."""
        args = [
            "-itreedir",
            str(itreefn),
            "-otreedir",
            str(otreefn),
            "-psetfn",
            str(psetfn),
        ]
        for k, v in kwargs.items():
            args.extend([f"-{k}", str(v)])

        cmd = Command(self._get_binary("prunetree"), args)
        self.add(cmd)
        return cmd

    def inc_comp(
        self,
        inmeanfn: Path,
        invarfn: Path,
        inmixwfn: Path,
        outmeanfn: Path,
        outvarfn: Path,
        outmixwfn: Path,
        **kwargs: Any,
    ) -> Command:
        """Build inc_comp command."""
        args = [
            "-inmeanfn",
            str(inmeanfn),
            "-invarfn",
            str(invarfn),
            "-inmixwfn",
            str(inmixwfn),
            "-outmeanfn",
            str(outmeanfn),
            "-outvarfn",
            str(outvarfn),
            "-outmixwfn",
            str(outmixwfn),
        ]
        for k, v in kwargs.items():
            args.extend([f"-{k}", str(v)])

        cmd = Command(self._get_binary("inc_comp"), args)
        self.add(cmd)
        return cmd

    def agg_seg(
        self,
        segdir: Path,
        moddeffn: Path,
        ctlfn: Path,
        **kwargs: Any,
    ) -> Command:
        """Build agg_seg command."""
        args = [
            "-segdir",
            str(segdir),
            "-moddeffn",
            str(moddeffn),
            "-ctlfn",
            str(ctlfn),
        ]
        for k, v in kwargs.items():
            args.extend([f"-{k}", str(v)])

        cmd = Command(self._get_binary("agg_seg"), args)
        self.add(cmd)
        return cmd

    def param_cnt(
        self,
        moddeffn: Path,
        dictfn: Path,
        ctlfn: Path,
        lsnfn: Path,
        paramtype: str = "state",
        **kwargs: Any,
    ) -> Command:
        """Build a core ``param_cnt`` command."""
        args = [
            "-moddeffn",
            str(moddeffn),
            "-dictfn",
            str(dictfn),
            "-ctlfn",
            str(ctlfn),
            "-lsnfn",
            str(lsnfn),
            "-paramtype",
            paramtype,
        ]
        for key, value in kwargs.items():
            args.extend([f"-{key}", str(value)])

        cmd = Command(self._get_binary("param_cnt"), args)
        self.add(cmd)
        return cmd

    def sphinx3_align(
        self,
        mdef: Path,
        mean: Path,
        var: Path,
        mixw: Path,
        tmat: Path,
        ctl: Path,
        insent: Path,
        dictfn: Path,
        stsegdir: Path | None = None,
        phsegdir: Path | None = None,
        wdsegdir: Path | None = None,
        **kwargs: Any,
    ) -> Command:
        """Build sphinx3_align command."""
        args = [
            "-mdef",
            str(mdef),
            "-mean",
            str(mean),
            "-var",
            str(var),
            "-mixw",
            str(mixw),
            "-tmat",
            str(tmat),
            "-ctl",
            str(ctl),
            "-insent",
            str(insent),
            "-dict",
            str(dictfn),
        ]
        if stsegdir:
            args.extend(["-stsegdir", str(stsegdir)])
        if phsegdir:
            args.extend(["-phsegdir", str(phsegdir)])
        if wdsegdir:
            args.extend(["-wdsegdir", str(wdsegdir)])
        for k, v in kwargs.items():
            args.extend([f"-{k}", str(v)])

        cmd = Command(self._get_binary("sphinx3_align"), args)
        self.add(cmd)
        return cmd

    def map_adapt(
        self,
        meanfn: Path,
        varfn: Path,
        mixwfn: Path,
        accumdir: Path,
        mapmeanfn: Path,
        mapvarfn: Path | None = None,
        mapmixwfn: Path | None = None,
        tau: float = 10.0,
        **kwargs: Any,
    ) -> Command:
        """Build map_adapt command."""
        _reject_comma_path(accumdir, "map_adapt -accumdir")
        args = [
            "-meanfn",
            str(meanfn),
            "-varfn",
            str(varfn),
            "-mixwfn",
            str(mixwfn),
            "-accumdir",
            str(accumdir),
            "-mapmeanfn",
            str(mapmeanfn),
            "-tau",
            str(tau),
        ]
        if mapvarfn:
            args.extend(["-mapvarfn", str(mapvarfn)])
        if mapmixwfn:
            args.extend(["-mapmixwfn", str(mapmixwfn)])
        for k, v in kwargs.items():
            args.extend([f"-{k}", str(v)])

        cmd = Command(self._get_binary("map_adapt"), args)
        self.add(cmd)
        return cmd

    def delint(
        self,
        accumdirs: list[Path],
        moddeffn: Path,
        mixwfn: Path,
        **kwargs: Any,
    ) -> Command:
        """Build delint command.

        The core ``ARG_STRING_LIST`` syntax uses commas as unescapable
        separators, so accumulator paths containing commas cannot be represented.
        """
        for path in accumdirs:
            _reject_comma_path(path, "delint -accumdirs")
        args = [
            "-moddeffn",
            str(moddeffn),
            "-mixwfn",
            str(mixwfn),
        ]
        args.extend(["-accumdirs", ",".join(str(d) for d in accumdirs)])
        for k, v in kwargs.items():
            args.extend([f"-{k}", str(v)])

        cmd = Command(self._get_binary("delint"), args)
        self.add(cmd)
        return cmd

    def kdtree(
        self,
        meanfn: Path,
        varfn: Path,
        outfn: Path,
        depth: int = 4,
        threshold: float = 0.0,
        **kwargs: Any,
    ) -> Command:
        """Build kdtree command."""
        args = [
            "-meanfn",
            str(meanfn),
            "-varfn",
            str(varfn),
            "-outfn",
            str(outfn),
            "-depth",
            str(depth),
            "-threshold",
            str(threshold),
        ]
        for k, v in kwargs.items():
            args.extend([f"-{k}", str(v)])

        cmd = Command(self._get_binary("kdtree"), args)
        self.add(cmd)
        return cmd

    def sphinx_cepview(
        self,
        input_file: Path,
        n_coeff: int = 13,
        display_cols: int = 10,
        start_frame: int = 0,
        end_frame: int | None = None,
        **kwargs: Any,
    ) -> Command:
        """Build sphinx_cepview command."""
        args = [
            "-f",
            str(input_file),
            "-i",
            str(n_coeff),
            "-d",
            str(display_cols),
            "-b",
            str(start_frame),
        ]
        if end_frame is not None:
            args.extend(["-e", str(end_frame)])
        for k, v in kwargs.items():
            args.extend([f"-{k}", str(v)])

        cmd = Command(self._get_binary("sphinx_cepview"), args)
        self.add(cmd)
        return cmd

    def printp(
        self,
        mixwfn: Path | None = None,
        tmatfn: Path | None = None,
        gaufn: Path | None = None,
        ts2cbfn: Path | None = None,
        sigfig: int = 6,
        **kwargs: Any,
    ) -> Command:
        """Build printp command."""
        args = ["-sigfig", str(sigfig)]
        if mixwfn:
            args.extend(["-mixwfn", str(mixwfn)])
        if tmatfn:
            args.extend(["-tmatfn", str(tmatfn)])
        if gaufn:
            args.extend(["-gaufn", str(gaufn)])
        if ts2cbfn:
            args.extend(["-ts2cbfn", str(ts2cbfn)])
        for k, v in kwargs.items():
            args.extend([f"-{k}", str(v)])

        cmd = Command(self._get_binary("printp"), args)
        self.add(cmd)
        return cmd

    def run_all(self) -> list[subprocess.CompletedProcess[str]]:
        """Run all queued commands.

        Returns:
            List of completed processes.

        Raises:
            subprocess.CalledProcessError: If any command fails.
        """
        if self.dry_run:
            for cmd in self._commands:
                print(cmd.to_shell())
            return []

        results = []
        for cmd in self._commands:
            results.append(cmd.run())
        return results

    def get_commands(self) -> list[Command]:
        """Get all queued commands."""
        return list(self._commands)

    def clear(self) -> None:
        """Clear the command queue."""
        self._commands.clear()

    def to_shell_script(self, shebang: bool = True) -> str:
        """Generate a shell script from queued commands.

        Args:
            shebang: Whether to include shebang line.

        Returns:
            Shell script as string.
        """
        lines = []
        if shebang:
            lines.append("#!/usr/bin/env bash")
            lines.append("set -euo pipefail")
            lines.append("")
        for cmd in self._commands:
            lines.append(cmd.to_shell())
        return "\n".join(lines)
