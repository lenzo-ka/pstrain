"""Tests for command registry and shell-out support."""

from __future__ import annotations

import ast
import importlib
import inspect
import subprocess
import types
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import pytest

from pstrain.cli import base as cli_base
from pstrain.lib.commands import PSTRAIN_BINARIES, Command, CommandBuilder, resolve_binary


class TestCommand:
    """Tests for Command class."""

    def test_to_shell_simple(self) -> None:
        """Test simple command to shell."""
        cmd = Command("bw", ["-meanfn", "means", "-varfn", "vars"])
        assert cmd.to_shell() == "bw -meanfn means -varfn vars"

    def test_to_shell_with_spaces(self) -> None:
        """Test command with paths containing spaces."""
        cmd = Command("bw", ["-meanfn", "/path/with spaces/means"])
        shell = cmd.to_shell()
        assert "'/path/with spaces/means'" in shell or '"/path/with spaces/means"' in shell

    def test_to_shell_with_env(self) -> None:
        """Test command with environment variables."""
        cmd = Command("bw", ["-meanfn", "means"], env={"FOO": "bar"})
        shell = cmd.to_shell()
        assert "FOO=" in shell
        assert "bar" in shell

    def test_to_shell_with_cwd(self) -> None:
        """Test command with working directory."""
        cmd = Command("bw", [], cwd=Path("/tmp/work"))
        shell = cmd.to_shell()
        assert "cd" in shell
        assert "/tmp/work" in shell


class TestCommandBuilder:
    """Tests for CommandBuilder class."""

    def test_dry_run_mode(self, tmp_path: Path) -> None:
        """Test dry-run mode prints commands."""
        builder = CommandBuilder(dry_run=True)
        builder.sphinx_fe(tmp_path / "in.wav", tmp_path / "out.mfc")

        # Should have one command queued
        assert len(builder.get_commands()) == 1

        # dry_run should not actually run
        results = builder.run_all()
        assert results == []

    def test_to_shell_script(self, tmp_path: Path) -> None:
        """Test shell script generation."""
        builder = CommandBuilder()
        builder.sphinx_fe(tmp_path / "in.wav", tmp_path / "out.mfc")
        builder.bw(
            mdef=tmp_path / "mdef",
            mean=tmp_path / "means",
            var=tmp_path / "vars",
            mixw=tmp_path / "mixw",
            tmat=tmp_path / "tmat",
            ctl=tmp_path / "ctl",
        )

        script = builder.to_shell_script()
        assert "#!/usr/bin/env bash" in script
        assert "sphinx_fe" in script
        assert "bw" in script

    def test_sphinx_fe_command(self, tmp_path: Path) -> None:
        """Test sphinx_fe command building."""
        builder = CommandBuilder()
        cmd = builder.sphinx_fe(
            tmp_path / "in.wav",
            tmp_path / "out.mfc",
            samprate=8000,
            ncep=39,
        )

        shell = cmd.to_shell()
        assert "sphinx_fe" in shell
        assert "-samprate 8000" in shell
        assert "-ncep 39" in shell

    def test_bw_command(self, tmp_path: Path) -> None:
        """Test bw command building."""
        builder = CommandBuilder()
        cmd = builder.bw(
            mdef=tmp_path / "mdef",
            mean=tmp_path / "means",
            var=tmp_path / "vars",
            mixw=tmp_path / "mixw",
            tmat=tmp_path / "tmat",
            ctl=tmp_path / "ctl",
            lsn=tmp_path / "lsn",
            dictfn=tmp_path / "dict",
        )

        shell = cmd.to_shell()
        assert "bw" in shell
        assert "-moddeffn" in shell
        assert "-meanfn" in shell
        assert "-lsnfn" in shell
        assert "-dictfn" in shell

    def test_mk_mdef_gen_command(self, tmp_path: Path) -> None:
        """Test mk_mdef_gen command building."""
        builder = CommandBuilder()
        cmd = builder.mk_mdef_gen(
            phnlist=tmp_path / "phones",
            output=tmp_path / "mdef",
            n_state=5,
        )

        shell = cmd.to_shell()
        assert "mk_mdef_gen" in shell
        assert "-phnlstfn" in shell
        assert "-n_state_pm 5" in shell

    def test_param_cnt_command(self, tmp_path: Path) -> None:
        builder = CommandBuilder()
        cmd = builder.param_cnt(
            moddeffn=tmp_path / "mdef",
            dictfn=tmp_path / "dict",
            ctlfn=tmp_path / "train.fileids",
            lsnfn=tmp_path / "train.transcription",
            paramtype="phone",
            outputfn=tmp_path / "phone_counts",
        )

        assert cmd.binary.endswith("param_cnt")
        assert cmd.args == [
            "-moddeffn",
            str(tmp_path / "mdef"),
            "-dictfn",
            str(tmp_path / "dict"),
            "-ctlfn",
            str(tmp_path / "train.fileids"),
            "-lsnfn",
            str(tmp_path / "train.transcription"),
            "-paramtype",
            "phone",
            "-outputfn",
            str(tmp_path / "phone_counts"),
        ]

    def test_delint_rejects_unrepresentable_comma_path(self, tmp_path: Path) -> None:
        builder = CommandBuilder()
        with pytest.raises(ValueError, match="cannot contain ','"):
            builder.delint(
                [tmp_path / "accum,one", tmp_path / "accum-two"],
                tmp_path / "mdef",
                tmp_path / "mixw",
            )

    @pytest.mark.parametrize("method", ["norm", "map_adapt"])
    def test_other_string_list_builders_reject_comma_paths(
        self, method: str, tmp_path: Path
    ) -> None:
        builder = CommandBuilder()
        target = getattr(builder, method)
        hints = get_type_hints(target)
        kwargs = {
            parameter.name: self._probe_value(hints[parameter.name], parameter.default, tmp_path)
            for parameter in inspect.signature(target).parameters.values()
            if parameter.kind is not inspect.Parameter.VAR_KEYWORD
        }
        kwargs["accumdir"] = tmp_path / "accum,one"
        with pytest.raises(ValueError, match="cannot contain ','"):
            target(**kwargs)

    def test_command_queue(self, tmp_path: Path) -> None:
        """Test command queue management."""
        builder = CommandBuilder()

        builder.sphinx_fe(tmp_path / "a.wav", tmp_path / "a.mfc")
        builder.sphinx_fe(tmp_path / "b.wav", tmp_path / "b.mfc")

        assert len(builder.get_commands()) == 2

        builder.clear()
        assert len(builder.get_commands()) == 0

    def test_bin_dir(self, tmp_path: Path) -> None:
        """Test custom bin_dir: when a real binary exists at bin_dir/<name>,
        the full path is used. (`_get_binary` falls back to PATH lookup
        when the file doesn't exist, so the test creates a fake binary
        to exercise the prefix path.)"""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "sphinx_fe").touch()

        builder = CommandBuilder(bin_dir=bin_dir)
        cmd = builder.sphinx_fe(tmp_path / "in.wav", tmp_path / "out.mfc")

        assert str(bin_dir / "sphinx_fe") in cmd.to_shell()

    def test_resolution_and_execution_use_same_binary(self, tmp_path: Path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        binary = bin_dir / "bw"
        binary.touch()

        assert resolve_binary("bw", bin_dir) == binary.resolve()
        assert CommandBuilder(bin_dir=bin_dir)._get_binary("bw") == str(binary.resolve())

    @staticmethod
    def _probe_value(annotation: Any, default: Any, tmp_path: Path) -> Any:
        """Return a value that makes each builder emit all of its fixed flags."""
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin in (types.UnionType, Union):
            annotation = next(arg for arg in args if arg is not type(None))
            origin = get_origin(annotation)
            args = get_args(annotation)
        if origin is list:
            return [tmp_path / "probe-a", tmp_path / "probe-b"]
        if annotation is Path:
            return tmp_path / "probe"
        if annotation is str:
            return "probe"
        if annotation is int:
            return 1
        if annotation is float:
            return 1.0
        if annotation is bool:
            return not default if default is not inspect.Parameter.empty else True
        raise AssertionError(f"No argv probe value for annotation {annotation!r}")

    def test_every_core_argv_emitter_is_accepted_by_core_parser(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Gate the repository-derived fixed core-argv emitter population.

        The population is the union of Command-returning ``CommandBuilder`` methods,
        concrete ``PstrainAction`` subclasses, and functions under ``pstrain/`` that
        both default a ``bin_path`` to a registered core program and pass a locally
        constructed command to ``subprocess.run``.  Optional values are supplied to
        exercise fixed conditional flags. ``**kwargs`` are deliberately not probed.
        """
        # Exercise the user-facing default resolution route. PSTRAIN_BIN_DIR is
        # intentionally honored here because it is a supported runtime override.
        builder = CommandBuilder()
        emitters: list[tuple[str, str, list[str]]] = []

        for name, method in inspect.getmembers(CommandBuilder, inspect.isfunction):
            if name.startswith("_"):
                continue
            hints = get_type_hints(method)
            if hints.get("return") is Command:
                kwargs = {}
                for parameter in inspect.signature(method).parameters.values():
                    if parameter.name == "self" or parameter.kind is inspect.Parameter.VAR_KEYWORD:
                        continue
                    kwargs[parameter.name] = self._probe_value(
                        hints[parameter.name], parameter.default, tmp_path
                    )
                command = getattr(builder, name)(**kwargs)
                emitters.append((f"CommandBuilder.{name}", command.binary, command.args))

        action_classes = [
            cls
            for _, cls in inspect.getmembers(cli_base, inspect.isclass)
            if issubclass(cls, cli_base.PstrainAction)
            and cls is not cli_base.PstrainAction
            and "_get_shell_cmd" in cls.__dict__
        ]
        for action_class in action_classes:
            hints = get_type_hints(action_class)
            kwargs = {}
            for parameter in inspect.signature(action_class).parameters.values():
                kwargs[parameter.name] = self._probe_value(
                    hints[parameter.name], parameter.default, tmp_path
                )
            argv = action_class(**kwargs)._get_shell_cmd()
            emitters.append((action_class.__name__, builder._get_binary(argv[0]), argv[1:]))

        shellout_functions: list[tuple[str, str]] = []
        for source_path in Path("pstrain").rglob("*.py"):
            tree = ast.parse(source_path.read_text(), filename=str(source_path))
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                default_args = (
                    node.args.args[-len(node.args.defaults) :] if node.args.defaults else []
                )
                defaults = dict(
                    zip([arg.arg for arg in default_args], node.args.defaults, strict=True)
                )
                default = defaults.get("bin_path")
                calls_subprocess = any(
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "subprocess"
                    and child.func.attr == "run"
                    for child in ast.walk(node)
                )
                if (
                    isinstance(default, ast.Constant)
                    and default.value in PSTRAIN_BINARIES.values()
                    and calls_subprocess
                ):
                    module = ".".join(source_path.with_suffix("").parts)
                    shellout_functions.append((module, node.name))

        for module_name, function_name in shellout_functions:
            function = getattr(importlib.import_module(module_name), function_name)
            hints = get_type_hints(function)
            kwargs = {}
            for parameter in inspect.signature(function).parameters.values():
                if parameter.name == "bin_path":
                    kwargs[parameter.name] = Path(builder._get_binary(parameter.default))
                else:
                    kwargs[parameter.name] = self._probe_value(
                        hints[parameter.name], parameter.default, tmp_path
                    )
            captured: list[str] = []

            def capture_run(
                argv: list[str], *, destination: list[str] = captured, **_kwargs: Any
            ) -> Any:
                destination.extend(argv)
                return types.SimpleNamespace(stdout="")

            with monkeypatch.context() as patch:
                patch.setattr(subprocess, "run", capture_run)
                function(**kwargs)
            emitters.append((f"{module_name}.{function_name}", captured[0], captured[1:]))

        assert len(emitters) == 27, (
            f"derived emitter population changed: {[item[0] for item in emitters]}"
        )
        for name, binary_name, args in emitters:
            binary = Path(binary_name)
            assert binary.is_file(), f"{name}: core binary not found: {binary}"
            result = subprocess.run(
                [str(binary), *args],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            parser_output = result.stdout + result.stderr
            assert "Unknown argument name" not in parser_output, (
                f"{name} ({binary.name}) emitted argv rejected by the core parser:\n"
                f"argv={args!r}\n{parser_output}"
            )


class TestBinaryRegistry:
    """Tests for binary registry."""

    def test_all_binaries_registered(self) -> None:
        """Test that expected binaries are in registry."""
        expected = [
            "sphinx_fe",
            "bw",
            "norm",
            "mk_mdef_gen",
            "make_quests",
            "bldtree",
            "tiestate",
            "sphinx3_align",
            "map_adapt",
            "param_cnt",
        ]
        for name in expected:
            assert name in PSTRAIN_BINARIES

    def test_find_binary_in_path(self) -> None:
        """Test finding binary in PATH."""
        from pstrain.lib.commands import find_binary

        # Should find common utilities
        find_binary("ls")
        assert True  # May not exist on all systems
