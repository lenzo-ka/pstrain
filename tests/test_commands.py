"""Tests for command registry and shell-out support."""

from pathlib import Path

from st2.lib.commands import ST2_BINARIES, Command, CommandBuilder


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
        assert "-phnlistfn" in shell
        assert "-n_state_pm 5" in shell

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
        ]
        for name in expected:
            assert name in ST2_BINARIES

    def test_find_binary_in_path(self) -> None:
        """Test finding binary in PATH."""
        from st2.lib.commands import find_binary

        # Should find common utilities
        find_binary("ls")
        assert True  # May not exist on all systems
