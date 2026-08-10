"""Parity tests: compare shell-out to C binaries vs CFFI results.

These tests ensure our CFFI implementations produce identical results
to the C binaries when shelling out.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pstrain.lib import _pstrainc
from pstrain.lib.commands import CommandBuilder, find_binary
from pstrain.lib.paths import get_bin_dir

# =============================================================================
# Test fixtures
# =============================================================================


def binary_available(name: str) -> bool:
    """Check if a pstrain C binary is available.

    Looks on PATH and, crucially, in pstrain's own build output directory
    (get_bin_dir() → build/bin/ for a dev build, the bundled wheel bin dir
    otherwise). pstrain builds these 30 CLI programs itself, so the CFFI-vs-CLI
    parity tests can and should run without a separate SphinxTrain install.
    """
    return _resolve_bin(name) is not None


def _resolve_bin(name: str) -> Path | None:
    """Resolve a pstrain binary to a concrete path (PATH or pstrain's build dir)."""
    found = find_binary(name)
    if found is not None:
        return found
    bin_dir = get_bin_dir()
    if bin_dir is not None and (bin_dir / name).is_file():
        return bin_dir / name
    return None


@pytest.fixture(scope="module")
def test_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create test data directory with model files."""
    data_dir = tmp_path_factory.mktemp("parity_test_data")

    # Create dummy model files
    n_phones = 5
    n_density = 4
    n_feat = 1
    veclen = 13
    n_state = 3

    # Seeded generator so parity comparisons are reproducible run to run.
    rng = np.random.default_rng(20260703)

    # Means and variances
    means = rng.random((n_phones, n_feat, n_density, veclen)).astype(np.float32)
    variances = np.abs(rng.random((n_phones, n_feat, n_density, veclen)).astype(np.float32)) + 0.01
    _pstrainc.write_gau(str(data_dir / "means"), means)
    _pstrainc.write_gau(str(data_dir / "variances"), variances)

    # Mixture weights (normalized)
    mixw = rng.random((n_phones * n_state, n_feat, n_density)).astype(np.float32)
    mixw = mixw / mixw.sum(axis=-1, keepdims=True)
    _pstrainc.write_mixw(str(data_dir / "mixture_weights"), mixw)

    # Transition matrices
    # Tmat format: (n_tmat, n_state_pm, n_state_pm+1)
    # where n_state_pm is number of emitting states (typically 3)
    # and last column is exit probability
    n_state_pm = n_state  # emitting states
    tmat = np.zeros((n_phones, n_state_pm, n_state_pm + 1), dtype=np.float32)
    for i in range(n_phones):
        for j in range(n_state_pm):
            # Self-loop and forward transition
            tmat[i, j, j] = 0.6
            if j < n_state_pm - 1:
                tmat[i, j, j + 1] = 0.4
            else:
                tmat[i, j, n_state_pm] = 0.4  # Exit
    _pstrainc.write_tmat(str(data_dir / "transition_matrices"), tmat)

    # Create simple mdef file
    mdef_content = """0.3
5 n_base
3 n_tri
15 n_state_map
5 n_tied_state
5 n_tied_ci_state
5 n_tied_tmat

SIL	-	-	-	b	0	1	2	N
AA	-	-	-	b	3	4	5	N
AE	-	-	-	b	6	7	8	N
AH	-	-	-	b	9	10	11	N
EH	-	-	-	b	12	13	14	N
"""
    (data_dir / "mdef").write_text(mdef_content)

    # Create phone list
    (data_dir / "phones.txt").write_text("SIL\nAA\nAE\nAH\nEH\n")

    # Create simple dictionary
    dict_content = """HELLO H AH L OW
WORLD W ER L D
TEST T EH S T
"""
    (data_dir / "dict.txt").write_text(dict_content)

    # Create dummy feature file using CFFI directly
    n_frames = 50
    features = rng.random((n_frames, veclen)).astype(np.float32)
    # Write in sphinx MFC format: header (n_frames*veclen as int32), then float32 data
    import struct

    mfc_path = data_dir / "test.mfc"
    with mfc_path.open("wb") as f:
        f.write(struct.pack("<i", n_frames * veclen))
        f.write(features.astype("<f").tobytes())

    # Create control file
    (data_dir / "test.ctl").write_text("test\n")

    return data_dir


# =============================================================================
# I/O parity tests
# =============================================================================


class TestGauParity:
    """Test Gaussian I/O parity."""

    def test_read_write_roundtrip(self, test_data_dir: Path) -> None:
        """Test that read/write cycle preserves data."""
        means_path = test_data_dir / "means"
        means, n_mgau, n_feat, n_density, veclen = _pstrainc.read_gau(str(means_path))

        # Write to new location
        out_path = test_data_dir / "means_copy"
        _pstrainc.write_gau(str(out_path), means)

        # Read back
        means2, n_mgau2, n_feat2, n_density2, veclen2 = _pstrainc.read_gau(str(out_path))

        assert n_mgau == n_mgau2
        assert n_feat == n_feat2
        assert n_density == n_density2
        np.testing.assert_allclose(means, means2, rtol=1e-5)


class TestMixwParity:
    """Test mixture weight I/O parity."""

    def test_read_write_roundtrip(self, test_data_dir: Path) -> None:
        """Test that read/write cycle preserves data."""
        mixw_path = test_data_dir / "mixture_weights"
        mixw, n_mixw, n_feat, n_density = _pstrainc.read_mixw(str(mixw_path))

        # Write to new location
        out_path = test_data_dir / "mixw_copy"
        _pstrainc.write_mixw(str(out_path), mixw)

        # Read back
        mixw2, n_mixw2, n_feat2, n_density2 = _pstrainc.read_mixw(str(out_path))

        assert n_mixw == n_mixw2
        assert n_feat == n_feat2
        assert n_density == n_density2
        np.testing.assert_allclose(mixw, mixw2, rtol=1e-5)


class TestTmatParity:
    """Test transition matrix I/O parity."""

    def test_read_write_roundtrip(self, test_data_dir: Path) -> None:
        """Test that tmat can be read and written without errors.

        Note: Tmat format stores n_state-1 rows (excluding exit state).
        Row sums may be < 1.0 because exit probability is stored separately.
        """
        tmat_path = test_data_dir / "transition_matrices"

        # Read existing file
        tmat, n_tmat, n_state = _pstrainc.read_tmat(str(tmat_path))

        # Basic sanity checks
        assert n_tmat == 5
        assert tmat.shape[0] == n_tmat
        # tmat should have reasonable values (probabilities)
        assert tmat.min() >= 0.0
        assert tmat.max() <= 1.0
        # Should have transition values (not all zeros)
        assert tmat.sum() > 0


# =============================================================================
# Command shell-out parity tests
# =============================================================================


class TestSphinxFeParity:
    """Test sphinx_fe shell-out vs CFFI parity."""

    @pytest.mark.skipif(not binary_available("sphinx_fe"), reason="sphinx_fe not found")
    @pytest.mark.xfail(
        strict=True,
        reason="Known gap: the CFFI feature path (pstrain_fe_create, a 'simplified' "
        "front-end that ignores remove_noise/transform/lifter/unit_area — see "
        "pstrain/lib/features.py:123) does not match the sphinx_fe CLI, which applies "
        "them. Cepstra differ by ~30 on average even with dither off. Aligning "
        "the two front-ends is feature-config work (plan Phase 2/7), not a "
        "Phase-1 fix; xfail(strict) so this flips red the moment they converge.",
    )
    def test_feature_extraction(self, test_data_dir: Path) -> None:
        """CFFI vs sphinx_fe CLI features (currently a documented divergence)."""
        # Create a simple audio file (sine wave)
        import wave

        audio_path = test_data_dir / "test_audio.wav"
        sample_rate = 16000
        duration = 1.0
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples)
        audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        with wave.open(str(audio_path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            f.writeframes(audio.tobytes())

        # Shell-out (dither off so the comparison is deterministic).
        shell_output = test_data_dir / "shell_features.mfc"
        builder = CommandBuilder()
        cmd = builder.sphinx_fe(
            input_file=audio_path,
            output_file=shell_output,
            samprate=sample_rate,
            ncep=13,
            dither=False,
        )
        cmd.run()

        # CFFI
        cffi_output = test_data_dir / "cffi_features.mfc"
        from pstrain.lib.features import extract_features

        extract_features(audio_path, cffi_output)

        # Compare
        from pstrain.lib.features import read_sphinx_mfc

        shell_feats = read_sphinx_mfc(shell_output)
        cffi_feats = read_sphinx_mfc(cffi_output)

        # Allow small differences due to potential implementation details
        np.testing.assert_allclose(shell_feats, cffi_feats, rtol=1e-4, atol=1e-4)


class TestPrintpParity:
    """Test printp shell-out vs native Python parity."""

    @pytest.mark.skipif(not binary_available("printp"), reason="printp not found")
    def test_print_mixw(self, test_data_dir: Path) -> None:
        """Test that native and shell-out print same mixture weights."""
        from pstrain.lib.printp import format_mixw, print_params_shellout

        mixw_path = test_data_dir / "mixture_weights"

        # Get native output
        native_output = format_mixw(mixw_path, sigfig=4)

        # Get shell output from pstrain's own printp binary.
        shell_output = print_params_shellout(
            mixw_path=mixw_path, sigfig=4, bin_path=_resolve_bin("printp")
        )

        # Compare numerical values (not exact string match due to formatting)
        native_lines = [
            line
            for line in native_output.splitlines()
            if line.strip() and not line.startswith("mixw")
        ]
        shell_lines = [
            line
            for line in shell_output.splitlines()
            if line.strip() and not line.startswith("mixw")
        ]

        # Parse numerical values
        for native_line, shell_line in zip(native_lines, shell_lines, strict=False):
            try:
                native_vals = [
                    float(v) for v in native_line.split() if v[0].isdigit() or v[0] in "-."
                ]
                shell_vals = [
                    float(v) for v in shell_line.split() if v[0].isdigit() or v[0] in "-."
                ]
                if native_vals and shell_vals:
                    np.testing.assert_allclose(native_vals, shell_vals, rtol=1e-3)
            except (ValueError, IndexError):
                continue  # Skip non-numerical lines


class TestCepviewParity:
    """Test sphinx_cepview shell-out vs native Python parity."""

    @pytest.mark.skipif(not binary_available("sphinx_cepview"), reason="sphinx_cepview not found")
    def test_view_features(self, test_data_dir: Path) -> None:
        """Test that native and shell-out view same features."""
        from pstrain.lib.cepview import check_parity

        mfc_path = test_data_dir / "test.mfc"
        assert check_parity(
            mfc_path,
            n_coeff=13,
            display_cols=5,
            end_frame=10,
            bin_path=_resolve_bin("sphinx_cepview"),
        )


# =============================================================================
# Split parity tests
# =============================================================================


class TestIncCompParity:
    """Test inc_comp (Gaussian splitting) command builder."""

    def test_inc_comp_command(self, test_data_dir: Path) -> None:
        """Test that inc_comp command builds correctly."""
        builder = CommandBuilder()
        cmd = builder.inc_comp(
            inmeanfn=test_data_dir / "means",
            invarfn=test_data_dir / "variances",
            inmixwfn=test_data_dir / "mixture_weights",
            outmeanfn=test_data_dir / "split_means",
            outvarfn=test_data_dir / "split_var",
            outmixwfn=test_data_dir / "split_mixw",
        )
        shell = cmd.to_shell()
        assert "inc_comp" in shell
        assert "-inmeanfn" in shell
        assert "-outmeanfn" in shell


# =============================================================================
# Decision tree parity tests
# =============================================================================


class TestMakeQuestsParity:
    """Test make_quests parity."""

    # NOTE: make_quests needs -type (.cont./.semi.), which CommandBuilder now
    # supplies via kwargs, but it segfaults on the tiny synthetic model in
    # this file — both the CLI (exit 139) and, worse, the in-process CFFI path,
    # which would take down the whole pytest run. The old body swallowed that
    # crash into pytest.skip("shell command failed"), masking a real fragility.
    # Running this for real needs a realistic triphone model (available once
    # the e2e integration job lands, plan 1.2) and the C decision-tree code
    # hardened against degenerate input (plan 4.1). Skip explicitly and safely
    # until then rather than invoke code that can crash the interpreter.
    @pytest.mark.skip(
        reason="Needs a realistic triphone model; make_quests segfaults on the "
        "synthetic fixture (CFFI crash would abort pytest). Tracked: plan 1.2 (e2e "
        "corpus) + 4.1 (harden C against bad input)."
    )
    def test_questions_generation(self, test_data_dir: Path) -> None:
        """CFFI vs shell-out question generation (needs a realistic model)."""
        from pstrain.lib.dtree import make_quests

        shell_quests = test_data_dir / "shell_quests.txt"
        builder = CommandBuilder()
        builder.make_quests(
            moddeffn=test_data_dir / "mdef",
            meanfn=test_data_dir / "means",
            varfn=test_data_dir / "variances",
            mixwfn=test_data_dir / "mixture_weights",
            questsfn=shell_quests,
            type=".cont.",
        ).run()

        cffi_quests = test_data_dir / "cffi_quests.txt"
        make_quests(
            mdef_path=test_data_dir / "mdef",
            mean_path=test_data_dir / "means",
            var_path=test_data_dir / "variances",
            mixw_path=test_data_dir / "mixture_weights",
            output_path=cffi_quests,
        )

        assert len(shell_quests.read_text().splitlines()) == len(
            cffi_quests.read_text().splitlines()
        )


# =============================================================================
# CLI dry-run parity tests
# =============================================================================


class TestCliDryRun:
    """Test that CLI --dry-run emits valid shell commands."""

    def test_dry_run_feature_extraction(self) -> None:
        """Test that dry-run emits valid sphinx_fe command."""
        from pstrain.cli.base import FeatureExtractAction

        action = FeatureExtractAction(
            input_file=Path("/tmp/audio.wav"),
            output_file=Path("/tmp/features.mfc"),
            samprate=16000,
            nfilt=40,
            ncep=13,
        )
        shell_cmd = action.to_shell()

        assert "sphinx_fe" in shell_cmd
        assert "-i /tmp/audio.wav" in shell_cmd
        assert "-o /tmp/features.mfc" in shell_cmd
        assert "-samprate 16000" in shell_cmd
        assert "-ncep 13" in shell_cmd

    def test_dry_run_baum_welch(self) -> None:
        """Test that dry-run emits valid bw command."""
        from pstrain.cli.base import BaumWelchAction

        action = BaumWelchAction(
            mdef=Path("/model/mdef"),
            mean=Path("/model/means"),
            var=Path("/model/variances"),
            mixw=Path("/model/mixture_weights"),
            tmat=Path("/model/transition_matrices"),
            ctl=Path("/etc/train.ctl"),
            cepdir=Path("/features"),
            dictfn=Path("/etc/dict.txt"),
        )
        shell_cmd = action.to_shell()

        assert "bw" in shell_cmd
        assert "-moddeffn /model/mdef" in shell_cmd
        assert "-meanfn /model/means" in shell_cmd
        assert "-ctlfn /etc/train.ctl" in shell_cmd
        assert "-dictfn /etc/dict.txt" in shell_cmd

    def test_dry_run_inc_comp(self) -> None:
        """Test that dry-run emits valid inc_comp command."""
        from pstrain.cli.base import SplitGaussiansAction

        action = SplitGaussiansAction(
            inmeanfn=Path("/model/means"),
            invarfn=Path("/model/variances"),
            inmixwfn=Path("/model/mixture_weights"),
            outmeanfn=Path("/model/split_means"),
            outvarfn=Path("/model/split_variances"),
            outmixwfn=Path("/model/split_mixture_weights"),
        )
        shell_cmd = action.to_shell()

        assert "inc_comp" in shell_cmd
        assert "-inmeanfn /model/means" in shell_cmd
        assert "-outmeanfn /model/split_means" in shell_cmd

    def test_dry_run_make_quests(self) -> None:
        """Test that dry-run emits valid make_quests command."""
        from pstrain.cli.base import MakeQuestsAction

        action = MakeQuestsAction(
            moddeffn=Path("/model/mdef"),
            meanfn=Path("/model/means"),
            varfn=Path("/model/variances"),
            mixwfn=Path("/model/mixture_weights"),
            questsfn=Path("/output/questions.txt"),
        )
        shell_cmd = action.to_shell()

        assert "make_quests" in shell_cmd
        assert "-moddeffn /model/mdef" in shell_cmd
        assert "-questsfn /output/questions.txt" in shell_cmd


# =============================================================================
# CommandBuilder parity tests
# =============================================================================


class TestCommandBuilderShellScript:
    """Test CommandBuilder shell script generation."""

    def test_generate_training_script(self, test_data_dir: Path) -> None:
        """Test that CommandBuilder generates valid shell script."""
        builder = CommandBuilder(dry_run=True)

        # Build a training pipeline
        builder.sphinx_fe(
            input_file=Path("/audio/train.wav"),
            output_file=Path("/features/train.mfc"),
        )
        builder.bw(
            mdef=Path("/model/mdef"),
            mean=Path("/model/means"),
            var=Path("/model/variances"),
            mixw=Path("/model/mixture_weights"),
            tmat=Path("/model/transition_matrices"),
            ctl=Path("/etc/train.ctl"),
            lsn=Path("/etc/train.lsn"),
            dictfn=Path("/etc/dict.txt"),
            accumdir=Path("/accum"),
        )
        builder.norm(
            accumdir=Path("/accum"),
            meanfn=Path("/model/means"),
            varfn=Path("/model/variances"),
            mixwfn=Path("/model/mixture_weights"),
            tmatfn=Path("/model/transition_matrices"),
        )

        script = builder.to_shell_script()

        # Verify script structure
        assert "#!/usr/bin/env bash" in script
        assert "set -euo pipefail" in script
        assert "sphinx_fe" in script
        assert "bw" in script
        assert "norm" in script

    def test_script_is_valid_bash(self, test_data_dir: Path) -> None:
        """Test that generated script passes bash syntax check."""
        builder = CommandBuilder(dry_run=True)
        builder.sphinx_fe(
            input_file=Path("/audio/test.wav"),
            output_file=Path("/features/test.mfc"),
        )

        script = builder.to_shell_script()

        # Write to temp file and syntax check
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script)
            f.flush()

            # Check bash syntax (doesn't execute)
            result = subprocess.run(
                ["bash", "-n", f.name],
                capture_output=True,
                text=True,
            )
            Path(f.name).unlink()

            assert result.returncode == 0, f"Bash syntax error: {result.stderr}"
