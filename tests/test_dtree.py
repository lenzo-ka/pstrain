"""Tests for decision tree building functionality."""

import contextlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip(
    "resource", reason="POSIX-only training resource accounting requires the resource module"
)

from pstrain.lib import _pstrainc, dtree, mdef, native_worker
from pstrain.lib.commands import resolve_binary
from pstrain.lib.steps import cd_pipeline

# Check if library exists
# libpstrainc availability comes from the shared helper (real loader-based
# detection); see tests/clib.py.
from tests.clib import C_LIBRARY_AVAILABLE as _lib_exists


@pytest.fixture
def ci_mdef_file(tmp_path: Path) -> Path:
    """Create a minimal CI mdef file."""
    mdef = tmp_path / "ci.mdef"
    mdef.write_text(
        """0.3
5 n_base
0 n_tri
5 n_state_map
5 n_tied_state
5 n_tied_ci_state
5 n_tied_tmat
AA - - - n/a 0 0 1 2 N
AE - - - n/a 1 3 4 5 N
SIL - - - filler 2 6 7 8 N
+NOISE+ - - - filler 3 9 10 11 N
+SPN+ - - - filler 4 12 13 14 N
"""
    )
    return mdef


class TestParseQuestions:
    """Tests for question file parsing."""

    def test_parse_basic(self, tmp_path: Path) -> None:
        """Test parsing a basic question file."""
        qfile = tmp_path / "questions.txt"
        qfile.write_text(
            """WDBNDRY_B
WDBNDRY_E
SILENCE SIL
QUESTION0 AA AE
QUESTION1 SIL
"""
        )
        result = dtree.parse_questions(qfile)

        assert "SILENCE" in result
        assert result["SILENCE"] == ["SIL"]
        assert "QUESTION0" in result
        assert result["QUESTION0"] == ["AA", "AE"]

    def test_parse_empty_lines(self, tmp_path: Path) -> None:
        """Test parsing handles empty lines."""
        qfile = tmp_path / "questions.txt"
        qfile.write_text(
            """QUESTION0 AA

QUESTION1 AE
"""
        )
        result = dtree.parse_questions(qfile)

        assert len(result) == 2
        assert "QUESTION0" in result
        assert "QUESTION1" in result


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestMakeQuests:
    """Tests for phonetic question generation."""

    def test_make_quests_requires_mean_var_for_continuous(
        self, ci_mdef_file: Path, tmp_path: Path
    ) -> None:
        """Test that continuous mode requires mean and var paths."""
        mixw = tmp_path / "mixw"
        mixw.touch()
        output = tmp_path / "questions.txt"

        with pytest.raises(ValueError, match="[Cc]ontinuous"):
            dtree.make_quests(
                ci_mdef_file,
                mixw,
                output,
                continuous=True,
                mean_path=None,
                var_path=None,
            )

    def test_make_quests_semi_continuous_no_mean_var(
        self, ci_mdef_file: Path, tmp_path: Path
    ) -> None:
        """Semi-continuous mode needs no mean/var, and bad input is contained.

        This replaces a skip ("C code crashes on invalid input files"): the
        native failure now arrives as a typed exception instead of taking the
        interpreter with it, so the argument contract is testable.
        """
        mixw = tmp_path / "mixw"
        mixw.write_bytes(b"not a mixture weight file")
        output = tmp_path / "questions.txt"

        with pytest.raises(native_worker.PstrainNativeError) as raised:
            dtree.make_quests(
                ci_mdef_file,
                mixw,
                output,
                continuous=False,
                mean_path=None,
                var_path=None,
            )
        assert raised.value.operation == "make_quests"
        assert raised.value.input_path == str(ci_mdef_file)
        assert raised.value.diagnostic

        # The interpreter survived and the next guarded call still works.
        phones = tmp_path / "phones"
        phones.write_text("AA\nSIL\n")
        recovered = tmp_path / "recovered.mdef"
        mdef.generate_ci_mdef(phones, recovered)
        assert recovered.exists()

    def test_make_quests_malformed_mdef_is_contained(self, tmp_path: Path) -> None:
        """A malformed mdef reaches the native reader and is contained."""
        bad_mdef = tmp_path / "bad.mdef"
        bad_mdef.write_text("this is not a model definition\n")
        mixw = tmp_path / "mixw"
        mixw.write_bytes(b"")

        with pytest.raises(native_worker.PstrainNativeError) as raised:
            dtree.make_quests(bad_mdef, mixw, tmp_path / "questions.txt", continuous=False)
        assert raised.value.operation == "make_quests"
        assert raised.value.diagnostic

    @staticmethod
    def _continuous_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
        model = tmp_path / "mdef"
        model.write_text(
            """0.3
2 n_base
0 n_tri
4 n_state_map
2 n_tied_state
2 n_tied_ci_state
2 n_tied_tmat
AA - - - n/a 0 0 N
SIL - - - filler 1 1 N
"""
        )
        mixw = tmp_path / "mixture_weights"
        means = tmp_path / "means"
        variances = tmp_path / "variances"
        assert _pstrainc.write_mixw(str(mixw), np.ones((2, 1, 2), dtype=np.float32)) == 0
        assert _pstrainc.write_gau(str(means), np.zeros((2, 1, 2, 3), dtype=np.float32)) == 0
        assert _pstrainc.write_gau(str(variances), np.ones((2, 1, 2, 3), dtype=np.float32)) == 0
        return model, mixw, means, variances

    @pytest.mark.parametrize(
        ("bad_artifact", "message"),
        [
            ("mixture_weights", "Invalid mixture weight"),
            ("means", "Non-finite mean"),
            ("variances", "Non-finite variance"),
            ("short_means_and_variances", "Mean/mixture-weight state-count mismatch"),
        ],
    )
    def test_make_quests_rejects_degraded_statistics(
        self, tmp_path: Path, bad_artifact: str, message: str
    ) -> None:
        """Malformed files must reach the real accumulator and stop artifact creation."""
        model, mixw, means, variances = self._continuous_inputs(tmp_path)
        if bad_artifact == "mixture_weights":
            values = np.ones((2, 1, 2), dtype=np.float32)
            values[0, 0, 0] = np.nan
            assert _pstrainc.write_mixw(str(mixw), values) == 0
        elif bad_artifact == "means":
            values = np.zeros((2, 1, 2, 3), dtype=np.float32)
            values[0, 0, 0, 0] = np.nan
            assert _pstrainc.write_gau(str(means), values) == 0
        elif bad_artifact == "variances":
            values = np.ones((2, 1, 2, 3), dtype=np.float32)
            values[0, 0, 0, 0] = np.inf
            assert _pstrainc.write_gau(str(variances), values) == 0
        else:
            short = np.ones((1, 1, 2, 3), dtype=np.float32)
            assert _pstrainc.write_gau(str(means), short) == 0
            assert _pstrainc.write_gau(str(variances), short) == 0

        output = tmp_path / "questions"
        with pytest.raises(native_worker.PstrainNativeFatalError) as raised:
            dtree.make_quests(model, mixw, output, means, variances)

        assert message in raised.value.diagnostic
        assert not output.exists()

    def test_make_quests_good_input_is_deterministic(self, tmp_path: Path) -> None:
        model, mixw, means, variances = self._continuous_inputs(tmp_path)
        first = tmp_path / "questions.first"
        second = tmp_path / "questions.second"

        dtree.make_quests(model, mixw, first, means, variances)
        dtree.make_quests(model, mixw, second, means, variances)

        assert first.read_bytes() == second.read_bytes()

    def test_make_quests_rejects_nonfinite_full_variance(self, tmp_path: Path) -> None:
        """The standalone full-covariance path is constructible and fails loudly."""
        model, mixw, means, variances = self._continuous_inputs(tmp_path)
        values = np.zeros((2, 1, 2, 3, 3), dtype=np.float32)
        values[..., range(3), range(3)] = 1.0
        values[0, 0, 0, 0, 0] = np.nan

        ffi = _pstrainc.get_ffi()
        lib = _pstrainc.get_lib()
        mgau = ffi.new("float32 ****[]", 2)
        feat = [ffi.new("float32 ***[]", 1) for _ in range(2)]
        density = [[ffi.new("float32 **[]", 2)] for _ in range(2)]
        rows = [[[ffi.new("float32 *[]", 3) for _ in range(2)]] for _ in range(2)]
        for m in range(2):
            mgau[m] = feat[m]
            feat[m][0] = density[m][0]
            for d in range(2):
                density[m][0][d] = rows[m][0][d]
                for row in range(3):
                    rows[m][0][d][row] = ffi.cast(
                        "float32 *", ffi.from_buffer(values[m, 0, d, row])
                    )
        veclen = ffi.new("uint32[]", [3])
        assert lib.s3gau_write_full(str(variances).encode(), mgau, 2, 1, 2, veclen) == 0

        output = tmp_path / "questions"
        executable = resolve_binary("make_quests")
        assert executable is not None
        result = subprocess.run(
            [
                str(executable),
                "-moddeffn",
                str(model),
                "-mixwfn",
                str(mixw),
                "-meanfn",
                str(means),
                "-varfn",
                str(variances),
                "-questfn",
                str(output),
                "-type",
                ".cont.",
                "-fullvar",
                "yes",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0
        assert "Non-finite full variance" in result.stderr
        assert not output.exists()


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
def test_init_mixw_rejects_uninitialized_destination(tmp_path: Path) -> None:
    """A destination phone absent from the source must prevent all output."""
    src_mdef = tmp_path / "src.mdef"
    src_mdef.write_text(
        """0.3
1 n_base
0 n_tri
2 n_state_map
1 n_tied_state
1 n_tied_ci_state
1 n_tied_tmat
AA - - - n/a 0 0 N
"""
    )
    dest_mdef = tmp_path / "dest.mdef"
    dest_mdef.write_text(
        """0.3
2 n_base
0 n_tri
4 n_state_map
2 n_tied_state
2 n_tied_ci_state
2 n_tied_tmat
AA - - - n/a 0 0 N
BB - - - n/a 1 1 N
"""
    )

    src_mixw = tmp_path / "src_mixw"
    src_mean = tmp_path / "src_mean"
    src_var = tmp_path / "src_var"
    src_tmat = tmp_path / "src_tmat"
    assert _pstrainc.write_mixw(str(src_mixw), np.ones((1, 1, 2), dtype=np.float32)) == 0
    assert _pstrainc.write_gau(str(src_mean), np.zeros((1, 1, 2, 3), dtype=np.float32)) == 0
    assert _pstrainc.write_gau(str(src_var), np.ones((1, 1, 2, 3), dtype=np.float32)) == 0
    assert (
        _pstrainc.write_tmat(str(src_tmat), np.array([[[0.5, 0.5], [0.0, 1.0]]], dtype=np.float32))
        == 0
    )

    outputs = [tmp_path / name for name in ("mixw", "means", "variances", "tmat")]
    with pytest.raises(native_worker.PstrainNativeError) as raised:
        dtree.init_mixw(
            src_mdef,
            src_mixw,
            src_mean,
            src_var,
            src_tmat,
            dest_mdef,
            outputs[0],
            outputs[1],
            outputs[2],
            outputs[3],
        )

    assert raised.value.operation == "init_mixw"
    assert raised.value.input_path == str(src_mdef)
    assert not any(path.exists() for path in outputs)


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
def test_init_mixw_rejects_source_tmat_count_mismatch(tmp_path: Path) -> None:
    """A source mdef cannot index beyond its transition-matrix file."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "mdef").write_text(
        """0.3
2 n_base
0 n_tri
4 n_state_map
2 n_tied_state
2 n_tied_ci_state
2 n_tied_tmat
AA - - - n/a 0 0 N
BB - - - n/a 1 1 N
"""
    )
    assert (
        _pstrainc.write_mixw(str(source / "mixture_weights"), np.ones((2, 1, 2), dtype=np.float32))
        == 0
    )
    assert _pstrainc.write_gau(str(source / "means"), np.zeros((2, 1, 2, 3), dtype=np.float32)) == 0
    assert (
        _pstrainc.write_gau(str(source / "variances"), np.ones((2, 1, 2, 3), dtype=np.float32)) == 0
    )
    assert (
        _pstrainc.write_tmat(
            str(source / "transition_matrices"),
            np.array([[[0.5, 0.5], [0.0, 1.0]]] * 2, dtype=np.float32),
        )
        == 0
    )
    transition_matrices = _pstrainc.read_tmat_counts(str(source / "transition_matrices"))[0]
    assert transition_matrices.shape[0] == 2
    truncated = np.pad(transition_matrices[:1], ((0, 0), (0, 1), (0, 0)))
    assert _pstrainc.write_tmat(str(source / "transition_matrices"), truncated) == 0

    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(
        native_worker.PstrainNativeError,
        match=r"Source model BB refers to transition matrix 1, but the source "
        r"transition-matrix file contains 1 matrices",
    ):
        dtree.init_mixw(
            source / "mdef",
            source / "mixture_weights",
            source / "means",
            source / "variances",
            source / "transition_matrices",
            source / "mdef",
            output / "mixture_weights",
            output / "means",
            output / "variances",
            output / "transition_matrices",
        )

    assert not any(output.iterdir())


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
def test_init_mixw_rejects_destination_tmat_count_mismatch(tmp_path: Path) -> None:
    """A destination mdef cannot index beyond its declared matrix count."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "mdef").write_text(
        """0.3
2 n_base
0 n_tri
4 n_state_map
2 n_tied_state
2 n_tied_ci_state
2 n_tied_tmat
AA - - - n/a 0 0 N
BB - - - n/a 1 1 N
"""
    )
    assert (
        _pstrainc.write_mixw(str(source / "mixture_weights"), np.ones((2, 1, 2), dtype=np.float32))
        == 0
    )
    assert _pstrainc.write_gau(str(source / "means"), np.zeros((2, 1, 2, 3), dtype=np.float32)) == 0
    assert (
        _pstrainc.write_gau(str(source / "variances"), np.ones((2, 1, 2, 3), dtype=np.float32)) == 0
    )
    assert (
        _pstrainc.write_tmat(
            str(source / "transition_matrices"),
            np.array([[[0.5, 0.5], [0.0, 1.0]]] * 2, dtype=np.float32),
        )
        == 0
    )

    destination_mdef = tmp_path / "destination.mdef"
    destination_mdef.write_text(
        """0.3
2 n_base
0 n_tri
4 n_state_map
2 n_tied_state
2 n_tied_ci_state
1 n_tied_tmat
AA - - - n/a 0 0 N
BB - - - n/a 1 1 N
"""
    )
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(
        native_worker.PstrainNativeError,
        match=r"Destination model BB refers to transition matrix 1, but the "
        r"destination model definition declares 1 matrices",
    ):
        dtree.init_mixw(
            source / "mdef",
            source / "mixture_weights",
            source / "means",
            source / "variances",
            source / "transition_matrices",
            destination_mdef,
            output / "mixture_weights",
            output / "means",
            output / "variances",
            output / "transition_matrices",
        )

    assert not any(output.iterdir())


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestBuildTree:
    """Tests for decision tree building."""

    def test_build_tree_requires_mean_var_for_continuous(
        self, ci_mdef_file: Path, tmp_path: Path
    ) -> None:
        """Test that continuous mode requires mean and var paths."""
        mixw = tmp_path / "mixw"
        mixw.touch()
        pset = tmp_path / "pset.txt"
        pset.touch()
        output = tmp_path / "tree.txt"

        with pytest.raises(ValueError, match="[Cc]ontinuous"):
            dtree.build_tree(
                ci_mdef_file,
                mixw,
                pset,
                output,
                phone="AA",
                state=0,
                continuous=True,
                mean_path=None,
                var_path=None,
            )

    def test_count_threshold_excludes_zero_occupancy_from_questions(self, tmp_path: Path) -> None:
        """A dictionary-only triphone cannot steer the selected question."""
        model = tmp_path / "mdef"
        model.write_text(
            """0.3
3 n_base
3 n_tri
12 n_state_map
6 n_tied_state
3 n_tied_ci_state
3 n_tied_tmat
AA - - - n/a 0 0 N
L1 - - - n/a 1 1 N
L2 - - - n/a 2 2 N
AA L1 L1 s n/a 0 3 N
AA L1 L2 s n/a 0 4 N
AA L2 L1 s n/a 0 5 N
"""
        )
        mixw = np.ones((6, 1, 2), dtype=np.float32)
        mixw[3] = (10.0, 0.0)
        mixw[4] = (0.0, 10.0)
        mixw[5] = (0.0, 0.0)
        mixw_path = tmp_path / "mixw"
        assert _pstrainc.write_mixw(str(mixw_path), mixw) == 0
        questions = tmp_path / "questions"
        questions.write_text("LEFT_ONE L1\nRIGHT_ONE L1\n")

        filtered = tmp_path / "filtered.dtree"
        dtree.build_tree(
            model,
            mixw_path,
            questions,
            filtered,
            "AA",
            0,
            continuous=False,
            ssplitmax=1,
            csplitmax=1,
            cntthresh=1e-5,
        )
        assert "(!LEFT_ONE 1)" in filtered.read_text().splitlines()[1]

        questions.write_text("ONLY_LEFT_L L1\n")
        directional = tmp_path / "directional.dtree"
        dtree.build_tree(
            model,
            mixw_path,
            questions,
            directional,
            "AA",
            0,
            continuous=False,
            ssplitmax=1,
            csplitmax=1,
            cntthresh=1e-5,
        )
        assert "ONLY_LEFT_L" not in directional.read_text()

        legacy = tmp_path / "legacy-directions.dtree"
        dtree.build_tree(
            model,
            mixw_path,
            questions,
            legacy,
            "AA",
            0,
            continuous=False,
            ssplitmax=1,
            csplitmax=1,
            cntthresh=1e-5,
            directional_questions=False,
        )
        assert "ONLY_LEFT_L" in legacy.read_text()

        with pytest.raises(
            ValueError,
            match="State weight count mismatch: expected 1 .* got 2",
        ):
            dtree.build_tree(
                model,
                mixw_path,
                questions,
                tmp_path / "wrong-weights.dtree",
                "AA",
                0,
                continuous=False,
                state_weights=np.ones(2, dtype=np.float32),
            )

    @staticmethod
    def _continuous_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
        model = tmp_path / "mdef"
        model.write_text(
            """0.3
3 n_base
3 n_tri
12 n_state_map
6 n_tied_state
3 n_tied_ci_state
3 n_tied_tmat
AA - - - n/a 0 0 N
L1 - - - n/a 1 1 N
L2 - - - n/a 2 2 N
AA L1 L1 s n/a 0 3 N
AA L1 L2 s n/a 0 4 N
AA L2 L1 s n/a 0 5 N
"""
        )
        mixw_path = tmp_path / "mixture_weights"
        mean_path = tmp_path / "means"
        var_path = tmp_path / "variances"
        questions = tmp_path / "questions"
        assert _pstrainc.write_mixw(str(mixw_path), np.ones((6, 1, 2), dtype=np.float32)) == 0
        assert _pstrainc.write_gau(str(mean_path), np.zeros((6, 1, 2, 3), dtype=np.float32)) == 0
        assert _pstrainc.write_gau(str(var_path), np.ones((6, 1, 2, 3), dtype=np.float32)) == 0
        questions.write_text("CONTEXT_L L1\n")
        return model, mixw_path, mean_path, var_path, questions

    def test_continuous_build_keeps_parsed_mixw_read_only(self, tmp_path: Path) -> None:
        """Exercise byte-identity, digest, and non-alias guards in native code."""
        model, mixw, means, variances, questions = self._continuous_inputs(tmp_path)

        output = tmp_path / "tree"
        dtree.build_tree(model, mixw, questions, output, "AA", 0, means, variances)

        assert output.read_text().startswith("n_node")

    @pytest.mark.parametrize(
        ("artifact", "shape", "message"),
        [
            ("means", (6, 2, 3, 3), "Mean/mixture-weight dimension mismatch"),
            ("variances", (6, 2, 3, 3), "Variance/mixture-weight dimension mismatch"),
            ("variances", (6, 1, 2, 4), "Mean/variance vector-length mismatch"),
            ("variances", (5, 1, 2, 3), "Mean/variance state-count mismatch"),
        ],
    )
    def test_continuous_dimension_mismatch_is_rejected(
        self,
        tmp_path: Path,
        artifact: str,
        shape: tuple[int, int, int, int],
        message: str,
    ) -> None:
        model, mixw, means, variances, questions = self._continuous_inputs(tmp_path)
        path = means if artifact == "means" else variances
        assert _pstrainc.write_gau(str(path), np.ones(shape, dtype=np.float32)) == 0

        with pytest.raises(native_worker.PstrainNativeError) as raised:
            dtree.build_tree(model, mixw, questions, tmp_path / "tree", "AA", 0, means, variances)

        assert message in raised.value.diagnostic

    def test_phone_set_requires_a_known_member(self, tmp_path: Path) -> None:
        model, mixw, means, variances, questions = self._continuous_inputs(tmp_path)
        questions.write_text("EMPTY NOT_A_PHONE\n")

        with pytest.raises(native_worker.PstrainNativeError) as raised:
            dtree.build_tree(model, mixw, questions, tmp_path / "tree", "AA", 0, means, variances)

        assert "expected at least one known phone member" in raised.value.diagnostic

    def test_nonuniform_model_topology_is_rejected(self, tmp_path: Path) -> None:
        model, mixw, means, variances, questions = self._continuous_inputs(tmp_path)
        model.write_text(
            model.read_text()
            .replace("12 n_state_map", "13 n_state_map")
            .replace("AA L1 L2 s n/a 0 4 N", "AA L1 L2 s n/a 0 4 5 N")
        )

        with pytest.raises(native_worker.PstrainNativeFatalError) as raised:
            dtree.build_tree(model, mixw, questions, tmp_path / "tree", "AA", 0, means, variances)

        assert "expected 1 emitting states, got 2" in raised.value.diagnostic


def test_build_tree_one_propagates_failure_without_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "AA-0.dtree"

    def fail(**kwargs: object) -> None:
        raise RuntimeError("malformed tree input")

    monkeypatch.setattr(cd_pipeline, "build_tree", fail)
    with pytest.raises(RuntimeError, match="malformed tree input"):
        cd_pipeline.build_tree_one(tmp_path, tmp_path / "questions", output, "AA", 0)
    assert not output.exists()


def test_build_tree_one_allows_normal_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "AA-0.dtree"

    def succeed(**kwargs: object) -> None:
        Path(kwargs["output_path"]).write_text("n_node 1\n")  # type: ignore[arg-type]

    monkeypatch.setattr(cd_pipeline, "build_tree", succeed)
    cd_pipeline.build_tree_one(tmp_path, tmp_path / "questions", output, "AA", 0)
    assert output.read_text() == "n_node 1\n"


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestTieStates:
    """Tests for state tying."""

    def test_tie_states_basic_validation(self, tmp_path: Path) -> None:
        """Test that tie_states accepts valid arguments."""
        # This will fail because the function is not implemented
        # but should not raise Python-level errors
        mdef = tmp_path / "mdef"
        mdef.touch()
        tree_dir = tmp_path / "trees"
        tree_dir.mkdir()
        pset = tmp_path / "pset.txt"
        pset.touch()
        output = tmp_path / "tied.mdef"

        try:
            dtree.tie_states(mdef, output, tree_dir, pset)
        except RuntimeError as e:
            # Expected - function not fully implemented yet
            assert "not yet implemented" in str(e).lower() or "Failed" in str(e)


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestPruneTree:
    """Tests for decision tree pruning."""

    def test_prune_tree_creates_output_dir(self, tmp_path: Path) -> None:
        """Test that prune_tree creates output directory if needed."""
        mdef = tmp_path / "mdef"
        mdef.touch()
        pset = tmp_path / "pset.txt"
        pset.touch()
        input_dir = tmp_path / "input_trees"
        input_dir.mkdir()
        output_dir = tmp_path / "output_trees"  # doesn't exist yet

        with contextlib.suppress(RuntimeError):
            dtree.prune_tree(
                mdef,
                pset,
                input_dir,
                output_dir,
                n_seno_target=100,
            )

        # Output directory should be created
        assert output_dir.exists()

    def test_prune_tree_accepts_min_occ(self, tmp_path: Path) -> None:
        """Test that prune_tree accepts min_occ parameter."""
        mdef = tmp_path / "mdef"
        mdef.touch()
        pset = tmp_path / "pset.txt"
        pset.touch()
        input_dir = tmp_path / "input_trees"
        input_dir.mkdir()
        output_dir = tmp_path / "output_trees"

        with contextlib.suppress(RuntimeError):
            dtree.prune_tree(
                mdef,
                pset,
                input_dir,
                output_dir,
                n_seno_target=100,
                min_occ=10.0,
                allphones=False,
            )

    @staticmethod
    def _tree_inputs(tmp_path: Path, tree_body: str | None) -> tuple[Path, Path, Path]:
        """Build a valid mdef/pset plus a tree dir holding ``tree_body``."""
        phones = tmp_path / "phones"
        phones.write_text("AA\nSIL\n")
        valid_mdef = tmp_path / "valid.mdef"
        mdef.generate_ci_mdef(phones, valid_mdef)
        pset = tmp_path / "pset"
        pset.write_text("")
        trees = tmp_path / "trees"
        trees.mkdir()
        if tree_body is not None:
            for state in range(3):
                (trees / f"AA-{state}.dtree").write_text(tree_body)
        return valid_mdef, pset, trees

    def test_stub_tree_segfault_is_contained_and_recovers(self, tmp_path: Path) -> None:
        """A truncated .dtree dereferences NULL in the reader without killing pytest.

        ``read_final_tree`` does not check ``lineiter_start_clean``, so a
        zero-length member file segfaults the native code. Before containment
        this took the whole interpreter down and the surrounding tests were
        skipped.
        """
        valid_mdef, pset, trees = self._tree_inputs(tmp_path, "")

        with pytest.raises(native_worker.PstrainNativeCrashError) as raised:
            dtree.prune_tree(valid_mdef, pset, trees, tmp_path / "out", 1)
        assert raised.value.operation == "prune_tree"
        assert raised.value.signal != 0
        assert raised.value.input_path == str(valid_mdef)

        # The interpreter survived; the helper respawns and the next call works.
        recovered = tmp_path / "recovered.mdef"
        mdef.generate_ci_mdef(tmp_path / "phones", recovered)
        assert recovered.exists()

    def test_malformed_tree_header_is_contained_as_fatal(self, tmp_path: Path) -> None:
        """A one-line stub tree with a bad header hits E_FATAL, not the parent."""
        valid_mdef, pset, trees = self._tree_inputs(tmp_path, "not_n_node 3\n")

        with pytest.raises(native_worker.PstrainNativeFatalError) as raised:
            dtree.prune_tree(valid_mdef, pset, trees, tmp_path / "out", 1)
        assert raised.value.operation == "prune_tree"
        assert raised.value.returncode is not None
        assert raised.value.returncode > 0
        assert "n_node" in raised.value.diagnostic

    def test_missing_member_tree_reports_the_failing_input(self, tmp_path: Path) -> None:
        """A missing member file surfaces the E_FATAL_SYSTEM text, same route."""
        valid_mdef, pset, trees = self._tree_inputs(tmp_path, None)

        with pytest.raises(native_worker.PstrainNativeFatalError) as raised:
            dtree.prune_tree(valid_mdef, pset, trees, tmp_path / "out", 1)
        assert raised.value.operation == "prune_tree"
        assert "Unable to open" in raised.value.diagnostic
