"""Tests for decision tree building functionality."""

import contextlib
from pathlib import Path

import numpy as np
import pytest

from pstrain.lib import _pstrainc, dtree, mdef, native_worker

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
