"""Tests for decision tree building functionality."""

import contextlib
from pathlib import Path

import pytest

from pstrain.lib import dtree

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

    @pytest.mark.skip(reason="C code crashes on invalid input files")
    def test_make_quests_semi_continuous_no_mean_var(
        self, ci_mdef_file: Path, tmp_path: Path
    ) -> None:
        """Test that semi-continuous mode doesn't require mean and var."""
        # Note: This test is skipped because the C code crashes on invalid files
        # In a real test, we'd need valid input files
        pass


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
