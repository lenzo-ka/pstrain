"""Tests for mdef generation (mk_mdef_gen)."""

import time
from pathlib import Path

import pytest

from pstrain.lib import mdef, native_worker

# Check if library is available
# libpstrainc availability comes from the shared helper (real loader-based
# detection); see tests/clib.py.
from tests.clib import C_LIBRARY_AVAILABLE as _lib_exists


@pytest.fixture
def phone_list(tmp_path: Path) -> Path:
    """Create a simple phone list."""
    phones = ["AA", "AE", "AH", "B", "D", "SIL"]
    phone_file = tmp_path / "phones.txt"
    phone_file.write_text("\n".join(phones) + "\n")
    return phone_file


@pytest.fixture
def dictionary(tmp_path: Path) -> Path:
    """Create a simple pronunciation dictionary."""
    entries = [
        "BAD B AE D",
        "DAD D AE D",
        "ADD AE D",
        "BAA B AA",
    ]
    dict_file = tmp_path / "dict.txt"
    dict_file.write_text("\n".join(entries) + "\n")
    return dict_file


@pytest.fixture
def filler_dict(tmp_path: Path) -> Path:
    """Create a filler dictionary."""
    entries = [
        "<s> SIL",
        "</s> SIL",
        "<sil> SIL",
    ]
    filler_file = tmp_path / "filler.txt"
    filler_file.write_text("\n".join(entries) + "\n")
    return filler_file


@pytest.fixture
def transcripts(tmp_path: Path) -> Path:
    """Create transcripts file."""
    lines = [
        "<s> BAD DAD </s> (utt1)",
        "<s> ADD BAA </s> (utt2)",
    ]
    trans_file = tmp_path / "transcripts.txt"
    trans_file.write_text("\n".join(lines) + "\n")
    return trans_file


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestGenerateCIMdef:
    """Test CI mdef generation."""

    def test_generates_ci_mdef(self, phone_list: Path, tmp_path: Path) -> None:
        """Test basic CI mdef generation."""
        output = tmp_path / "ci.mdef"
        mdef.generate_ci_mdef(phone_list, output)
        assert output.exists()

    def test_ci_mdef_content(self, phone_list: Path, tmp_path: Path) -> None:
        """Test CI mdef contains expected phones."""
        output = tmp_path / "ci.mdef"
        mdef.generate_ci_mdef(phone_list, output)

        content = output.read_text()
        # Should contain all phones
        assert "AA" in content
        assert "AE" in content
        assert "SIL" in content
        # Should have version header
        assert "0.3" in content or "mdef" in content.lower()

    def test_ci_mdef_is_byte_reproducible(self, phone_list: Path, tmp_path: Path) -> None:
        """Identical inputs produce identical mdef bytes across wall-clock seconds."""
        first = tmp_path / "first.mdef"
        second = tmp_path / "second.mdef"

        mdef.generate_ci_mdef(phone_list, first)
        time.sleep(1.1)
        mdef.generate_ci_mdef(phone_list, second)

        assert first.read_bytes() == second.read_bytes()
        assert first.read_text().splitlines()[0] == "0.3"

    def test_ci_mdef_custom_states(self, phone_list: Path, tmp_path: Path) -> None:
        """Test CI mdef with custom state count."""
        output = tmp_path / "ci.mdef"
        mdef.generate_ci_mdef(phone_list, output, n_state=5)
        assert output.exists()
        # Content should reflect 5 states per phone
        content = output.read_text()
        # Check that tied state count is phones * states
        lines = content.strip().split("\n")
        # Find the line with state count (format: "N n_tied_state")
        for line in lines:
            if "n_tied_state" in line:
                parts = line.split()
                n_tied = int(parts[0])  # Number is first
                # 6 phones * 5 states = 30
                assert n_tied == 30
                break

    def test_invalid_phone_list_fails(self, tmp_path: Path) -> None:
        """Test that invalid phone list path fails."""
        output = tmp_path / "ci.mdef"
        with pytest.raises(native_worker.PstrainNativeFatalError) as raised:
            mdef.generate_ci_mdef(tmp_path / "nonexistent", output)
        assert raised.value.operation == "mdef_gen_ci"
        assert raised.value.input_path == str(tmp_path / "nonexistent")
        assert raised.value.diagnostic


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestGenerateAllTriphonesMdef:
    """Test all-triphones mdef generation."""

    def test_generates_alltriphones_mdef(
        self, phone_list: Path, dictionary: Path, tmp_path: Path
    ) -> None:
        """Test basic all-triphones mdef generation."""
        output = tmp_path / "alltriphones.mdef"
        mdef.generate_alltriphones_mdef(phone_list, dictionary, output)
        assert output.exists()

    def test_alltriphones_has_triphones(
        self, phone_list: Path, dictionary: Path, tmp_path: Path
    ) -> None:
        """Test that all-triphones mdef contains triphones."""
        output = tmp_path / "alltriphones.mdef"
        mdef.generate_alltriphones_mdef(phone_list, dictionary, output)

        content = output.read_text()
        # Should have triphone entries (base left right position)
        # e.g. "AE B D" for the middle phone of "BAD"
        assert content.count("\n") > 10  # More than just CI phones

    def test_with_filler_dict(
        self, phone_list: Path, dictionary: Path, filler_dict: Path, tmp_path: Path
    ) -> None:
        """Test all-triphones with filler dictionary."""
        output = tmp_path / "alltriphones.mdef"
        mdef.generate_alltriphones_mdef(phone_list, dictionary, output, filler_dict=filler_dict)
        assert output.exists()

    def test_ignore_word_position(self, phone_list: Path, dictionary: Path, tmp_path: Path) -> None:
        """Test with word position ignored."""
        output_with_wpos = tmp_path / "with_wpos.mdef"
        output_no_wpos = tmp_path / "no_wpos.mdef"

        mdef.generate_alltriphones_mdef(phone_list, dictionary, output_with_wpos, ignore_wpos=False)
        mdef.generate_alltriphones_mdef(phone_list, dictionary, output_no_wpos, ignore_wpos=True)

        # With word position should have more triphones (b, e, i, s variants)
        size_with = output_with_wpos.stat().st_size
        size_without = output_no_wpos.stat().st_size
        assert size_with >= size_without


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestGenerateUntiedMdef:
    """Test untied mdef generation."""

    def test_generates_untied_mdef(
        self,
        phone_list: Path,
        dictionary: Path,
        filler_dict: Path,
        transcripts: Path,
        tmp_path: Path,
    ) -> None:
        """Test basic untied mdef generation."""
        output = tmp_path / "untied.mdef"
        mdef.generate_untied_mdef(
            phone_list, dictionary, transcripts, output, filler_dict=filler_dict
        )
        assert output.exists()

    def test_transcript_reachable_matches_multipron_graph_domain(self, tmp_path: Path) -> None:
        """Reachable inventory includes both sides of variant boundaries."""
        phones = tmp_path / "phones.txt"
        phones.write_text("L\nM\nX\nY\nP\nQ\nN\nO\nZ\nSIL\n")
        dictionary = tmp_path / "dictionary.dict"
        dictionary.write_text("LEFT L X\nLEFT(2) M Y\nRIGHT P N\nRIGHT(2) Q O\nUNUSED Z\n")
        filler = tmp_path / "filler.dict"
        filler.write_text("<s> SIL\n</s> SIL\n")
        transcripts = tmp_path / "train.transcription"
        transcripts.write_text("utt1 LEFT RIGHT\n")
        reachable_path = tmp_path / "reachable.mdef"
        all_path = tmp_path / "all.mdef"
        mdef.generate_untied_mdef(
            phones,
            dictionary,
            transcripts,
            reachable_path,
            filler_dict=filler,
            inventory_policy="transcript-reachable",
        )
        mdef.generate_alltriphones_mdef(phones, dictionary, all_path, filler_dict=filler)

        def rows(path: Path) -> set[tuple[str, str, str, str]]:
            return {
                tuple(fields[:4])
                for line in path.read_text().splitlines()
                if len(fields := line.split()) >= 4
                and not fields[0].startswith("#")
                and fields[1] != "-"
            }

        # Computed independently from the four pronunciation pairs. Fan-out
        # gives X/Y both P/Q right contexts; fan-in gives P/Q both X/Y lefts.
        expected = {
            ("L", "SIL", "X", "b"),
            ("M", "SIL", "Y", "b"),
            ("X", "L", "P", "e"),
            ("X", "L", "Q", "e"),
            ("Y", "M", "P", "e"),
            ("Y", "M", "Q", "e"),
            ("P", "X", "N", "b"),
            ("P", "Y", "N", "b"),
            ("Q", "X", "O", "b"),
            ("Q", "Y", "O", "b"),
            ("N", "P", "SIL", "e"),
            ("O", "Q", "SIL", "e"),
        }
        reachable = rows(reachable_path)
        all_rows = rows(all_path)
        assert reachable == expected
        assert reachable < all_rows
        assert "utt1" not in reachable_path.read_text()

    @pytest.mark.parametrize("multipron", [False, True])
    def test_untied_inventory_matches_training_mode(
        self,
        phone_list: Path,
        dictionary: Path,
        filler_dict: Path,
        transcripts: Path,
        tmp_path: Path,
        multipron: bool,
    ) -> None:
        """Linear is occurrence-pruned; multipron covers the graph domain."""
        output = tmp_path / "untied.mdef"
        mdef.generate_untied_mdef(
            phone_list,
            dictionary,
            transcripts,
            output,
            filler_dict=filler_dict,
            multipron=multipron,
        )

        # Untied should have fewer triphones than all-triphones
        all_output = tmp_path / "all.mdef"
        mdef.generate_alltriphones_mdef(phone_list, dictionary, all_output, filler_dict=filler_dict)

        def cd_rows(path: Path) -> set[tuple[str, str, str, str]]:
            return {
                (fields[0], fields[1], fields[2], fields[3])
                for line in path.read_text().splitlines()
                if len(fields := line.split()) >= 4 and fields[1] != "-"
            }

        untied_rows = cd_rows(output)
        all_rows = cd_rows(all_output)
        if multipron:
            assert untied_rows == all_rows
        else:
            assert untied_rows < all_rows
            # This dictionary-producible BAA start would require BAA to follow
            # BAA; that sequence is absent from the transcript.
            assert ("B", "AA", "AA", "b") in all_rows
            assert ("B", "AA", "AA", "b") not in untied_rows


@pytest.mark.skipif(not _lib_exists, reason="libpstrainc not built")
class TestCountTriphones:
    """Test triphone counting."""

    def test_counts_triphones(
        self,
        phone_list: Path,
        dictionary: Path,
        filler_dict: Path,
        transcripts: Path,
        tmp_path: Path,
    ) -> None:
        """Test triphone counting."""
        output = tmp_path / "counts.txt"
        mdef.count_triphones(phone_list, dictionary, transcripts, output, filler_dict=filler_dict)
        assert output.exists()

    def test_counts_file_format(
        self,
        phone_list: Path,
        dictionary: Path,
        filler_dict: Path,
        transcripts: Path,
        tmp_path: Path,
    ) -> None:
        """Test counts file has expected format."""
        output = tmp_path / "counts.txt"
        mdef.count_triphones(phone_list, dictionary, transcripts, output, filler_dict=filler_dict)

        content = output.read_text()
        # Should have phone counts
        assert len(content) > 0
        # Content should have numbers (counts)
        assert any(c.isdigit() for c in content)
