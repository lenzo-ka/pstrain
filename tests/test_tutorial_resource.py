"""Tests for resolving the tutorial from a source checkout."""

from pathlib import Path

from pstrain.api import TUTORIAL_FILENAME, copy_tutorial


def test_tutorial_uses_checkout_notebook_when_packaged_copy_is_absent(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    packaged = repository_root / "pstrain" / "data" / "notebooks" / TUTORIAL_FILENAME
    source = repository_root / "notebooks" / TUTORIAL_FILENAME
    output = tmp_path / TUTORIAL_FILENAME

    assert not packaged.exists()
    assert copy_tutorial(output)["status"] == "written"
    assert output.read_bytes() == source.read_bytes()
