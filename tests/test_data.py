"""Test package data access."""

from pathlib import Path

from pstrain.data import get_data_file, get_data_path


def test_data_path_exists() -> None:
    """Test that data path exists."""
    path = get_data_path()
    assert path.exists()
    assert path.is_dir()


def test_get_data_file() -> None:
    """Test getting a data file path."""
    path = get_data_file("test.txt")
    assert isinstance(path, Path)
    # File may not exist, but path should be valid
    assert path.parent == get_data_path()


def test_data_path_is_absolute() -> None:
    """Test that data path is absolute."""
    path = get_data_path()
    assert path.is_absolute()
