"""Package data access utilities."""

from pathlib import Path


def get_data_path() -> Path:
    """Return the path to the package data directory."""
    return Path(__file__).parent


def get_data_file(name: str) -> Path:
    """Return the path to a specific data file."""
    return get_data_path() / name


def get_sample_path() -> Path:
    """Return the path to the sample data directory."""
    return get_data_path() / "sample"


def get_sample_file(name: str) -> Path:
    """Return the path to a specific sample file."""
    return get_sample_path() / name


def get_sample_audio() -> Path:
    """Return the path to the sample audio file."""
    return get_sample_file("kevin-alice-16k.wav")


def get_sample_transcript() -> Path:
    """Return the path to the sample transcript file."""
    return get_sample_file("kevin-alice-16k.txt")
