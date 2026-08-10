"""Test sample data access and usage."""

from pstrain.data import (
    get_sample_audio,
    get_sample_path,
    get_sample_transcript,
)


def test_sample_path_exists() -> None:
    """Test that sample data path exists."""
    path = get_sample_path()
    assert path.exists()
    assert path.is_dir()


def test_sample_audio_exists() -> None:
    """Test that sample audio file exists."""
    audio_path = get_sample_audio()
    assert audio_path.exists()
    assert audio_path.is_file()
    assert audio_path.suffix == ".wav"
    assert audio_path.name == "kevin-alice-16k.wav"


def test_sample_transcript_exists() -> None:
    """Test that sample transcript file exists."""
    transcript_path = get_sample_transcript()
    assert transcript_path.exists()
    assert transcript_path.is_file()
    assert transcript_path.suffix == ".txt"
    assert transcript_path.name == "kevin-alice-16k.txt"


def test_sample_transcript_content() -> None:
    """Test that sample transcript has expected content."""
    transcript_path = get_sample_transcript()
    content = transcript_path.read_text()

    # Should contain Alice in Wonderland text
    assert "Alice" in content
    assert len(content) > 100  # Should be substantial text


def test_sample_audio_size() -> None:
    """Test that sample audio file has reasonable size."""
    audio_path = get_sample_audio()
    size = audio_path.stat().st_size

    # Should be a reasonable audio file size (not empty, not huge)
    assert size > 1000  # At least 1KB
    assert size < 100 * 1024 * 1024  # Less than 100MB


def test_sample_files_paired() -> None:
    """Test that audio and transcript files are paired correctly."""
    audio_path = get_sample_audio()
    transcript_path = get_sample_transcript()

    # Base names should match (without extension)
    assert audio_path.stem == transcript_path.stem
    assert audio_path.stem == "kevin-alice-16k"


def test_sample_audio_readable() -> None:
    """Test that sample audio file exists and has correct format.

    Note: Actual audio reading is handled by C code (sphinx_fe), not Python.
    """
    audio_path = get_sample_audio()

    # Just verify it's a WAV file with reasonable size
    # The C code will handle actual audio processing
    assert audio_path.exists()
    assert audio_path.suffix == ".wav"
    size = audio_path.stat().st_size
    assert size > 0
