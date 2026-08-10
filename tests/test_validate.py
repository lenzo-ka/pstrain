"""Tests for project validation."""

from pathlib import Path

from st2.lib.validate import validate_project


def test_nested_audio_reports_transcription_fileid_mismatch(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "audio" / "spk1").mkdir(parents=True)
    (project / "audio" / "spk1" / "utt1.wav").touch()
    (project / "shared").mkdir()
    etc = project / "experiments" / "default" / "etc"
    etc.mkdir(parents=True)
    (etc / "train.transcription").write_text("utt1 HELLO\n")

    report = validate_project(project)

    assert report.audio_files == 0
    assert report.missing_audio == ["utt1"]
    assert any(
        "Audio files not referenced by any transcription fileid: 1 (e.g., spk1/utt1.wav)" in warning
        for warning in report.warnings
    )


def test_nested_audio_matches_relative_transcription_fileid(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "audio" / "spk1").mkdir(parents=True)
    (project / "audio" / "spk1" / "utt1.wav").touch()
    (project / "shared").mkdir()
    etc = project / "experiments" / "default" / "etc"
    etc.mkdir(parents=True)
    (etc / "train.transcription").write_text("spk1/utt1 HELLO\n")

    report = validate_project(project)

    assert report.audio_files == 1
    assert report.missing_audio == []
    assert not any("Audio files not referenced" in warning for warning in report.warnings)
