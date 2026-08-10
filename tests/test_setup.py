"""Pure-Python tests for project setup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from pytest import MonkeyPatch

from st2.cli.cli import main
from st2.lib.dictionary import Dictionary
from st2.lib.phoneset import Phoneset
from st2.lib.pipeline.context import DEFAULT_CONFIGS, PipelineContext
from st2.lib.setup import setup_project
from st2.lib.validate import validate_project

FIXTURE = Path(__file__).parent / "fixtures" / "mini_arctic"


def test_setup_cli_creates_project_with_wideband_config(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """The CLI delegates a complete setup to the library implementation."""
    project = tmp_path / "project"
    argv = [
        "st2",
        "setup",
        str(project),
        "--transcription",
        str(FIXTURE / "transcription.txt"),
        "--audio",
        str(FIXTURE / "wav"),
        "--dictionary",
        str(FIXTURE / "dictionary.dict"),
        "--filler-dict",
        str(FIXTURE / "filler.dict"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert main() == 0
    assert (project / "etc" / "all.transcription").is_file()
    assert (project / "shared" / "dictionary.dict").is_file()
    assert (project / "shared" / "phoneset.txt").is_file()
    assert (project / "shared" / "filler.dict").is_file()
    assert len(list((project / "audio").glob("*.wav"))) == 10
    assert (project / "etc" / "config.yaml").is_file()
    assert (project / "etc" / "configs.yaml").is_file()
    assert PipelineContext.from_config(project, config_name="wideband").feat.samprate == 16000


def test_setup_without_phoneset_covers_dictionary_and_filler_phones(tmp_path: Path) -> None:
    """Extracted phones include every regular and filler dictionary phone."""
    project = tmp_path / "project"
    setup_project(
        project,
        dictionary_path=FIXTURE / "dictionary.dict",
        filler_dict_path=FIXTURE / "filler.dict",
    )

    phones = Phoneset.from_file(project / "shared" / "phoneset.txt").phones()
    expected = Dictionary.from_file(project / "shared" / "dictionary.dict").phonemes()
    expected.update(Dictionary.from_file(project / "shared" / "filler.dict").phonemes())
    assert "SIL" in phones
    assert expected <= phones


def test_generated_config_profiles_all_load(tmp_path: Path) -> None:
    """Generated named profiles serialize defaults and all construct contexts."""
    project = tmp_path / "project"
    setup_project(project)

    with open(project / "etc" / "configs.yaml", encoding="utf-8") as f:
        generated = yaml.safe_load(f)
    assert generated == DEFAULT_CONFIGS
    for config_name in generated:
        context = PipelineContext.from_config(project, config_name=config_name)
        assert context.config_name == config_name


def test_setup_link_audio_on_fresh_project(tmp_path: Path) -> None:
    project = tmp_path / "project"

    setup_project(project, audio_path=FIXTURE / "wav", link_audio=True)

    assert (project / "audio").is_symlink()
    assert len(list((project / "audio").rglob("*.wav"))) == 10


def test_setup_copies_nested_audio_tree(tmp_path: Path) -> None:
    source = tmp_path / "corpus"
    (source / "spk1").mkdir(parents=True)
    (source / "spk2").mkdir()
    (source / "spk1" / "utt1.wav").write_bytes(b"one")
    (source / "spk2" / "utt2.wav").write_bytes(b"two")
    (source / "spk1" / "notes.txt").write_text("metadata")
    project = tmp_path / "project"

    setup_project(project, audio_path=source)

    assert len(list((project / "audio").rglob("*.wav"))) == 2
    assert (project / "audio" / "spk1" / "utt1.wav").read_bytes() == b"one"
    assert (project / "audio" / "spk2" / "utt2.wav").read_bytes() == b"two"
    assert (project / "audio" / "spk1" / "notes.txt").read_text() == "metadata"


def test_complete_fresh_setup_validates_before_split(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    project = tmp_path / "project"
    args = [
        "st2",
        "setup",
        str(project),
        "--transcription",
        str(FIXTURE / "transcription.txt"),
        "--audio",
        str(FIXTURE / "wav"),
        "--dictionary",
        str(FIXTURE / "dictionary.dict"),
        "--filler-dict",
        str(FIXTURE / "filler.dict"),
        "--validate",
    ]
    monkeypatch.setattr(sys, "argv", args)

    assert main() == 0
    report = validate_project(project)
    assert report.is_valid
    assert "Split outputs are missing; run st2 split" in report.warnings
    assert (project / "experiments" / "default" / "etc").is_dir()


def test_setup_dry_run_describes_link_and_creates_nothing(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    args = [
        "st2",
        "setup",
        str(project),
        "--audio",
        str(FIXTURE / "wav"),
        "--link",
        "--clobber",
        "--validate",
        "--dry-run",
    ]
    monkeypatch.setattr(sys, "argv", args)

    assert main() == 0
    output = capsys.readouterr().out
    assert f"Link audio: {(FIXTURE / 'wav').resolve()} -> {project / 'audio'}" in output
    assert "Clobber enabled" in output
    assert "Write default config.yaml" in output
    assert "Write configs.yaml profiles" in output
    assert "Validate project after setup" in output
    assert not project.exists()


def test_setup_missing_config_errors(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    missing = tmp_path / "missing.yaml"
    project = tmp_path / "project"

    with pytest.raises(FileNotFoundError, match=str(missing)):
        setup_project(project, config_path=missing)
    assert not project.exists()

    monkeypatch.setattr(
        sys, "argv", ["st2", "setup", str(project), "--config", str(missing)]
    )
    assert main() == 1
    assert not project.exists()


def test_setup_dry_run_missing_dictionary_errors_without_creating_project(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    missing = tmp_path / "missing.dict"
    project = tmp_path / "project"
    monkeypatch.setattr(
        sys,
        "argv",
        ["st2", "setup", str(project), "--dictionary", str(missing), "--dry-run"],
    )

    assert main() == 1
    assert not project.exists()


@pytest.mark.parametrize("dry_run", [False, True])
def test_setup_cli_rejects_transcription_directory_without_creating_project(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dry_run: bool,
) -> None:
    transcription_dir = tmp_path / "transcription"
    transcription_dir.mkdir()
    project = tmp_path / "project"
    argv = ["st2", "setup", str(project), "--transcription", str(transcription_dir)]
    if dry_run:
        argv.append("--dry-run")
    monkeypatch.setattr(sys, "argv", argv)

    assert main() == 1
    assert f"Transcription must be a file: {transcription_dir}" in capsys.readouterr().err
    assert not project.exists()


def test_setup_cli_accepts_audio_directory(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    audio_dir = tmp_path / "audio-source"
    audio_dir.mkdir()
    (audio_dir / "sample.wav").write_bytes(b"audio")
    project = tmp_path / "project"
    monkeypatch.setattr(
        sys, "argv", ["st2", "setup", str(project), "--audio", str(audio_dir)]
    )

    assert main() == 0
    assert (project / "audio" / "sample.wav").read_bytes() == b"audio"


def test_setup_clobber_link_rejects_source_inside_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "audio"
    source.mkdir(parents=True)
    wav = source / "keep.wav"
    wav.write_bytes(b"keep")

    with pytest.raises(ValueError, match="around the project directory"):
        setup_project(project, audio_path=source, link_audio=True, clobber=True)

    assert source.is_dir()
    assert not source.is_symlink()
    assert wav.read_bytes() == b"keep"


def test_setup_link_rejects_source_containing_project(tmp_path: Path) -> None:
    source = tmp_path / "corpus"
    project = source / "project"
    project.mkdir(parents=True)

    with pytest.raises(ValueError, match="around the project directory"):
        setup_project(project, audio_path=source, link_audio=True)

    assert not (project / "audio").exists()


def test_setup_copy_without_clobber_rejects_linked_project_audio(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    external_wav = external / "sample.wav"
    external_wav.write_bytes(b"external")
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "sample.wav").write_bytes(b"incoming")
    project = tmp_path / "project"
    setup_project(project, audio_path=external, link_audio=True)

    with pytest.raises(FileExistsError, match="Project audio is a link"):
        setup_project(project, audio_path=incoming)

    assert (project / "audio").is_symlink()
    assert external_wav.read_bytes() == b"external"


def test_setup_copy_with_clobber_replaces_link_without_touching_source(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    external_wav = external / "sample.wav"
    external_wav.write_bytes(b"external")
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "sample.wav").write_bytes(b"incoming")
    project = tmp_path / "project"
    setup_project(project, audio_path=external, link_audio=True)

    setup_project(project, audio_path=incoming, clobber=True)

    assert (project / "audio").is_dir()
    assert not (project / "audio").is_symlink()
    assert (project / "audio" / "sample.wav").read_bytes() == b"incoming"
    assert external_wav.read_bytes() == b"external"


def test_validate_corpusless_project_is_invalid(tmp_path: Path) -> None:
    project = tmp_path / "project"
    setup_project(project)

    report = validate_project(project)

    assert not report.is_valid
    assert "no transcription; project has no corpus to train on" in report.errors
    assert "Split outputs are missing; run st2 split" not in report.warnings


def test_validate_empty_transcription_is_invalid(tmp_path: Path) -> None:
    project = tmp_path / "project"
    setup_project(project)
    (project / "etc" / "all.transcription").write_text("")

    report = validate_project(project)

    assert not report.is_valid
    assert "no transcription; project has no corpus to train on" in report.errors
    assert "Split outputs are missing; run st2 split" not in report.warnings


def test_validate_partial_split_is_invalid(tmp_path: Path) -> None:
    project = tmp_path / "project"
    setup_project(project, transcription_path=FIXTURE / "transcription.txt")
    (project / "experiments" / "default" / "etc" / "train.fileids").write_text("")

    report = validate_project(project)

    assert not report.is_valid
    assert "Missing split output: test.fileids" in report.errors
    assert "Missing split output: train.transcription" in report.errors
    assert "Missing split output: test.transcription" in report.errors


def test_validate_reports_unique_dictionary_base_words(tmp_path: Path) -> None:
    project = tmp_path / "project"
    setup_project(project)
    (project / "shared" / "dictionary.dict").write_text(
        "word W ER D\nword(2) W AO R D\n"
    )
    (project / "shared" / "phoneset.txt").write_text("W\nER\nD\nAO\nR\n")

    report = validate_project(project)

    assert report.dictionary_entries == 2
    assert report.dictionary_base_words == 1
