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


def test_setup_clobber_link_rejects_source_inside_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "audio"
    source.mkdir(parents=True)
    wav = source / "keep.wav"
    wav.write_bytes(b"keep")

    with pytest.raises(ValueError, match="inside the project directory"):
        setup_project(project, audio_path=source, link_audio=True, clobber=True)

    assert source.is_dir()
    assert not source.is_symlink()
    assert wav.read_bytes() == b"keep"
