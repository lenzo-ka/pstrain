"""Pure-Python tests for project setup."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from pytest import MonkeyPatch

from st2.cli.cli import main
from st2.lib.dictionary import Dictionary
from st2.lib.phoneset import Phoneset
from st2.lib.pipeline.context import DEFAULT_CONFIGS, PipelineContext
from st2.lib.setup import setup_project

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
