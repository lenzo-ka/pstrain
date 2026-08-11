"""Tests for the one-command training input contract and destination semantics."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from pstrain.cli.cli import main
from pstrain.lib.one_command import PromptFormatError, detect_prompt_format, parse_prompts

FIXTURE = Path(__file__).parent / "fixtures" / "mini_arctic"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("utt1 HELLO WORLD\n", "leading-id"),
        ("<s> HELLO WORLD </s> (utt1)\n", "sphinx"),
        ("utt1\tHELLO WORLD\n", "tsv"),
        ('"utt1","HELLO, WORLD"\n', "csv"),
    ],
)
def test_unambiguous_prompt_format_detection(tmp_path: Path, content: str, expected: str) -> None:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(content)
    assert detect_prompt_format(prompts) == expected


def test_comma_requires_explicit_prompt_format(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("utt1,HELLO WORLD\n")
    with pytest.raises(PromptFormatError, match="ambiguous"):
        detect_prompt_format(prompts)
    selected, parsed = parse_prompts(prompts, "csv")
    assert selected == "csv"
    assert (parsed[0].fileid, parsed[0].text) == ("utt1", "HELLO WORLD")


def _invoke(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> int:
    monkeypatch.setattr(sys, "argv", ["pstrain", "train", *arguments])
    return main()


def _base_arguments(project: Path, prompts: Path | None = None) -> tuple[str, ...]:
    return (
        str(project),
        "--audio",
        str(FIXTURE / "wav"),
        "--prompts",
        str(prompts or FIXTURE / "transcription.txt"),
        "--dictionary",
        str(FIXTURE / "dictionary.dict"),
        "--phoneset",
        str(FIXTURE / "phoneset.txt"),
        "--filler-dict",
        str(FIXTURE / "filler.dict"),
        "-j",
        "1",
    )


def test_oov_blocks_before_setup_and_names_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prompts = tmp_path / "oov.txt"
    source = (FIXTURE / "transcription.txt").read_text()
    prompts.write_text(source.replace("author", "NOT_IN_THE_DICTIONARY", 1))
    project = tmp_path / "project"
    assert _invoke(monkeypatch, *_base_arguments(project, prompts)) == 1
    error = capsys.readouterr().err
    assert "OOV validation blocked training" in error
    assert str(project / "reports" / "oov.txt") in error
    assert not (project / "etc").exists()
    report = json.loads((project / "reports" / "input-validation.json").read_text())
    assert "NOT_IN_THE_DICTIONARY" in report["oov"]


def test_resume_and_replace_input_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    args = _base_arguments(project)
    assert _invoke(monkeypatch, *args, "--dry-run") == 0
    assert not project.exists()

    # Seed a compatible destination without paying the model-training cost here.
    from pstrain.lib.one_command import input_identity
    from pstrain.lib.setup import setup_project

    setup_project(
        project,
        transcription_path=FIXTURE / "transcription.txt",
        audio_path=FIXTURE / "wav",
        dictionary_path=FIXTURE / "dictionary.dict",
        phoneset_path=FIXTURE / "phoneset.txt",
        filler_dict_path=FIXTURE / "filler.dict",
    )

    identity = input_identity(
        FIXTURE / "wav",
        FIXTURE / "transcription.txt",
        FIXTURE / "dictionary.dict",
        FIXTURE / "filler.dict",
        FIXTURE / "phoneset.txt",
    )
    (project / "etc" / "input-identity.json").write_text(json.dumps(identity))
    assert _invoke(monkeypatch, *args, "--resume", "--dry-run") == 0

    changed = tmp_path / "changed.txt"
    shutil.copyfile(FIXTURE / "transcription.txt", changed)
    changed.write_text(changed.read_text() + "\n")
    changed_args = _base_arguments(project, changed)
    assert _invoke(monkeypatch, *changed_args, "--resume", "--dry-run") == 1
    assert _invoke(monkeypatch, *changed_args, "--replace-inputs", "--dry-run") == 0


def test_one_command_trains_ci_1g_and_decodes_without_transcript_munging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    assert _invoke(monkeypatch, *_base_arguments(project)) == 0
    model = project / "shared" / "models" / "ci-1g" / "default" / "mdef"
    decoder_transcript = project / "experiments" / "default" / "etc" / "test.decoder.transcription"
    assert model.is_file()
    assert decoder_transcript.read_text().startswith("<s>")

    monkeypatch.setattr(
        sys,
        "argv",
        ["pstrain", "test", "ci-1g", "--project-dir", str(project), "--no-lm"],
    )
    assert main() == 0
