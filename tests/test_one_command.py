"""Tests for the one-command training input contract and destination semantics."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from pstrain.cli.cli import main
from pstrain.lib.one_command import (
    PromptFormatError,
    detect_prompt_format,
    input_identity,
    installed_corpus_identity,
    parse_prompts,
    validate_inputs,
    write_validation_reports,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mini_arctic"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("utt1 HELLO WORLD\n", "leading-id"),
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


@pytest.mark.parametrize(
    ("content", "explicit", "expected"),
    [
        ("<s> HELLO WORLD </s> (utt1)\n", "sphinx", ("utt1", "HELLO WORLD")),
        ("<s> HELLO </s>\n", "leading-id", ("<s>", "HELLO </s>")),
    ],
)
def test_sphinx_tokens_require_explicit_format(
    tmp_path: Path, content: str, explicit: str, expected: tuple[str, str]
) -> None:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(content)
    with pytest.raises(PromptFormatError, match="Ambiguous"):
        detect_prompt_format(prompts)
    selected, parsed = parse_prompts(prompts, explicit)
    assert selected == explicit
    assert (parsed[0].fileid, parsed[0].text) == expected


def test_prompt_bom_is_rejected_explicitly(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("\ufeffutt1 HELLO\n")
    with pytest.raises(PromptFormatError, match="BOM"):
        parse_prompts(prompts, "leading-id")


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


def _seed_project(project: Path) -> None:
    from pstrain.lib.setup import setup_project

    setup_project(
        project,
        transcription_path=FIXTURE / "transcription.txt",
        audio_path=FIXTURE / "wav",
        dictionary_path=FIXTURE / "dictionary.dict",
        phoneset_path=FIXTURE / "phoneset.txt",
        filler_dict_path=FIXTURE / "filler.dict",
    )
    source = input_identity(
        FIXTURE / "wav",
        FIXTURE / "transcription.txt",
        FIXTURE / "dictionary.dict",
        FIXTURE / "filler.dict",
        FIXTURE / "phoneset.txt",
    )
    installed = installed_corpus_identity(project, audio_ownership="copy")
    manifest = {"version": 2, "source": source, "installed": installed}
    (project / "etc" / "input-identity.json").write_text(json.dumps(manifest))


def _seed_link_project(project: Path) -> None:
    from pstrain.lib.setup import setup_project

    setup_project(
        project,
        transcription_path=FIXTURE / "transcription.txt",
        audio_path=FIXTURE / "wav",
        dictionary_path=FIXTURE / "dictionary.dict",
        phoneset_path=FIXTURE / "phoneset.txt",
        filler_dict_path=FIXTURE / "filler.dict",
        link_audio=True,
    )
    source = input_identity(
        FIXTURE / "wav",
        FIXTURE / "transcription.txt",
        FIXTURE / "dictionary.dict",
        FIXTURE / "filler.dict",
        FIXTURE / "phoneset.txt",
        True,
    )
    installed = installed_corpus_identity(project, audio_ownership="link")
    manifest = {"version": 2, "source": source, "installed": installed}
    (project / "etc" / "input-identity.json").write_text(json.dumps(manifest))


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
    _seed_project(project)
    assert _invoke(monkeypatch, *args, "--resume", "--dry-run") == 0

    changed = tmp_path / "changed.txt"
    shutil.copyfile(FIXTURE / "transcription.txt", changed)
    changed.write_text(changed.read_text() + "\n")
    changed_args = _base_arguments(project, changed)
    assert _invoke(monkeypatch, *changed_args, "--resume", "--dry-run") == 1
    assert _invoke(monkeypatch, *changed_args, "--replace-inputs", "--dry-run") == 0


def test_resume_authenticates_installed_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    _seed_project(project)
    changed = project / "audio" / "arctic_a0001.wav"
    changed.write_bytes(changed.read_bytes() + b"changed")
    assert _invoke(monkeypatch, *_base_arguments(project), "--resume", "--dry-run") == 1
    assert "audio/arctic_a0001.wav" in capsys.readouterr().err


def test_resume_refuses_repointed_installed_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    _seed_project(project)
    installed = project / "audio" / "arctic_a0001.wav"
    installed.unlink()
    installed.symlink_to(FIXTURE / "wav" / "arctic_a0002.wav")
    assert _invoke(monkeypatch, *_base_arguments(project), "--resume", "--dry-run") == 1
    error = capsys.readouterr().err
    assert "Unexpected symlink" in error
    assert "arctic_a0001.wav" in error


@pytest.mark.parametrize(
    "relative",
    ["shared/dictionary.dict", "etc/all.transcription", "audio/arctic_a0001.wav"],
)
def test_replace_refuses_installed_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    relative: str,
) -> None:
    project = tmp_path / "project"
    _seed_project(project)
    installed = project / relative
    external = tmp_path / f"external-{installed.name}"
    external.write_bytes(installed.read_bytes())
    installed.unlink()
    installed.symlink_to(external)
    assert _invoke(monkeypatch, *_base_arguments(project), "--replace-inputs") == 1
    error = capsys.readouterr().err
    assert "containing symlink" in error
    assert relative in error
    assert external.read_bytes()


def test_replace_refuses_project_root_symlink_without_touching_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    external = tmp_path / "external" / "project"
    _seed_project(external)
    marker = external / "untouched.marker"
    marker.write_text("untouched")
    alias = tmp_path / "project"
    alias.symlink_to(external, target_is_directory=True)

    assert _invoke(monkeypatch, *_base_arguments(alias), "--replace-inputs") == 1
    assert "project path that is a symlink" in capsys.readouterr().err
    assert marker.read_text() == "untouched"
    assert alias.is_symlink()


def test_link_audio_project_can_be_replaced_only_in_link_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import pstrain.cli.train as train_module

    project = tmp_path / "project"
    _seed_link_project(project)
    assert _invoke(monkeypatch, *_base_arguments(project), "--replace-inputs") == 1
    error = capsys.readouterr().err
    assert "Project is in link-mode" in error
    assert "--link-audio" in error

    class SuccessfulPipeline:
        def run(self, *args: object, **kwargs: object) -> int:
            return 0

    monkeypatch.setattr(train_module, "build_pipeline", lambda context: SuccessfulPipeline())
    assert _invoke(monkeypatch, *_base_arguments(project), "--replace-inputs", "--link-audio") == 0
    assert (project / "audio").is_symlink()


@pytest.mark.parametrize("complete_staging", [False, True])
def test_interrupted_swap_is_recovered_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete_staging: bool
) -> None:
    project = tmp_path / "project"
    _seed_project(project)
    (project / "generation").write_text("old")
    backup = tmp_path / ".project.previous-abcdefgh"
    project.rename(backup)
    staging = tmp_path / ".project.replacement-abcdefgh"
    if complete_staging:
        _seed_project(staging)
        (staging / "etc" / "prompts.source").write_text("complete")
        (staging / "generation").write_text("new")
    journal = tmp_path / ".project.pstrain-swap.json"
    journal.write_text(
        json.dumps(
            {
                "version": 1,
                "project": "project",
                "staging": staging.name,
                "backup": backup.name,
            }
        )
    )

    assert _invoke(monkeypatch, *_base_arguments(project), "--resume", "--dry-run") == 0
    assert (project / "generation").read_text() == ("new" if complete_staging else "old")
    assert not backup.exists()
    assert not staging.exists()
    assert not journal.exists()


@pytest.mark.parametrize(
    ("staging_name", "backup_name"),
    [("unrelated-staging", "unrelated-backup"), ("project", "project")],
)
def test_crafted_swap_journal_cannot_delete_unrelated_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    staging_name: str,
    backup_name: str,
) -> None:
    project = tmp_path / "project"
    _seed_project(project)
    (project / "untouched.marker").write_text("project")
    unrelated_staging = tmp_path / "unrelated-staging"
    unrelated_backup = tmp_path / "unrelated-backup"
    unrelated_staging.mkdir()
    unrelated_backup.mkdir()
    (unrelated_staging / "untouched.marker").write_text("staging")
    (unrelated_backup / "untouched.marker").write_text("backup")
    journal = tmp_path / ".project.pstrain-swap.json"
    journal.write_text(
        json.dumps(
            {
                "version": 1,
                "project": "project",
                "staging": staging_name,
                "backup": backup_name,
            }
        )
    )

    assert _invoke(monkeypatch, *_base_arguments(project), "--resume", "--dry-run", "--json") == 1
    result = json.loads(capsys.readouterr().out)
    assert result["code"] == "swap_recovery_failed"
    assert (project / "untouched.marker").read_text() == "project"
    assert (unrelated_staging / "untouched.marker").read_text() == "staging"
    assert (unrelated_backup / "untouched.marker").read_text() == "backup"
    assert journal.exists()


def test_interrupted_swap_restores_backup_when_staging_manifest_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    _seed_project(project)
    (project / "generation").write_text("old")
    backup = tmp_path / ".project.previous-1234abcd"
    project.rename(backup)
    staging = tmp_path / ".project.replacement-1234abcd"
    (staging / "etc").mkdir(parents=True)
    (staging / "etc" / "input-identity.json").write_text("{}")
    journal = tmp_path / ".project.pstrain-swap.json"
    journal.write_text(
        json.dumps(
            {
                "version": 1,
                "project": "project",
                "staging": staging.name,
                "backup": backup.name,
            }
        )
    )

    assert _invoke(monkeypatch, *_base_arguments(project), "--resume", "--dry-run") == 0
    assert (project / "generation").read_text() == "old"
    assert not staging.exists()
    assert not backup.exists()
    assert not journal.exists()


def test_failed_replacement_leaves_original_project_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pstrain.cli.train as train_module

    project = tmp_path / "project"
    _seed_project(project)
    marker = project / "original.marker"
    marker.write_text("intact")

    def fail_setup(*args: object, **kwargs: object) -> None:
        target = Path(str(kwargs["project_dir"]))
        (target / "partial").write_text("partial")
        raise ValueError("injected setup failure")

    monkeypatch.setattr(train_module, "setup_project", fail_setup)
    assert _invoke(monkeypatch, *_base_arguments(project), "--replace-inputs") == 1
    assert marker.read_text() == "intact"
    assert not (project / "partial").exists()


def test_oov_report_lists_every_utterance_and_accepts_filler_words(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.txt"
    rows = [f"utt{number} UNKNOWN" for number in range(7)]
    rows.append("utt7 <sil>")
    prompts.write_text("\n".join(rows) + "\n")
    audio = tmp_path / "audio"
    audio.mkdir()
    source_wav = FIXTURE / "wav" / "arctic_a0001.wav"
    for number in range(8):
        shutil.copyfile(source_wav, audio / f"utt{number}.wav")
    report, _ = validate_inputs(
        audio,
        prompts,
        FIXTURE / "dictionary.dict",
        "leading-id",
        FIXTURE / "phoneset.txt",
        FIXTURE / "filler.dict",
    )
    assert "<sil>" not in report.oov
    assert report.oov["UNKNOWN"]["examples"] == [f"utt{number}" for number in range(7)]
    _, oov_path = write_validation_reports(tmp_path / "project", report)
    assert oov_path.read_text().splitlines()[1].endswith("utt0,utt1,utt2,utt3,utt4,utt5,utt6")


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
