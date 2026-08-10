"""Project setup implementation for ST2."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from st2.lib.config import ST2Config
from st2.lib.dictionary import Dictionary
from st2.lib.phoneset import Phoneset
from st2.lib.pipeline.context import DEFAULT_CONFIGS

__all__ = ["setup_project"]


def setup_project(
    project_dir: Path,
    transcription_path: Path | None = None,
    audio_path: Path | None = None,
    dictionary_path: Path | None = None,
    phoneset_path: Path | None = None,
    filler_dict_path: Path | None = None,
    config_path: Path | None = None,
    link_audio: bool = False,
    clobber: bool = False,
) -> dict[str, Any]:
    """Set up a new ST2 project.

    Args:
        project_dir: Project directory (create if needed)
        transcription_path: Path to transcription file
        audio_path: Path to audio directory or file (optional)
        dictionary_path: Path to dictionary file
        phoneset_path: Path to phoneset file (or extract from dictionary)
        filler_dict_path: Path to filler dictionary (optional)
        config_path: Path to config file (or create default)
        link_audio: If True and audio_path provided, symlink instead of copying
        clobber: If True, overwrite existing files; if False, skip existing files

    Returns:
        Dict with setup status and paths
    """
    project_dir = project_dir.resolve()

    # Create directory structure
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "etc").mkdir(exist_ok=True)
    (project_dir / "audio").mkdir(exist_ok=True)
    (project_dir / "shared").mkdir(exist_ok=True)
    (project_dir / "shared" / "features").mkdir(exist_ok=True)
    (project_dir / "experiments").mkdir(exist_ok=True)

    # Create or load configuration
    config_file = project_dir / "etc" / "config.yaml"
    if config_path and config_path.exists():
        if clobber or not config_file.exists():
            config = ST2Config.from_yaml(config_path)
            config.to_yaml(config_file)
    elif clobber or not config_file.exists():
        project_name = project_dir.name
        config = ST2Config(name=project_name)
        config.bind_to_project(project_dir)
        config.to_yaml(config_file)

    configs_file = project_dir / "etc" / "configs.yaml"
    if clobber or not configs_file.exists():
        with open(configs_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(DEFAULT_CONFIGS, f, sort_keys=False)

    # Copy transcription file
    if transcription_path:
        transcription_path = Path(transcription_path).resolve()
        dest_transcription = project_dir / "etc" / "all.transcription"
        if transcription_path != dest_transcription:
            if clobber or not dest_transcription.exists():
                shutil.copy(transcription_path, dest_transcription)

    # Handle audio files
    audio_dir = project_dir / "audio"
    if audio_path:
        audio_path = Path(audio_path).resolve()

        if link_audio:
            # Symlink entire audio directory
            if clobber and audio_dir.exists():
                if audio_dir.is_symlink():
                    audio_dir.unlink()
                elif audio_dir.is_dir():
                    shutil.rmtree(audio_dir)
            if not audio_dir.exists():
                try:
                    audio_dir.symlink_to(audio_path)
                except OSError:
                    # Fall back to individual file symlinks if directory symlink fails
                    audio_dir.mkdir(exist_ok=True)
                    if audio_path.is_dir():
                        for audio_file in audio_path.glob("*.wav"):
                            link_path = audio_dir / audio_file.name
                            if clobber or not link_path.exists():
                                if link_path.exists():
                                    link_path.unlink()
                                link_path.symlink_to(audio_file)
        else:
            # Copy audio files
            if audio_path.is_dir():
                for audio_file in audio_path.glob("*.wav"):
                    dest_file = audio_dir / audio_file.name
                    if clobber or not dest_file.exists():
                        shutil.copy(audio_file, dest_file)
            else:
                dest_file = audio_dir / audio_path.name
                if clobber or not dest_file.exists():
                    shutil.copy(audio_path, dest_file)
    # If no audio_path, do nothing (directory already created above)

    # Copy dictionary
    if dictionary_path:
        dictionary_path = Path(dictionary_path).resolve()
        dest_dict = project_dir / "shared" / "dictionary.dict"
        if dictionary_path != dest_dict:
            if clobber or not dest_dict.exists():
                shutil.copy(dictionary_path, dest_dict)

    # Copy filler dictionary
    dest_filler = project_dir / "shared" / "filler.dict"
    if filler_dict_path:
        filler_dict_path = Path(filler_dict_path).resolve()
        if filler_dict_path != dest_filler:
            if clobber or not dest_filler.exists():
                shutil.copy(filler_dict_path, dest_filler)
    else:
        # Use default filler dictionary from package data
        if clobber or not dest_filler.exists():
            from st2.data import get_data_file

            default_filler = get_data_file("filler.dict")
            shutil.copy(default_filler, dest_filler)

    # Extract or copy phoneset after installing the filler dictionary so an
    # extracted inventory covers every trainable dictionary entry.
    dest_phoneset = project_dir / "shared" / "phoneset.txt"
    if phoneset_path:
        phoneset_path = Path(phoneset_path).resolve()
        if phoneset_path != dest_phoneset:
            if clobber or not dest_phoneset.exists():
                shutil.copy(phoneset_path, dest_phoneset)
    elif dictionary_path:
        dict_file = project_dir / "shared" / "dictionary.dict"
        if dict_file.exists() and (clobber or not dest_phoneset.exists()):
            phones = Dictionary.from_file(dict_file).phonemes()
            if dest_filler.exists():
                phones.update(Dictionary.from_file(dest_filler).phonemes())
            Phoneset(phones).to_file(dest_phoneset)

    return {
        "project_dir": str(project_dir),
        "config_file": str(project_dir / "etc" / "config.yaml"),
        "configs_file": str(project_dir / "etc" / "configs.yaml"),
        "transcription_file": str(project_dir / "etc" / "all.transcription"),
        "dictionary_file": str(project_dir / "shared" / "dictionary.dict"),
        "phoneset_file": str(project_dir / "shared" / "phoneset.txt"),
        "audio_dir": str(project_dir / "audio"),
    }
