"""Step 00: Project setup.

Creates project structure, copies/validates dictionary, phoneset, config, etc.

Note: The actual execution order is determined by file dependencies
(inputs/outputs), not by the step number. The pipeline runner computes the
partial order from the dependency graph.

Can be used as:
1. Library module: from pstrain.lib.steps.setup import step_00_setup, run_step_00
2. Runnable script: python -m pstrain.lib.steps.setup [args]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pstrain.lib.setup import setup_project


def step_00_setup(
    project_dir: Path | str,
    transcription_path: Path | str | None = None,
    audio_path: Path | str | None = None,
    dictionary_path: Path | str | None = None,
    phoneset_path: Path | str | None = None,
    filler_dict_path: Path | str | None = None,
    config_path: Path | str | None = None,
    link_audio: bool = False,
) -> dict[str, Any]:
    """Get rule definition for stage 00: project setup.

    The execution order is determined by the inputs/outputs, not by the step
    number. The pipeline runner will automatically figure out that other
    steps depend on this one based on the output file paths (dictionary,
    phoneset, config, etc.).

    Args:
        project_dir: Project directory path
        transcription_path: Path to transcription file (input)
        audio_path: Path to audio directory (input, optional)
        dictionary_path: Path to dictionary file (input)
        phoneset_path: Path to phoneset file (input, optional - can extract from dict)
        filler_dict_path: Path to filler dictionary (input, optional)
        config_path: Path to config file (input, optional - will create default)
        link_audio: If True, symlink audio instead of copying

    Returns:
        Dictionary with rule definition:
        - inputs: list of input file paths (external files needed)
        - outputs: list of output file paths (dictionary, phoneset, config, etc.)
        - params: dict of parameters
        - script: "setup" (Python function)
        - number: reference number (for documentation, not execution order)
    """
    project_dir = Path(project_dir)

    # Inputs: external files needed for setup
    inputs = []
    if transcription_path:
        inputs.append(str(transcription_path))
    if audio_path:
        # Audio can be a directory or a single file
        audio_path_obj = Path(audio_path)
        if audio_path_obj.is_dir():
            # For directories, we'll need to track the directory itself
            inputs.append(str(audio_path_obj))
        else:
            inputs.append(str(audio_path_obj))
    if dictionary_path:
        inputs.append(str(dictionary_path))
    if phoneset_path:
        inputs.append(str(phoneset_path))
    if filler_dict_path:
        inputs.append(str(filler_dict_path))
    if config_path:
        inputs.append(str(config_path))

    # Outputs: files created by setup that other steps depend on
    outputs = [
        str(project_dir / "shared" / "dictionary.dict"),
        str(project_dir / "shared" / "phoneset.txt"),
        str(project_dir / "shared" / "filler.dict"),
        str(project_dir / "etc" / "config.yaml"),
        str(project_dir / "etc" / "all.transcription"),  # If transcription is provided
    ]

    params = {
        "transcription_path": str(transcription_path) if transcription_path else None,
        "audio_path": str(audio_path) if audio_path else None,
        "dictionary_path": str(dictionary_path) if dictionary_path else None,
        "phoneset_path": str(phoneset_path) if phoneset_path else None,
        "filler_dict_path": str(filler_dict_path) if filler_dict_path else None,
        "config_path": str(config_path) if config_path else None,
        "link_audio": link_audio,
    }

    return {
        "name": "setup",
        "script": "setup",  # Python function, not C program
        "inputs": inputs,  # The runner ensures these exist before this rule runs
        "outputs": outputs,  # The runner tracks these - other rules can depend on them
        "params": params,
        "description": "Set up project structure, dictionary, phoneset, and config",
        # The runner automatically computes partial order from inputs/outputs
    }


def run_step_00(
    project_dir: Path | str,
    transcription_path: Path | str | None = None,
    audio_path: Path | str | None = None,
    dictionary_path: Path | str | None = None,
    phoneset_path: Path | str | None = None,
    filler_dict_path: Path | str | None = None,
    config_path: Path | str | None = None,
    link_audio: bool = False,
    clobber: bool = False,
) -> int:
    """Execute stage 00: project setup.

    Creates project directory structure, copies/validates input files,
    and prepares the project for training.

    Args:
        project_dir: Project directory path
        transcription_path: Path to transcription file (optional)
        audio_path: Path to audio directory or file (optional)
        dictionary_path: Path to dictionary file (optional)
        phoneset_path: Path to phoneset file (optional, can extract from dictionary)
        filler_dict_path: Path to filler dictionary (optional)
        config_path: Path to config file (optional, will create default)
        link_audio: If True, symlink audio files instead of copying
        clobber: If True, overwrite existing files

    Returns:
        Exit code: 0 for success, 1 for warnings/errors

    Raises:
        FileNotFoundError: If required input files are missing
        OSError: If directories cannot be created or files cannot be copied
    """
    project_dir = Path(project_dir)

    # Convert Path | str to Path objects
    transcription: Path | None = Path(transcription_path) if transcription_path else None
    audio: Path | None = Path(audio_path) if audio_path else None
    dictionary: Path | None = Path(dictionary_path) if dictionary_path else None
    phoneset: Path | None = Path(phoneset_path) if phoneset_path else None
    filler_dict: Path | None = Path(filler_dict_path) if filler_dict_path else None
    config: Path | None = Path(config_path) if config_path else None

    try:
        result = setup_project(
            project_dir=project_dir,
            transcription_path=transcription,
            audio_path=audio,
            dictionary_path=dictionary,
            phoneset_path=phoneset,
            filler_dict_path=filler_dict,
            config_path=config,
            link_audio=link_audio,
            clobber=clobber,
        )
        if result.get("errors"):
            print(f"Setup completed with warnings: {result['errors']}", file=sys.stderr)
            return 1
        return 0
    except Exception as e:
        print(f"Setup failed: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point for the setup step script."""
    parser = argparse.ArgumentParser(description="Step 00: Project setup")
    parser.add_argument(
        "--project-dir",
        type=str,
        default=Path.cwd(),
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--transcription-path",
        type=str,
        help="Path to transcription file",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        help="Path to audio directory or file",
    )
    parser.add_argument(
        "--dictionary-path",
        type=str,
        help="Path to dictionary file",
    )
    parser.add_argument(
        "--phoneset-path",
        type=str,
        help="Path to phoneset file (optional - can extract from dictionary)",
    )
    parser.add_argument(
        "--filler-dict-path",
        type=str,
        dest="filler_dict_path",
        help="Path to filler dictionary (optional)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to config file (optional - will create default)",
    )
    parser.add_argument(
        "--link-audio",
        action="store_true",
        help="Symlink audio files instead of copying",
    )
    parser.add_argument(
        "--clobber",
        action="store_true",
        help="Overwrite existing files",
    )

    args = parser.parse_args()

    # Convert args to function parameters
    project_dir = Path(args.project_dir) if hasattr(args, "project_dir") else Path.cwd()
    transcription_path = (
        Path(args.transcription_path)
        if hasattr(args, "transcription_path") and args.transcription_path
        else None
    )
    audio_path = Path(args.audio_path) if hasattr(args, "audio_path") and args.audio_path else None
    dictionary_path = (
        Path(args.dictionary_path)
        if hasattr(args, "dictionary_path") and args.dictionary_path
        else None
    )
    phoneset_path = (
        Path(args.phoneset_path) if hasattr(args, "phoneset_path") and args.phoneset_path else None
    )
    filler_dict_path = (
        Path(args.filler_dict_path)
        if hasattr(args, "filler_dict_path") and args.filler_dict_path
        else None
    )
    config_path = (
        Path(args.config_path) if hasattr(args, "config_path") and args.config_path else None
    )
    link_audio = getattr(args, "link_audio", False)
    clobber = getattr(args, "clobber", False)

    return run_step_00(
        project_dir=project_dir,
        transcription_path=transcription_path,
        audio_path=audio_path,
        dictionary_path=dictionary_path,
        phoneset_path=phoneset_path,
        filler_dict_path=filler_dict_path,
        config_path=config_path,
        link_audio=link_audio,
        clobber=clobber,
    )


if __name__ == "__main__":
    sys.exit(main())
