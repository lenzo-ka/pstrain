"""Project validation for ST2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from st2.lib.dictionary import Dictionary
from st2.lib.phoneset import Phoneset
from st2.lib.transcription import get_fileids, parse_transcription_file

__all__ = ["ValidationError", "ValidationReport", "validate_project", "validate_files_exist"]


class ValidationError(Exception):
    """Validation error."""

    pass


def validate_files_exist(files: list[Path], context: str = "") -> None:
    """Validate that all files in a list exist.

    Args:
        files: List of file paths to check
        context: Optional context string for error messages

    Raises:
        FileNotFoundError: If any file does not exist
    """
    for f in files:
        if not f.exists():
            msg = f"Required file not found: {f}"
            if context:
                msg += f" ({context})"
            raise FileNotFoundError(msg)


@dataclass
class ValidationReport:
    """Validation report with errors and stats."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Corpus stats
    train_utterances: int = 0
    test_utterances: int = 0
    dev_utterances: int = 0
    total_utterances: int = 0

    # Vocabulary stats
    vocabulary_size: int = 0
    missing_words: list[str] = field(default_factory=list)

    # Dictionary stats
    dictionary_entries: int = 0
    dictionary_base_words: int = 0

    # Phoneset stats
    phoneset_size: int = 0
    has_silence: bool = False
    missing_phones: list[str] = field(default_factory=list)

    # Audio stats
    audio_files: int = 0
    missing_audio: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no errors."""
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary (JSON-serializable)."""
        from dataclasses import asdict

        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: Path) -> None:
        """Save report as JSON file."""
        path.write_text(self.to_json())

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("ST2 Project Validation Report")
        lines.append("=" * 60)

        # Corpus
        lines.append("\nCorpus:")
        lines.append(f"  Train utterances:  {self.train_utterances}")
        lines.append(f"  Test utterances:   {self.test_utterances}")
        if self.dev_utterances:
            lines.append(f"  Dev utterances:    {self.dev_utterances}")
        lines.append(f"  Total:             {self.total_utterances}")

        # Vocabulary
        lines.append("\nVocabulary:")
        lines.append(f"  Unique words:      {self.vocabulary_size}")
        if self.missing_words:
            lines.append(f"  Missing from dict: {len(self.missing_words)}")

        # Dictionary
        lines.append("\nDictionary:")
        lines.append(f"  Entries:           {self.dictionary_entries}")
        lines.append(f"  Base words:        {self.dictionary_base_words}")

        # Phoneset
        lines.append("\nPhoneset:")
        lines.append(f"  Phones:            {self.phoneset_size}")
        lines.append(f"  Has silence (SIL): {self.has_silence}")
        if self.missing_phones:
            lines.append(f"  Missing phones:    {len(self.missing_phones)}")

        # Audio
        lines.append("\nAudio:")
        lines.append(f"  Files found:       {self.audio_files}")
        if self.missing_audio:
            lines.append(f"  Missing files:     {len(self.missing_audio)}")

        # Errors
        if self.errors:
            lines.append(f"\n{'=' * 60}")
            lines.append(f"ERRORS ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"  ✗ {err}")

        # Warnings
        if self.warnings:
            lines.append(f"\nWARNINGS ({len(self.warnings)}):")
            for warn in self.warnings:
                lines.append(f"  ⚠ {warn}")

        # Status
        lines.append(f"\n{'=' * 60}")
        if self.is_valid:
            lines.append("✓ Project is VALID")
        else:
            lines.append("✗ Project has ERRORS")

        return "\n".join(lines)


def validate_project(project_dir: Path, experiment: str = "default") -> ValidationReport:
    """Validate project structure and files, return detailed report.

    Args:
        project_dir: Project directory to validate
        experiment: Experiment name (default: "default")

    Returns:
        ValidationReport with errors, warnings, and stats
    """
    report = ValidationReport()
    project_dir = Path(project_dir)
    experiment_dir = project_dir / "experiments" / experiment

    # Check required directories
    for dir_name in ["audio", "shared"]:
        if not (project_dir / dir_name).exists():
            report.errors.append(f"Missing directory: {dir_name}/")

    if not experiment_dir.exists():
        report.errors.append(f"Missing experiment directory: experiments/{experiment}/")

    # Check shared files
    dict_file = project_dir / "shared" / "dictionary.dict"
    phoneset_file = project_dir / "shared" / "phoneset.txt"

    if not dict_file.exists():
        report.errors.append("Missing: shared/dictionary.dict")
    if not phoneset_file.exists():
        report.errors.append("Missing: shared/phoneset.txt")

    # Load dictionary
    dictionary = None
    if dict_file.exists():
        try:
            dictionary = Dictionary.from_file(dict_file)
            report.dictionary_entries = len(dictionary)
            report.dictionary_base_words = len(dictionary.base_words())
        except Exception as e:
            report.errors.append(f"Error loading dictionary: {e}")

    # Load phoneset
    phoneset = None
    if phoneset_file.exists():
        try:
            phoneset = Phoneset.from_file(phoneset_file)
            report.phoneset_size = len(phoneset)
            report.has_silence = phoneset.has_sil()
        except Exception as e:
            report.errors.append(f"Error loading phoneset: {e}")

    # Validate phoneset vs dictionary
    if dictionary and phoneset:
        is_valid, phones_missing = phoneset.validate_dictionary(dictionary)
        if not is_valid:
            report.missing_phones = sorted(phones_missing)
            report.errors.append(
                f"Phones in dictionary not in phoneset: {len(phones_missing)} "
                f"(e.g., {', '.join(sorted(phones_missing)[:5])})"
            )

    # Check transcripts and collect vocabulary
    all_vocab: set[str] = set()
    split_artifacts = [
        experiment_dir / "etc" / name
        for name in [
            "train.fileids",
            "test.fileids",
            "train.transcription",
            "test.transcription",
        ]
    ]
    present_split_artifacts = [path for path in split_artifacts if path.exists()]
    if not present_split_artifacts:
        if (project_dir / "etc" / "all.transcription").exists():
            report.warnings.append("Split outputs are missing; run st2 split")
        else:
            report.errors.append("no transcription; project has no corpus to train on")
    elif len(present_split_artifacts) != len(split_artifacts):
        for path in split_artifacts:
            if not path.exists():
                report.errors.append(f"Missing split output: {path.name}")

    split_files = {
        split: experiment_dir / "etc" / f"{split}.transcription"
        for split in ["train", "test", "dev"]
    }
    for split, trans_file in split_files.items():
        if trans_file.exists():
            try:
                transcripts = parse_transcription_file(trans_file)
                count = len(transcripts)

                if split == "train":
                    report.train_utterances = count
                elif split == "test":
                    report.test_utterances = count
                elif split == "dev":
                    report.dev_utterances = count

                # Collect vocabulary
                for text in transcripts.values():
                    all_vocab.update(text.split())
            except Exception as e:
                report.errors.append(f"Error reading {split}.transcription: {e}")

    report.total_utterances = (
        report.train_utterances + report.test_utterances + report.dev_utterances
    )
    report.vocabulary_size = len(all_vocab)

    # Check dictionary coverage
    if dictionary and all_vocab:
        dictionary_words = dictionary.words()
        words_missing = [w for w in all_vocab if w not in dictionary_words]
        if words_missing:
            report.missing_words = sorted(set(words_missing))
            report.errors.append(
                f"Words not in dictionary: {len(report.missing_words)} "
                f"(e.g., {', '.join(report.missing_words[:5])})"
            )

    # Check audio files
    audio_dir = project_dir / "audio"
    if audio_dir.exists():
        discovered_fileids = {
            path.relative_to(audio_dir).with_suffix("").as_posix()
            for path in audio_dir.rglob("*.wav")
        }

        # Get all expected fileids from transcripts
        all_fileids: list[str] = []
        for split in ["train", "test", "dev"]:
            trans_file = experiment_dir / "etc" / f"{split}.transcription"
            if trans_file.exists():
                try:
                    all_fileids.extend(get_fileids(trans_file))
                except Exception:
                    pass

        # Check which audio files exist
        found = 0
        audio_missing: list[str] = []
        for fileid in all_fileids:
            audio_file = audio_dir / f"{fileid}.wav"
            if audio_file.exists():
                found += 1
            else:
                audio_missing.append(fileid)

        report.audio_files = found
        if audio_missing:
            report.missing_audio = audio_missing[:20]  # Limit list size
            report.errors.append(
                f"Missing audio files: {len(audio_missing)} (e.g., {audio_missing[0]}.wav)"
            )

        unreachable = sorted(discovered_fileids - set(all_fileids))
        if unreachable:
            report.warnings.append(
                f"Audio files not referenced by any transcription fileid: "
                f"{len(unreachable)} (e.g., {unreachable[0]}.wav). "
                "For nested audio, fileids must include relative directory components."
            )

    return report
