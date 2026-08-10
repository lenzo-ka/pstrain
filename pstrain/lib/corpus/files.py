"""I/O utilities for file operations."""

from __future__ import annotations

import csv
import re
from pathlib import Path


def load_filelist(path: Path) -> list[str]:
    """Load filelist, skipping empty lines.

    Args:
        path: Path to filelist file

    Returns:
        List of file IDs (one per line, stripped)
    """
    if not path.exists():
        return []
    try:
        # Read line by line instead of loading entire file into memory
        result = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    result.append(stripped)
        return result
    except FileNotFoundError:
        # File was deleted between exists() check and read
        return []


def save_filelist(path: Path, filelist: list[str]) -> None:
    """Save filelist to file.

    Args:
        path: Path to output file
        filelist: List of file IDs to save
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for file_id in filelist:
            f.write(f"{file_id}\n")


def extract_vocabulary(transcript_files: list[Path]) -> set[str]:
    """Extract unique vocabulary from transcript files.

    Args:
        transcript_files: Paths to .transcription files

    Returns:
        Set of unique words (case-sensitive)

    Note:
        Expects Sphinx transcript format: <s> text </s> (utt_id)
    """
    vocabulary: set[str] = set()

    for transcript_file in transcript_files:
        if not transcript_file.exists():
            continue

        with transcript_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Match: <s> text </s> (utt_id)
                match = re.match(r"<s>\s+(.*?)\s+</s>\s+\(([^)]+)\)", line)
                if match:
                    text = match.group(1)
                    vocabulary.update(text.split())

    return vocabulary


# =============================================================================
# Transcript format conversion
# =============================================================================


def detect_transcript_format(path: Path) -> str:
    """Detect transcript file format.

    Args:
        path: Path to transcript file

    Returns:
        Format name: "sphinx", "csv", "tsv", or "unknown"
    """
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Sphinx format: <s> text </s> (utt_id)
            if line.startswith("<s>") and "</s>" in line:
                return "sphinx"

            # TSV: fileid\ttext
            if "\t" in line:
                return "tsv"

            # CSV: fileid,text (but be careful with commas in text)
            if "," in line:
                return "csv"

            break

    return "unknown"


def convert_to_sphinx(
    input_path: Path,
    output_path: Path,
    input_format: str | None = None,
) -> int:
    """Convert transcript file to Sphinx format.

    Supported input formats:
    - sphinx: Already in format, just copy/validate
    - csv: fileid,text (first column is fileid, rest is text)
    - tsv: fileid<TAB>text

    Output format: <s> text </s> (fileid)

    Args:
        input_path: Input transcript file
        output_path: Output Sphinx-format transcript
        input_format: Format name, or None to auto-detect

    Returns:
        Number of utterances converted
    """
    if input_format is None:
        input_format = detect_transcript_format(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_format == "sphinx":
        return _convert_sphinx_to_sphinx(input_path, output_path)
    if input_format == "tsv":
        return _convert_tsv_to_sphinx(input_path, output_path)
    if input_format == "csv":
        return _convert_csv_to_sphinx(input_path, output_path)
    raise ValueError(f"Unknown transcript format: {input_format}")


def _convert_sphinx_to_sphinx(input_path: Path, output_path: Path) -> int:
    """Copy/validate Sphinx format transcript."""
    count = 0
    with (
        input_path.open(encoding="utf-8") as f_in,
        output_path.open("w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Validate format
            if re.match(r"<s>\s+.*\s+</s>\s+\([^)]+\)", line):
                f_out.write(line + "\n")
                count += 1

    return count


def _convert_tsv_to_sphinx(input_path: Path, output_path: Path) -> int:
    """Convert TSV (fileid<TAB>text) to Sphinx format."""
    count = 0
    with (
        input_path.open(encoding="utf-8") as f_in,
        output_path.open("w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t", 1)
            if len(parts) == 2:
                fileid, text = parts
                f_out.write(f"<s> {text.strip()} </s> ({fileid.strip()})\n")
                count += 1

    return count


def _convert_csv_to_sphinx(input_path: Path, output_path: Path) -> int:
    """Convert CSV (fileid,text) to Sphinx format."""
    count = 0
    with (
        input_path.open(encoding="utf-8", newline="") as f_in,
        output_path.open("w", encoding="utf-8") as f_out,
    ):
        reader = csv.reader(f_in)
        for row in reader:
            if len(row) >= 2:
                fileid = row[0].strip()
                # Join remaining columns in case text had commas
                text = ",".join(row[1:]).strip()
                if fileid and text:
                    f_out.write(f"<s> {text} </s> ({fileid})\n")
                    count += 1

    return count
