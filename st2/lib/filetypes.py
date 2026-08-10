"""File type detection for Sphinx/ST2 files.

Detects acoustic model files, feature files, and other artifacts
based on filename patterns and file content.
"""

from __future__ import annotations

import struct
from enum import Enum
from pathlib import Path

from st2.lib.model import MODEL_FILES_REQUIRED

__all__ = [
    "FileType",
    "detect_file_type",
    "describe_file",
    "is_sphinx_binary",
    "get_sphinx_header",
    "validate_file_type",
    "assert_file_type",
]


class FileType(Enum):
    """Known file types in ST2/Sphinx ecosystem."""

    # Feature files
    FEATURES = "features"  # .mfc files

    # Model parameter files
    MEANS = "means"
    VARIANCES = "variances"
    MIXTURE_WEIGHTS = "mixture_weights"
    TRANSITION_MATRICES = "transition_matrices"
    MDEF = "mdef"
    SENDUMP = "sendump"

    # Accumulator files
    GAUDEN_COUNTS = "gauden_counts"

    # Model directory
    MODEL = "model"

    # Dictionary files
    DICTIONARY = "dictionary"
    FILLER_DICT = "filler_dict"

    # Transcription/control files
    TRANSCRIPTION = "transcription"
    FILEIDS = "fileids"
    PHONESET = "phoneset"

    # Language model
    LM_ARPA = "lm_arpa"

    # Unknown
    UNKNOWN = "unknown"


# Filename patterns for detection
_FILENAME_PATTERNS: dict[str, FileType] = {
    "mdef": FileType.MDEF,
    "means": FileType.MEANS,
    "variances": FileType.VARIANCES,
    "mixture_weights": FileType.MIXTURE_WEIGHTS,
    "transition_matrices": FileType.TRANSITION_MATRICES,
    "sendump": FileType.SENDUMP,
    "global_mean": FileType.MEANS,
    "global_var": FileType.VARIANCES,
    "gauden_counts": FileType.GAUDEN_COUNTS,
}

_EXTENSION_PATTERNS: dict[str, FileType] = {
    ".mfc": FileType.FEATURES,
    ".dict": FileType.DICTIONARY,
    ".arpa": FileType.LM_ARPA,
    ".lm": FileType.LM_ARPA,
    ".transcription": FileType.TRANSCRIPTION,
    ".fileids": FileType.FILEIDS,
}


def get_sphinx_header(path: Path) -> tuple[str, str] | None:
    """Read Sphinx binary file header.

    Sphinx binary files start with a text header like:
    s3
    version 1.0
    chksum0 yes
    ...
    endhdr

    Returns:
        Tuple of (file_type_hint, version) or None if not a Sphinx file
    """
    try:
        with path.open("rb") as f:
            header = f.read(512)

        if not header.startswith(b"s3\n"):
            return None

        # Parse header lines
        lines = header.split(b"\n")
        version = ""
        for line in lines:
            line_str = line.decode("ascii", errors="ignore").strip()
            if line_str.startswith("version"):
                version = line_str.split()[1] if len(line_str.split()) > 1 else ""
            if line_str == "endhdr":
                break

        return ("sphinx_binary", version)
    except Exception:
        return None


def is_sphinx_binary(path: Path) -> bool:
    """Check if file is a Sphinx binary format file."""
    return get_sphinx_header(path) is not None


def _check_mfc_format(path: Path) -> bool:
    """Check if file appears to be in Sphinx .mfc format."""
    try:
        with path.open("rb") as f:
            header = f.read(4)
        if len(header) < 4:
            return False

        n_floats = struct.unpack("<i", header)[0]
        if not (0 < n_floats < 100_000_000):  # Reasonable range
            return False

        file_size = path.stat().st_size
        expected_size = 4 + n_floats * 4
        return bool(file_size == expected_size)
    except Exception:
        return False


def _is_model_directory(path: Path) -> bool:
    """Check if directory contains a complete acoustic model."""
    return all((path / f).exists() for f in MODEL_FILES_REQUIRED)


def detect_file_type(path: Path) -> FileType:
    """Detect file type from path and content.

    Detection strategy:
    1. Check if it's a directory (model)
    2. Check filename against known patterns
    3. Check file extension
    4. Check file content/magic bytes

    Args:
        path: Path to file or directory

    Returns:
        FileType enum value
    """
    path = Path(path)

    # Directory check
    if path.is_dir():
        if _is_model_directory(path):
            return FileType.MODEL
        return FileType.UNKNOWN

    if not path.exists():
        return FileType.UNKNOWN

    name = path.name.lower()
    stem = path.stem.lower()
    suffix = path.suffix.lower()

    # Check exact filename matches
    if name in _FILENAME_PATTERNS:
        return _FILENAME_PATTERNS[name]

    # Check extension patterns
    if suffix in _EXTENSION_PATTERNS:
        return _EXTENSION_PATTERNS[suffix]

    # Check if stem matches (for files like "means.1" or "variances.dat")
    for pattern, file_type in _FILENAME_PATTERNS.items():
        if stem == pattern or stem.startswith(pattern + "."):
            return file_type

    # Content-based detection
    if is_sphinx_binary(path):
        # Try to infer type from parent directory or sibling files
        parent = path.parent
        if (parent / "mdef").exists():
            # Likely a model parameter file
            if "mean" in name:
                return FileType.MEANS
            if "var" in name:
                return FileType.VARIANCES
            if "mixw" in name:
                return FileType.MIXTURE_WEIGHTS
            if "tmat" in name:
                return FileType.TRANSITION_MATRICES
        return FileType.UNKNOWN

    # Check for MFC format
    if _check_mfc_format(path):
        return FileType.FEATURES

    # Check for text-based files
    try:
        with path.open(encoding="utf-8") as f:
            first_line = f.readline().strip()

        # ARPA LM format
        if first_line.startswith("\\data\\"):
            return FileType.LM_ARPA

        # Transcription format: <s> ... </s> (uttid)
        if first_line.startswith("<s>") and "(" in first_line:
            return FileType.TRANSCRIPTION

        # Phoneset: one phone per line
        if len(first_line.split()) == 1 and first_line.isupper():
            return FileType.PHONESET

    except Exception:
        pass

    return FileType.UNKNOWN


def describe_file(path: Path) -> str:
    """Get human-readable description of file type."""
    file_type = detect_file_type(path)
    descriptions = {
        FileType.FEATURES: "MFCC feature file",
        FileType.MEANS: "Gaussian means",
        FileType.VARIANCES: "Gaussian variances",
        FileType.MIXTURE_WEIGHTS: "Mixture weights",
        FileType.TRANSITION_MATRICES: "Transition matrices",
        FileType.MDEF: "Model definition",
        FileType.SENDUMP: "Sendump (compressed model)",
        FileType.GAUDEN_COUNTS: "Gaussian accumulator counts",
        FileType.MODEL: "Acoustic model directory",
        FileType.DICTIONARY: "Pronunciation dictionary",
        FileType.FILLER_DICT: "Filler dictionary",
        FileType.TRANSCRIPTION: "Transcription file",
        FileType.FILEIDS: "File ID list",
        FileType.PHONESET: "Phone set",
        FileType.LM_ARPA: "ARPA language model",
        FileType.UNKNOWN: "Unknown file type",
    }
    return descriptions.get(file_type, "Unknown")


def validate_file_type(
    path: Path,
    expected: FileType,
    deep: bool = False,
) -> tuple[bool, str]:
    """Validate that a file matches the expected type.

    Args:
        path: Path to file or directory
        expected: Expected FileType
        deep: If True, attempt to load the file to validate (slower but more thorough)

    Returns:
        Tuple of (is_valid, message)
    """
    path = Path(path)

    if not path.exists():
        return False, f"File not found: {path}"

    # Quick detection first
    detected = detect_file_type(path)

    # Check for compatible types (means/variances are both gaussian)
    compatible_groups = [
        {FileType.MEANS, FileType.VARIANCES},
    ]

    def types_compatible(a: FileType, b: FileType) -> bool:
        if a == b:
            return True
        return any(a in group and b in group for group in compatible_groups)

    if not types_compatible(detected, expected):
        return False, f"Expected {expected.value}, detected {detected.value}"

    # Deep validation - try to load the file
    if deep:
        try:
            if expected == FileType.FEATURES:
                from st2.lib.features import read_sphinx_mfc

                data = read_sphinx_mfc(path)
                return True, f"Valid features: {data.shape}"

            if expected in (FileType.MEANS, FileType.VARIANCES):
                from st2.lib._cffi import read_gau

                data, n_cb, n_density, veclen, _ = read_gau(str(path))
                return True, f"Valid gaussians: {data.shape}"

            if expected == FileType.MIXTURE_WEIGHTS:
                from st2.lib._cffi import read_mixw

                data, n_mixw, n_feat, n_density = read_mixw(str(path))
                return True, f"Valid mixw: {data.shape}"

            if expected == FileType.TRANSITION_MATRICES:
                from st2.lib._cffi import read_tmat

                data, n_tmat, n_state = read_tmat(str(path))
                return True, f"Valid tmat: {data.shape}"

            if expected == FileType.MODEL:
                # Check required files exist
                missing = [f for f in MODEL_FILES_REQUIRED if not (path / f).exists()]
                if missing:
                    return False, f"Model missing: {', '.join(missing)}"
                return True, "Valid model directory"

            if expected == FileType.MDEF:
                # Try to parse mdef header
                text = path.read_text()
                if "version" not in text.lower() and "n_tied" not in text.lower():
                    return False, "Does not appear to be a valid mdef file"
                return True, "Valid mdef"

        except Exception as e:
            return False, f"Failed to load: {e}"

    return True, f"Detected as {detected.value}"


def assert_file_type(path: Path, expected: FileType, deep: bool = False) -> None:
    """Assert that a file matches the expected type.

    Args:
        path: Path to file or directory
        expected: Expected FileType
        deep: If True, attempt to load the file

    Raises:
        ValueError: If file doesn't match expected type
        FileNotFoundError: If file doesn't exist
    """
    valid, message = validate_file_type(path, expected, deep)
    if not valid:
        raise ValueError(f"{path}: {message}")
