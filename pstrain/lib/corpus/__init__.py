"""Corpus utilities for text normalization, file handling, and splitting."""

from pstrain.lib.corpus.files import (
    convert_to_sphinx,
    detect_transcript_format,
    extract_vocabulary,
    load_filelist,
    save_filelist,
)
from pstrain.lib.corpus.normalize import (
    normalize_transcript,
    normalize_transcript_file,
    strip_boundary_punct,
)
from pstrain.lib.corpus.split import SplitResult, train_test_split

__all__ = [
    # Filelist I/O
    "load_filelist",
    "save_filelist",
    # Vocabulary
    "extract_vocabulary",
    # Transcript format conversion
    "convert_to_sphinx",
    "detect_transcript_format",
    # Normalization
    "normalize_transcript",
    "normalize_transcript_file",
    "strip_boundary_punct",
    # Train/test split
    "SplitResult",
    "train_test_split",
]
