"""Public API for model testing and evaluation operations."""

from pathlib import Path

from pstrain.lib.testing import TestReport, TestResult
from pstrain.lib.testing import check_pocketsphinx as _check_pocketsphinx
from pstrain.lib.testing import create_report as _create_report
from pstrain.lib.testing import load_transcripts as _load_transcripts
from pstrain.lib.testing import test_model as _test_model


def check_pocketsphinx() -> tuple[bool, str]:
    """Check decoder availability through a concrete public-API call frame."""
    return _check_pocketsphinx()


def create_report(
    result: TestResult,
    title: str | None = None,
    corpus_name: str = "",
    test_set_name: str = "",
) -> TestReport:
    """Create a test report through a concrete public-API call frame."""
    return _create_report(result, title, corpus_name, test_set_name)


def load_transcripts(transcript_file: Path) -> dict[str, str]:
    """Load transcripts through a concrete public-API call frame."""
    return _load_transcripts(transcript_file)


def test_model(
    model_dir: Path,
    test_audio_dir: Path,
    test_transcripts: dict[str, str],
    dict_file: Path,
    filler_dict: Path | None = None,
    lm: Path | None = None,
    verbose: bool = False,
    compute_cer: bool = False,
    jobs: int | None = None,
) -> TestResult:
    """Test a model through a concrete public-API call frame."""
    return _test_model(
        model_dir,
        test_audio_dir,
        test_transcripts,
        dict_file,
        filler_dict,
        lm,
        verbose,
        compute_cer,
        jobs,
    )


__all__ = [
    "check_pocketsphinx",
    "create_report",
    "load_transcripts",
    "test_model",
]
