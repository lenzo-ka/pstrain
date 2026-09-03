"""Model testing and WER evaluation."""

from __future__ import annotations

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pstrain.lib.model import MODEL_FILES_REQUIRED, require_complete_model
from pstrain.lib.native_worker import PstrainError
from pstrain.lib.testing.decoder import Decoder, DecodingResult, check_pocketsphinx
from pstrain.lib.testing.wer import WERResult, aggregate_wer, calculate_wer

logger = logging.getLogger(__name__)

MAX_DECODE_WORKERS = 12


@dataclass(frozen=True)
class _DecodeConfig:
    model_dir: Path
    dict_file: Path
    filler_dict: Path | None
    lm: Path | None


def _decode_shard(
    config: _DecodeConfig, shard: list[tuple[int, str, Path]]
) -> list[tuple[int, str, DecodingResult]]:
    """Decode one shard with one decoder, returning source positions."""
    decoder = Decoder(
        model_dir=config.model_dir,
        dict_file=config.dict_file,
        filler_dict=config.filler_dict,
        lm=config.lm,
    )
    try:
        return [(position, utt_id, decoder.decode_file(path)) for position, utt_id, path in shard]
    finally:
        decoder.close()


def _resolve_decode_jobs(jobs: int | None) -> int:
    if jobs is None or jobs == -1:
        return min(MAX_DECODE_WORKERS, os.cpu_count() or 1)
    if jobs < 1:
        raise ValueError("jobs must be -1 (auto) or a positive integer")
    return jobs


def _decode_files(
    config: _DecodeConfig, files: list[tuple[str, Path]], jobs: int | None
) -> list[tuple[str, DecodingResult]]:
    """Decode files serially or in stable process shards, preserving input order."""
    if not files:
        return []

    worker_count = min(_resolve_decode_jobs(jobs), len(files))
    indexed = [(position, utt_id, path) for position, (utt_id, path) in enumerate(files)]
    if worker_count <= 1:
        decoded = _decode_shard(config, indexed)
    else:
        shards = [indexed[worker::worker_count] for worker in range(worker_count)]
        # PocketSphinx owns native process-global state. A fresh interpreter keeps
        # workers independent of any decoder previously used by the parent.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            futures = [executor.submit(_decode_shard, config, shard) for shard in shards]
            decoded = [item for future in futures for item in future.result()]
    decoded.sort(key=lambda item: item[0])
    return [(utt_id, result) for _, utt_id, result in decoded]


@dataclass
class TestResult:
    """Result from model testing with all jiwer metrics.

    Attributes:
        model_dir: Path to the model directory tested
        model_name: Name of the model (e.g., "cd-8g")
        n_utterances: Number of test utterances
        n_decoded: Number of successfully decoded utterances
        wer_result: Aggregated WER metrics (source of truth for all WER fields)
        timestamp: When the test was run
        per_utterance: Per-utterance results (optional)
    """

    model_dir: Path
    model_name: str
    n_utterances: int
    n_decoded: int
    wer_result: WERResult
    timestamp: datetime = field(default_factory=datetime.now)
    per_utterance: dict[str, dict[str, Any]] | None = None

    # Delegation properties for backward compatibility
    @property
    def wer(self) -> float:
        return self.wer_result.wer

    @property
    def mer(self) -> float:
        return self.wer_result.mer

    @property
    def wil(self) -> float:
        return self.wer_result.wil

    @property
    def wip(self) -> float:
        return self.wer_result.wip

    @property
    def hits(self) -> int:
        return self.wer_result.hits

    @property
    def substitutions(self) -> int:
        return self.wer_result.substitutions

    @property
    def deletions(self) -> int:
        return self.wer_result.deletions

    @property
    def insertions(self) -> int:
        return self.wer_result.insertions

    @property
    def ref_words(self) -> int:
        return self.wer_result.ref_words

    @property
    def hyp_words(self) -> int:
        return self.wer_result.hyp_words

    @property
    def cer(self) -> float | None:
        return self.wer_result.cer

    @property
    def accuracy(self) -> float:
        return self.wer_result.accuracy

    @property
    def errors(self) -> int:
        return self.wer_result.errors

    @property
    def total_words(self) -> int:
        return self.wer_result.total_words

    @property
    def correct(self) -> int:
        return self.wer_result.correct

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "model_dir": str(self.model_dir),
            "model_name": self.model_name,
            "n_utterances": self.n_utterances,
            "n_decoded": self.n_decoded,
            "timestamp": self.timestamp.isoformat(),
            "per_utterance": self.per_utterance,
        }
        result.update(self.wer_result.to_dict())
        return result


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
    """Test an acoustic model and calculate all WER metrics.

    Args:
        model_dir: Path to acoustic model directory
        test_audio_dir: Directory containing test audio files
        test_transcripts: Dict mapping utterance_id to reference transcript
        dict_file: Path to pronunciation dictionary
        filler_dict: Optional filler dictionary
        lm: Optional language model (ARPA format)
        verbose: Store per-utterance results
        compute_cer: Also compute Character Error Rate
        jobs: Decoder worker processes. None auto-selects up to 12.

    Returns:
        TestResult with all jiwer metrics

    Raises:
        ImportError: If PocketSphinx not available
        RuntimeError: If testing fails
    """
    model_dir = Path(model_dir)
    test_audio_dir = Path(test_audio_dir)
    dict_file = Path(dict_file)

    # Check PocketSphinx availability
    available, msg = check_pocketsphinx()
    if not available:
        raise ImportError(msg)

    # Validate model directory. Testing consumes the model for decoding, so the
    # complete-model contract applies: the front-end record must be present and
    # readable, not merely the files training writes. This is checked here rather
    # than left to decoder construction, which does not happen when there is
    # nothing to decode.
    for fname in MODEL_FILES_REQUIRED:
        if not (model_dir / fname).exists():
            raise FileNotFoundError(f"Model file not found: {model_dir / fname}")
    require_complete_model(model_dir)

    # Determine model name from directory
    model_name = model_dir.parent.name if model_dir.name == "default" else model_dir.name

    logger.info("Testing model: %s", model_dir)
    logger.info("Test utterances: %d", len(test_transcripts))

    decode_config = _DecodeConfig(
        model_dir=model_dir,
        dict_file=dict_file,
        filler_dict=filler_dict,
        lm=lm,
    )

    # Decode test utterances and calculate WER
    wer_results: list[WERResult] = []
    per_utterance: dict[str, dict[str, Any]] = {}
    n_decoded = 0
    decode_failures: list[tuple[str, str]] = []

    decode_inputs: list[tuple[str, Path]] = []
    missing_audio: list[str] = []
    for utt_id in test_transcripts:
        audio_file = test_audio_dir / f"{utt_id}.wav"
        if not audio_file.exists():
            logger.warning("Audio file not found: %s", audio_file)
            missing_audio.append(utt_id)
            continue
        decode_inputs.append((utt_id, audio_file))

    logger.info("Decoder workers: %d", min(_resolve_decode_jobs(jobs), len(decode_inputs)))
    for utt_id, result in _decode_files(decode_config, decode_inputs, jobs):
        reference = test_transcripts[utt_id]
        if result.success:
            n_decoded += 1
            hypothesis = result.hypothesis

            # Calculate WER for this utterance
            wer_result = calculate_wer(reference, hypothesis, compute_cer=compute_cer)
            wer_results.append(wer_result)

            if verbose:
                per_utterance[utt_id] = {
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "wer": wer_result.wer,
                    "mer": wer_result.mer,
                    "wil": wer_result.wil,
                    "wip": wer_result.wip,
                    "cer": wer_result.cer,
                    "hits": wer_result.hits,
                    "substitutions": wer_result.substitutions,
                    "deletions": wer_result.deletions,
                    "insertions": wer_result.insertions,
                }
        else:
            reason = result.error or "unknown decoder failure"
            decode_failures.append((utt_id, reason))
            logger.warning("Decoding failed for %s: %s", utt_id, reason)

    if test_transcripts and n_decoded == 0:
        # Utterances were requested and none produced a hypothesis. Report every
        # reason, including audio that was never found: a corpus whose audio is
        # all missing decodes exactly as little as one that all failed, and
        # neither has a word error rate worth reporting.
        reasons = [f"{utt_id}: {reason}" for utt_id, reason in decode_failures]
        reasons += [f"{utt_id}: audio file not found" for utt_id in missing_audio]
        raise PstrainError(
            f"Nothing decoded: all {len(test_transcripts)} requested utterances failed: "
            + "; ".join(reasons)
        )

    # Aggregate results
    if wer_results:
        total = aggregate_wer(wer_results)
    else:
        total = WERResult(
            wer=1.0,
            mer=1.0,
            wil=1.0,
            wip=0.0,
            hits=0,
            substitutions=0,
            deletions=0,
            insertions=0,
            ref_words=0,
            hyp_words=0,
        )

    logger.info("WER: %.2f%% (%d/%d decoded)", total.wer * 100, n_decoded, len(test_transcripts))

    return TestResult(
        model_dir=model_dir,
        model_name=model_name,
        n_utterances=len(test_transcripts),
        n_decoded=n_decoded,
        wer_result=total,
        per_utterance=per_utterance if verbose else None,
    )


def load_transcripts(transcript_file: Path) -> dict[str, str]:
    """Load transcripts from a Sphinx-format transcription file.

    Format: <s> word word word </s> (utterance_id)

    Args:
        transcript_file: Path to transcription file

    Returns:
        Dict mapping utterance_id to transcript text
    """
    transcripts = {}

    with transcript_file.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse: <s> text </s> (utt_id)
            if line.startswith("<s>") and "(" in line:
                # Extract utterance ID
                paren_start = line.rfind("(")
                paren_end = line.rfind(")")
                if paren_start > 0 and paren_end > paren_start:
                    utt_id = line[paren_start + 1 : paren_end].strip()

                    # Extract text between <s> and </s>
                    text_part = line[:paren_start].strip()
                    if text_part.startswith("<s>"):
                        text_part = text_part[3:]
                    if text_part.endswith("</s>"):
                        text_part = text_part[:-4]

                    transcripts[utt_id] = text_part.strip()

    return transcripts
