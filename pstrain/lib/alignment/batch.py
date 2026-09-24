"""Batch alignment for corpus processing.

Aligns an entire corpus of audio files to their transcripts with a
single long-lived :class:`Aligner`, so the acoustic model is loaded
exactly once per corpus pass.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

from pstrain.lib.alignment.acceptance import (
    DEFAULT_RETRY_ACCEPTANCE_TARGET,
    AlignmentRejectedError,
    RetryCalibration,
    RungYield,
    rejection_message,
)
from pstrain.lib.alignment.core import (
    DEFAULT_BEAM,
    DEFAULT_RETRY_BEAM_FACTOR,
    AlignmentResult,
    RetryOutcome,
)
from pstrain.lib.alignment.coverage import AlignmentCoverage, alignment_coverage
from pstrain.lib.alignment.native import Aligner
from pstrain.lib.lexicon_check import (
    UnsupportedPhoneReport,
    base_word,
    check_model_lexicon,
)
from pstrain.lib.native_worker import PstrainNativeError, PstrainWorkerError
from pstrain.lib.retry_ladder import RetryBeamFactor

logger = logging.getLogger(__name__)

# Cadence for "Progress: N/total" log lines during a long batch.
_PROGRESS_LOG_EVERY = 100
_ERROR_MESSAGE_LIMIT = 200
_BRIEF_ERROR_LIMIT = 80


def _error_message(exc: Exception) -> str:
    """Keep the actionable end of a bounded native-worker diagnostic."""
    message = str(exc)
    if len(message) <= _ERROR_MESSAGE_LIMIT:
        return message
    return f"...{message[-(_ERROR_MESSAGE_LIMIT - 3) :]}"


@dataclass
class AlignmentJob:
    """Result of a batch alignment job.

    Attributes:
        model_dir: Path to acoustic model used
        n_utterances: Total utterances to align
        n_aligned: Successfully aligned utterances
        n_failed: Failed alignments
        results: Dict mapping utterance_id to AlignmentResult
        errors: Dict mapping utterance_id to error message
        timestamp: When the job was run
        phone_report: Pronunciations the model's phone inventory cannot
            support, collected before the run. ``None`` when the model
            definition could not be read.
        retry_yield: Per-rung ``(factor, attempted, recovered, rejected)``
            for the wider-beam final-state retry, in ladder order.
            ``attempted`` counts utterances that reached the rung;
            ``recovered`` counts those it aligned, and ``rejected`` those of
            them the acceptance check rejected. Empty when the retry is
            disabled or the aligner never started.
        retry_rejections: The retry-recovered alignments the acceptance check
            rejected, by utterance. Each is also in ``errors`` with its reason.
        retry_calibration: The acceptance threshold calibrated for each rung
            that recovered anything, when the run calibrated its own. Empty
            when nothing was recovered, the check is off, or a threshold was
            supplied.
        retry_acceptance_target: The check's target, or ``None`` when it was off.
        coverage: Mass and coverage by outcome, with thin-phone and other
            flags (:class:`~pstrain.lib.alignment.coverage.AlignmentCoverage`).
            Reporting only: it is computed after every acceptance decision and
            changes none. ``None`` when the report was not requested or could
            not be built.
    """

    model_dir: Path
    n_utterances: int
    n_aligned: int
    n_failed: int
    results: dict[str, AlignmentResult] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    phone_report: UnsupportedPhoneReport | None = None
    retry_yield: tuple[RungYield, ...] = ()
    retry_rejections: dict[str, RetryOutcome] = field(default_factory=dict)
    retry_calibration: tuple[RetryCalibration, ...] = ()
    retry_acceptance_target: float | None = DEFAULT_RETRY_ACCEPTANCE_TARGET
    coverage: AlignmentCoverage | None = None

    @property
    def success_rate(self) -> float:
        """Alignment success rate (0.0-1.0)."""
        if self.n_utterances == 0:
            return 0.0
        return self.n_aligned / self.n_utterances


def collect_phone_report(
    model_dir: Path,
    dict_path: Path,
    filler_dict: Path | None = None,
) -> UnsupportedPhoneReport | None:
    """Collect the pronunciations this model's phone inventory cannot support.

    Args:
        model_dir: Acoustic model directory.
        dict_path: Pronunciation dictionary.
        filler_dict: Filler dictionary. Optional.

    Returns:
        The collected report, or ``None`` when the model definition or a
        dictionary could not be read. An unreadable input is the aligner's
        problem to report, not this check's: it must never be the reason a
        corpus pass does not start. Skipping is said out loud, because a
        silent skip turns the whole check off with no trace.
    """
    try:
        return check_model_lexicon(model_dir, dict_path, filler_dict)
    except (OSError, ValueError) as exc:
        logger.warning(
            "Not checking the dictionary against the model's phone inventory: %s. "
            "Pronunciations using phones the model does not define will not be "
            "reported before the run.",
            exc,
        )
        return None


def explain_failure(
    message: str,
    transcript: str,
    phone_report: UnsupportedPhoneReport | None,
) -> str:
    """Name the phone inventory as the cause of a failure when it is the cause.

    A word whose every pronunciation was dropped at load time is missing from
    the lexicon by the time the aligner looks for it, so the native failure
    reads as an out-of-vocabulary one. Say what actually happened.

    Args:
        message: The native failure message.
        transcript: Transcript of the failed utterance.
        phone_report: Report collected before the run, if any.

    Returns:
        The message, with the phone-inventory cause appended when one applies.
    """
    if phone_report is None or not phone_report.unresolvable_words:
        return message

    unresolvable = phone_report.unresolvable_words
    tokens = [base_word(token) for token in dict.fromkeys(transcript.split())]
    affected = [token for token in dict.fromkeys(tokens) if token in unresolvable]
    if not affected:
        return message
    return (
        f"{message} [no pronunciation survived the model's phone inventory for: "
        f"{_name_causes(affected, phone_report)}]"
    )


def _name_causes(words: list[str], phone_report: UnsupportedPhoneReport) -> str:
    """Name each word with every phone its dropped pronunciations needed."""
    named = []
    for word in words:
        phones = phone_report.missing_by_base.get(word, ())
        detail = f" ({', '.join(phones)})" if phones else ""
        named.append(f"{word}{detail}")
    return "; ".join(named)


def align_corpus(
    transcripts: dict[str, str],
    audio_dir: Path,
    model_dir: Path,
    dict_path: Path,
    filler_dict: Path | None = None,
    audio_ext: str = ".wav",
    include_phones: bool = True,
    beam: float = DEFAULT_BEAM,
    retry_beam_factor: RetryBeamFactor = DEFAULT_RETRY_BEAM_FACTOR,
    failed_alignment: Literal["recover", "abort", "omit"] = "recover",
    verbatim_tokens: bool = False,
    phone_report: UnsupportedPhoneReport | None = None,
    retry_acceptance_target: float | None = DEFAULT_RETRY_ACCEPTANCE_TARGET,
    retry_acceptance_threshold: float | Sequence[float] | None = None,
    coverage_report: bool = True,
) -> AlignmentJob:
    """Align an entire corpus.

    Loads the acoustic model once, keeps it resident for all utterances.

    Args:
        transcripts: Dict mapping utterance_id to transcript text.
        audio_dir: Directory containing audio files.
        model_dir: Path to acoustic model directory.
        dict_path: Path to pronunciation dictionary.
        filler_dict: Path to filler dictionary (optional).
        audio_ext: Audio file extension (default ``".wav"``).
        include_phones: Capture phone-level segmentation.
        beam: Viterbi pruning beam.
        retry_beam_factor: Factor for one wider-beam final-state retry, or an
            ascending sequence of factors tried in order until one succeeds.
        failed_alignment: Whether final-state failures are retried before being recorded.
        verbatim_tokens: Honor explicit pronunciation variants exactly.
        phone_report: An already-collected report of pronunciations the
            model cannot support, so a caller that reported it before the
            run does not pay for the check or report it twice. When
            omitted, the check runs here and any finding is logged.
        retry_acceptance_target: The retry acceptance check's quantile
            (default 0.05); ``None`` accepts retries unchecked. A
            retry-recovered alignment is accepted only if its speech score
            reaches the threshold for its rung's beam; otherwise it is
            recorded as failed, with the reason. First-pass alignments are
            never checked. Unless a threshold is supplied, the run calibrates
            its own after the corpus pass, and only if a retry recovered
            anything: it realigns at most ``RETRY_ACCEPTANCE_MAX_SAMPLES``
            (200) of its first-pass alignments, evenly spaced, at each such
            rung's beam and takes this quantile of their scores. With fewer
            than ``RETRY_ACCEPTANCE_MIN_SAMPLES`` (20) scored, every recovery
            at that rung is rejected.
        retry_acceptance_threshold: A threshold per rung, in nats per speech
            frame, to use instead of calibrating.
        coverage_report: Attach the mass-and-coverage report
            (:attr:`AlignmentJob.coverage`). It is built after the pass from
            the outcomes already decided and never changes one.

    Returns:
        :class:`AlignmentJob` with all alignment results.

    Example:
        >>> transcripts = {"utt001": "hello world", "utt002": "goodbye"}
        >>> job = align_corpus(transcripts, audio_dir, model_dir, dict_path)
        >>> print(f"Aligned {job.n_aligned}/{job.n_utterances}")
    """
    audio_dir = Path(audio_dir)
    model_dir = Path(model_dir)
    dict_path = Path(dict_path)

    results: dict[str, AlignmentResult] = {}
    errors: dict[str, str] = {}
    rejections: dict[str, RetryOutcome] = {}
    calibration: tuple[RetryCalibration, ...] = ()
    n_aligned = 0
    n_failed = 0

    if phone_report is None:
        phone_report = collect_phone_report(model_dir, dict_path, filler_dict)
        if phone_report:
            logger.error("%s", phone_report.format())

    def finish(job: AlignmentJob) -> AlignmentJob:
        if coverage_report:
            job.coverage = _coverage(job, transcripts, dict_path, filler_dict, audio_dir, audio_ext)
        return job

    total = len(transcripts)
    if total == 0:
        return finish(
            AlignmentJob(
                model_dir=model_dir,
                n_utterances=0,
                n_aligned=0,
                n_failed=0,
                results=results,
                errors=errors,
                phone_report=phone_report,
                retry_acceptance_target=retry_acceptance_target,
            )
        )

    logger.info("Aligning %d utterances...", total)
    if retry_acceptance_target is None and failed_alignment == "recover":
        logger.warning(
            "The retry acceptance check is off: alignments a wider-beam retry "
            "recovers are accepted unchecked."
        )
    # With the check on and no threshold supplied, the corpus calibrates its
    # own: recoveries wait, unjudged, until the pass is over.
    deferred = retry_acceptance_target is not None and retry_acceptance_threshold is None

    # If model_dir is missing required files Aligner raises before the
    # loop; surface that as a corpus-wide failure rather than per-utt.
    try:
        aligner = Aligner(
            model_dir,
            dict_path,
            filler_dict=filler_dict,
            beam=beam,
            retry_beam_factor=retry_beam_factor,
            failed_alignment=failed_alignment,
            retry_acceptance_target=retry_acceptance_target,
            retry_acceptance_threshold=retry_acceptance_threshold,
            retry_acceptance_deferred=deferred,
            include_phones=include_phones,
            verbatim_tokens=verbatim_tokens,
        )
    except (FileNotFoundError, RuntimeError) as e:
        init_error = f"Aligner init failed: {e}"
        logger.error("%s", init_error)
        for utt_id in transcripts:
            errors[utt_id] = init_error
        return finish(
            AlignmentJob(
                model_dir=model_dir,
                n_utterances=total,
                n_aligned=0,
                n_failed=total,
                results=results,
                errors=errors,
                phone_report=phone_report,
                retry_acceptance_target=retry_acceptance_target,
            )
        )

    first_pass: list[str] = []
    pending: dict[str, AlignmentResult] = {}
    deferred_rejections = [0] * len(aligner.retry_yield())
    try:
        for i, (utt_id, transcript) in enumerate(transcripts.items(), 1):
            audio_path = audio_dir / f"{utt_id}{audio_ext}"

            if i % _PROGRESS_LOG_EVERY == 0 or i == total:
                logger.info("  Progress: %d/%d (%.1f%%)", i, total, 100 * i / total)

            if not audio_path.exists():
                errors[utt_id] = f"Audio file not found: {audio_path}"
                n_failed += 1
                continue

            try:
                result = aligner.align_audio(audio_path, transcript, utterance_id=utt_id)
            except AlignmentRejectedError as e:
                rejections[utt_id] = e.outcome
                errors[utt_id] = rejection_message(None, e.outcome)
                n_failed += 1
                logger.warning("Alignment rejected for %s: %s", utt_id, e)
                continue
            except Exception as e:
                message = explain_failure(_error_message(e), transcript, phone_report)
                errors[utt_id] = message
                n_failed += 1
                logger.warning("Alignment failed for %s: %s", utt_id, message)
                continue
            if result.retry is None:
                first_pass.append(utt_id)
            elif deferred:
                pending[utt_id] = result
                continue
            results[utt_id] = result
            n_aligned += 1

        if pending:
            calibration = _calibrate_pending(
                aligner,
                pending,
                first_pass,
                transcripts,
                audio_dir,
                audio_ext,
                beam,
                retry_acceptance_target or DEFAULT_RETRY_ACCEPTANCE_TARGET,
            )
            thresholds = {c.rung: c for c in calibration}
            for utt_id, result in pending.items():
                assert result.retry is not None
                rung = thresholds[result.retry.rung]
                judged = replace(result.retry, threshold=rung.threshold, basis=rung.basis)
                if (
                    judged.threshold is not None
                    and judged.score is not None
                    and judged.score >= judged.threshold
                ):
                    result.retry = judged
                    results[utt_id] = result
                    n_aligned += 1
                    continue
                rejections[utt_id] = judged
                errors[utt_id] = rejection_message(None, judged)
                deferred_rejections[judged.rung - 1] += 1
                n_failed += 1
                logger.warning("Alignment rejected for %s: %s", utt_id, errors[utt_id])
        retry_yield = tuple(
            rung._replace(rejected=rung.rejected + deferred_rejections[index])
            for index, rung in enumerate(aligner.retry_yield())
        )
    finally:
        aligner.close()

    # Judging recoveries after the pass must not reorder the corpus.
    results = {utt_id: results[utt_id] for utt_id in transcripts if utt_id in results}
    errors = {utt_id: errors[utt_id] for utt_id in transcripts if utt_id in errors}

    logger.info(
        "Alignment complete: %d/%d successful (%.1f%%)",
        n_aligned,
        total,
        100 * n_aligned / total,
    )

    return finish(
        AlignmentJob(
            model_dir=model_dir,
            n_utterances=total,
            n_aligned=n_aligned,
            n_failed=n_failed,
            results=results,
            errors=errors,
            phone_report=phone_report,
            retry_yield=retry_yield,
            retry_rejections=rejections,
            retry_calibration=calibration,
            retry_acceptance_target=retry_acceptance_target,
        )
    )


def _coverage(
    job: AlignmentJob,
    transcripts: dict[str, str],
    dict_path: Path,
    filler_dict: Path | None,
    audio_dir: Path,
    audio_ext: str,
) -> AlignmentCoverage | None:
    """Build the coverage report; a failure to build it never fails the run."""
    try:
        return alignment_coverage(
            job, transcripts, dict_path, filler_dict, audio_dir=audio_dir, audio_ext=audio_ext
        )
    except Exception as exc:  # noqa: BLE001 - a report must never fail the alignment
        logger.warning("Not reporting alignment mass and coverage: %s", exc)
        return None


def _brief_error(exc: Exception) -> str:
    """Name an error in a few words: its type and the end of its diagnostic."""
    detail = exc.diagnostic if isinstance(exc, PstrainNativeError) else str(exc)
    detail = " ".join(detail.split())
    if len(detail) > _BRIEF_ERROR_LIMIT:
        detail = f"...{detail[-(_BRIEF_ERROR_LIMIT - 3) :]}"
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _calibrate_pending(
    aligner: Aligner,
    pending: dict[str, AlignmentResult],
    first_pass: list[str],
    transcripts: dict[str, str],
    audio_dir: Path,
    audio_ext: str,
    beam: float,
    target: float,
) -> tuple[RetryCalibration, ...]:
    """Calibrate the threshold of every rung that recovered something.

    Realigns this run's own first-pass successes at each such rung's beam; the
    aligner samples at most ``RETRY_ACCEPTANCE_MAX_SAMPLES`` of them.
    """
    items = [(audio_dir / f"{utt_id}{audio_ext}", transcripts[utt_id]) for utt_id in first_pass]
    rungs = sorted({result.retry.rung for result in pending.values() if result.retry is not None})
    calibration = []
    for rung in rungs:
        try:
            result = aligner.calibrate_rung(items, rung)
        except Exception as exc:  # noqa: BLE001 - the recoveries are rejected, with the reason
            factor = aligner.retry_yield()[rung - 1].factor
            result = RetryCalibration(
                rung=rung,
                factor=factor,
                beam=beam / factor,
                target=target,
                threshold=None,
                n_scored=0,
                error=_brief_error(exc),
                aligner_lost=isinstance(exc, (PstrainNativeError, PstrainWorkerError)),
            )
        logger.info(
            "Retry acceptance, rung %d: %s",
            rung,
            f"threshold {result.threshold:.3f} nats per speech frame, {result.basis}"
            if result.threshold is not None
            else result.basis,
        )
        calibration.append(result)
    return tuple(calibration)


def load_transcripts(transcript_file: Path) -> dict[str, str]:
    """Load transcripts from a Sphinx-format transcription file.

    Format: <s> word word word </s> (utterance_id)

    Args:
        transcript_file: Path to transcription file

    Returns:
        Dict mapping utterance_id to transcript text (with sentence markers)
    """
    transcripts = {}

    with transcript_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse: <s> text </s> (utt_id)
            if "(" in line:
                paren_start = line.rfind("(")
                paren_end = line.rfind(")")
                if paren_start > 0 and paren_end > paren_start:
                    utt_id = line[paren_start + 1 : paren_end].strip()
                    transcript = line[:paren_start].strip()
                    transcripts[utt_id] = transcript

    return transcripts
