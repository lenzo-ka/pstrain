"""BW training step function.

Orchestrates Baum-Welch training using the CFFI-wrapped BWTrainer.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

from pstrain.lib.bw import BWConfig, BWTrainer
from pstrain.lib.features import read_sphinx_mfc
from pstrain.lib.model import MODEL_FILES_REQUIRED
from pstrain.lib.transcription import parse_transcription_file
from pstrain.lib.validate import validate_files_exist

logger = logging.getLogger(__name__)

__all__ = ["run_bw_training", "TrainingIteration", "TrainingResult"]

_CHECKPOINT_FILES = (
    "mdef",
    "means",
    "variances",
    "mixture_weights",
    "transition_matrices",
    "gauden_counts",
)
_TELEMETRY_FILENAME = "bw_telemetry.json"


class TerminalAlignmentError(RuntimeError):
    """An utterance still cannot reach its final state after retry."""


def _exact_zero_codebooks(model_dir: Path) -> int:
    """Count codebooks whose serialized mean and variance are entirely zero."""
    import numpy as np

    from pstrain.lib import _pstrainc

    means = _pstrainc.read_gau(str(model_dir / "means"))[0]
    variances = _pstrainc.read_gau(str(model_dir / "variances"))[0]
    parameter_axes = tuple(range(1, means.ndim))
    return int(
        np.count_nonzero(
            np.all(means == 0, axis=parameter_axes) & np.all(variances == 0, axis=parameter_axes)
        )
    )


def _accept_arctic_a0302_exception(
    *, fileid: str, model_dir: Path, band: tuple[int, int] | None
) -> int | None:
    """Accept only the named sentinel when its governing occupancy is in band."""
    if fileid != "arctic_a0302" or band is None:
        return None
    value = _exact_zero_codebooks(model_dir)
    lower, upper = band
    if not lower <= value <= upper:
        raise TerminalAlignmentError(
            "arctic_a0302 accepted-exception band refused: "
            f"exact_zero_codebooks={value} is outside inclusive band [{lower}, {upper}]"
        )
    logger.warning(
        "ACCEPTED EXCEPTION arctic_a0302: exact_zero_codebooks=%d is inside "
        "inclusive band [%d, %d]; continuing without this utterance update",
        value,
        lower,
        upper,
    )
    return value


def _write_telemetry(
    output_dir: Path, rows: list[dict[str, object]], *, schema_version: int = 1
) -> None:
    """Atomically retain the completed BW passes without affecting training."""
    destination = output_dir / _TELEMETRY_FILENAME
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(
                {"schema_version": schema_version, "passes": rows},
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except Exception as exc:
        logger.warning("Could not write BW telemetry to %s: %s", destination, exc)
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def _checkpoint_iteration(output_dir: Path, iteration: int) -> None:
    """Retain the complete model produced by one BW pass for validation."""
    checkpoint = output_dir / "iterations" / f"{iteration:02d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    for filename in _CHECKPOINT_FILES:
        shutil.copyfile(output_dir / filename, checkpoint / filename)


def _convergence_delta(current: float, previous: float) -> float:
    """Return SphinxTrain's signed per-frame log-likelihood delta."""
    if previous == 0:
        return 1.0 if current > 0 else -1.0 if current < 0 else 0.0
    return current - previous


def _has_converged(
    current: float,
    previous: float,
    iteration: int,
    threshold: float,
    min_iterations: int,
) -> bool:
    """Apply the upstream convergence decision after a non-initial iteration."""
    return _convergence_delta(current, previous) <= threshold and iteration >= min_iterations


def _process_with_final_state_retry(
    trainer: BWTrainer,
    mfcc: npt.NDArray[np.float32],
    transcript: str,
    normal_beam: float,
    retry_beam_factor: float,
    fileid: str,
) -> bool:
    """Process an update, retrying only a forward-final-state pruning failure.

    ``BWTrainer`` is a mutable native session and must not be shared between
    threads. The debug assertion makes concurrent entry at this mutation seam
    fail instead of allowing another call to observe the temporary beam.
    """
    assert not trainer._retry_transaction_active, "BWTrainer cannot be shared across threads"
    trainer._last_process_retried = False
    trainer._retry_transaction_active = True
    try:
        success = trainer.process_utterance_mfcc(mfcc, transcript)
        if success:
            return success

        if not trainer.final_state_not_reached:
            return False

        if retry_beam_factor <= 1.0:
            raise TerminalAlignmentError(
                f"Final state not reached for {fileid} with a_beam={normal_beam:.3g}; "
                "retry is disabled"
            )

        retry_beam = normal_beam / retry_beam_factor
        trainer._last_process_retried = True
        logger.warning(
            "Final state not reached for %s; retrying once with a_beam=%.3g",
            fileid,
            retry_beam,
        )
        previous_beam = trainer.set_a_beam(retry_beam)
        try:
            success = trainer.process_utterance_mfcc(mfcc, transcript)
            if not success:
                raise TerminalAlignmentError(
                    f"Final state not reached for {fileid} after retry: "
                    f"expected a complete alignment at a_beam={retry_beam:.3g}"
                )
            return True
        finally:
            trainer.set_a_beam(previous_beam)
    finally:
        trainer._retry_transaction_active = False


@dataclass
class TrainingIteration:
    """Numerical and accounting telemetry for one BW pass."""

    iteration: int
    total_log_lik: float
    avg_log_prob: float
    per_frame_delta: float | None
    frames: int
    input_utts: int
    processed_utts: int
    retried_utts: int
    skipped_utts: int
    excluded_by_schedule: int = 0


@dataclass
class TrainingResult:
    """Result from BW training."""

    iterations: int
    converged: bool
    final_likelihood: float
    final_frames: int
    final_utts: int
    total_skipped: int = 0
    trajectory: tuple[TrainingIteration, ...] = ()


def _config_for_iteration(
    config: BWConfig,
    *,
    multipron: bool,
    iteration: int,
    first_pass_2passvar: bool,
) -> BWConfig:
    """Resolve caller configuration with the mandatory stage variance policy."""
    from dataclasses import replace

    resolved = config
    if iteration == 1:
        resolved = replace(resolved, pass2var=first_pass_2passvar)
    if resolved.multipron != multipron:
        resolved = replace(resolved, multipron=multipron)
    return resolved


def run_bw_training(
    model_dir: Path,
    output_dir: Path,
    features_dir: Path,
    train_fileids: Path,
    transcription: Path,
    dictionary: Path,
    first_pass_2passvar: bool,
    config: BWConfig,
    filler_dict: Path | None = None,
    n_iter: int = 10,
    convergence_ratio: float = 0.001,
    min_iterations: int = 1,
    multipron: bool = True,
    max_skip_fraction: float = 0.05,
    retry_beam_factor: float = 1e10,
    exclusion_schedule: dict[int | str, list[str]] | None = None,
    arctic_a0302_zero_codebook_band: tuple[int, int] | None = None,
) -> TrainingResult:
    """Run Baum-Welch training iterations.

    Args:
        model_dir: Directory containing initial model (mdef, means, variances,
            mixture_weights, transition_matrices)
        output_dir: Directory to write trained model
        features_dir: Directory containing .mfc feature files
        train_fileids: File listing utterance IDs (one per line)
        transcription: Transcription file (Sphinx format)
        dictionary: Pronunciation dictionary path
        filler_dict: Filler dictionary path (optional)
        n_iter: Maximum training iterations
        convergence_ratio: Signed per-frame likelihood delta threshold, using
            each pass's own frame count. A delta strictly greater than this
            threshold continues training; otherwise training converges once
            ``min_iterations`` is satisfied.
        min_iterations: Minimum number of completed iterations before convergence
        config: BW training configuration, including the explicit variance
            policy retained after the stage-specific first iteration.
        max_skip_fraction: Fail when skipped utterances exceed this fraction.
        retry_beam_factor: Widen the forward beam by this factor for one retry
            when pruning prevents the final state from being reached. Set to 1
            to disable retries.
        first_pass_2passvar: Required stage policy for the first iteration.
            ``True`` selects centered two-pass accumulation and ``False``
            selects one-pass variance accumulation.
        exclusion_schedule: Experimental mapping of one-based pass numbers or
            ``"*"`` to utterance IDs that must not reach BW accumulation.

    Returns:
        TrainingResult with training statistics

    Raises:
        FileNotFoundError: If required files are missing
        RuntimeError: If training fails
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    features_dir = Path(features_dir)

    # Validate inputs
    validate_files_exist(
        [model_dir / f for f in MODEL_FILES_REQUIRED] + [train_fileids, transcription, dictionary],
        context="BW training",
    )
    # Load transcriptions
    transcripts = parse_transcription_file(transcription)
    logger.info("Loaded %d transcripts", len(transcripts))

    # Load fileids
    with train_fileids.open() as f:
        fileids = [line.strip() for line in f if line.strip()]
    logger.info("Training on %d utterances", len(fileids))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_enabled = os.environ.get("PSTRAIN_BW_CHECKPOINTS") == "1"
    if checkpoints_enabled:
        shutil.rmtree(output_dir / "iterations", ignore_errors=True)

    # Copy mdef (unchanged during training)
    shutil.copy(model_dir / "mdef", output_dir / "mdef")

    prev_likelihood = float("-inf")
    current_model = model_dir
    last_frames = 0
    last_utts = 0
    total_skipped = 0
    trajectory: list[TrainingIteration] = []
    telemetry_rows: list[dict[str, object]] = []
    exclusion_schedule = exclusion_schedule or {}

    for iteration in range(1, n_iter + 1):
        logger.info("Starting iteration %d/%d...", iteration, n_iter)

        # SphinxTrain's policy is stage-specific: CI and tied stages begin with
        # one-pass variance, while stage 30 CD-untied uses -2passvar yes from
        # its first pass (baum_welch.pl's unconditional $var2pass = "yes").
        iter_config = _config_for_iteration(
            config,
            multipron=multipron,
            iteration=iteration,
            first_pass_2passvar=first_pass_2passvar,
        )
        if iteration == 1:
            logger.info(
                "Using stage policy: %s-pass variance for iteration 1",
                2 if first_pass_2passvar else 1,
            )
        # Create trainer for this iteration
        trainer = BWTrainer(
            mdef_path=current_model / "mdef",
            means_path=current_model / "means",
            vars_path=current_model / "variances",
            mixw_path=current_model / "mixture_weights",
            tmat_path=current_model / "transition_matrices",
            config=iter_config,
        )

        # Set dictionary for text-based processing
        trainer.set_dict(dictionary, filler_dict)

        # Process all utterances
        processed = 0
        skipped = 0
        retried = 0
        excluded = 0
        skip_reasons = {
            "excluded_by_schedule": 0,
            "feature_not_found": 0,
            "transcript_not_found": 0,
            "feature_dimension": 0,
            "alignment_failure": 0,
            "exception": 0,
        }
        terminal_skips: list[dict[str, str]] = []
        accepted_exceptions: list[dict[str, object]] = []
        excluded_fileids = set(exclusion_schedule.get("*", ()))
        excluded_fileids.update(exclusion_schedule.get(iteration, ()))
        excluded_fileids.update(exclusion_schedule.get(str(iteration), ()))
        for fileid in fileids:
            if fileid in excluded_fileids:
                logger.info("Skipping %s on iteration %d: excluded_by_schedule", fileid, iteration)
                skipped += 1
                excluded += 1
                skip_reasons["excluded_by_schedule"] += 1
                continue
            # Load features
            mfc_path = features_dir / f"{fileid}.mfc"
            if not mfc_path.exists():
                logger.warning("Features not found: %s", mfc_path)
                skipped += 1
                skip_reasons["feature_not_found"] += 1
                continue

            # Get transcript
            if fileid not in transcripts:
                logger.warning("Transcript not found: %s", fileid)
                skipped += 1
                skip_reasons["transcript_not_found"] += 1
                continue

            try:
                # Load raw MFCC features (13-dim)
                # C code handles CMN and delta computation via feat module
                mfcc = read_sphinx_mfc(mfc_path)
                if mfcc.shape[1] != 13:
                    logger.warning("Unexpected feature dimension %d for %s", mfcc.shape[1], fileid)
                    skipped += 1
                    skip_reasons["feature_dimension"] += 1
                    continue

                # Get transcript and add <s> / </s> markers for C code
                text = transcripts[fileid]
                transcript = f"<s> {text} </s>"

                # Use process_utterance_mfcc - C handles CMN+deltas
                success = _process_with_final_state_retry(
                    trainer,
                    mfcc,
                    transcript,
                    iter_config.a_beam,
                    retry_beam_factor,
                    fileid,
                )
                if success:
                    if trainer._last_process_retried:
                        retried += 1
                    else:
                        processed += 1
                else:
                    logger.warning("Failed to process: %s", fileid)
                    skipped += 1
                    skip_reasons["alignment_failure"] += 1
                    terminal_skips.append({"utterance": fileid, "reason": "alignment_failure"})
            except TerminalAlignmentError:
                occupancy = _accept_arctic_a0302_exception(
                    fileid=fileid,
                    model_dir=current_model,
                    band=arctic_a0302_zero_codebook_band,
                )
                if occupancy is None:
                    raise
                assert arctic_a0302_zero_codebook_band is not None
                skipped += 1
                skip_reasons["alignment_failure"] += 1
                terminal_skips.append({"utterance": fileid, "reason": "accepted_exception_band"})
                accepted_exceptions.append(
                    {
                        "sentinel": fileid,
                        "quantity": "exact_zero_codebooks",
                        "value": occupancy,
                        "band": list(arctic_a0302_zero_codebook_band),
                    }
                )
            except Exception as e:
                logger.warning("Error processing %s: %s", fileid, e)
                skipped += 1
                skip_reasons["exception"] += 1
                terminal_skips.append({"utterance": fileid, "reason": "exception"})

        total_skipped += skipped
        if skipped:
            logger.warning(
                "WARNING: iteration %d skipped %d/%d utterance updates (%.2f%%)",
                iteration,
                skipped,
                len(fileids),
                100.0 * skipped / len(fileids),
            )
        else:
            logger.info(
                "Iteration %d processed %d utterances with zero skips",
                iteration,
                processed + retried,
            )

        skip_fraction = skipped / len(fileids) if fileids else 1.0
        if skip_fraction > max_skip_fraction:
            raise RuntimeError(
                f"Iteration {iteration}: skipped {skipped}/{len(fileids)} utterances "
                f"({skip_fraction:.2%}), above configured limit {max_skip_fraction:.2%}"
            )

        if processed + retried == 0:
            raise RuntimeError("No utterances processed successfully")

        # Get statistics BEFORE normalization (normalize resets stats)
        stats = trainer.get_stats()
        last_frames = stats.total_frames
        last_utts = stats.total_utts
        logger.info(
            "Iteration %d: likelihood=%.2f, frames=%d, utts=%d, avg=%.4f",
            iteration,
            stats.total_log_lik,
            stats.total_frames,
            stats.total_utts,
            stats.avg_log_prob,
        )
        per_frame_delta = (
            None if iteration == 1 else _convergence_delta(stats.avg_log_prob, prev_likelihood)
        )
        stop_decision = (
            "converged"
            if iteration > 1
            and _has_converged(
                stats.avg_log_prob,
                prev_likelihood,
                iteration,
                convergence_ratio,
                min_iterations,
            )
            else "cap"
            if iteration == n_iter
            else "continued"
        )
        trajectory.append(
            TrainingIteration(
                iteration=iteration,
                total_log_lik=stats.total_log_lik,
                avg_log_prob=stats.avg_log_prob,
                per_frame_delta=per_frame_delta,
                frames=stats.total_frames,
                input_utts=len(fileids),
                processed_utts=processed,
                retried_utts=retried,
                skipped_utts=skipped,
                excluded_by_schedule=excluded,
            )
        )
        telemetry_row: dict[str, object] = {
            "pass": iteration,
            "total_log_likelihood": stats.total_log_lik,
            "total_frames": stats.total_frames,
            "per_frame_log_likelihood": stats.avg_log_prob,
            "signed_convergence_delta": per_frame_delta,
            "stop_decision": stop_decision,
        }
        telemetry_row["accounting"] = {
            "input_utts": len(fileids),
            "processed_utts": processed,
            "retried_utts": retried,
            "skipped_utts": skipped,
            "skip_reasons": skip_reasons,
            "terminal_skips": terminal_skips,
            "accepted_exceptions": accepted_exceptions,
        }
        telemetry_rows.append(telemetry_row)
        logger.info(
            "BW telemetry: pass=%d total_log_likelihood=%.6f total_frames=%d "
            "per_frame_log_likelihood=%.6f signed_convergence_delta=%s stop=%s",
            iteration,
            stats.total_log_lik,
            stats.total_frames,
            stats.avg_log_prob,
            "null" if per_frame_delta is None else f"{per_frame_delta:.6f}",
            stop_decision,
        )
        _write_telemetry(
            output_dir,
            telemetry_rows,
            schema_version=2,
        )

        # Check for degenerate training (no successful utterances)
        if stats.total_frames == 0:
            raise RuntimeError(
                f"Iteration {iteration}: No frames processed. "
                "Model may be degenerate (check flat model initialization)."
            )

        # Save density counts BEFORE normalization (normalization clears accumulators)
        trainer.save_density_counts(output_dir / "gauden_counts")

        # Normalize accumulators (also resets stats for next iteration)
        trainer.normalize()

        # Save model
        trainer.save(
            means_path=output_dir / "means",
            vars_path=output_dir / "variances",
            mixw_path=output_dir / "mixture_weights",
            tmat_path=output_dir / "transition_matrices",
        )
        if checkpoints_enabled:
            _checkpoint_iteration(output_dir, iteration)

        # Check convergence
        if iteration > 1:
            change = _convergence_delta(stats.avg_log_prob, prev_likelihood)
            logger.info("Convergence ratio: %.6f (threshold: %.6f)", change, convergence_ratio)
            if change < 0:
                logger.warning(
                    "WARNING: negative convergence ratio at iteration %d; check BW inputs and logs",
                    iteration,
                )
            # SphinxTrain continues only for a strictly greater delta, and
            # otherwise enforces CFG_MIN_ITERATIONS before declaring convergence.
            if _has_converged(
                stats.avg_log_prob,
                prev_likelihood,
                iteration,
                convergence_ratio,
                min_iterations,
            ):
                if total_skipped:
                    logger.warning(
                        "WARNING: BW training skipped %d utterance updates in total",
                        total_skipped,
                    )
                logger.info("Converged after %d iterations", iteration)
                return TrainingResult(
                    iterations=iteration,
                    converged=True,
                    final_likelihood=stats.avg_log_prob,
                    final_frames=stats.total_frames,
                    final_utts=stats.total_utts,
                    total_skipped=total_skipped,
                    trajectory=tuple(trajectory),
                )

        prev_likelihood = stats.avg_log_prob

        # Next iteration reads from output
        current_model = output_dir

        # Clean up trainer
        del trainer

    if total_skipped:
        logger.warning("WARNING: BW training skipped %d utterance updates in total", total_skipped)
    logger.info(
        "Completed %d iterations (did not converge); total skipped=%d", n_iter, total_skipped
    )
    return TrainingResult(
        iterations=n_iter,
        converged=False,
        final_likelihood=prev_likelihood,
        final_frames=last_frames,
        final_utts=last_utts,
        total_skipped=total_skipped,
        trajectory=tuple(trajectory),
    )
