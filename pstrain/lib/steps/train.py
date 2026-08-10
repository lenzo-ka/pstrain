"""BW training step function.

Orchestrates Baum-Welch training using the CFFI-wrapped BWTrainer.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from pstrain.lib.bw import BWConfig, BWTrainer
from pstrain.lib.features import read_sphinx_mfc
from pstrain.lib.model import MODEL_FILES_REQUIRED
from pstrain.lib.transcription import parse_transcription_file
from pstrain.lib.validate import validate_files_exist

logger = logging.getLogger(__name__)

__all__ = ["run_bw_training", "TrainingResult"]


@dataclass
class TrainingResult:
    """Result from BW training."""

    iterations: int
    converged: bool
    final_likelihood: float
    final_frames: int
    final_utts: int
    total_skipped: int = 0


def run_bw_training(
    model_dir: Path,
    output_dir: Path,
    features_dir: Path,
    train_fileids: Path,
    transcription: Path,
    dictionary: Path,
    filler_dict: Path | None = None,
    n_iter: int = 10,
    convergence_ratio: float = 0.001,
    config: BWConfig | None = None,
    multipron: bool = True,
    max_skip_fraction: float = 0.05,
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
        convergence_ratio: Convergence threshold (relative change in likelihood)
        config: BW training configuration
        max_skip_fraction: Fail when skipped utterances exceed this fraction.

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

    # Copy mdef (unchanged during training)
    shutil.copy(model_dir / "mdef", output_dir / "mdef")

    prev_likelihood = float("-inf")
    current_model = model_dir
    last_frames = 0
    last_utts = 0
    total_skipped = 0

    for iteration in range(1, n_iter + 1):
        logger.info("Starting iteration %d/%d...", iteration, n_iter)

        # Use 1-pass variance for first iteration, 2-pass thereafter
        # This matches SphinxTrain behavior which uses -2passvar no for iter 1
        from dataclasses import replace

        if config is None:
            iter_config = BWConfig(pass2var=(iteration > 1), multipron=multipron)
        else:
            iter_config = config
            if iteration == 1 and config.pass2var:
                iter_config = replace(iter_config, pass2var=False)
                logger.info("Using 1-pass variance for iteration 1 (SphinxTrain-compatible)")
            if multipron != iter_config.multipron:
                iter_config = replace(iter_config, multipron=multipron)

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
        for fileid in fileids:
            # Load features
            mfc_path = features_dir / f"{fileid}.mfc"
            if not mfc_path.exists():
                logger.warning("Features not found: %s", mfc_path)
                skipped += 1
                continue

            # Get transcript
            if fileid not in transcripts:
                logger.warning("Transcript not found: %s", fileid)
                skipped += 1
                continue

            try:
                # Load raw MFCC features (13-dim)
                # C code handles CMN and delta computation via feat module
                mfcc = read_sphinx_mfc(mfc_path)
                if mfcc.shape[1] != 13:
                    logger.warning("Unexpected feature dimension %d for %s", mfcc.shape[1], fileid)
                    skipped += 1
                    continue

                # Get transcript and add <s> / </s> markers for C code
                text = transcripts[fileid]
                transcript = f"<s> {text} </s>"

                # Use process_utterance_mfcc - C handles CMN+deltas
                if trainer.process_utterance_mfcc(mfcc, transcript):
                    processed += 1
                else:
                    logger.warning("Failed to process: %s", fileid)
                    skipped += 1
            except Exception as e:
                logger.warning("Error processing %s: %s", fileid, e)
                skipped += 1

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
                "Iteration %d processed %d utterances with zero skips", iteration, processed
            )

        skip_fraction = skipped / len(fileids) if fileids else 1.0
        if skip_fraction > max_skip_fraction:
            raise RuntimeError(
                f"Iteration {iteration}: skipped {skipped}/{len(fileids)} utterances "
                f"({skip_fraction:.2%}), above configured limit {max_skip_fraction:.2%}"
            )

        if processed == 0:
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

        # Check convergence
        if iteration > 1 and prev_likelihood != 0:
            change = abs(stats.avg_log_prob - prev_likelihood) / abs(prev_likelihood)
            logger.info("Relative change: %.6f (threshold: %.6f)", change, convergence_ratio)
            if change < convergence_ratio:
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
    )
