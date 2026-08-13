"""BW training step function.

Orchestrates Baum-Welch training using the CFFI-wrapped BWTrainer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from dataclasses import asdict, dataclass
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
    fallback_senone_ids: tuple[int, ...] = ()
    accumulation_wall_seconds: float = 0.0
    worker_user_cpu_seconds: float = 0.0


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


@dataclass
class _ShardResult:
    index: int
    accum_dir: Path
    metadata: dict[str, object]


_SHARD_SCHEMA = 1
_SHARD_METADATA = "pstrain_bw_artifact.json"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_fingerprint(value: object) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def _model_fingerprint(model_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("mdef", "means", "variances", "mixture_weights", "transition_matrices"):
        digest.update(name.encode())
        with (model_dir / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _parameter_shapes(model_dir: Path) -> dict[str, list[int]]:
    from pstrain.lib.bw import HMM

    model = HMM.load(model_dir)
    return {
        "means": list(model.means.shape),
        "variances": list(model.variances.shape),
        "mixture_weights": list(model.mixw.shape),
        "transition_matrices": list(model.tmat.shape),
    }


def _write_shard_metadata(directory: Path, metadata: dict[str, object]) -> None:
    path = directory / _SHARD_METADATA
    path.write_text(
        json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_shard_artifacts(
    results: list[_ShardResult],
    *,
    expected_ids: list[str],
    pass_number: int,
    model_fingerprint: str,
    config_fingerprint: str,
    manifest_fingerprint: str,
    parameter_shapes: dict[str, list[int]],
) -> list[_ShardResult]:
    """Reject incomplete, overlapping, stale, or incompatible shard collections."""
    expected = set(expected_ids)
    if len(expected) != len(expected_ids):
        raise RuntimeError("BW manifest contains duplicate utterance identities")
    expected_contract: dict[str, object] = {
        "schema_version": _SHARD_SCHEMA,
        "pass_number": pass_number,
        "model_fingerprint": model_fingerprint,
        "config_fingerprint": config_fingerprint,
        "manifest_fingerprint": manifest_fingerprint,
        "parameter_shapes": parameter_shapes,
    }
    ordered = sorted(results, key=lambda result: result.index)
    indexes: set[int] = set()
    covered: set[str] = set()
    previous_end = 0
    for result in ordered:
        path = result.accum_dir / _SHARD_METADATA
        if not path.is_file():
            raise RuntimeError(f"missing BW shard metadata: {path}")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata != result.metadata:
            raise RuntimeError(f"BW shard metadata changed after worker completion: {path}")
        for key, value in expected_contract.items():
            if metadata.get(key) != value:
                raise RuntimeError(f"incompatible BW shard {result.index}: {key}")
        index = metadata.get("shard_index")
        if not isinstance(index, int) or index != result.index or index in indexes:
            raise RuntimeError(f"duplicate or invalid BW shard index: {index!r}")
        indexes.add(index)
        assigned = metadata.get("assigned_ids")
        processed = metadata.get("processed_ids")
        skipped = metadata.get("skipped")
        contributions = metadata.get("log_lik_contributions")
        span = metadata.get("serial_span")
        if not isinstance(assigned, list) or not all(isinstance(x, str) for x in assigned):
            raise RuntimeError(f"invalid assigned identities in BW shard {index}")
        if len(assigned) != len(set(assigned)) or covered.intersection(assigned):
            raise RuntimeError(f"overlapping BW shard coverage at shard {index}")
        if not isinstance(processed, list) or not all(isinstance(x, str) for x in processed):
            raise RuntimeError(f"invalid processed identities in BW shard {index}")
        if not isinstance(skipped, list) or not all(
            isinstance(x, dict) and isinstance(x.get("utterance"), str) for x in skipped
        ):
            raise RuntimeError(f"invalid skipped identities in BW shard {index}")
        if (
            not isinstance(contributions, list)
            or len(contributions) != len(processed)
            or not all(isinstance(value, float) for value in contributions)
        ):
            raise RuntimeError(f"invalid likelihood contributions in BW shard {index}")
        skipped_ids = [item["utterance"] for item in skipped]
        if set(processed).intersection(skipped_ids) or set(processed) | set(skipped_ids) != set(
            assigned
        ):
            raise RuntimeError(f"BW shard {index} accounting is not exactly once")
        if metadata.get("attempted_count") != len(assigned):
            raise RuntimeError(f"BW shard {index} attempted count mismatch")
        if metadata.get("accumulated_count") != len(processed):
            raise RuntimeError(f"BW shard {index} accumulated count mismatch")
        if not isinstance(span, list) or len(span) != 2 or span[0] != previous_end:
            raise RuntimeError(f"BW shard {index} is not in serial utterance order")
        if span[1] - span[0] != len(assigned) or assigned != expected_ids[span[0] : span[1]]:
            raise RuntimeError(f"BW shard {index} serial span does not match manifest")
        previous_end = span[1]
        covered.update(assigned)
    if covered != expected or previous_end != len(expected_ids):
        missing = sorted(expected - covered)
        extra = sorted(covered - expected)
        raise RuntimeError(f"incomplete BW shard coverage: missing={missing}, extra={extra}")
    return ordered


def _run_bw_shard(
    index: int,
    entries: list[tuple[str, Path, str]],
    accum_dir: Path,
    model_dir: Path,
    dictionary: Path,
    filler_dict: Path | None,
    config: BWConfig,
    retry_beam_factor: float,
    contract: dict[str, object],
    serial_span: tuple[int, int],
) -> _ShardResult:
    """Accumulate one contiguous corpus shard in an isolated native session."""
    from pstrain.lib import native_worker

    # A process-pool shard is already a crash-isolated native process.  Mark it
    # as such so contained CFFI calls execute here instead of recursively
    # starting a second helper process (which can deadlock during spawn).
    native_worker._inside_worker = True
    trainer = BWTrainer(
        model_dir / "mdef",
        model_dir / "means",
        model_dir / "variances",
        model_dir / "mixture_weights",
        model_dir / "transition_matrices",
        config,
    )
    trainer.set_dict(dictionary, filler_dict)
    processed: list[str] = []
    retried: list[str] = []
    failures: list[dict[str, str]] = []
    log_lik_contributions: list[float] = []
    for fileid, mfc_path, transcript in entries:
        try:
            mfcc = read_sphinx_mfc(mfc_path)
            if mfcc.shape[1] != 13:
                failures.append({"utterance": fileid, "reason": "feature_dimension"})
                continue
            if _process_with_final_state_retry(
                trainer, mfcc, transcript, config.a_beam, retry_beam_factor, fileid
            ):
                if trainer._last_process_retried:
                    retried.append(fileid)
                processed.append(fileid)
                log_lik_contributions.append(trainer.last_log_lik())
            else:
                failures.append({"utterance": fileid, "reason": "alignment_failure"})
        except TerminalAlignmentError:
            failures.append({"utterance": fileid, "reason": "terminal_alignment"})
        except Exception:
            logger.exception("Error processing %s in BW shard %d", fileid, index)
            failures.append({"utterance": fileid, "reason": "exception"})
    if not trainer.dump_accumulators(accum_dir):
        raise RuntimeError(f"Failed to dump BW shard {index}")
    metadata = {
        **contract,
        "shard_index": index,
        "serial_span": list(serial_span),
        "assigned_ids": [entry[0] for entry in entries],
        "processed_ids": processed,
        "retried_ids": retried,
        "skipped": failures,
        "attempted_count": len(entries),
        "accumulated_count": len(processed),
        "skipped_count": len(failures),
        "log_lik_contributions": log_lik_contributions,
    }
    _write_shard_metadata(accum_dir, metadata)
    return _ShardResult(index, accum_dir, metadata)


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
    accept_arctic_a0587_pass: int | None = None,
    jobs: int | None = 1,
    shard_boundaries: tuple[int, ...] | None = None,
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
        jobs: Independent BW accumulator shards. ``None`` uses CPU count minus two.
            Zero is reserved for the inline serial-reference path used by parity gates.
        shard_boundaries: Optional serial-position boundaries for parity experiments;
            must contain ``jobs + 1`` nondecreasing offsets from zero through
            the eligible utterance count.

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
    if len(fileids) != len(set(fileids)):
        raise ValueError("training file-ID manifest contains duplicate utterance identities")
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
        assigned_ids: list[str] = []
        accumulated_ids: list[str] = []
        accepted_exceptions: list[dict[str, object]] = []
        excluded_fileids = set(exclusion_schedule.get("*", ()))
        excluded_fileids.update(exclusion_schedule.get(iteration, ()))
        excluded_fileids.update(exclusion_schedule.get(str(iteration), ()))
        requested_jobs = max(1, (os.cpu_count() or 2) - 2) if jobs is None else jobs
        if requested_jobs < 0:
            raise ValueError("jobs must be a nonnegative integer or None")
        shapes = _parameter_shapes(current_model)
        start_times = os.times()
        accumulation_started = time.perf_counter()
        fileids_for_serial: list[str] = fileids if requested_jobs == 0 else []
        if requested_jobs >= 1:
            entries: list[tuple[str, Path, str]] = []
            for fileid in fileids:
                if fileid in excluded_fileids:
                    logger.info(
                        "Skipping %s on iteration %d: excluded_by_schedule", fileid, iteration
                    )
                    skipped += 1
                    excluded += 1
                    skip_reasons["excluded_by_schedule"] += 1
                    terminal_skips.append({"utterance": fileid, "reason": "excluded_by_schedule"})
                    continue
                mfc_path = features_dir / f"{fileid}.mfc"
                if not mfc_path.exists():
                    logger.warning("Features not found: %s", mfc_path)
                    skipped += 1
                    skip_reasons["feature_not_found"] += 1
                    terminal_skips.append({"utterance": fileid, "reason": "feature_not_found"})
                    continue
                if fileid not in transcripts:
                    logger.warning("Transcript not found: %s", fileid)
                    skipped += 1
                    skip_reasons["transcript_not_found"] += 1
                    terminal_skips.append({"utterance": fileid, "reason": "transcript_not_found"})
                    continue
                entries.append((fileid, mfc_path, f"<s> {transcripts[fileid]} </s>"))

            worker_count = requested_jobs
            if worker_count:
                # Contiguous partitions preserve corpus order within every shard;
                # numbered merge order is independent of worker completion order.
                shard_size = (len(entries) + worker_count - 1) // worker_count if entries else 0
                if shard_boundaries is not None and (
                    len(shard_boundaries) != worker_count + 1
                    or shard_boundaries[0] != 0
                    or shard_boundaries[-1] != len(entries)
                    or any(
                        left > right
                        for left, right in zip(shard_boundaries, shard_boundaries[1:], strict=False)
                    )
                ):
                    raise ValueError(
                        "shard_boundaries must be jobs + 1 nondecreasing serial offsets "
                        "from zero through the eligible utterance count"
                    )
                with tempfile.TemporaryDirectory(prefix="pstrain-bw-shards-") as temp_root:
                    root = Path(temp_root)
                    futures = []
                    context = multiprocessing.get_context("spawn")
                    manifest_ids = [entry[0] for entry in entries]
                    assigned_ids = manifest_ids
                    model_fingerprint = _model_fingerprint(current_model)
                    config_fingerprint = _json_fingerprint(asdict(iter_config))
                    manifest_fingerprint = _json_fingerprint(manifest_ids)
                    contract: dict[str, object] = {
                        "schema_version": _SHARD_SCHEMA,
                        "pass_number": iteration,
                        "model_fingerprint": model_fingerprint,
                        "config_fingerprint": config_fingerprint,
                        "manifest_fingerprint": manifest_fingerprint,
                        "parameter_shapes": shapes,
                    }
                    with ProcessPoolExecutor(
                        max_workers=worker_count, mp_context=context
                    ) as executor:
                        for index in range(worker_count):
                            if shard_boundaries is None:
                                start = min(index * shard_size, len(entries))
                                end = min((index + 1) * shard_size, len(entries))
                            else:
                                start, end = shard_boundaries[index : index + 2]
                            shard_entries = entries[start:end]
                            accum_dir = root / f"{index:06d}"
                            futures.append(
                                executor.submit(
                                    _run_bw_shard,
                                    index,
                                    shard_entries,
                                    accum_dir,
                                    current_model,
                                    dictionary,
                                    filler_dict,
                                    iter_config,
                                    retry_beam_factor,
                                    contract,
                                    (start, end),
                                )
                            )
                        shard_results = _validate_shard_artifacts(
                            [future.result() for future in futures],
                            expected_ids=manifest_ids,
                            pass_number=iteration,
                            model_fingerprint=model_fingerprint,
                            config_fingerprint=config_fingerprint,
                            manifest_fingerprint=manifest_fingerprint,
                            parameter_shapes=shapes,
                        )
                    if not trainer.merge_accumulators(
                        [result.accum_dir for result in shard_results]
                    ):
                        raise RuntimeError("Failed to merge BW accumulator shards")
                    canonical_log_lik = 0.0
                    for result in shard_results:
                        contributions = result.metadata["log_lik_contributions"]
                        assert isinstance(contributions, list)
                        for contribution in contributions:
                            canonical_log_lik += float(contribution)
                    trainer.set_total_log_lik(canonical_log_lik)
                    for result in shard_results:
                        processed_ids = result.metadata["processed_ids"]
                        retried_ids = result.metadata["retried_ids"]
                        failures = result.metadata["skipped"]
                        assert isinstance(processed_ids, list)
                        assert isinstance(retried_ids, list)
                        assert isinstance(failures, list)
                        processed += len(processed_ids) - len(retried_ids)
                        retried += len(retried_ids)
                        for fileid in retried_ids:
                            logger.warning(
                                "Final state not reached for %s; retrying once in BW shard",
                                fileid,
                            )
                        accumulated_ids.extend(str(item) for item in processed_ids)
                        for failure in failures:
                            assert isinstance(failure, dict)
                            fileid = str(failure["utterance"])
                            reason = str(failure["reason"])
                            if reason == "terminal_alignment":
                                if (
                                    fileid == "arctic_a0587"
                                    and iteration == accept_arctic_a0587_pass
                                ):
                                    reason = "alignment_failure"
                                else:
                                    occupancy = _accept_arctic_a0302_exception(
                                        fileid=fileid,
                                        model_dir=current_model,
                                        band=arctic_a0302_zero_codebook_band,
                                    )
                                    if occupancy is None:
                                        raise TerminalAlignmentError(
                                            f"Final state not reached for {fileid} after retry"
                                        )
                                    reason = "accepted_exception_band"
                                    assert arctic_a0302_zero_codebook_band is not None
                                    accepted_exceptions.append(
                                        {
                                            "sentinel": fileid,
                                            "quantity": "exact_zero_codebooks",
                                            "value": occupancy,
                                            "band": list(arctic_a0302_zero_codebook_band),
                                        }
                                    )
                            skipped += 1
                            key = (
                                "alignment_failure"
                                if reason in {"alignment_failure", "accepted_exception_band"}
                                else reason
                            )
                            skip_reasons[key] += 1
                            terminal_skips.append({"utterance": fileid, "reason": reason})

        for fileid in fileids_for_serial:
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
                if fileid == "arctic_a0587" and iteration == accept_arctic_a0587_pass:
                    logger.warning(
                        "KNOWN EXCEPTION arctic_a0587: terminal alignment failure at "
                        "ratified pass %d; continuing without this utterance update",
                        iteration,
                    )
                    skipped += 1
                    skip_reasons["alignment_failure"] += 1
                    terminal_skips.append({"utterance": fileid, "reason": "alignment_failure"})
                    continue
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
        accumulation_wall_seconds = time.perf_counter() - accumulation_started
        end_times = os.times()
        worker_user_cpu_seconds = (
            end_times.user - start_times.user
            if requested_jobs == 0
            else end_times.children_user - start_times.children_user
        )
        fallback_senone_ids = tuple(
            senone
            for senone in range(shapes["mixture_weights"][0])
            if trainer.fallback_senone_active(senone)
        )
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
                fallback_senone_ids=fallback_senone_ids,
                accumulation_wall_seconds=accumulation_wall_seconds,
                worker_user_cpu_seconds=worker_user_cpu_seconds,
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
            "assigned_ids": assigned_ids,
            "accumulated_ids": accumulated_ids,
            "skipped_ids": [item["utterance"] for item in terminal_skips],
        }
        telemetry_row["fallback_senone_ids"] = list(fallback_senone_ids)
        telemetry_row["parallelism"] = {
            "accumulation_wall_seconds": accumulation_wall_seconds,
            "worker_user_cpu_seconds": worker_user_cpu_seconds,
            "user_cpu_per_wall": (
                worker_user_cpu_seconds / accumulation_wall_seconds
                if accumulation_wall_seconds
                else 0.0
            ),
            "declared_workers": requested_jobs,
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
