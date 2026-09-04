"""BW training step function.

Orchestrates Baum-Welch training using the CFFI-wrapped BWTrainer.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import multiprocessing
import os
import resource
import shutil
import sys
import time
import uuid
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

from pstrain.lib import native_worker
from pstrain.lib.bw import HMM, BWConfig, BWResult, BWTrainer
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
_ACCUMULATOR_FILES = ("gauden_counts", "mixw_counts", "tmat_counts")
_COPIED_TRAINING_OUTPUTS = ("mdef",)


def _flush_stdout() -> None:
    """Flush Python and C stdio before changing the process stdout fd."""
    sys.stdout.flush()
    # fflush(NULL) intentionally flushes every C stream in this process.
    ctypes.CDLL(None).fflush(None)


@contextmanager
def _redirect_stdout_fd(destination: Path) -> Iterator[None]:
    """Temporarily send fd 1, including native ``printf`` output, to a file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    saved_stdout = os.dup(1)
    redirected = False
    restored = False
    try:
        with destination.open("ab", buffering=0) as stream:
            _flush_stdout()
            os.dup2(stream.fileno(), 1)
            redirected = True
            try:
                yield
            finally:
                with suppress(Exception):
                    _flush_stdout()
                try:
                    os.dup2(saved_stdout, 1)
                except OSError:
                    logger.warning(
                        "Could not restore stdout fd; retaining saved fd %d", saved_stdout
                    )
                    raise
                else:
                    restored = True
    finally:
        if not redirected or restored:
            os.close(saved_stdout)


@contextmanager
def _redirect_bw_stdout(destination: Path) -> Iterator[None]:
    """Redirect native BW output in whichever process owns the trainer."""
    if native_worker.in_worker():
        with _redirect_stdout_fd(destination):
            yield
        return
    native_worker.call("stdout_redirect", (str(destination),), (destination,))
    try:
        yield
    finally:
        native_worker.call("stdout_restore", (), ())


_BW_COLUMN_HEADER = (
    b"""column defns
\t<seq>
\t<id>
\t<n_frame_in>
\t<n_frame_del>
\t<n_state_shmm>
\t<avg_states_alpha>
\t<avg_states_beta>
\t<avg_states_reest>
\t<avg_posterior_prune>
\t<frame_log_lik>
\t<utt_log_lik>
"""
    + b"\t... timing info ... \n"
)


class TerminalAlignmentError(RuntimeError):
    """An utterance still cannot reach its final state after retry."""


def _sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _fingerprint_model(model_dir: Path) -> str:
    return _sha256_files([model_dir / name for name in MODEL_FILES_REQUIRED])


def _fingerprint_config(config: BWConfig) -> str:
    return hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _fingerprint_manifest(fileids: list[str]) -> str:
    return hashlib.sha256("".join(f"{fileid}\n" for fileid in fileids).encode()).hexdigest()


@dataclass(frozen=True)
class _ShardResult:
    shard: int
    assigned_ids: tuple[str, ...]
    processed_ids: tuple[str, ...]
    retried_ids: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...]
    total_log_lik: float
    total_frames: int
    accum_dir: Path
    accepted_exceptions: tuple[tuple[str, int, int, int], ...] = ()
    user_cpu_seconds: float = 0.0


def _write_shard_metadata(
    result: _ShardResult,
    *,
    iteration: int,
    model_fingerprint: str,
    config_fingerprint: str,
    manifest_fingerprint: str,
    shapes: dict[str, list[int]],
) -> None:
    payload_files = [result.accum_dir / name for name in _ACCUMULATOR_FILES]
    metadata = {
        "schema_version": 1,
        "pass": iteration,
        "shard": result.shard,
        "model_fingerprint": model_fingerprint,
        "config_fingerprint": config_fingerprint,
        "manifest_fingerprint": manifest_fingerprint,
        "parameter_shapes": shapes,
        "assigned_ids": list(result.assigned_ids),
        "processed_ids": list(result.processed_ids),
        "retried_ids": list(result.retried_ids),
        "skipped": [list(item) for item in result.skipped],
        "accepted_exceptions": [list(item) for item in result.accepted_exceptions],
        "payload_sha256": _sha256_files(payload_files),
    }
    (result.accum_dir / "artifact.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _validate_shard_artifacts(
    accum_dirs: list[Path],
    *,
    iteration: int,
    fileids: list[str],
    model_fingerprint: str,
    config_fingerprint: str,
    manifest_fingerprint: str,
    shapes: dict[str, list[int]],
) -> list[dict[str, object]]:
    expected_ids = set(fileids)
    if len(expected_ids) != len(fileids):
        raise RuntimeError("BW manifest contains duplicate utterance IDs")
    seen_ids: set[str] = set()
    seen_shards: set[int] = set()
    metadata_rows: list[dict[str, object]] = []
    for accum_dir in accum_dirs:
        metadata_path = accum_dir / "artifact.json"
        if not metadata_path.is_file():
            raise RuntimeError(f"Missing BW shard metadata: {metadata_path}")
        row = json.loads(metadata_path.read_text(encoding="utf-8"))
        shard = int(row["shard"])
        if shard in seen_shards:
            raise RuntimeError(f"Duplicate BW shard artifact: {shard}")
        seen_shards.add(shard)
        expected = {
            "pass": iteration,
            "model_fingerprint": model_fingerprint,
            "config_fingerprint": config_fingerprint,
            "manifest_fingerprint": manifest_fingerprint,
            "parameter_shapes": shapes,
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise RuntimeError(f"Incompatible BW shard {shard}: {key}")
        payload_files = [accum_dir / name for name in _ACCUMULATOR_FILES]
        if not all(path.is_file() for path in payload_files):
            raise RuntimeError(f"Missing accumulator payload for BW shard {shard}")
        if row.get("payload_sha256") != _sha256_files(payload_files):
            raise RuntimeError(f"Accumulator payload digest mismatch for BW shard {shard}")
        assigned = set(row["assigned_ids"])
        overlap = seen_ids & assigned
        if overlap:
            raise RuntimeError(f"Overlapping BW shard coverage: {sorted(overlap)}")
        seen_ids.update(assigned)
        outcomes = (
            set(row["processed_ids"])
            | set(row["retried_ids"])
            | {item[0] for item in row["skipped"]}
        )
        if outcomes != assigned:
            raise RuntimeError(f"BW shard {shard} outcomes do not match assigned IDs")
        metadata_rows.append(row)
    if seen_ids != expected_ids:
        raise RuntimeError(f"Missing BW shard coverage: {sorted(expected_ids - seen_ids)}")
    return metadata_rows


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
    failed_alignment: Literal["recover", "abort", "omit"] = "recover",
) -> bool:
    """Process an update, retrying only a forward-final-state pruning failure.

    ``BWTrainer`` is a mutable native session and must not be shared between
    threads. The debug assertion makes concurrent entry at this mutation seam
    fail instead of allowing another call to observe the temporary beam.
    """
    assert not trainer._retry_transaction_active, "BWTrainer cannot be shared across threads"
    trainer._last_process_retried = False
    trainer._retry_transaction_active = True

    def process() -> bool:
        if isinstance(trainer, BWTrainer):
            return trainer.process_utterance_mfcc(mfcc, transcript, fileid)
        return trainer.process_utterance_mfcc(mfcc, transcript)

    try:
        success = process()
        if success:
            return success

        if not trainer.final_state_not_reached:
            if failed_alignment == "abort":
                raise TerminalAlignmentError(f"Alignment failed for {fileid}")
            if failed_alignment == "omit":
                logger.error("%s ignored after failed alignment", fileid)
            return False

        if failed_alignment == "omit":
            logger.error("%s ignored after failed alignment", fileid)
            return False

        if failed_alignment == "abort" or retry_beam_factor <= 1.0:
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
            success = process()
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


def _partition_manifest(
    fileids: list[str],
    n_shards: int,
    partition_position: Literal["remainder-first", "remainder-last"] = "remainder-first",
) -> list[list[str]]:
    """Construct contiguous ranges using the declared remainder position."""
    if n_shards < 1:
        raise ValueError("n_shards must be at least 1")
    quotient, remainder = divmod(len(fileids), n_shards)
    partitions: list[list[str]] = []
    start = 0
    if partition_position not in {"remainder-first", "remainder-last"}:
        raise ValueError(f"unknown partition position: {partition_position}")
    for shard in range(n_shards):
        if partition_position == "remainder-first":
            width = quotient + (1 if shard < remainder else 0)
        else:
            width = quotient + (remainder if shard == n_shards - 1 else 0)
        partitions.append(fileids[start : start + width])
        start += width
    return partitions


def _ordered_shard_results(futures: Iterable[object]) -> list[_ShardResult]:
    """Collect worker results and pin every downstream merge to shard index."""
    results = [future.result() for future in futures]  # type: ignore[attr-defined]
    return sorted(results, key=lambda result: result.shard)


def _effective_bw_shard_count(n_shards: int, *, multipron: bool) -> int:
    """Resolve the loud multipron fallback before a training pass starts."""
    if n_shards < 1:
        raise ValueError("n_shards must be at least 1")
    if n_shards > 1 and multipron:
        logger.warning(
            "BW sharding disabled because multipron_training=true: fallback_senone "
            "is pass-wide state; running serially"
        )
        return 1
    return n_shards


def _initialize_bw_pool_worker() -> None:
    """Make the process-pool worker itself the native containment boundary."""
    from pstrain.lib import native_worker

    native_worker._inside_worker = True


def _run_bw_shard(
    shard: int,
    assigned_ids: list[str],
    current_model: Path,
    features_dir: Path,
    transcripts: dict[str, str],
    dictionary: Path,
    filler_dict: Path | None,
    iter_config: BWConfig,
    retry_beam_factor: float,
    failed_alignment: Literal["recover", "abort", "omit"],
    excluded_fileids: set[str],
    accum_dir: Path,
    iteration: int,
    arctic_a0302_zero_codebook_band: tuple[int, int] | None,
    accept_arctic_a0587_pass: int | None,
    diagnostic_log: Path,
) -> _ShardResult:
    user_cpu_start = resource.getrusage(resource.RUSAGE_SELF).ru_utime
    trainer = BWTrainer(
        mdef_path=current_model / "mdef",
        means_path=current_model / "means",
        vars_path=current_model / "variances",
        mixw_path=current_model / "mixture_weights",
        tmat_path=current_model / "transition_matrices",
        config=iter_config,
    )
    trainer.set_dict(dictionary, filler_dict)
    processed: list[str] = []
    retried: list[str] = []
    skipped: list[tuple[str, str]] = []
    accepted_exceptions: list[tuple[str, int, int, int]] = []
    diagnostic_log.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_log.write_bytes(_BW_COLUMN_HEADER)
    for fileid in assigned_ids:
        if fileid in excluded_fileids:
            skipped.append((fileid, "excluded_by_schedule"))
            continue
        mfc_path = features_dir / f"{fileid}.mfc"
        if not mfc_path.exists():
            skipped.append((fileid, "feature_not_found"))
            continue
        if fileid not in transcripts:
            skipped.append((fileid, "transcript_not_found"))
            continue
        try:
            mfcc = read_sphinx_mfc(mfc_path)
            if mfcc.shape[1] != 13:
                skipped.append((fileid, "feature_dimension"))
                continue
            with _redirect_bw_stdout(diagnostic_log):
                success = _process_with_final_state_retry(
                    trainer,
                    mfcc,
                    f"<s> {transcripts[fileid]} </s>",
                    iter_config.a_beam,
                    retry_beam_factor,
                    fileid,
                    failed_alignment,
                )
            if not success:
                skipped.append((fileid, "alignment_failure"))
            elif trainer._last_process_retried:
                retried.append(fileid)
            else:
                processed.append(fileid)
        except TerminalAlignmentError:
            if fileid == "arctic_a0587" and iteration == accept_arctic_a0587_pass:
                skipped.append((fileid, "alignment_failure"))
                continue
            occupancy = _accept_arctic_a0302_exception(
                fileid=fileid,
                model_dir=current_model,
                band=arctic_a0302_zero_codebook_band,
            )
            if occupancy is None:
                raise
            assert arctic_a0302_zero_codebook_band is not None
            lower, upper = arctic_a0302_zero_codebook_band
            accepted_exceptions.append((fileid, occupancy, lower, upper))
            skipped.append((fileid, "alignment_failure"))
        except Exception:
            logger.exception("Error processing %s in BW shard %d", fileid, shard)
            skipped.append((fileid, "exception"))
    stats = trainer.get_stats()
    trainer.dump_accumulators(accum_dir)
    return _ShardResult(
        shard=shard,
        assigned_ids=tuple(assigned_ids),
        processed_ids=tuple(processed),
        retried_ids=tuple(retried),
        skipped=tuple(skipped),
        total_log_lik=stats.total_log_lik,
        total_frames=stats.total_frames,
        accum_dir=accum_dir,
        accepted_exceptions=tuple(accepted_exceptions),
        user_cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime - user_cpu_start,
    )


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
    failed_alignment: Literal["recover", "abort", "omit"] = "recover",
    checkpoint_iterations: bool = False,
    exclusion_schedule: dict[int | str, list[str]] | None = None,
    arctic_a0302_zero_codebook_band: tuple[int, int] | None = None,
    accept_arctic_a0587_pass: int | None = None,
    n_shards: int = 1,
    partition_position: Literal["remainder-first", "remainder-last"] = "remainder-first",
    project_dir: Path | None = None,
    stage: str | None = None,
    _in_process_reference: bool = False,
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
        failed_alignment: Recover with one widened-beam retry, abort the stage,
            or report and omit the utterance while continuing.
        checkpoint_iterations: Retain the compact model files produced by each
            pass. The deprecated ``PSTRAIN_BW_CHECKPOINTS=1`` environment
            variable can also enable retention, but cannot disable this setting.
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
    n_shards = _effective_bw_shard_count(n_shards, multipron=multipron)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    features_dir = Path(features_dir)
    project_dir = Path(project_dir) if project_dir is not None else output_dir.parent
    stage = stage or output_dir.name
    diagnostic_dir = project_dir / ".pstrain" / "bw" / stage
    print(f"bw-logs\t{diagnostic_dir}")

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

    checkpoints_enabled = checkpoint_iterations or os.environ.get("PSTRAIN_BW_CHECKPOINTS") == "1"
    if checkpoints_enabled:
        shutil.rmtree(output_dir / "iterations", ignore_errors=True)

    # Copy structural inputs unchanged; these are not produced by training.
    for filename in _COPIED_TRAINING_OUTPUTS:
        shutil.copy(model_dir / filename, output_dir / filename)

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
        pass_wall_start = time.perf_counter()
        pass_self_start = resource.getrusage(resource.RUSAGE_SELF).ru_utime
        pass_children_start = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime

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
        merged_stats: BWResult | None = None
        shard_metadata: list[dict[str, object]] = []
        iteration_fileids = fileids
        serial_diagnostic_log = diagnostic_dir / f"pass-{iteration:02d}-shard-00.log"
        if iteration_fileids:
            serial_diagnostic_log.parent.mkdir(parents=True, exist_ok=True)
            serial_diagnostic_log.write_bytes(_BW_COLUMN_HEADER)
        if not multipron and not _in_process_reference:
            iteration_fileids = []
            pass_root = output_dir / ".bw-accum" / f"pass-{iteration:02d}"
            shutil.rmtree(pass_root, ignore_errors=True)
            pass_root.mkdir(parents=True)
            partitions = _partition_manifest(fileids, n_shards, partition_position)
            shard_dirs = [pass_root / f"shard-{index:05d}" for index in range(n_shards)]
            arguments = [
                (
                    index,
                    assigned,
                    current_model,
                    features_dir,
                    transcripts,
                    dictionary,
                    filler_dict,
                    iter_config,
                    retry_beam_factor,
                    failed_alignment,
                    excluded_fileids,
                    shard_dirs[index],
                    iteration,
                    arctic_a0302_zero_codebook_band,
                    accept_arctic_a0587_pass,
                    diagnostic_dir / f"pass-{iteration:02d}-shard-{index:02d}.log",
                )
                for index, assigned in enumerate(partitions)
            ]
            with ProcessPoolExecutor(
                max_workers=n_shards,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_bw_pool_worker,
            ) as pool:
                futures = [pool.submit(_run_bw_shard, *argument) for argument in arguments]
                shard_results = _ordered_shard_results(futures)
            model_fingerprint = _fingerprint_model(current_model)
            config_fingerprint = _fingerprint_config(iter_config)
            manifest_fingerprint = _fingerprint_manifest(fileids)
            hmm = HMM.load(current_model)
            shapes = {
                "means": list(hmm.means.shape),
                "variances": list(hmm.variances.shape),
                "mixture_weights": list(hmm.mixw.shape),
                "transition_matrices": list(hmm.tmat.shape),
            }
            for result in shard_results:
                _write_shard_metadata(
                    result,
                    iteration=iteration,
                    model_fingerprint=model_fingerprint,
                    config_fingerprint=config_fingerprint,
                    manifest_fingerprint=manifest_fingerprint,
                    shapes=shapes,
                )
            shard_metadata = _validate_shard_artifacts(
                shard_dirs,
                iteration=iteration,
                fileids=fileids,
                model_fingerprint=model_fingerprint,
                config_fingerprint=config_fingerprint,
                manifest_fingerprint=manifest_fingerprint,
                shapes=shapes,
            )
            trainer.restore_accumulators(shard_dirs)
            processed_ids = [item for result in shard_results for item in result.processed_ids]
            retried_ids = [item for result in shard_results for item in result.retried_ids]
            skipped_items = [item for result in shard_results for item in result.skipped]
            processed = len(processed_ids)
            retried = len(retried_ids)
            skipped = len(skipped_items)
            excluded = sum(reason == "excluded_by_schedule" for _, reason in skipped_items)
            for fileid, reason in skipped_items:
                skip_reasons[reason] += 1
                if reason not in {
                    "excluded_by_schedule",
                    "feature_not_found",
                    "transcript_not_found",
                    "feature_dimension",
                }:
                    terminal_skips.append({"utterance": fileid, "reason": reason})
            for result in shard_results:
                for fileid, shard_occupancy, lower, upper in result.accepted_exceptions:
                    accepted_exceptions.append(
                        {
                            "sentinel": fileid,
                            "quantity": "exact_zero_codebooks",
                            "value": shard_occupancy,
                            "band": [lower, upper],
                        }
                    )
            total_log_lik = sum(result.total_log_lik for result in shard_results)
            total_frames = sum(result.total_frames for result in shard_results)
            merged_stats = BWResult(
                total_log_lik=total_log_lik,
                total_frames=total_frames,
                total_utts=processed + retried,
                avg_log_prob=total_log_lik / total_frames if total_frames else 0.0,
            )
        for fileid in iteration_fileids:
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
                with _redirect_bw_stdout(serial_diagnostic_log):
                    success = _process_with_final_state_retry(
                        trainer,
                        mfcc,
                        transcript,
                        iter_config.a_beam,
                        retry_beam_factor,
                        fileid,
                        failed_alignment,
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
        stats = merged_stats or trainer.get_stats()
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
        pass_wall = time.perf_counter() - pass_wall_start
        pass_user_cpu = (
            resource.getrusage(resource.RUSAGE_SELF).ru_utime
            - pass_self_start
            + resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
            - pass_children_start
        )
        if not multipron and not _in_process_reference:
            pass_user_cpu = sum(result.user_cpu_seconds for result in shard_results)
        telemetry_row["performance"] = {
            "wall_seconds": pass_wall,
            "user_cpu_seconds": pass_user_cpu,
            "parallelism_user_cpu_per_wall": pass_user_cpu / pass_wall if pass_wall else 0.0,
            "workers": n_shards if not multipron and not _in_process_reference else 1,
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
        if shard_metadata:
            telemetry_row["shards"] = shard_metadata
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
