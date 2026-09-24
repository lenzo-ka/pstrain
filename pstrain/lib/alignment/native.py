"""In-process forced alignment via the pstrain_align CFFI bindings.

Replaces both the PocketSphinx-based path and the former subprocess
wrapper around the standalone ``sphinx3_align`` binary. One
:class:`Aligner` instance keeps the acoustic model loaded for the
duration of a corpus, so per-utterance cost is just feature
extraction + Viterbi search.

Only one ``Aligner`` may be alive at a time per process; the underlying C
aligner holds module-static state. The second concurrent construction
raises ``RuntimeError`` until the first is closed.

Typical use::

    from pstrain.lib.alignment import Aligner

    with Aligner(model_dir, dict_path) as aligner:
        result = aligner.align_audio(audio_path, transcript)
"""

from __future__ import annotations

import contextlib
import numbers
import struct
import wave
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import numpy as np

from pstrain.lib import native_worker
from pstrain.lib._cffi.core import _init
from pstrain.lib.alignment.acceptance import (
    DEFAULT_RETRY_ACCEPTANCE_TARGET,
    RETRY_ACCEPTANCE_MAX_SAMPLES,
    AlignmentRejectedError,
    RetryCalibration,
    RungYield,
    calibration_from_scores,
    evenly_spaced,
    read_filler_words,
    rejection_message,
    speech_score,
)
from pstrain.lib.alignment.core import (
    DEFAULT_RETRY_BEAM_FACTOR,
    AlignedSegment,
    AlignmentResult,
    RetryOutcome,
)
from pstrain.lib.feasibility import infeasible_frames_message
from pstrain.lib.features import FeatureExtractor
from pstrain.lib.model import MODEL_FILES_REQUIRED, read_complete_model_feat_params
from pstrain.lib.pipeline.feat_params import feature_extractor_config_from_record
from pstrain.lib.retry_ladder import RetryBeamFactor, retry_ladder

if TYPE_CHECKING:
    import numpy.typing as npt


_DEFAULT_BEAM = 1e-64
_DEFAULT_RETRY_BEAM_FACTOR = DEFAULT_RETRY_BEAM_FACTOR
_DEFAULT_FEAT_TYPE = "1s_c_d_dd"
_DEFAULT_CMN = "batch"
_DEFAULT_AGC = "none"

# Why a final-state failure was not retried: the retry's recovery could not
# have been judged.
_RETRY_WITHHELD = (
    "; the wider-beam retry was not run, because an alignment it recovers must "
    "pass the retry acceptance check, which needs a threshold: pass "
    "retry_acceptance_threshold (Aligner.calibrate_retry_acceptance computes one "
    "from known-good utterances), align the corpus with align_corpus, which "
    "calibrates one, or set retry_acceptance_target=None to accept retries unchecked"
)

# A calibration item: audio (a WAV path) or an MFCC matrix, and its transcript.
CalibrationItem = tuple["Path | str | npt.NDArray[np.float32]", str]


class Aligner:
    """In-process forced aligner backed by ``libpstrainc.pstrain_align_*``.

    Args:
        model_dir: Acoustic model directory. Must contain ``mdef``,
            ``means``, ``variances``, ``mixture_weights``,
            ``transition_matrices``, and ``feat.params``.
        dict_path: Pronunciation dictionary path.
        filler_dict: Filler / non-speech dictionary path. Optional.
        beam: Pruning beam (default 1e-64, matches sphinx3_align).
        retry_beam_factor: Factor that widens the beam for one retry after a
            final-state failure; values at or below 1 disable the retry. The
            default, 1e136, retries once at 1e-200 on the default beam. An
            ascending sequence of factors, each greater than 1 and relative to
            ``beam``, is a ladder: its rungs run in order and the first success
            ends it. :meth:`retry_yield` reports what each rung bought.
        retry_acceptance_target: The retry acceptance check (see
            :mod:`pstrain.lib.alignment.acceptance`). A retry-recovered
            alignment is accepted only if its speech score reaches a threshold
            calibrated at this quantile (default 0.05) of normal alignments at
            the rung's beam; a rejected one raises
            :class:`~pstrain.lib.alignment.acceptance.AlignmentRejectedError`,
            a ``RuntimeError``, as a failed retry would. First-pass alignments
            are never checked. ``None`` turns the check off and accepts
            retries unchecked. With the check on, the retry runs only when a
            threshold is supplied, or when ``retry_acceptance_deferred`` is set;
            otherwise a final-state failure is raised unretried, and says why.
        retry_acceptance_threshold: The threshold, in nats per speech frame at
            the rung's beam: one number per ladder rung, or one number for a
            single retry. :meth:`calibrate_retry_acceptance` computes it from
            known-good utterances, realigning at most
            ``RETRY_ACCEPTANCE_MAX_SAMPLES`` (200) and needing at least
            ``RETRY_ACCEPTANCE_MIN_SAMPLES`` (20) of them to align.
        retry_acceptance_deferred: Run the retry and return what it recovers
            unjudged, with its score in ``result.retry``, for a caller that
            calibrates and judges it itself, as
            :func:`~pstrain.lib.alignment.batch.align_corpus` does.
        failed_alignment: ``"recover"`` retries final-state failures;
            ``"abort"`` and ``"omit"`` raise without retrying.
        insert_sil: Insert optional inter-word silences (default ``True``).
        include_phones: Return phone segmentation in the result.
        include_states: Return per-frame state segmentation.
        cmn: Cepstral mean normalization mode. When omitted, use the
            validated model ``feat.params`` value. An explicit value
            deliberately overrides the model record. The native parser treats
            ``"current"`` as an exact alias for ``"batch"``.
        cmninit: Initial mean vector used when ``cmn="live"``. When omitted,
            use the validated model ``feat.params`` value; an explicit value
            deliberately overrides the model record.
        agc: Automatic gain control mode (default ``"none"``).
        varnorm: Apply cepstral variance normalization.
        feat_type: Feature stream spec (default ``"1s_c_d_dd"``).
        frate: Frame rate in Hz (default 100, i.e. 10 ms frames).
        lts_mismatch: Use CMU letter-to-sound rules for OOV words.
        verbatim_tokens: Honor an explicit pronunciation token such as
            ``WORD(2)`` exactly. The default is ``False``: suffixes collapse
            to the base word and Viterbi considers its full alternative chain.
            Unknown explicit variants fail with the token named. This option
            affects forced alignment only, never Baum-Welch training.
    """

    _active: Aligner | None = None

    def __init__(
        self,
        model_dir: Path | str,
        dict_path: Path | str,
        *,
        filler_dict: Path | str | None = None,
        beam: float = _DEFAULT_BEAM,
        retry_beam_factor: RetryBeamFactor = _DEFAULT_RETRY_BEAM_FACTOR,
        failed_alignment: Literal["recover", "abort", "omit"] = "recover",
        retry_acceptance_target: float | None = DEFAULT_RETRY_ACCEPTANCE_TARGET,
        retry_acceptance_threshold: float | Sequence[float] | None = None,
        retry_acceptance_deferred: bool = False,
        insert_sil: bool = True,
        include_phones: bool = True,
        include_states: bool = False,
        cmn: str | None = None,
        cmninit: str | None = None,
        agc: str = _DEFAULT_AGC,
        varnorm: bool = False,
        feat_type: str = _DEFAULT_FEAT_TYPE,
        frate: int = 100,
        lts_mismatch: bool = False,
        verbatim_tokens: bool = False,
    ) -> None:
        if Aligner._active is not None:
            raise RuntimeError(
                "Another Aligner is already active in this process. "
                "Call .close() on the existing instance first."
            )

        # An invalid factor list fails here, before any model is loaded.
        self._retry_rungs = retry_ladder(retry_beam_factor)
        self._retry_beam_factor = retry_beam_factor
        # The nominal beam, kept on both sides of the helper boundary: the
        # rung beams reported and calibrated here are relative to it.
        self._beam = beam
        # Per-rung counts live in the object the caller holds. Behind the
        # native worker that is this parent-side proxy, so a worker that dies
        # mid-corpus takes none of the counts already gathered with it.
        self._last_alignment_retried = False
        self._last_retry_rungs = 0
        self._retry_rung_attempts = [0] * len(self._retry_rungs)
        self._retry_rung_recoveries = [0] * len(self._retry_rungs)
        self._retry_rung_rejections = [0] * len(self._retry_rungs)
        if retry_acceptance_target is not None and not (0.0 < retry_acceptance_target < 0.5):
            raise ValueError(
                "retry_acceptance_target must be greater than 0 and less than 0.5, or None "
                f"to turn the check off; got {retry_acceptance_target!r}"
            )
        self._retry_acceptance_target = retry_acceptance_target
        self._retry_thresholds = self._rung_thresholds(retry_acceptance_threshold)
        self._retry_acceptance_deferred = retry_acceptance_deferred
        self._retry_withheld = False
        self._filler_words = read_filler_words(filler_dict)
        model_dir = Path(model_dir)
        dict_path = Path(dict_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        for name in MODEL_FILES_REQUIRED:
            model_file = model_dir / name
            if not model_file.exists():
                raise FileNotFoundError(f"Model file missing: {model_file}")
        feat_record = read_complete_model_feat_params(model_dir)
        feat_params = model_dir / "feat.params"
        self._fe_config = feature_extractor_config_from_record(feat_record)
        if cmn is None:
            cmn = feat_record["-cmn"]
        if cmninit is None:
            cmninit = feat_record["-cmninit"]
        if not dict_path.exists():
            raise FileNotFoundError(f"Dictionary not found: {dict_path}")

        if not native_worker.in_worker():
            self._proxy = native_worker.NativeObjectProxy(
                __name__,
                "Aligner",
                (model_dir, dict_path),
                {
                    "filler_dict": filler_dict,
                    "beam": beam,
                    "retry_beam_factor": retry_beam_factor,
                    "failed_alignment": failed_alignment,
                    # Judging happens here, beside the counts. The helper runs
                    # the retry unchecked when this side can judge or defer
                    # what it recovers, and withholds it otherwise.
                    "retry_acceptance_target": (
                        None if self._retry_permitted else retry_acceptance_target
                    ),
                    "insert_sil": insert_sil,
                    "include_phones": include_phones,
                    "include_states": include_states,
                    "cmn": cmn,
                    "cmninit": cmninit,
                    "agc": agc,
                    "varnorm": varnorm,
                    "feat_type": feat_type,
                    "frate": frate,
                    "lts_mismatch": lts_mismatch,
                    "verbatim_tokens": verbatim_tokens,
                },
                (model_dir, dict_path),
            )
            Aligner._active = self
            return

        ffi, lib = _init()
        self._ffi = ffi
        self._lib = lib

        mdef = model_dir / "mdef"
        means = model_dir / "means"
        var = model_dir / "variances"
        mixw = model_dir / "mixture_weights"
        tmat = model_dir / "transition_matrices"
        cfg = ffi.new("pstrain_align_config_t *")
        lib.pstrain_align_config_default(cfg)
        cfg.beam = float(beam)
        cfg.insert_sil = 1 if insert_sil else 0
        cfg.compute_phones = 1 if include_phones else 0
        cfg.compute_states = 1 if include_states else 0
        cfg.varnorm = 1 if feat_record["-varnorm"][0] in "ytYT1" else 0
        cfg.ceplen = int(feat_record["-ceplen"])
        cfg.frate = int(feat_record["-frate"])
        cfg.lts_mismatch = 1 if lts_mismatch else 0
        cfg.verbatim_tokens = 1 if verbatim_tokens else 0

        self._feat_type_b = feat_record["-feat"].encode()
        self._cmn_b = cmn.encode()
        self._cmninit_b = cmninit.encode()
        self._agc_b = feat_record["-agc"].encode()
        cfg.feat_type = ffi.cast("const char *", ffi.from_buffer(self._feat_type_b))
        cfg.cmn = ffi.cast("const char *", ffi.from_buffer(self._cmn_b))
        cfg.cmninit = ffi.cast("const char *", ffi.from_buffer(self._cmninit_b))
        cfg.agc = ffi.cast("const char *", ffi.from_buffer(self._agc_b))

        ctx = lib.pstrain_align_init(
            str(mdef).encode(),
            str(means).encode(),
            str(var).encode(),
            str(mixw).encode(),
            str(tmat).encode(),
            str(feat_params).encode(),
            str(dict_path).encode(),
            str(filler_dict).encode() if filler_dict else ffi.NULL,
            cfg,
        )
        if ctx == ffi.NULL:
            err = self._last_error()
            raise RuntimeError(f"pstrain_align_init failed: {err or 'unknown'}")

        self._ctx = ctx
        self._ceplen = int(feat_record["-ceplen"])
        self._beam = beam
        self._failed_alignment = failed_alignment
        self._fe: FeatureExtractor | None = None
        self._sample_rate = int(self._fe_config["samprate"])
        self._frame_rate = int(feat_record["-frate"])
        Aligner._active = self

    def _last_error(self) -> str | None:
        if hasattr(self, "_proxy"):
            result = self._proxy.call("_last_error")
            return str(result) if result is not None else None
        ptr = self._lib.pstrain_align_last_error()
        if ptr == self._ffi.NULL:
            return None
        msg: str = self._ffi.string(ptr).decode("utf-8", errors="replace")
        return msg

    def close(self) -> None:
        """Release the underlying C state. Idempotent."""
        if hasattr(self, "_proxy"):
            self._proxy.close()
            if Aligner._active is self:
                Aligner._active = None
            return
        if getattr(self, "_ctx", None) is not None and self._ctx != self._ffi.NULL:
            self._lib.pstrain_align_free(self._ctx)
            self._ctx = self._ffi.NULL
        if self._fe is not None:
            self._fe.close()
            self._fe = None
        if Aligner._active is self:
            Aligner._active = None

    def set_beam(self, beam: float) -> float:
        """Set the live pruning beam and return its previous value."""
        if hasattr(self, "_proxy"):
            return float(self._proxy.call("set_beam", beam))
        if self._ctx == self._ffi.NULL:
            raise RuntimeError("Aligner is closed")
        return float(self._lib.pstrain_align_set_beam(self._ctx, beam))

    def minimum_frames(self, transcript: str) -> int:
        """Fewest feature frames that could possibly align to this transcript.

        A shortest path over the sentence HMM this aligner builds for the
        transcript, counting one frame per emitting state. It is exact for this
        model and dictionary, not an estimate from the word count, and it is
        measured on the aligner's own graph -- Baum-Welch builds a different
        utterance HMM and its minimum is a different number.
        """
        if hasattr(self, "_proxy"):
            return int(self._proxy.call("minimum_frames", transcript))
        if self._ctx == self._ffi.NULL:
            raise RuntimeError("Aligner is closed")
        out = self._ffi.new("uint32 *")
        rc = int(self._lib.pstrain_align_min_frames(self._ctx, transcript.encode(), out))
        if rc != 0:
            err = self._last_error()
            raise RuntimeError(f"pstrain_align_min_frames failed: {err or f'rc={rc}'}")
        return int(out[0])

    def _mfc_frame_count(self, mfc_path: Path) -> int:
        """Frame count from a Sphinx ``.mfc`` header, without reading the data."""
        with mfc_path.open("rb") as stream:
            n_floats = int(struct.unpack("<i", stream.read(4))[0])
        if n_floats < 0 or n_floats % self._ceplen:
            # Flooring a partial frame would put a made-up number into a
            # diagnosis. Refuse to guess; the caller falls back to the
            # engine's own final-state message.
            raise ValueError(
                f"{mfc_path}: header declares {n_floats} floats, "
                f"not a multiple of ceplen={self._ceplen}"
            )
        return n_floats // self._ceplen

    @staticmethod
    def _infeasible_message(utterance_id: str, shortfall: tuple[int, int]) -> str:
        return (
            infeasible_frames_message(utterance_id, *shortfall)
            + "; no wider beam can change that, so the retry was skipped"
        )

    def _infeasible_frames_for_mfc(
        self, rc: int, transcript: str, mfc_path: Path
    ) -> tuple[int, int] | None:
        """``_infeasible_frames`` for a cepstrum file, reading its header lazily.

        The header is only consulted on a final-state failure, and a header this
        cannot trust yields no diagnosis at all rather than a guessed one: the
        caller then reports the engine's own message.
        """
        if rc != -3:
            return None
        try:
            frames = self._mfc_frame_count(mfc_path)
        except (OSError, ValueError, struct.error):
            return None
        return self._infeasible_frames(rc, transcript, frames)

    def _infeasible_frames(self, rc: int, transcript: str, n_frames: int) -> tuple[int, int] | None:
        """Return ``(minimum, available)`` when no beam width could have worked.

        An utterance HMM spends at least one frame in every emitting state it
        passes through, so audio shorter than the shortest path through the
        sentence HMM cannot reach the final state however wide the beam is. That
        makes the wider-beam retry pure waste, and it makes the generic
        final-state message misleading: nothing was mis-pruned.
        """
        if rc != -3:
            return None
        try:
            required = self.minimum_frames(transcript)
        except Exception:  # noqa: BLE001 - a diagnosis must never replace the failure
            return None
        if n_frames >= required:
            return None
        return required, n_frames

    def _rung_thresholds(
        self, threshold: float | Sequence[float] | None
    ) -> tuple[float, ...] | None:
        """One supplied acceptance threshold per ladder rung, or ``None``."""
        if threshold is None:
            return None
        if isinstance(threshold, numbers.Real):
            values: tuple[float, ...] = (float(threshold),)
        else:
            values = tuple(float(value) for value in cast("Sequence[float]", threshold))
        if len(values) != len(self._retry_rungs):
            raise ValueError(
                f"retry_acceptance_threshold needs one threshold per retry rung: "
                f"{len(self._retry_rungs)} rung(s), {len(values)} threshold(s)"
            )
        return values

    @property
    def _retry_permitted(self) -> bool:
        """Whether whatever the retry recovers can be judged, or need not be."""
        return (
            self._retry_acceptance_target is None
            or self._retry_thresholds is not None
            or self._retry_acceptance_deferred
        )

    def _final_state_retry_beam(self, rc: int, rung: int = 0) -> float | None:
        """Return the beam for retry ladder rung ``rung``, or ``None`` to stop.

        Mirrors Baum-Welch training: a final-state pruning failure (``rc == -3``)
        under the ``recover`` policy is retried at ``beam / factor`` for each
        factor of the ladder in turn (a smaller value is a wider beam). Any other
        rc or policy, a single factor at or below 1, or a ladder already climbed
        ends the retry. Asking for rung 0 starts a new utterance's accounting.
        So does a retry whose recovery could not be judged: with the acceptance
        check on and no threshold, no rung runs (``_retry_withheld``).
        """
        if rung == 0:
            self._last_alignment_retried = False
            self._last_retry_rungs = 0
            self._retry_withheld = False
        if rc != -3 or self._failed_alignment != "recover":
            return None
        rungs = retry_ladder(self._retry_beam_factor)
        if rung >= len(rungs):
            return None
        if not self._retry_permitted:
            self._retry_withheld = True
            return None
        self._last_alignment_retried = True
        return self._beam / rungs[rung]

    def _climb_retry_ladder(self, rc: int, attempt: Callable[[], int]) -> int:
        """Retry a final-state failure up the ladder; return the final rc.

        Each rung runs ``attempt`` at its own beam and the nominal beam is
        restored after every rung, success or not. The ladder stops at the first
        rung that succeeds, and at any failure that is not a final-state failure,
        since no wider beam answers that. When every rung fails, the native
        error left behind is the last rung's: the widest search that ran, and
        the same message a single retry has always reported.
        """
        rung = 0
        while (beam := self._final_state_retry_beam(rc, rung)) is not None:
            previous_beam = self.set_beam(beam)
            try:
                rc = attempt()
            finally:
                self.set_beam(previous_beam)
            rung += 1
            self._last_retry_rungs = rung
        self._record_retry(rung, recovered=rc == 0)
        return rc

    def _record_retry(self, rungs_run: int, *, recovered: bool) -> None:
        """Credit one utterance's climb: every rung that ran, and the one that recovered it."""
        for rung in range(min(rungs_run, len(self._retry_rung_attempts))):
            self._retry_rung_attempts[rung] += 1
        if recovered and 0 < rungs_run <= len(self._retry_rung_recoveries):
            self._retry_rung_recoveries[rungs_run - 1] += 1

    def _failure_message(self, prefix: str, rc: int) -> str:
        """The native failure, and why no retry ran when one was withheld."""
        err = self._last_error()
        message = f"{prefix}: {err or f'rc={rc}'}"
        if self._retry_withheld:
            message += _RETRY_WITHHELD
        return message

    def _recovered(self, result: AlignmentResult) -> AlignmentResult:
        """Attach the retry's rung, beam and speech score to what it recovered."""
        rung = self._last_retry_rungs
        if rung <= 0:
            return result
        factor = self._retry_rungs[rung - 1]
        result.retry = RetryOutcome(
            rung=rung,
            factor=factor,
            beam=self._beam / factor,
            score=speech_score(result, self._filler_words),
        )
        return result

    def _judge_retry(self, result: AlignmentResult) -> AlignmentResult:
        """Apply the acceptance check to a retry-recovered alignment.

        Runs once, in the object the caller holds, beside the per-rung counts.
        A first-pass alignment passes untouched. With the check off, or
        deferred to the caller, the recovery is returned as it is.
        """
        outcome = result.retry
        if outcome is None:
            return result
        if self._retry_acceptance_target is None:
            result.retry = replace(outcome, basis="acceptance check off")
            return result
        if self._retry_thresholds is None:
            # Deferred: the caller calibrates and judges.
            result.retry = replace(outcome, threshold=None, basis="not judged")
            return result
        threshold = self._retry_thresholds[outcome.rung - 1]
        judged = replace(outcome, threshold=threshold, basis="supplied threshold")
        if judged.score is None or judged.score < threshold:
            self._record_rejection(judged.rung)
            raise AlignmentRejectedError(rejection_message(result.utterance_id, judged), judged)
        result.retry = judged
        return result

    def _record_rejection(self, rung: int) -> None:
        """Count a recovery the acceptance check rejected against its rung."""
        if 0 < rung <= len(self._retry_rung_rejections):
            self._retry_rung_rejections[rung - 1] += 1

    def _scores_at_rung(self, items: Sequence[CalibrationItem], rung: int) -> list[float | None]:
        """Align each item directly at a rung's beam, with no retry, and score it.

        Returns each alignment's speech score, or ``None`` for an item that did
        not align there. Runs where the native aligner lives.
        """
        if hasattr(self, "_proxy"):
            return list(self._proxy.call("_scores_at_rung", list(items), rung))
        factor = self._retry_rungs[rung - 1]
        previous_beam = self.set_beam(self._beam / factor)
        saved = (self._retry_beam_factor, self._last_retry_rungs, self._last_alignment_retried)
        self._retry_beam_factor = 1.0
        scores: list[float | None] = []
        try:
            for index, (source, transcript) in enumerate(items):
                utterance_id = f"calibration-{index}"
                try:
                    if isinstance(source, np.ndarray):
                        result = self.align_mfcc(source, transcript, utterance_id)
                    else:
                        result = self.align_audio(source, transcript, utterance_id)
                except (RuntimeError, OSError, ValueError):
                    scores.append(None)
                    continue
                scores.append(speech_score(result, self._filler_words))
        finally:
            self._retry_beam_factor, self._last_retry_rungs, self._last_alignment_retried = saved
            self.set_beam(previous_beam)
        return scores

    def calibrate_rung(
        self,
        items: Sequence[CalibrationItem],
        rung: int,
        target: float | None = None,
    ) -> RetryCalibration:
        """Calibrate one ladder rung's acceptance threshold.

        Realigns at most ``RETRY_ACCEPTANCE_MAX_SAMPLES`` (200) of ``items``,
        evenly spaced, directly at the rung's beam with no retry, and takes the
        ``target`` quantile of their speech scores (by default this aligner's
        target, or 0.05 when its check is off). With fewer than
        ``RETRY_ACCEPTANCE_MIN_SAMPLES`` (20) scored, the threshold is ``None``.
        """
        if not 0 < rung <= len(self._retry_rungs):
            raise ValueError(f"rung {rung} is not a rung of this aligner's retry ladder")
        if target is None:
            target = self._retry_acceptance_target or DEFAULT_RETRY_ACCEPTANCE_TARGET
        sample = evenly_spaced(list(items), RETRY_ACCEPTANCE_MAX_SAMPLES)
        factor = self._retry_rungs[rung - 1]
        return calibration_from_scores(
            self._scores_at_rung(sample, rung),
            rung=rung,
            factor=factor,
            beam=self._beam / factor,
            target=target,
        )

    def calibrate_retry_acceptance(
        self,
        items: Iterable[CalibrationItem],
        target: float | None = None,
    ) -> tuple[float, ...]:
        """Acceptance thresholds for every rung, from known-good utterances.

        ``items`` are ``(audio, transcript)`` pairs, where audio is a WAV path
        or an MFCC matrix, that align normally. Each rung realigns at most
        ``RETRY_ACCEPTANCE_MAX_SAMPLES`` (200) of them at its own beam. Pass the
        result as ``retry_acceptance_threshold``.

        Raises:
            ValueError: No retry is configured, or fewer than
                ``RETRY_ACCEPTANCE_MIN_SAMPLES`` (20) items aligned at a rung.
        """
        if not self._retry_rungs:
            raise ValueError("no wider-beam retry is configured, so there is nothing to calibrate")
        pairs = list(items)
        thresholds: list[float] = []
        for rung in range(1, len(self._retry_rungs) + 1):
            calibration = self.calibrate_rung(pairs, rung, target)
            if calibration.threshold is None:
                raise ValueError(calibration.basis)
            thresholds.append(calibration.threshold)
        return tuple(thresholds)

    def _align_reporting_rungs(self, method: str, *args: Any) -> tuple[AlignmentResult, int]:
        """Worker side: align, and say how many rungs the climb ran."""
        self._last_retry_rungs = 0
        result = getattr(self, method)(*args)
        return result, self._last_retry_rungs

    def _reported_retry_rungs(self) -> int:
        """Worker side: how many rungs the last alignment ran before it failed."""
        return self._last_retry_rungs

    def _count_proxied_success(self, outcome: tuple[AlignmentResult, int]) -> AlignmentResult:
        """Parent side: keep the rung counts of a climb the worker ran."""
        result, rungs = outcome
        assert isinstance(result, AlignmentResult)
        self._last_retry_rungs = int(rungs)
        self._last_alignment_retried = self._last_retry_rungs > 0
        self._record_retry(self._last_retry_rungs, recovered=True)
        return self._judge_retry(result)

    def _count_proxied_failure(self) -> None:
        """Parent side: count the rungs a failed climb ran, if the worker can still say.

        The failure itself is what the caller needs. Its rung count is
        bookkeeping: if the worker died with it, there is none to fetch, and
        the counts already kept here are unaffected.
        """
        rungs: int | None = None
        with contextlib.suppress(Exception):
            rungs = int(self._proxy.call("_reported_retry_rungs"))
        if rungs is not None:
            self._last_retry_rungs = rungs
            self._last_alignment_retried = rungs > 0
            self._record_retry(rungs, recovered=False)

    def retry_yield(self) -> tuple[RungYield, ...]:
        """Per-rung ``(factor, attempted, recovered, rejected)`` over this aligner's life.

        One entry per ladder rung, in order. ``attempted`` counts utterances
        that reached the rung, so it counts only retries that actually ran;
        ``recovered`` counts the utterances that rung aligned, and ``rejected``
        those of them the acceptance check then rejected. With a single factor
        the one entry is the whole retry's yield.
        """
        return tuple(
            RungYield(*counts)
            for counts in zip(
                self._retry_rungs,
                self._retry_rung_attempts,
                self._retry_rung_recoveries,
                self._retry_rung_rejections,
                strict=True,
            )
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def align_mfcc(
        self,
        mfcc: npt.NDArray[np.float32],
        transcript: str,
        utterance_id: str = "utt",
    ) -> AlignmentResult:
        """Align an MFCC matrix against a transcript.

        Args:
            mfcc: Row-major MFCC matrix, shape ``(n_frames, ncep)``,
                dtype ``float32``. ``ncep`` must match the model. It is not
                modified: every attempt, retry rungs included, normalizes
                its own copy.
            transcript: Word-level reference (sphinx ``<s>/</s>``
                markers are tolerated and stripped).
            utterance_id: Identifier used in logging and stored on the
                returned result.

        Returns:
            An :class:`AlignmentResult` with word- (and phone-/state-)
            level segments. A variant suffix such as ``reading(2)`` labels the
            dictionary pronunciation selected by Viterbi. By default, a
            suffix in ``transcript`` does not constrain that selection. With
            ``verbatim_tokens=True``, an explicit suffix selects exactly that
            pronunciation and is preserved.
        """
        if hasattr(self, "_proxy"):
            try:
                outcome = self._proxy.call(
                    "_align_reporting_rungs", "align_mfcc", mfcc, transcript, utterance_id
                )
            except Exception:
                self._count_proxied_failure()
                raise
            return self._count_proxied_success(outcome)
        if self._ctx == self._ffi.NULL:
            raise RuntimeError("Aligner is closed")
        arr = np.ascontiguousarray(mfcc, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"mfcc must be 2-D, got shape {arr.shape}")
        n_frames, ncep = arr.shape

        out_pp = self._ffi.new("pstrain_align_result_t **")
        cepstra = self._ffi.cast("float *", self._ffi.from_buffer(arr))
        rc = int(
            self._lib.pstrain_align_mfcc(
                self._ctx,
                cepstra,
                n_frames,
                ncep,
                transcript.encode(),
                utterance_id.encode(),
                out_pp,
            )
        )
        shortfall = self._infeasible_frames(rc, transcript, int(n_frames))
        if shortfall is not None:
            self._last_alignment_retried = False
            self._last_retry_rungs = 0
            raise RuntimeError(self._infeasible_message(utterance_id, shortfall))
        rc = self._climb_retry_ladder(
            rc,
            lambda: int(
                self._lib.pstrain_align_mfcc(
                    self._ctx,
                    cepstra,
                    n_frames,
                    ncep,
                    transcript.encode(),
                    utterance_id.encode(),
                    out_pp,
                )
            ),
        )
        if rc != 0:
            raise RuntimeError(self._failure_message("pstrain_align_mfcc failed", rc))
        try:
            result = self._recovered(self._unpack_result(out_pp[0], utterance_id, transcript))
        finally:
            self._lib.pstrain_align_result_free(out_pp[0])
        return self._judge_retry(result)

    def align_mfc_file(
        self,
        mfc_path: Path | str,
        transcript: str,
        utterance_id: str | None = None,
    ) -> AlignmentResult:
        """Align a Sphinx-format ``.mfc`` cepstrum file against a transcript.

        Convenience for parity checking against the standalone
        ``sphinx3_align`` binary, which also accepts ``.mfc`` input.
        """
        if hasattr(self, "_proxy"):
            try:
                outcome = self._proxy.call(
                    "_align_reporting_rungs", "align_mfc_file", mfc_path, transcript, utterance_id
                )
            except Exception:
                self._count_proxied_failure()
                raise
            return self._count_proxied_success(outcome)
        if self._ctx == self._ffi.NULL:
            raise RuntimeError("Aligner is closed")
        mfc_path = Path(mfc_path)
        utt_id = utterance_id or mfc_path.stem
        out_pp = self._ffi.new("pstrain_align_result_t **")
        rc = int(
            self._lib.pstrain_align_mfc_file(
                self._ctx,
                str(mfc_path).encode(),
                transcript.encode(),
                utt_id.encode(),
                out_pp,
            )
        )
        shortfall = self._infeasible_frames_for_mfc(rc, transcript, mfc_path)
        if shortfall is not None:
            self._last_alignment_retried = False
            self._last_retry_rungs = 0
            raise RuntimeError(self._infeasible_message(utt_id, shortfall))
        rc = self._climb_retry_ladder(
            rc,
            lambda: int(
                self._lib.pstrain_align_mfc_file(
                    self._ctx,
                    str(mfc_path).encode(),
                    transcript.encode(),
                    utt_id.encode(),
                    out_pp,
                )
            ),
        )
        if rc != 0:
            raise RuntimeError(self._failure_message("pstrain_align_mfc_file failed", rc))
        try:
            result = self._recovered(self._unpack_result(out_pp[0], utt_id, transcript))
        finally:
            self._lib.pstrain_align_result_free(out_pp[0])
        return self._judge_retry(result)

    def align_audio(
        self,
        audio_path: Path | str,
        transcript: str,
        utterance_id: str | None = None,
    ) -> AlignmentResult:
        """Align a 16 kHz mono WAV against a transcript.

        Runs feature extraction (via :class:`FeatureExtractor`) before
        handing the MFCCs off to the aligner. The feature extractor is
        reused across calls.
        """
        if hasattr(self, "_proxy"):
            try:
                outcome = self._proxy.call(
                    "_align_reporting_rungs", "align_audio", audio_path, transcript, utterance_id
                )
            except Exception:
                self._count_proxied_failure()
                raise
            return self._count_proxied_success(outcome)
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        utt_id = utterance_id or audio_path.stem

        with wave.open(str(audio_path), "rb") as wf:
            if wf.getnchannels() != 1:
                raise ValueError(f"{audio_path}: expected mono, got {wf.getnchannels()} channels")
            sample_rate = wf.getframerate()
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

        if sample_rate != self._sample_rate:
            raise ValueError(
                f"{audio_path}: sample rate {sample_rate} does not match model feat.params "
                f"-samprate {self._sample_rate}"
            )
        if self._fe is None:
            self._fe = FeatureExtractor(**self._fe_config)

        mfcc = self._fe.process_audio(audio)
        return self.align_mfcc(mfcc, transcript, utterance_id=utt_id)

    def _unpack_result(
        self,
        c_result: Any,
        utterance_id: str,
        transcript: str,
    ) -> AlignmentResult:
        # c_result is a cffi cdata struct pointer (pstrain_align_result_t *),
        # which mypy can't introspect. Treated as Any here.
        ffi = self._ffi
        result = c_result

        words: list[AlignedSegment] = []
        for i in range(result.n_words):
            s = result.words[i]
            words.append(
                AlignedSegment(
                    name=ffi.string(s.name).decode("utf-8", errors="replace"),
                    start_frame=int(s.start_frame),
                    end_frame=int(s.end_frame),
                    score=int(s.score),
                )
            )

        phones: list[AlignedSegment] = []
        for i in range(result.n_phones):
            s = result.phones[i]
            phones.append(
                AlignedSegment(
                    name=ffi.string(s.name).decode("utf-8", errors="replace"),
                    start_frame=int(s.start_frame),
                    end_frame=int(s.end_frame),
                    score=int(s.score),
                )
            )

        states: list[AlignedSegment] = []
        for i in range(result.n_states):
            s = result.states[i]
            states.append(
                AlignedSegment(
                    name=ffi.string(s.name).decode("utf-8", errors="replace"),
                    start_frame=int(s.start_frame),
                    end_frame=int(s.end_frame),
                    score=int(s.score),
                )
            )

        return AlignmentResult(
            utterance_id=utterance_id,
            words=words,
            phones=phones,
            states=states,
            total_score=int(result.total_score),
            n_frames=int(result.n_frames),
            transcript=transcript,
            frame_rate=self._frame_rate,
        )


__all__ = ["Aligner", "CalibrationItem"]
