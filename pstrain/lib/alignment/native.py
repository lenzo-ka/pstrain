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
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from pstrain.lib import native_worker
from pstrain.lib._cffi.core import _init
from pstrain.lib.alignment.core import AlignedSegment, AlignmentResult
from pstrain.lib.features import FeatureExtractor
from pstrain.lib.model import MODEL_FILES_REQUIRED, read_complete_model_feat_params
from pstrain.lib.pipeline.feat_params import feature_extractor_config_from_record

if TYPE_CHECKING:
    import numpy.typing as npt


_DEFAULT_BEAM = 1e-64
_DEFAULT_FEAT_TYPE = "1s_c_d_dd"
_DEFAULT_CMN = "batch"
_DEFAULT_AGC = "none"


class Aligner:
    """In-process forced aligner backed by ``libpstrainc.pstrain_align_*``.

    Args:
        model_dir: Acoustic model directory. Must contain ``mdef``,
            ``means``, ``variances``, ``mixture_weights``,
            ``transition_matrices``, and ``feat.params``.
        dict_path: Pronunciation dictionary path.
        filler_dict: Filler / non-speech dictionary path. Optional.
        beam: Pruning beam (default 1e-64, matches sphinx3_align).
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
    """

    _active: Aligner | None = None

    def __init__(
        self,
        model_dir: Path | str,
        dict_path: Path | str,
        *,
        filler_dict: Path | str | None = None,
        beam: float = _DEFAULT_BEAM,
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
    ) -> None:
        if Aligner._active is not None:
            raise RuntimeError(
                "Another Aligner is already active in this process. "
                "Call .close() on the existing instance first."
            )

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
                dtype ``float32``. ``ncep`` must match the model.
            transcript: Word-level reference (sphinx ``<s>/</s>``
                markers are tolerated and stripped).
            utterance_id: Identifier used in logging and stored on the
                returned result.

        Returns:
            An :class:`AlignmentResult` with word- (and phone-/state-)
            level segments. A variant suffix such as ``reading(2)`` labels the
            dictionary pronunciation selected by Viterbi. A suffix in
            ``transcript`` does not constrain that selection and need not be
            preserved in the output.
        """
        if hasattr(self, "_proxy"):
            result = self._proxy.call("align_mfcc", mfcc, transcript, utterance_id)
            assert isinstance(result, AlignmentResult)
            return result
        if self._ctx == self._ffi.NULL:
            raise RuntimeError("Aligner is closed")
        arr = np.ascontiguousarray(mfcc, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"mfcc must be 2-D, got shape {arr.shape}")
        n_frames, ncep = arr.shape

        out_pp = self._ffi.new("pstrain_align_result_t **")
        rc = self._lib.pstrain_align_mfcc(
            self._ctx,
            self._ffi.cast("float *", self._ffi.from_buffer(arr)),
            n_frames,
            ncep,
            transcript.encode(),
            utterance_id.encode(),
            out_pp,
        )
        if rc != 0:
            err = self._last_error()
            raise RuntimeError(f"pstrain_align_mfcc failed: {err or f'rc={rc}'}")
        try:
            return self._unpack_result(out_pp[0], utterance_id, transcript)
        finally:
            self._lib.pstrain_align_result_free(out_pp[0])

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
            result = self._proxy.call("align_mfc_file", mfc_path, transcript, utterance_id)
            assert isinstance(result, AlignmentResult)
            return result
        if self._ctx == self._ffi.NULL:
            raise RuntimeError("Aligner is closed")
        mfc_path = Path(mfc_path)
        utt_id = utterance_id or mfc_path.stem
        out_pp = self._ffi.new("pstrain_align_result_t **")
        rc = self._lib.pstrain_align_mfc_file(
            self._ctx,
            str(mfc_path).encode(),
            transcript.encode(),
            utt_id.encode(),
            out_pp,
        )
        if rc != 0:
            err = self._last_error()
            raise RuntimeError(f"pstrain_align_mfc_file failed: {err or f'rc={rc}'}")
        try:
            return self._unpack_result(out_pp[0], utt_id, transcript)
        finally:
            self._lib.pstrain_align_result_free(out_pp[0])

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
            result = self._proxy.call("align_audio", audio_path, transcript, utterance_id)
            assert isinstance(result, AlignmentResult)
            return result
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


__all__ = ["Aligner"]
