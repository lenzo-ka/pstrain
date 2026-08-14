"""PocketSphinx decoder wrapper for model testing."""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pstrain.lib.config.models import FeatureConfig

logger = logging.getLogger(__name__)


@dataclass
class DecodingResult:
    """Result of decoding a single utterance."""

    utterance_id: str
    hypothesis: str
    success: bool
    error: str | None = None
    # Monotonic wall duration for this utterance task, excluding decoder construction.
    task_wall_seconds: float = 0.0


def pocketsphinx_version() -> str:
    """Return the linked PocketSphinx version and verified source commit."""
    from pstrain.lib._cffi.core import get_ffi, get_lib

    return cast(bytes, get_ffi().string(get_lib().pstrain_pocketsphinx_version())).decode()


def check_pocketsphinx() -> tuple[bool, str]:
    """Check if PocketSphinx is available.

    Returns:
        Tuple of (available, message)
    """
    try:
        version = pocketsphinx_version()
        return (True, f"PocketSphinx C library available ({version})")
    except Exception:
        pass

    return (False, "PocketSphinx is not present in libpstrainc; rebuild the native library")


class Decoder:
    """PocketSphinx decoder wrapper.

    Creates a decoder instance configured with an acoustic model, dictionary,
    and optional language model. The decoder can then decode multiple audio files
    efficiently by reusing the same instance.

    Defaults match SphinxTrain's ``decode/psdecode.pl`` recipe.
    """

    DEFAULT_BEAM = 1e-80
    DEFAULT_WBEAM = 1e-40

    def __init__(
        self,
        model_dir: Path,
        dict_file: Path,
        filler_dict: Path | None = None,
        lm: Path | None = None,
        beam: float | None = None,
        wbeam: float | None = None,
        pl_window: int | None = None,
        lw: float = 10.0,
        wip: float = 0.2,
        pbeam: float = 1e-80,
        lpbeam: float = 1e-80,
        lponlybeam: float = 1e-80,
        fwdflatbeam: float = 1e-80,
        fwdflatwbeam: float = 1e-40,
        feature_config: FeatureConfig | None = None,
    ):
        """Initialize decoder.

        Args:
            model_dir: Path to acoustic model directory (contains mdef, means, etc.)
            dict_file: Path to pronunciation dictionary
            filler_dict: Optional path to filler dictionary
            lm: Optional language model file (ARPA format)
            beam: Main beam width (None = auto-detect based on model type)
            wbeam: Word beam width (None = auto-detect based on model type)
            pl_window: Phone lookahead window (None = use default 5)
            feature_config: Canonical schema-owned acoustic front-end settings.

        Raises:
            ImportError: If PocketSphinx not available
            RuntimeError: If decoder initialization fails
        """
        self.model_dir = Path(model_dir)
        self.dict_file = Path(dict_file)
        self.filler_dict = Path(filler_dict) if filler_dict else None
        self.lm = Path(lm) if lm else None
        self._decoder: object | None = None
        feature_config = feature_config or FeatureConfig()

        if beam is None:
            beam = self.DEFAULT_BEAM
        if wbeam is None:
            wbeam = self.DEFAULT_WBEAM

        logger.info("Using decode beam settings: beam=%s, wbeam=%s", beam, wbeam)

        # Validate paths - don't silently ignore missing files
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        if not self.dict_file.exists():
            raise FileNotFoundError(f"Dictionary not found: {dict_file}")
        if self.filler_dict and not self.filler_dict.exists():
            raise FileNotFoundError(f"Filler dictionary not found: {filler_dict}")
        if self.lm and not self.lm.exists():
            raise FileNotFoundError(f"Language model not found: {lm}")

        try:
            from pstrain.lib._cffi.core import get_ffi, get_lib, path_or_null

            self._ffi = get_ffi()
            self._lib = get_lib()
            config = self._ffi.new("pstrain_decoder_config_t *")
            # Keep CFFI-owned strings alive through ps_init(), which copies them.
            strings = [
                self._ffi.new("char[]", str(self.model_dir).encode()),
                self._ffi.new("char[]", str(self.dict_file).encode()),
            ]
            config.hmm, config.dict = strings
            if self.filler_dict:
                strings.append(self._ffi.new("char[]", str(self.filler_dict).encode()))
                config.fdict = strings[-1]
            else:
                config.fdict = path_or_null(None)
            if self.lm:
                strings.append(self._ffi.new("char[]", str(self.lm).encode()))
                config.lm = strings[-1]
            else:
                config.lm = path_or_null(None)
            config.beam = beam
            config.wbeam = wbeam
            config.lw = lw
            config.wip = wip
            config.pbeam = pbeam
            config.lpbeam = lpbeam
            config.lponlybeam = lponlybeam
            config.fwdflatbeam = fwdflatbeam
            config.fwdflatwbeam = fwdflatwbeam
            config.pl_window = 5 if pl_window is None else pl_window
            config.samprate = feature_config.samprate
            for value in (feature_config.agc, feature_config.cmn, feature_config.cmninit):
                strings.append(self._ffi.new("char[]", value.encode()))
            config.agc, config.cmn, config.cmninit = strings[-3:]
            config.varnorm = feature_config.varnorm == "yes"
            config.remove_noise = feature_config.remove_noise
            self._decoder = self._lib.pstrain_decoder_create(config)
            if self._decoder == self._ffi.NULL:
                raise RuntimeError("PocketSphinx decoder initialization failed")
        except Exception as e:
            error_msg = str(e)
            # Check for senone limit error
            if "senone" in error_msg.lower() and (
                "32767" in error_msg or "exceed" in error_msg.lower()
            ):
                raise RuntimeError(
                    "Model exceeds PocketSphinx senone limit (32767). "
                    "This model has too many tied states for PocketSphinx."
                ) from e
            raise RuntimeError(f"Failed to initialize decoder: {e}") from e

    def close(self) -> None:
        """Release the linked decoder."""
        if self._decoder is not None:
            self._lib.pstrain_decoder_free(self._decoder)
            self._decoder = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def decode_file(self, audio_file: Path) -> DecodingResult:
        """Decode a single audio file.

        Args:
            audio_file: Path to WAV file (16kHz, 16-bit, mono)

        Returns:
            DecodingResult with hypothesis and metadata
        """
        audio_file = Path(audio_file)
        utterance_id = audio_file.stem
        started = time.perf_counter()

        if not audio_file.exists():
            return DecodingResult(
                utterance_id=utterance_id,
                hypothesis="",
                success=False,
                error=f"Audio file not found: {audio_file}",
                task_wall_seconds=time.perf_counter() - started,
            )

        if self._decoder is None:
            raise RuntimeError("Decoder is not initialized")

        try:
            import wave

            with wave.open(str(audio_file), "rb") as wf:
                audio_data = wf.readframes(wf.getnframes())

            samples = self._ffi.from_buffer("int16[]", audio_data)
            if self._lib.pstrain_decoder_start_utt(self._decoder) < 0:
                raise RuntimeError("Failed to start utterance processing")
            if (
                self._lib.pstrain_decoder_process_raw(
                    self._decoder, samples, len(audio_data) // 2, 0, 1
                )
                < 0
            ):
                raise RuntimeError("Failed to process raw audio")
            if self._lib.pstrain_decoder_end_utt(self._decoder) < 0:
                raise RuntimeError("Failed to end utterance processing")
            hypothesis = self._lib.pstrain_decoder_hyp(self._decoder)
            hyp_text = self._ffi.string(hypothesis).decode() if hypothesis else ""

            return DecodingResult(
                utterance_id=utterance_id,
                hypothesis=hyp_text,
                success=True,
                task_wall_seconds=time.perf_counter() - started,
            )

        except Exception as e:
            return DecodingResult(
                utterance_id=utterance_id,
                hypothesis="",
                success=False,
                error=str(e)[:200],
                task_wall_seconds=time.perf_counter() - started,
            )

    def decode_batch(self, audio_files: list[Path]) -> dict[str, DecodingResult]:
        """Decode multiple audio files.

        Args:
            audio_files: List of audio file paths

        Returns:
            Dictionary mapping utterance_id to DecodingResult
        """
        results = {}
        for audio_file in audio_files:
            result = self.decode_file(audio_file)
            results[result.utterance_id] = result
        return results
