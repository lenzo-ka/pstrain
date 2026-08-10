"""PocketSphinx decoder wrapper for model testing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DecodingResult:
    """Result of decoding a single utterance."""

    utterance_id: str
    hypothesis: str
    success: bool
    error: str | None = None


def check_pocketsphinx() -> tuple[bool, str]:
    """Check if PocketSphinx is available.

    Returns:
        Tuple of (available, message)
    """
    try:
        import importlib.util

        if importlib.util.find_spec("pocketsphinx") is not None:
            return (True, "PocketSphinx Python module available")
    except (ImportError, AttributeError):
        pass

    # Try CLI
    try:
        import subprocess

        result = subprocess.run(
            ["pocketsphinx", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode in (0, 1):
            return (True, "PocketSphinx CLI available")
    except FileNotFoundError:
        pass
    except Exception:
        pass

    return (False, "PocketSphinx not found. Install with: pip install pocketsphinx")


class Decoder:
    """PocketSphinx decoder wrapper.

    Creates a decoder instance configured with an acoustic model, dictionary,
    and optional language model. The decoder can then decode multiple audio files
    efficiently by reusing the same instance.

    Beam Width Configuration:
        CI models need wider beams (1e-80) due to lower acoustic discriminability.
        CD models can use narrower beams (1e-48) for faster decoding.
        The decoder auto-detects CI vs CD from the model path if beams not specified.
    """

    # Default beams for CI models (wider - prevents DAG errors)
    CI_BEAM = 1e-80
    CI_WBEAM = 1e-40

    # Default beams for CD models (narrower - faster, PocketSphinx defaults)
    CD_BEAM = 1e-48
    CD_WBEAM = 7e-29

    def __init__(
        self,
        model_dir: Path,
        dict_file: Path,
        filler_dict: Path | None = None,
        lm: Path | None = None,
        beam: float | None = None,
        wbeam: float | None = None,
        pl_window: int | None = None,
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

        Raises:
            ImportError: If PocketSphinx not available
            RuntimeError: If decoder initialization fails
        """
        self.model_dir = Path(model_dir)
        self.dict_file = Path(dict_file)
        self.filler_dict = Path(filler_dict) if filler_dict else None
        self.lm = Path(lm) if lm else None
        self._decoder = None

        # Auto-detect model type (CI vs CD) for beam selection
        # Use specific patterns to avoid false matches (e.g., "citadel" matching "ci")
        model_path_str = str(self.model_dir).lower()
        is_ci_model = (
            "/ci-" in model_path_str or "/ci/" in model_path_str or model_path_str.endswith("/ci")
        )

        # Use provided beams or auto-select based on model type
        if beam is None:
            beam = self.CI_BEAM if is_ci_model else self.CD_BEAM
        if wbeam is None:
            wbeam = self.CI_WBEAM if is_ci_model else self.CD_WBEAM

        model_type = "CI" if is_ci_model else "CD"
        logger.info(f"Using {model_type} beam settings: beam={beam}, wbeam={wbeam}")

        # Validate paths - don't silently ignore missing files
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        if not self.dict_file.exists():
            raise FileNotFoundError(f"Dictionary not found: {dict_file}")
        if self.filler_dict and not self.filler_dict.exists():
            raise FileNotFoundError(f"Filler dictionary not found: {filler_dict}")
        if self.lm and not self.lm.exists():
            raise FileNotFoundError(f"Language model not found: {lm}")

        # Try to initialize PocketSphinx
        try:
            from pocketsphinx import Decoder as PSDecoder

            config = {
                "hmm": str(self.model_dir),
                "dict": str(self.dict_file),
                "beam": beam,
                "wbeam": wbeam,
            }

            if pl_window is not None:
                config["pl_window"] = pl_window

            if self.filler_dict:
                config["fdict"] = str(self.filler_dict)

            if self.lm:
                config["lm"] = str(self.lm)

            self._decoder = PSDecoder(**config)

        except ImportError as e:
            raise ImportError(
                "PocketSphinx not available. Install with: pip install pocketsphinx"
            ) from e
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

    def decode_file(self, audio_file: Path) -> DecodingResult:
        """Decode a single audio file.

        Args:
            audio_file: Path to WAV file (16kHz, 16-bit, mono)

        Returns:
            DecodingResult with hypothesis and metadata
        """
        audio_file = Path(audio_file)
        utterance_id = audio_file.stem

        if not audio_file.exists():
            return DecodingResult(
                utterance_id=utterance_id,
                hypothesis="",
                success=False,
                error=f"Audio file not found: {audio_file}",
            )

        if self._decoder is None:
            raise RuntimeError("Decoder is not initialized")

        try:
            import wave

            with wave.open(str(audio_file), "rb") as wf:
                audio_data = wf.readframes(wf.getnframes())

            self._decoder.start_utt()
            self._decoder.process_raw(audio_data, no_search=False, full_utt=True)
            self._decoder.end_utt()

            hypothesis = self._decoder.hyp()
            hyp_text = hypothesis.hypstr if hypothesis else ""

            return DecodingResult(
                utterance_id=utterance_id,
                hypothesis=hyp_text,
                success=True,
            )

        except Exception as e:
            return DecodingResult(
                utterance_id=utterance_id,
                hypothesis="",
                success=False,
                error=str(e)[:200],
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
