"""Public API for language-model construction.

Decoding needs a language model, so building one is part of the supported
surface rather than an internal detail. Callers should use these functions
instead of reaching into ``pstrain.lib.lm``.
"""

from pstrain.lib.lm import build_lm, build_lm_from_file, load_transcripts

__all__ = [
    "build_lm",
    "build_lm_from_file",
    "load_transcripts",
]
