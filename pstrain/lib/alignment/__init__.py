"""Forced alignment module.

Aligns audio to transcripts using the sphinx3 forced aligner compiled
into ``libpstrainc`` and driven through CFFI. The previous PocketSphinx-based
path and the ``sphinx3_align`` subprocess wrapper have both been
replaced by :class:`pstrain.lib.alignment.native.Aligner`.

Use cases:

1. **Alignment-focused workflow.** Train on all data, then drive a
   long-lived :class:`Aligner` across the corpus.
2. **Phone-level alignment for phonetic research.** Set
   ``include_phones=True`` (default).
3. **Word-level alignment for subtitling/caption sync.** Set
   ``include_phones=False`` for ~2x throughput.
"""

from __future__ import annotations

from pstrain.lib.alignment.batch import AlignmentJob, align_corpus, load_transcripts
from pstrain.lib.alignment.core import (
    AlignedSegment,
    AlignmentResult,
    align_utterance,
)
from pstrain.lib.alignment.export import (
    save_ctm,
    save_textgrid,
    to_ctm,
    to_sphinx_segments,
    to_textgrid,
)
from pstrain.lib.alignment.native import Aligner

__all__ = [
    "AlignedSegment",
    "AlignmentJob",
    "AlignmentResult",
    "Aligner",
    "align_corpus",
    "align_utterance",
    "load_transcripts",
    "save_ctm",
    "save_textgrid",
    "to_ctm",
    "to_sphinx_segments",
    "to_textgrid",
]
