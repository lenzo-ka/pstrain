"""The wider-beam retry ladder shared by forced alignment and Baum-Welch training.

Under the ``recover`` policy, an utterance whose search fails to reach its final
state is retried at a wider beam. ``retry_beam_factor`` names those beams, each
relative to the nominal beam (a smaller beam value is a wider search):

* A single number is a one-rung ladder. The retry runs once at
  ``beam / factor``, and a factor at or below 1 disables it.
* An ascending list of factors, each greater than 1, is a ladder of several
  rungs. They run in order, the first success ends the ladder, and no later
  rung runs.

Either way the nominal beam is restored afterward, and an utterance whose audio
is too short for its transcript runs no rung at all, because no beam can change
that outcome.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import TypeAlias

__all__ = [
    "RetryBeamFactor",
    "format_retry_factor",
    "retry_ladder",
    "validate_retry_ladder",
]

RetryBeamFactor: TypeAlias = float | Sequence[float]


def validate_retry_ladder(factors: Sequence[float]) -> tuple[float, ...]:
    """Return a list of retry factors as a ladder, or raise ``ValueError``.

    A ladder is non-empty, every factor is greater than 1 (a factor of 1 or
    less would not widen the beam), and the factors strictly ascend, so each
    rung searches wider than the one before it.
    """
    rungs = tuple(float(factor) for factor in factors)
    if not rungs:
        raise ValueError("a retry beam factor list needs at least one factor")
    for factor in rungs:
        if not (factor > 1.0 and math.isfinite(factor)):
            raise ValueError(
                f"each retry beam factor in a list must be a finite number greater than 1; "
                f"got {factor!r}"
            )
    for narrower, wider in pairwise(rungs):
        if not wider > narrower:
            raise ValueError(
                f"retry beam factors must strictly ascend; {wider!r} does not exceed {narrower!r}"
            )
    return rungs


def retry_ladder(factor: RetryBeamFactor) -> tuple[float, ...]:
    """The factors to try, in order, after a final-state failure.

    A single number keeps its long-standing meaning: one retry at that factor,
    or none when it is at or below 1. A sequence must be a valid ladder.
    """
    if isinstance(factor, int | float):
        return (float(factor),) if factor > 1.0 else ()
    return validate_retry_ladder(factor)


def format_retry_factor(factor: float) -> str:
    """Render a factor compactly for summaries and logs, such as ``1e+48``."""
    return f"{factor:.3g}"
