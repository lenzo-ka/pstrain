"""Whether an utterance has enough frames to reach its HMM's final state.

A forced alignment that never reaches the final state is usually reported as a
search failure, and a wider beam is the usual remedy. One family of failures is
not a search failure at all: an utterance HMM spends at least one frame in every
emitting state it passes through, so an utterance whose audio has fewer frames
than the shortest path through its HMM cannot reach the final state at any beam
width. That condition is arithmetic, it can be decided exactly from the graph
the engine already built, and no retry can recover it.

A large multi-locale alignment audit found this family to be a substantial share
of real final-state failures, which is why the engines name it instead of
reporting only the generic message.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol

__all__ = [
    "NON_EMITTING_MIXW",
    "infeasible_frames_message",
    "minimum_emitting_states",
]

# ``TYING_NON_EMITTING`` from csrc/include/s3/s3.h: the mixture-weight id that
# marks a state which consumes no frame.
NON_EMITTING_MIXW = 0xFFFFFFFF


class _State(Protocol):
    """The structural part of an utterance-HMM state view."""

    @property
    def mixw(self) -> int: ...

    @property
    def next_state(self) -> tuple[int, ...]: ...


def minimum_emitting_states(states: list[_State]) -> int | None:
    """Return the fewest emitting states on any path from the first to the last.

    The first entry is the utterance HMM's initial state and the last is its
    final state, so this is the fewest frames the utterance could occupy. Skip
    transitions and alternative pronunciations are honored because this is a
    shortest path over the graph as built, not an estimate from the transcript:
    emitting states cost one frame, non-emitting states cost nothing, and a
    0-1 breadth-first search over that weighting is exact.

    Return ``None`` when the graph is empty or its final state is unreachable.
    """
    if not states:
        return None

    final = len(states) - 1
    distance: list[int | None] = [None] * len(states)
    distance[0] = 0 if states[0].mixw == NON_EMITTING_MIXW else 1
    queue: deque[int] = deque([0])

    while queue:
        index = queue.popleft()
        here = distance[index]
        assert here is not None
        for successor in states[index].next_state:
            if not 0 <= successor < len(states):
                continue
            cost = 0 if states[successor].mixw == NON_EMITTING_MIXW else 1
            candidate = here + cost
            known = distance[successor]
            if known is not None and known <= candidate:
                continue
            distance[successor] = candidate
            if cost:
                queue.append(successor)
            else:
                queue.appendleft(successor)

    return distance[final]


def infeasible_frames_message(utterance: str, required: int, frames: int) -> str:
    """Phrase the arithmetic impossibility the same way wherever it is found."""
    return (
        f"{utterance} cannot be aligned at any beam width: its transcript needs at "
        f"least {required} frames to reach the final state, and the audio has {frames}"
    )
