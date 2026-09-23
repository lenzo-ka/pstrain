"""Tests for the arithmetic frame budget of an utterance HMM."""

from __future__ import annotations

from dataclasses import dataclass

from pstrain.lib.feasibility import (
    NON_EMITTING_MIXW,
    infeasible_frames_message,
    minimum_emitting_states,
)


@dataclass(frozen=True)
class _State:
    """Stand-in for the structural half of ``pstrain.lib.bw.StateInfo``."""

    mixw: int
    next_state: tuple[int, ...]


def _chain(n_emitting: int) -> list[_State]:
    """A strict left-to-right chain ending in one non-emitting final state."""
    states = [_State(index, (index, index + 1)) for index in range(n_emitting)]
    return [*states, _State(NON_EMITTING_MIXW, ())]


def test_straight_chain_costs_one_frame_per_emitting_state() -> None:
    assert minimum_emitting_states(_chain(5)) == 5


def test_self_loops_do_not_inflate_the_minimum() -> None:
    """Every state in a real utterance HMM can loop; the minimum ignores that."""
    assert minimum_emitting_states(_chain(1)) == 1


def test_skip_transitions_shorten_the_minimum() -> None:
    """The answer is a shortest path, not a count of states in the graph."""
    states = _chain(5)
    # State 0 may jump straight to state 3, bypassing states 1 and 2.
    states[0] = _State(0, (0, 1, 3))
    assert minimum_emitting_states(states) == 3


def test_alternative_branches_take_the_shorter_one() -> None:
    """Pronunciation variants fan out; the budget follows the cheapest."""
    states = [
        _State(0, (0, 1, 2)),  # 0: entry, branches into a long and a short arm
        _State(1, (1, 4)),  # 1: short arm, one emitting state
        _State(2, (2, 3)),  # 2: long arm, two emitting states
        _State(3, (3, 4)),  # 3
        _State(NON_EMITTING_MIXW, ()),  # 4: final
    ]
    assert minimum_emitting_states(states) == 2


def test_non_emitting_states_on_the_path_are_free() -> None:
    states = [
        _State(0, (0, 1)),
        _State(NON_EMITTING_MIXW, (2,)),
        _State(1, (2, 3)),
        _State(NON_EMITTING_MIXW, ()),
    ]
    assert minimum_emitting_states(states) == 2


def test_an_emitting_final_state_still_costs_its_own_frame() -> None:
    states = [_State(0, (0, 1)), _State(1, (1,))]
    assert minimum_emitting_states(states) == 2


def test_empty_and_unreachable_graphs_report_no_answer() -> None:
    assert minimum_emitting_states([]) is None
    assert minimum_emitting_states([_State(0, ()), _State(NON_EMITTING_MIXW, ())]) is None


def test_the_message_names_both_numbers() -> None:
    message = infeasible_frames_message("utt-1", 231, 180)
    assert "utt-1" in message
    assert "231" in message
    assert "180" in message
