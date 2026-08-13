"""Phone-boundary accuracy for hand-labelled corpora such as TIMIT.

The scorer sequence-aligns phone identities before comparing time.  An
internal reference boundary is comparable only when both phones flanking it
match two consecutive hypothesis phones.  Boundaries adjacent to an
insertion, deletion, or substitution are misses, rather than being silently
dropped from tolerance accuracy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import inf
from statistics import mean, median

TOLERANCES_MS = (10.0, 20.0, 25.0, 50.0)


@dataclass(frozen=True)
class PhoneInterval:
    """A labelled half-open phone interval, in seconds."""

    label: str
    start: float
    end: float


@dataclass(frozen=True)
class BoundaryScore:
    """Failure-aware phone-boundary summary.

    ``mean_absolute_error_ms`` and ``median_absolute_error_ms`` describe only
    comparable boundaries and must always be reported with ``coverage``.
    Tolerance recall uses *all* reference boundaries as its denominator, so
    sequence mismatches cannot improve it by removing difficult boundaries.
    Precision uses all hypothesis boundaries.  ``f1_within`` combines them.
    """

    mean_absolute_error_ms: float
    median_absolute_error_ms: float
    reference_boundaries: int
    hypothesis_boundaries: int
    comparable_boundaries: int
    insertions: int
    deletions: int
    substitutions: int
    absolute_errors_ms: tuple[float, ...]
    recall_within: dict[float, float]
    precision_within: dict[float, float]
    f1_within: dict[float, float]

    @property
    def coverage(self) -> float:
        return _ratio(self.comparable_boundaries, self.reference_boundaries)


def score_phone_boundaries(
    reference: Sequence[PhoneInterval],
    hypothesis: Sequence[PhoneInterval],
    *,
    tolerances_ms: Sequence[float] = TOLERANCES_MS,
) -> BoundaryScore:
    """Compare internal phone boundaries after deterministic edit alignment.

    Utterance start and end are excluded because forced aligners commonly pin
    them to the audio extent.  A comparable internal boundary must be between
    two identity matches which are consecutive in both sequences.  Its
    hypothesis time is the midpoint between the left interval's end and the
    right interval's start; the same definition is used for the reference.
    """
    _validate(reference, "reference")
    _validate(hypothesis, "hypothesis")
    pairs, insertions, deletions, substitutions = _edit_align(
        [p.label for p in reference], [p.label for p in hypothesis]
    )
    ref_to_hyp = {ri: hi for ri, hi in pairs if ri is not None and hi is not None}
    errors: list[float] = []
    for right_ref in range(1, len(reference)):
        left_hyp = ref_to_hyp.get(right_ref - 1)
        right_hyp = ref_to_hyp.get(right_ref)
        if left_hyp is None or right_hyp != left_hyp + 1:
            continue
        ref_time = (reference[right_ref - 1].end + reference[right_ref].start) / 2
        hyp_time = (hypothesis[left_hyp].end + hypothesis[right_hyp].start) / 2
        errors.append(abs(ref_time - hyp_time) * 1000)

    n_ref = max(0, len(reference) - 1)
    n_hyp = max(0, len(hypothesis) - 1)
    recall: dict[float, float] = {}
    precision: dict[float, float] = {}
    f1: dict[float, float] = {}
    for tolerance in tolerances_ms:
        # Decimal seconds cannot represent every millisecond exactly.  The
        # epsilon keeps an intended 20.0 ms boundary in the 20 ms band.
        hits = sum(error <= tolerance + 1e-9 for error in errors)
        r = _ratio(hits, n_ref)
        p = _ratio(hits, n_hyp)
        recall[float(tolerance)] = r
        precision[float(tolerance)] = p
        f1[float(tolerance)] = 0.0 if p + r == 0 else 2 * p * r / (p + r)

    return BoundaryScore(
        mean_absolute_error_ms=mean(errors) if errors else inf,
        median_absolute_error_ms=median(errors) if errors else inf,
        reference_boundaries=n_ref,
        hypothesis_boundaries=n_hyp,
        comparable_boundaries=len(errors),
        insertions=insertions,
        deletions=deletions,
        substitutions=substitutions,
        absolute_errors_ms=tuple(errors),
        recall_within=recall,
        precision_within=precision,
        f1_within=f1,
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _validate(intervals: Sequence[PhoneInterval], label: str) -> None:
    for i, phone in enumerate(intervals):
        if not phone.label:
            raise ValueError(f"{label} phone {i} has an empty label")
        if phone.start < 0 or phone.end < phone.start:
            raise ValueError(f"{label} phone {i} has invalid times")
        if i and phone.start < intervals[i - 1].start:
            raise ValueError(f"{label} phones are not ordered at index {i}")


def _edit_align(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> tuple[list[tuple[int | None, int | None]], int, int, int]:
    """Levenshtein alignment with deterministic match/sub/delete/insert ties."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]
    back = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost[i][0], back[i][0] = i, "D"
    for j in range(1, cols):
        cost[0][j], back[0][j] = j, "I"
    for i in range(1, rows):
        for j in range(1, cols):
            same = reference[i - 1] == hypothesis[j - 1]
            candidates = [
                (cost[i - 1][j - 1] + (not same), "M" if same else "S", 0),
                (cost[i - 1][j] + 1, "D", 1),
                (cost[i][j - 1] + 1, "I", 2),
            ]
            best, op, _ = min(candidates, key=lambda item: (item[0], item[2]))
            cost[i][j], back[i][j] = int(best), op

    pairs: list[tuple[int | None, int | None]] = []
    insertions = deletions = substitutions = 0
    i, j = len(reference), len(hypothesis)
    while i or j:
        op = back[i][j]
        if op in {"M", "S"}:
            i -= 1
            j -= 1
            if op == "M":
                pairs.append((i, j))
            else:
                substitutions += 1
        elif op == "D":
            i -= 1
            deletions += 1
        else:
            j -= 1
            insertions += 1
    pairs.reverse()
    return pairs, insertions, deletions, substitutions
