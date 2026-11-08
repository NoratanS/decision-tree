"""Candidate Tests at a Node, and Threshold search (spec R4, R5.1).

Cases arrive as (attribute value, Decision Class index, Case Weight) triples,
the value None for a Missing Value. A test is scored on the Cases with a
known value (spec R4.5, ADR 0002).
"""

from bisect import bisect_right
from collections.abc import Iterable, Sequence
from math import log2
from typing import NamedTuple

from decision_tree.measures import EPSILON, Partition, gain, known_counts
from decision_tree.tree import DiscreteTest, Test, ThresholdTest

type WeightedValue[V] = tuple[V | None, int, float]
"""One Case at a Node: (attribute value, Decision Class index, Case Weight)."""


class CandidateTest(NamedTuple):
    """A Test allowed at a Node: at least two of its branches hold `min_cases`
    or more of the known weight."""

    test: Test
    partition: Partition
    """Weighted class counts per branch, known values only."""
    missing: float
    """Weight of the Cases with a Missing Value for the attribute."""
    gain: float
    """Gain on the known Cases, scaled by their fraction of the weight."""


def discrete_candidate(
    test: DiscreteTest,
    cases: Sequence[WeightedValue[str]],
    n_classes: int,
    *,
    min_cases: float,
) -> CandidateTest | None:
    """`test` as a Candidate Test on `cases`, or None if it is not allowed.

    The partition holds a branch for each value present among the known
    Cases, in order of first appearance.
    """
    known, missing = _known(cases)
    branches: dict[str, list[float]] = {}
    for value, k, weight in known:
        counts = branches.setdefault(value, [0.0] * n_classes)
        counts[k] += weight
    partition = list(branches.values())
    if not _enough_cases([sum(branch) for branch in partition], min_cases):
        return None
    return _scored(test, partition, missing)


def threshold_candidate(
    attribute: str,
    index: int,
    cases: Sequence[WeightedValue[float]],
    n_classes: int,
    training_values: Sequence[float],
    *,
    min_cases: float,
) -> CandidateTest | None:
    """The allowed Threshold test on `attribute` (column `index`) with the
    highest Gain, or None if no boundary is allowed.

    Every boundary between adjacent distinct known values is tried. The
    Threshold is the largest of `training_values` (every value of the
    attribute in the Training Set, ascending) at or below the midpoint of the
    boundary, as C4.5 prints it. Ties go to the lower Threshold.

    The known Cases are sorted once; one sweep then moves them left of the
    boundary one at a time. Info(T) and the known fraction are the same at
    every boundary, so the sweep minimises Info(X,T) instead of computing
    Gain, and keeps it in O(1) per Case (see `_info_split_sweep`).
    """
    known, missing = _known(cases)
    ordered = sorted(known, key=lambda case: case[0])
    total = [0.0] * n_classes
    for _, k, weight in ordered:
        total[k] += weight

    best_index = _info_split_sweep(ordered, total, min_cases)
    if best_index is None:
        return None
    left = [0.0] * n_classes
    for _, k, weight in ordered[: best_index + 1]:
        left[k] += weight
    partition = (left, [t - c for t, c in zip(total, left, strict=True)])
    midpoint = (ordered[best_index][0] + ordered[best_index + 1][0]) / 2
    threshold = training_values[bisect_right(training_values, midpoint) - 1]
    return _scored(ThresholdTest(attribute, index, threshold), partition, missing)


def _known[V](
    cases: Sequence[WeightedValue[V]],
) -> tuple[list[tuple[V, int, float]], float]:
    """The Cases with a known value, and the weight of those without."""
    known = [case for case in cases if case[0] is not None]
    if len(known) == len(cases):
        return known, 0.0
    return known, sum(weight for value, _, weight in cases if value is None)


def _enough_cases(branch_weights: Iterable[float], min_cases: float) -> bool:
    """The `-m` rule: at least two branches hold `min_cases` or more."""
    return sum(1 for weight in branch_weights if weight >= min_cases) >= 2


def _scored(test: Test, partition: Partition, missing: float) -> CandidateTest:
    """`test` with its Gain on the known Cases, scaled by the known fraction."""
    score = gain(known_counts(partition), partition, missing=missing)
    return CandidateTest(test, partition, missing, score)


def _info_split_sweep(
    ordered: Sequence[tuple[float, int, float]],
    total: Sequence[float],
    min_cases: float,
) -> int | None:
    """Index of the last `<=` Case at the allowed boundary with the lowest
    Info(X,T); the first such boundary on a tie. None if none is allowed.

    With f(x) = x·log2 x, a branch of weight n and class counts c holds
    n·Info = f(n) − Σ f(c). The sweep keeps Σ f(c) of both branches as running
    sums: moving one Case changes a single class count on each side. This is
    the hot loop of tree growth, so f is inlined and each f(c) is kept.
    """
    total_weight = sum(total)
    left = [0.0] * len(total)
    right = list(total)
    xlogx_left = [0.0] * len(total)  # f(c) of each class count
    xlogx_right = list(map(_xlog2x, total))
    left_sum, right_sum = 0.0, sum(xlogx_right)  # Σ f(c) of each branch
    left_weight = 0.0
    best, best_info = None, 0.0
    for i in range(len(ordered) - 1):
        value, k, weight = ordered[i]
        count = left[k] = left[k] + weight
        xlogx = count * log2(count) if count > 0 else 0.0
        left_sum += xlogx - xlogx_left[k]
        xlogx_left[k] = xlogx
        count = right[k] = right[k] - weight
        xlogx = count * log2(count) if count > 0 else 0.0
        right_sum += xlogx - xlogx_right[k]
        xlogx_right[k] = xlogx
        left_weight += weight
        if value == ordered[i + 1][0]:
            continue
        right_weight = total_weight - left_weight
        info_split = (
            _xlog2x(left_weight) - left_sum + _xlog2x(right_weight) - right_sum
        ) / total_weight
        # The `-m` rule is checked only on a would-be best boundary: calling it
        # at every boundary costs a fifth of tree growth.
        if (best is None or info_split < best_info - EPSILON) and _enough_cases(
            (left_weight, right_weight), min_cases
        ):
            best, best_info = i, info_split
    return best


def _xlog2x(x: float) -> float:
    """x·log2 x, taken as 0 at 0 (and for rounding noise below it)."""
    return x * log2(x) if x > 0 else 0.0
