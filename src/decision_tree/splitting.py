"""Threshold search on a Continuous Attribute (spec R4.2, R4.6)."""

from bisect import bisect_right
from collections.abc import Iterable, Sequence
from math import log2
from typing import NamedTuple

from decision_tree.measures import EPSILON, gain

type WeightedValue = tuple[float, int, float]
"""One Case at a Node: (attribute value, Decision Class index, Case Weight)."""

type ThresholdPartition = tuple[list[float], list[float]]
"""Weighted class counts of the `<=` and `>` branches."""


class ThresholdTest(NamedTuple):
    threshold: float
    """Cases with a value `<= threshold` take the first branch, the rest the second."""
    partition: ThresholdPartition
    gain: float


def best_threshold(
    cases: Iterable[WeightedValue],
    n_classes: int,
    training_values: Sequence[float],
    *,
    min_cases: float,
) -> ThresholdTest | None:
    """The Threshold with the highest Gain, or None if no boundary is allowed.

    Every boundary between adjacent distinct values is tried, keeping those
    where both branches hold `min_cases` or more. The Threshold is the largest of
    `training_values` (every value of the attribute in the Training Set,
    ascending) at or below the midpoint of the boundary, as C4.5 prints it.
    Ties go to the lower Threshold.

    The cases are sorted once; one sweep then moves them left of the
    boundary one at a time. Info(T) is the same at every boundary, so the
    sweep minimises Info(X,T) instead of computing Gain, and keeps it in O(1)
    per Case (see `_info_split_sweep`). Gain is computed once, for the winner.
    """
    ordered = sorted(cases, key=lambda case: case[0])
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
    return ThresholdTest(threshold, partition, gain(total, partition))


def _info_split_sweep(
    ordered: Sequence[WeightedValue], total: Sequence[float], min_cases: float
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
        if left_weight < min_cases or right_weight < min_cases:
            continue
        info_split = (
            _xlog2x(left_weight) - left_sum + _xlog2x(right_weight) - right_sum
        ) / total_weight
        if best is None or info_split < best_info - EPSILON:
            best, best_info = i, info_split
    return best


def _xlog2x(x: float) -> float:
    """x·log2 x, taken as 0 at 0 (and for rounding noise below it)."""
    return x * log2(x) if x > 0 else 0.0
