"""Threshold search on a Continuous Attribute (spec R4.2, R4.6)."""

from bisect import bisect_right
from collections.abc import Iterable, Sequence
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
    boundary one at a time, so each candidate costs O(classes).
    """
    ordered = sorted(cases, key=lambda case: case[0])
    total = [0.0] * n_classes
    for _, k, weight in ordered:
        total[k] += weight
    total_weight = sum(total)

    left = [0.0] * n_classes
    left_weight = 0.0
    best: tuple[int, ThresholdPartition, float] | None = None
    for i, (value, k, weight) in enumerate(ordered[:-1]):
        left[k] += weight
        left_weight += weight
        if value == ordered[i + 1][0]:
            continue
        if left_weight < min_cases or total_weight - left_weight < min_cases:
            continue
        partition = (left.copy(), [t - c for t, c in zip(total, left, strict=True)])
        score = gain(total, partition)
        if best is None or score > best[2] + EPSILON:
            best = (i, partition, score)

    if best is None:
        return None
    i, partition, score = best
    midpoint = (ordered[i][0] + ordered[i + 1][0]) / 2
    threshold = training_values[bisect_right(training_values, midpoint) - 1]
    return ThresholdTest(threshold, partition, score)
