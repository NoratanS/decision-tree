"""Columns, Candidate Tests at a Node, and Threshold search (spec R4, R5.1).

`columns` reads each Conditional Attribute of a Decision Table once per tree.
At a Node, a column offers its Candidate Test on the Cases there, and the
winning Candidate Test sends those Cases down its branches. A test is scored
on the Cases with a known value (spec R4.5, ADR 0002).
"""

from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import log2
from typing import NamedTuple

from decision_tree.measures import EPSILON, Partition, gain, known_counts
from decision_tree.table import DecisionTable
from decision_tree.tree import DiscreteTest, Test, ThresholdTest

type WeightedCases = list[tuple[int, float]]
"""The Cases at a Node, as (case index, Case Weight) pairs."""

type WeightedValue[V] = tuple[V | None, int, float]
"""One Case at a Node: (attribute value, Decision Class index, Case Weight)."""


class CandidateTest(NamedTuple):
    """A Test allowed at a Node: at least two of its branches hold `min_cases`
    or more of the known weight.

    `partition` leaves out a discrete branch with no known weight, while
    `branches` (and so `Node.branches`) keeps one per training value. Gain and
    Split Info are the same either way: 0·log 0 = 0.
    """

    test: Test
    partition: Partition
    """Weighted class counts per branch, known values only."""
    missing: float
    """Weight of the Cases with a Missing Value for the attribute."""
    gain: float
    """Gain on the known Cases, scaled by their fraction of the weight."""
    column: Column
    """The column of the tested attribute, which `branches` routes."""

    def branches(self, cases: WeightedCases) -> dict[str, WeightedCases]:
        """The Cases of each branch of `test`, keyed by `test.branch_keys`.

        A Case with a Missing Value goes down every branch, its weight scaled by
        the branch's share of the known weight (the Missing Value weight rule in
        `tree.py`, spec R5.5). Zero shares are dropped, so an empty branch stays
        empty.
        """
        test, values = self.test, self.column.values
        branches: dict[str, WeightedCases] = {key: [] for key in test.branch_keys}
        missing: WeightedCases = []
        for i, weight in cases:
            key = test.route(values[i])
            if key is None:
                missing.append((i, weight))
            else:
                branches[key].append((i, weight))
        if missing:
            branch_weight = {
                key: sum(w for _, w in branch) for key, branch in branches.items()
            }
            # Not zero: the `-m` rule (`_enough_cases`) gave this Candidate Test
            # two branches of at least `min_cases` > 0 known weight.
            total = sum(branch_weight.values())
            for key, branch in branches.items():
                if share := branch_weight[key] / total:
                    branch.extend((i, weight * share) for i, weight in missing)
        return branches


def columns(table: DecisionTable) -> list[Column]:
    """One column per Conditional Attribute of `table`, in column order."""
    class_index = {c: k for k, c in enumerate(table.classes)}
    class_of = [class_index[d] for d in table.decisions]
    return [
        ContinuousColumn.of(table, a, class_of)
        if kind == "continuous"
        else DiscreteColumn.of(table, a, class_of)
        for a, kind in enumerate(table.kinds)
    ]


@dataclass(frozen=True, eq=False)
class DiscreteColumn:
    """A Discrete Attribute's value in every Case, as text; None for a
    Missing Value."""

    values: Sequence[str | None]
    test: DiscreteTest
    """One branch per value in the Training Set, in order of first appearance."""
    class_of: Sequence[int]
    """Each Case's Decision Class, as its index in `table.classes`."""
    n_classes: int

    @classmethod
    def of(
        cls, table: DecisionTable, attribute: int, class_of: Sequence[int]
    ) -> DiscreteColumn:
        values = [case[attribute] for case in table.cases]
        test = DiscreteTest(
            table.attributes[attribute], attribute, table.values(attribute)
        )
        return cls(values, test, class_of, len(table.classes))

    def candidate(
        self, cases: WeightedCases, *, min_cases: float
    ) -> CandidateTest | None:
        """The Candidate Test on this attribute at a Node holding `cases`, or
        None if it is not allowed.

        The partition holds a branch for each value present among the known
        Cases, in the order of `test.branch_keys`.
        """
        known, missing = _known(_weighted(self.values, self.class_of, cases))
        branches: dict[str, list[float]] = {}
        for value, k, weight in known:
            counts = branches.setdefault(value, [0.0] * self.n_classes)
            counts[k] += weight
        partition = [branches[v] for v in self.test.values if v in branches]
        if not _enough_cases([sum(branch) for branch in partition], min_cases):
            return None
        return _scored(self.test, partition, missing, self)


@dataclass(frozen=True, eq=False)
class ContinuousColumn:
    """A Continuous Attribute's value in every Case, parsed once; None for a
    Missing Value."""

    attribute: str
    """Name of the attribute."""
    index: int
    """Column of the attribute in a Case."""
    values: Sequence[float | None]
    training_values: Sequence[float]
    """Distinct known values in the Training Set, ascending."""
    class_of: Sequence[int]
    """Each Case's Decision Class, as its index in `table.classes`."""
    n_classes: int

    @classmethod
    def of(
        cls, table: DecisionTable, attribute: int, class_of: Sequence[int]
    ) -> ContinuousColumn:
        values = [
            None if (value := case[attribute]) is None else float(value)
            for case in table.cases
        ]
        training_values = sorted({v for v in values if v is not None})
        return cls(
            table.attributes[attribute],
            attribute,
            values,
            training_values,
            class_of,
            len(table.classes),
        )

    def candidate(
        self, cases: WeightedCases, *, min_cases: float
    ) -> CandidateTest | None:
        """The allowed Threshold test on this attribute with the highest Gain
        at a Node holding `cases`, or None if no boundary is allowed.

        Every boundary between adjacent distinct known values is tried. The
        Threshold is the largest training value at or below the midpoint of
        the boundary, as C4.5 prints it. Ties go to the lower Threshold.

        The known Cases are sorted once; one sweep then moves them left of the
        boundary one at a time. Info(T) and the known fraction are the same at
        every boundary, so the sweep minimises Info(X,T) instead of computing
        Gain, and keeps it in O(1) per Case (see `_info_split_sweep`).
        """
        n_classes = self.n_classes
        known, missing = _known(_weighted(self.values, self.class_of, cases))
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
        training_values = self.training_values
        threshold = training_values[bisect_right(training_values, midpoint) - 1]
        test = ThresholdTest(self.attribute, self.index, threshold)
        return _scored(test, partition, missing, self)


type Column = DiscreteColumn | ContinuousColumn
"""One Conditional Attribute read for tree growth; `columns` picks the kind."""


def _weighted[V](
    values: Sequence[V | None], class_of: Sequence[int], cases: WeightedCases
) -> list[WeightedValue[V]]:
    """`cases` as (attribute value, Decision Class index, Case Weight) triples."""
    return [(values[i], class_of[i], weight) for i, weight in cases]


def _known[V](
    cases: Sequence[WeightedValue[V]],
) -> tuple[list[tuple[V, int, float]], float]:
    """The Cases with a known value, and the weight of those without."""
    known = [case for case in cases if case[0] is not None]
    if len(known) == len(cases):
        return known, 0.0
    return known, sum(weight for value, _, weight in cases if value is None)


def _enough_cases(branch_weights: Iterable[float], min_cases: float) -> bool:
    """The `-m` rule: at least two branches hold `min_cases` or more.

    `CandidateTest.branches` divides by the known weight, which this rule keeps
    above 0 (with `min_cases` > 0, checked by `BuildOptions`)."""
    return sum(1 for weight in branch_weights if weight >= min_cases) >= 2


def _scored(
    test: Test, partition: Partition, missing: float, column: Column
) -> CandidateTest:
    """`test` with its Gain on the known Cases, scaled by the known fraction."""
    score = gain(known_counts(partition), partition, missing=missing)
    return CandidateTest(test, partition, missing, score, column)


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
