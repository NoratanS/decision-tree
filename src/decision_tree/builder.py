"""Recursive C4.5 tree growth (spec R4, R5).

Cases travel as (case index, Case Weight) pairs so every count is weighted
(ADR 0002). In this version every Case Weight is 1.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, NamedTuple

from decision_tree.measures import EPSILON, ClassCounts, Partition, gain, gain_ratio
from decision_tree.splitting import best_threshold
from decision_tree.table import DecisionTable
from decision_tree.tree import (
    ClassDistribution,
    Leaf,
    Node,
    Tree,
    majority,
    threshold_branch,
)

type WeightedCases = list[tuple[int, float]]
# A plain alias, not a `type` statement, so the CLI can use it as a Typer option.
Criterion = Literal["gain", "gain-ratio"]
"""How tests are ranked: Gain (ID3) or Gain Ratio (C4.5)."""


class _Candidate(NamedTuple):
    attribute: int
    partition: Partition
    gain: float
    threshold: float | None = None
    """The Threshold of a test on a Continuous Attribute; None for a discrete test."""


_SCORES: dict[Criterion, Callable[[ClassCounts, _Candidate], float]] = {
    "gain": lambda counts, candidate: candidate.gain,
    "gain-ratio": lambda counts, candidate: gain_ratio(counts, candidate.partition),
}


def build_tree(
    table: DecisionTable,
    *,
    criterion: Criterion = "gain-ratio",
    quinlan_gain_filter: bool = False,
    min_cases: float = 2,
) -> Tree:
    """Grow an unpruned C4.5 tree from a Training Set.

    A test is a candidate when at least two of its branches hold `min_cases`
    or more cases (`-m`, spec R5.1). Among candidates with positive Gain the
    best `criterion` wins. With `quinlan_gain_filter`, a test must also have
    at least the average Gain of all candidates (spec R4.4).
    """
    if criterion not in _SCORES:
        raise ValueError(f"unknown criterion {criterion!r}")
    cases = [(i, 1.0) for i in range(len(table.cases))]
    available = frozenset(range(len(table.attributes)))
    builder = _Builder(table, criterion, quinlan_gain_filter, min_cases)
    return builder.grow(cases, available, table.classes[0])


@dataclass(frozen=True)
class _Builder:
    table: DecisionTable
    criterion: Criterion
    quinlan_gain_filter: bool
    min_cases: float

    def grow(
        self, cases: WeightedCases, available: frozenset[int], parent_majority: str
    ) -> Tree:
        distribution = self._distribution(cases)
        if sum(distribution.values()) <= 0:
            return Leaf(parent_majority, distribution)

        most_common = majority(distribution)
        pure = sum(1 for w in distribution.values() if w > 0) == 1
        counts = list(distribution.values())
        test = None if pure else self._best_test(cases, counts, available)
        if test is None:
            return Leaf(most_common, distribution)

        table, attribute = self.table, test.attribute
        if test.threshold is None:
            subsets: dict[str, WeightedCases] = {v: [] for v in table.values(attribute)}
            for i, weight in cases:
                subsets[table.cases[i][attribute]].append((i, weight))
            # A Discrete Attribute is tested at most once per path (spec R5.2).
            remaining = available - {attribute}
        else:
            numbers = self._numbers[attribute]
            subsets = {"<=": [], ">": []}
            for i, weight in cases:
                subsets[threshold_branch(numbers[i], test.threshold)].append(
                    (i, weight)
                )
            # A Continuous Attribute may be tested again with another Threshold.
            remaining = available
        return Node(
            attribute=table.attributes[attribute],
            index=attribute,
            branches={
                value: self.grow(subset, remaining, most_common)
                for value, subset in subsets.items()
            },
            distribution=distribution,
            threshold=test.threshold,
        )

    def _best_test(
        self, cases: WeightedCases, counts: ClassCounts, available: frozenset[int]
    ) -> _Candidate | None:
        """Best candidate by the criterion; the leftmost column wins ties.

        Candidates are the tests that pass the `min_cases` check; a Continuous
        Attribute offers its best Threshold by Gain. The Quinlan filter
        averages Gain over all of them, zero Gains included (as C4.5).
        """
        candidates = []
        for attribute in sorted(available):
            if self.table.kinds[attribute] == "continuous":
                candidate = self._threshold_test(cases, attribute)
                if candidate is not None:
                    candidates.append(candidate)
                continue
            partition = self._partition(cases, attribute)
            if self._enough_cases(partition):
                candidates.append(
                    _Candidate(attribute, partition, gain(counts, partition))
                )
        if not candidates:
            return None

        min_gain = EPSILON
        if self.quinlan_gain_filter:
            average = sum(c.gain for c in candidates) / len(candidates)
            min_gain = max(min_gain, average - EPSILON)

        best, best_score = None, 0.0
        for candidate in candidates:
            if candidate.gain <= min_gain:
                continue
            score = _SCORES[self.criterion](counts, candidate)
            # A later column must win by more than rounding noise.
            if best is None or score > best_score + EPSILON:
                best, best_score = candidate, score
        return best

    def _threshold_test(
        self, cases: WeightedCases, attribute: int
    ) -> _Candidate | None:
        numbers, decisions = self._numbers[attribute], self.table.decisions
        class_index = self._class_index
        found = best_threshold(
            ((numbers[i], class_index[decisions[i]], weight) for i, weight in cases),
            len(class_index),
            self._training_values[attribute],
            min_cases=self.min_cases,
        )
        if found is None:
            return None
        return _Candidate(attribute, found.partition, found.gain, found.threshold)

    def _enough_cases(self, partition: Partition) -> bool:
        """At least two branches hold `min_cases` or more cases."""
        return sum(1 for branch in partition if sum(branch) >= self.min_cases) >= 2

    def _partition(self, cases: WeightedCases, attribute: int) -> Partition:
        """Weighted class counts per value of `attribute`."""
        rows, decisions = self.table.cases, self.table.decisions
        class_index, n_classes = self._class_index, len(self._class_index)
        branches: dict[str, list[float]] = {}
        for i, weight in cases:
            counts = branches.setdefault(rows[i][attribute], [0.0] * n_classes)
            counts[class_index[decisions[i]]] += weight
        return list(branches.values())

    @cached_property
    def _numbers(self) -> dict[int, list[float]]:
        """Each Continuous Attribute's value in every Case, parsed once."""
        table = self.table
        return {
            a: [float(case[a]) for case in table.cases]
            for a, kind in enumerate(table.kinds)
            if kind == "continuous"
        }

    @cached_property
    def _training_values(self) -> dict[int, list[float]]:
        """Distinct values of each Continuous Attribute, ascending."""
        return {a: sorted(set(numbers)) for a, numbers in self._numbers.items()}

    @cached_property
    def _class_index(self) -> dict[str, int]:
        """Position of each Decision Class in a class counts list."""
        return {c: k for k, c in enumerate(self.table.classes)}

    def _distribution(self, cases: WeightedCases) -> ClassDistribution:
        decisions = self.table.decisions
        distribution = dict.fromkeys(self.table.classes, 0.0)
        for i, weight in cases:
            distribution[decisions[i]] += weight
        return distribution
