"""Recursive C4.5 tree growth (spec R4.3, R4.4, R4.6, R5).

Cases travel as (case index, Case Weight) pairs so every count is weighted
(ADR 0002). In this version every Case Weight is 1.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, NamedTuple

from decision_tree.measures import ClassCounts, Partition, gain, gain_ratio
from decision_tree.table import DecisionTable
from decision_tree.tree import ClassDistribution, Leaf, Node, Tree

type WeightedCases = list[tuple[int, float]]
# A plain alias, not a `type` statement, so the CLI can use it as a Typer option.
Criterion = Literal["gain", "gain-ratio"]
"""How tests are ranked: Gain (ID3) or Gain Ratio (C4.5)."""

# Gains, and differences between scores, this small are rounding noise.
_EPSILON = 1e-12


class _Candidate(NamedTuple):
    attribute: int
    partition: Partition
    gain: float


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

        majority = _majority(distribution)
        pure = sum(1 for w in distribution.values() if w > 0) == 1
        counts = list(distribution.values())
        attribute = None if pure else self._best_attribute(cases, counts, available)
        if attribute is None:
            return Leaf(majority, distribution)

        table = self.table
        subsets: dict[str, WeightedCases] = {v: [] for v in table.values(attribute)}
        for i, weight in cases:
            subsets[table.cases[i][attribute]].append((i, weight))
        remaining = available - {attribute}
        return Node(
            attribute=table.attributes[attribute],
            index=attribute,
            branches={
                value: self.grow(subset, remaining, majority)
                for value, subset in subsets.items()
            },
            distribution=distribution,
        )

    def _best_attribute(
        self, cases: WeightedCases, counts: ClassCounts, available: frozenset[int]
    ) -> int | None:
        """Best candidate by the criterion; the leftmost column wins ties.

        Candidates are the tests that pass the `min_cases` check. The Quinlan
        filter averages Gain over all of them, zero Gains included (as C4.5).
        """
        candidates = []
        for attribute in sorted(available):
            partition = self._partition(cases, attribute)
            if self._enough_cases(partition):
                candidates.append(
                    _Candidate(attribute, partition, gain(counts, partition))
                )
        if not candidates:
            return None

        min_gain = _EPSILON
        if self.quinlan_gain_filter:
            average = sum(c.gain for c in candidates) / len(candidates)
            min_gain = max(min_gain, average - _EPSILON)

        best, best_score = None, 0.0
        for candidate in candidates:
            if candidate.gain <= min_gain:
                continue
            score = _SCORES[self.criterion](counts, candidate)
            # A later column must win by more than rounding noise.
            if best is None or score > best_score + _EPSILON:
                best, best_score = candidate.attribute, score
        return best

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
    def _class_index(self) -> dict[str, int]:
        """Position of each Decision Class in a class counts list."""
        return {c: k for k, c in enumerate(self.table.classes)}

    def _distribution(self, cases: WeightedCases) -> ClassDistribution:
        decisions = self.table.decisions
        distribution = dict.fromkeys(self.table.classes, 0.0)
        for i, weight in cases:
            distribution[decisions[i]] += weight
        return distribution


def _majority(distribution: ClassDistribution) -> str:
    """Most common Decision Class; the first in file order wins ties."""
    return max(distribution, key=lambda c: distribution[c])
