"""Recursive C4.5 tree growth with Gain Ratio (spec R5).

Cases travel as (case index, Case Weight) pairs so every count is weighted
(ADR 0002). In this version every Case Weight is 1.
"""

from decision_tree.measures import gain, gain_ratio
from decision_tree.table import DecisionTable
from decision_tree.tree import ClassDistribution, Leaf, Node, Tree

type WeightedCases = list[tuple[int, float]]

# Gains this small are rounding noise, not information.
_MIN_GAIN = 1e-12


def build_tree(table: DecisionTable) -> Tree:
    """Grow an unpruned C4.5 tree from a Training Set."""
    cases = [(i, 1.0) for i in range(len(table.cases))]
    available = frozenset(range(len(table.attributes)))
    return _grow(table, cases, available, parent_majority=table.classes[0])


def _grow(
    table: DecisionTable,
    cases: WeightedCases,
    available: frozenset[int],
    parent_majority: str,
) -> Tree:
    distribution = _distribution(table, cases)
    if sum(distribution.values()) <= 0:
        return Leaf(parent_majority, distribution)

    majority = _majority(distribution)
    pure = sum(1 for w in distribution.values() if w > 0) == 1
    counts = list(distribution.values())
    attribute = None if pure else _best_attribute(table, cases, counts, available)
    if attribute is None:
        return Leaf(majority, distribution)

    subsets: dict[str, WeightedCases] = {v: [] for v in table.values(attribute)}
    for i, weight in cases:
        subsets[table.cases[i][attribute]].append((i, weight))
    remaining = available - {attribute}
    return Node(
        attribute=table.attributes[attribute],
        index=attribute,
        branches={
            value: _grow(table, subset, remaining, majority)
            for value, subset in subsets.items()
        },
        distribution=distribution,
    )


def _best_attribute(
    table: DecisionTable,
    cases: WeightedCases,
    counts: list[float],
    available: frozenset[int],
) -> int | None:
    """Highest Gain Ratio among tests with positive Gain; leftmost wins ties."""
    best, best_ratio = None, 0.0
    for attribute in sorted(available):
        partition = _partition(table, cases, attribute)
        if gain(counts, partition) <= _MIN_GAIN:
            continue
        ratio = gain_ratio(counts, partition)
        if best is None or ratio > best_ratio:
            best, best_ratio = attribute, ratio
    return best


def _partition(
    table: DecisionTable, cases: WeightedCases, attribute: int
) -> list[list[float]]:
    """Weighted class counts per value of `attribute`."""
    class_index = {c: k for k, c in enumerate(table.classes)}
    branches: dict[str, list[float]] = {}
    for i, weight in cases:
        counts = branches.setdefault(
            table.cases[i][attribute], [0.0] * len(class_index)
        )
        counts[class_index[table.decisions[i]]] += weight
    return list(branches.values())


def _distribution(table: DecisionTable, cases: WeightedCases) -> ClassDistribution:
    distribution = dict.fromkeys(table.classes, 0.0)
    for i, weight in cases:
        distribution[table.decisions[i]] += weight
    return distribution


def _majority(distribution: ClassDistribution) -> str:
    """Most common Decision Class; the first in file order wins ties."""
    return max(distribution, key=lambda c: distribution[c])
