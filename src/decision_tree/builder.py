"""Recursive C4.5 tree growth (spec R4, R5).

Cases travel as (case index, Case Weight) pairs so every count is weighted
(ADR 0002). Every Case starts with weight 1; a Case with a Missing Value for
the tested attribute goes down every branch with a fraction of its weight.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, NamedTuple

from decision_tree.measures import EPSILON, Partition, gain, gain_ratio
from decision_tree.splitting import best_threshold
from decision_tree.table import DecisionTable
from decision_tree.tree import (
    ClassDistribution,
    DiscreteTest,
    Leaf,
    Node,
    Test,
    ThresholdTest,
    Tree,
    majority,
)

type WeightedCases = list[tuple[int, float]]
# A plain alias, not a `type` statement, so the CLI can use it as a Typer option.
Criterion = Literal["gain", "gain-ratio"]
"""How tests are ranked: Gain (ID3) or Gain Ratio (C4.5)."""


class _Candidate(NamedTuple):
    test: Test
    partition: Partition
    """Weighted class counts per branch, known values only."""
    missing: float
    """Weight of the Cases with a Missing Value for the attribute."""
    gain: float


_SCORES: dict[Criterion, Callable[[_Candidate], float]] = {
    "gain": lambda c: c.gain,
    "gain-ratio": lambda c: gain_ratio(
        _known_counts(c.partition), c.partition, missing=c.missing
    ),
}


def _known_counts(partition: Partition) -> list[float]:
    """Class counts of the Cases a partition covers: its column sums."""
    return [sum(column) for column in zip(*partition, strict=True)]


@dataclass(frozen=True)
class BuildOptions:
    """How a tree is grown; pruning is a separate stage (spec R5, R10).

    A test is a candidate when at least two of its branches hold `min_cases`
    or more cases (`-m`, spec R5.1). Among candidates with positive Gain the
    best `criterion` wins. With `quinlan_gain_filter`, a test must also have
    at least the average Gain of all candidates (spec R4.4).

    Raises ValueError for an unknown `criterion` or a `min_cases` of 0 or less.
    """

    criterion: Criterion = "gain-ratio"
    quinlan_gain_filter: bool = False
    min_cases: float = 2

    def __post_init__(self) -> None:
        if self.criterion not in _SCORES:
            raise ValueError(f"unknown criterion {self.criterion!r}")
        if self.min_cases <= 0:
            raise ValueError(f"min_cases must be above 0, got {self.min_cases!r}")


DEFAULT_BUILD_OPTIONS = BuildOptions()
"""C4.5's defaults: Gain Ratio, no Quinlan filter, `-m 2`."""


def build_tree(
    table: DecisionTable, options: BuildOptions = DEFAULT_BUILD_OPTIONS
) -> Tree:
    """Grow an unpruned C4.5 tree from a Training Set, as `options` say."""
    cases = [(i, 1.0) for i in range(len(table.cases))]
    available = frozenset(range(len(table.attributes)))
    builder = _Builder(table, options)
    return builder.grow(cases, available, table.classes[0])


@dataclass(frozen=True)
class _Builder:
    table: DecisionTable
    options: BuildOptions

    def grow(
        self, cases: WeightedCases, available: frozenset[int], parent_majority: str
    ) -> Tree:
        distribution = self._distribution(cases)
        if sum(distribution.values()) <= 0:
            return Leaf(parent_majority, distribution)

        most_common = majority(distribution)
        pure = sum(1 for w in distribution.values() if w > 0) == 1
        best = None if pure else self._best_test(cases, available)
        if best is None:
            return Leaf(most_common, distribution)

        test = best.test
        remaining = available if test.keeps_attribute else available - {test.index}
        subsets = self._subsets(cases, test)
        return Node(
            test=test,
            branches={
                key: self.grow(subset, remaining, most_common)
                for key, subset in subsets.items()
            },
            distribution=distribution,
        )

    def _subsets(self, cases: WeightedCases, test: Test) -> dict[str, WeightedCases]:
        """The Cases of each branch of `test`, keyed as in `Node.branches`.

        A Case with a Missing Value goes down every branch, its weight scaled by
        the branch's share of the known weight (the Missing Value weight rule in
        `tree.py`, spec R5.5). Zero shares are dropped, so an empty branch stays
        empty.
        """
        subsets: dict[str, WeightedCases] = {key: [] for key in test.branch_keys}
        column = self._columns[test.index]
        missing: WeightedCases = []
        for i, weight in cases:
            key = test.route(column[i])
            if key is None:
                missing.append((i, weight))
            else:
                subsets[key].append((i, weight))
        if missing:
            branch_weight = {
                key: sum(w for _, w in subset) for key, subset in subsets.items()
            }
            total = sum(branch_weight.values())
            for key, subset in subsets.items():
                if share := branch_weight[key] / total:
                    subset.extend((i, weight * share) for i, weight in missing)
        return subsets

    def _best_test(
        self, cases: WeightedCases, available: frozenset[int]
    ) -> _Candidate | None:
        """Best candidate by the criterion; the leftmost column wins ties.

        Candidates are the tests that pass the `min_cases` check on the Cases
        whose value is known; a Continuous Attribute offers its best Threshold
        by Gain. The Quinlan filter averages Gain over all of them, zero Gains
        included (as C4.5).
        """
        candidates = []
        for attribute in sorted(available):
            if self.table.kinds[attribute] == "continuous":
                candidate = self._threshold_test(cases, attribute)
            else:
                candidate = self._discrete_test(cases, attribute)
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            return None

        min_gain = EPSILON
        if self.options.quinlan_gain_filter:
            average = sum(c.gain for c in candidates) / len(candidates)
            min_gain = max(min_gain, average - EPSILON)

        score_of = _SCORES[self.options.criterion]
        best, best_score = None, 0.0
        for candidate in candidates:
            if candidate.gain <= min_gain:
                continue
            score = score_of(candidate)
            # A later column must win by more than rounding noise.
            if best is None or score > best_score + EPSILON:
                best, best_score = candidate, score
        return best

    def _discrete_test(self, cases: WeightedCases, attribute: int) -> _Candidate | None:
        """The test on a Discrete Attribute, scored on the Cases with a known
        value; None if it fails the `min_cases` check."""
        rows, decisions = self.table.cases, self.table.decisions
        class_index, n_classes = self._class_index, len(self._class_index)
        branches: dict[str, list[float]] = {}
        missing = 0.0
        for i, weight in cases:
            value = rows[i][attribute]
            if value is None:
                missing += weight
                continue
            counts = branches.setdefault(value, [0.0] * n_classes)
            counts[class_index[decisions[i]]] += weight
        partition = list(branches.values())
        if not self._enough_cases(partition):
            return None
        test = DiscreteTest(
            self.table.attributes[attribute], attribute, self._values[attribute]
        )
        return self._candidate(test, partition, missing)

    def _threshold_test(
        self, cases: WeightedCases, attribute: int
    ) -> _Candidate | None:
        """The best Threshold test on the Cases with a known value."""
        numbers, decisions = self._numbers[attribute], self.table.decisions
        class_index = self._class_index
        known = []
        missing = 0.0
        for i, weight in cases:
            number = numbers[i]
            if number is None:
                missing += weight
            else:
                known.append((number, class_index[decisions[i]], weight))
        if not known:
            return None
        found = best_threshold(
            known,
            len(class_index),
            self._training_values[attribute],
            min_cases=self.options.min_cases,
        )
        if found is None:
            return None
        test = ThresholdTest(
            self.table.attributes[attribute], attribute, found.threshold
        )
        return self._candidate(test, found.partition, missing)

    def _candidate(
        self, test: Test, partition: Partition, missing: float
    ) -> _Candidate:
        """A test scored on the known Cases, Gain scaled by the known fraction."""
        score = gain(_known_counts(partition), partition, missing=missing)
        return _Candidate(test, partition, missing, score)

    def _enough_cases(self, partition: Partition) -> bool:
        """At least two branches hold `min_cases` or more cases."""
        min_cases = self.options.min_cases
        return sum(1 for branch in partition if sum(branch) >= min_cases) >= 2

    @cached_property
    def _values(self) -> dict[int, tuple[str, ...]]:
        """Known values of each Discrete Attribute, in file order."""
        table = self.table
        return {
            a: table.values(a)
            for a, kind in enumerate(table.kinds)
            if kind == "discrete"
        }

    @cached_property
    def _numbers(self) -> dict[int, list[float | None]]:
        """Each Continuous Attribute's value in every Case, parsed once; None
        for a Missing Value."""
        table = self.table
        return {
            a: [None if case[a] is None else float(case[a]) for case in table.cases]
            for a, kind in enumerate(table.kinds)
            if kind == "continuous"
        }

    @cached_property
    def _columns(self) -> dict[int, Sequence[str | float | None]]:
        """Each attribute's value in every Case, as its Test routes it: parsed
        once for a Continuous Attribute, as text for a Discrete one."""
        table = self.table
        return {
            a: self._numbers[a] if kind == "continuous" else [c[a] for c in table.cases]
            for a, kind in enumerate(table.kinds)
        }

    @cached_property
    def _training_values(self) -> dict[int, list[float]]:
        """Distinct known values of each Continuous Attribute, ascending."""
        return {
            a: sorted({n for n in numbers if n is not None})
            for a, numbers in self._numbers.items()
        }

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
