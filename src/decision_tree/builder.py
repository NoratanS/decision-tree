"""Recursive C4.5 tree growth (spec R4, R5).

Cases travel as (case index, Case Weight) pairs so every count is weighted
(ADR 0002). Every Case starts with weight 1; a Case with a Missing Value for
the tested attribute goes down every branch with a fraction of its weight.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from decision_tree.measures import EPSILON, gain_ratio_of
from decision_tree.splitting import (
    CandidateTest,
    discrete_candidate,
    threshold_candidate,
)
from decision_tree.table import DecisionTable
from decision_tree.tree import (
    ClassDistribution,
    DiscreteTest,
    Leaf,
    Node,
    Test,
    Tree,
    majority,
)

type WeightedCases = list[tuple[int, float]]
# A plain alias, not a `type` statement, so the CLI can use it as a Typer option.
Criterion = Literal["gain", "gain-ratio"]
"""How tests are ranked: Gain (ID3) or Gain Ratio (C4.5)."""


_SCORES: dict[Criterion, Callable[[CandidateTest], float]] = {
    "gain": lambda c: c.gain,
    "gain-ratio": lambda c: gain_ratio_of(c.gain, c.partition, c.missing),
}


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
    ) -> CandidateTest | None:
        """Best Candidate Test by the criterion; the leftmost column wins ties.

        A Continuous Attribute offers its best Threshold by Gain. The Quinlan
        filter averages Gain over all Candidate Tests, zero Gains included (as
        C4.5).
        """
        candidates = []
        for attribute in sorted(available):
            candidate = self._candidate_test(cases, attribute)
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

    def _candidate_test(
        self, cases: WeightedCases, attribute: int
    ) -> CandidateTest | None:
        """The Candidate Test on `attribute` at a Node holding `cases`."""
        column, class_of = self._columns[attribute], self._class_of
        values = [(column[i], class_of[i], weight) for i, weight in cases]
        n_classes, min_cases = len(self.table.classes), self.options.min_cases
        if self.table.kinds[attribute] == "continuous":
            return threshold_candidate(
                self.table.attributes[attribute],
                attribute,
                values,
                n_classes,
                self._training_values[attribute],
                min_cases=min_cases,
            )
        return discrete_candidate(
            self._discrete_tests[attribute], values, n_classes, min_cases=min_cases
        )

    @cached_property
    def _discrete_tests(self) -> dict[int, DiscreteTest]:
        """The test on each Discrete Attribute: one branch per value in the
        Training Set, in file order."""
        table = self.table
        return {
            a: DiscreteTest(table.attributes[a], a, table.values(a))
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
    def _class_of(self) -> list[int]:
        """Each Case's Decision Class, as its position in a class counts list."""
        class_index = {c: k for k, c in enumerate(self.table.classes)}
        return [class_index[d] for d in self.table.decisions]

    def _distribution(self, cases: WeightedCases) -> ClassDistribution:
        decisions = self.table.decisions
        distribution = dict.fromkeys(self.table.classes, 0.0)
        for i, weight in cases:
            distribution[decisions[i]] += weight
        return distribution
