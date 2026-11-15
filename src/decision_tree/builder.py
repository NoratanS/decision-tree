"""Recursive C4.5 tree growth (spec R4, R5).

Cases travel as (case index, Case Weight) pairs so every count is weighted
(ADR 0002). Every Case starts with weight 1; a Case with a Missing Value for
the tested attribute goes down every branch with a fraction of its weight.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from decision_tree.measures import EPSILON
from decision_tree.splitting import CandidateTest, Column, WeightedCases, columns
from decision_tree.table import DecisionTable
from decision_tree.tree import ClassDistribution, Leaf, Node, Tree

# A plain alias, not a `type` statement, so the CLI can use it as a Typer option.
Criterion = Literal["gain", "gain-ratio"]
"""How tests are ranked: Gain (ID3) or Gain Ratio (C4.5)."""


_SCORES: dict[Criterion, Callable[[CandidateTest], float]] = {
    "gain": lambda c: c.gain,
    "gain-ratio": lambda c: c.gain_ratio,
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
    builder = _Builder(table, options, columns(table))
    return builder.grow(cases, available, table.classes[0])


@dataclass(frozen=True)
class _Builder:
    table: DecisionTable
    options: BuildOptions
    columns: list[Column]
    """Each Conditional Attribute, read once per tree."""

    def grow(
        self, cases: WeightedCases, available: frozenset[int], parent_majority: str
    ) -> Tree:
        distribution = self._distribution(cases)
        if distribution.total <= 0:
            return Leaf(parent_majority, distribution)

        most_common = distribution.majority()
        best = None if distribution.is_pure else self._best_test(cases, available)
        if best is None:
            return Leaf(most_common, distribution)

        test = best.test
        remaining = available if test.keeps_attribute else available - {test.index}
        return Node(
            test=test,
            branches={
                key: self.grow(subset, remaining, most_common)
                for key, subset in best.branches(cases).items()
            },
            distribution=distribution,
        )

    def _best_test(
        self, cases: WeightedCases, available: frozenset[int]
    ) -> CandidateTest | None:
        """Best Candidate Test by the criterion; the leftmost column wins ties.

        A Continuous Attribute offers its best Threshold by Gain. The Quinlan
        filter averages Gain over all Candidate Tests, zero Gains included (as
        C4.5).
        """
        min_cases = self.options.min_cases
        candidates = []
        for attribute in sorted(available):
            candidate = self.columns[attribute].candidate(cases, min_cases=min_cases)
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

    def _distribution(self, cases: WeightedCases) -> ClassDistribution:
        decisions = self.table.decisions
        return ClassDistribution.count(
            self.table.classes, ((decisions[i], weight) for i, weight in cases)
        )
