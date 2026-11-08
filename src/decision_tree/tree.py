"""Tests, Node and Leaf types, and classification (spec R5.6, R7)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

type ClassDistribution = dict[str, float]
"""Weight of each Decision Class, in file order (zero weights included)."""


@dataclass(frozen=True)
class Leaf:
    decision: str
    distribution: ClassDistribution

    @property
    def total(self) -> float:
        return sum(self.distribution.values())

    @property
    def errors(self) -> float:
        """Weight of the training cases here that are not of `decision`."""
        return self.total - self.distribution[self.decision]


# A Case with a Missing Value for the tested attribute (or, when classifying, a
# discrete value never seen in training) goes down every branch of the Node,
# weighted by each branch's share of the weight (spec R5.5, R7). As in C4.5:
# growth shares the Case's weight by the known weight of each branch
# (`builder._subsets`, C4.5's `Group` in `build.c`); classification shares the
# branch's Class Distribution by its training weight, `child.total`
# (`_combine_branches`, C4.5's `Classify` in `classify.c`).


@dataclass(frozen=True)
class DiscreteTest:
    """A test on a Discrete Attribute: one branch per value."""

    attribute: str
    """Name of the tested Conditional Attribute."""
    index: int
    """Column of the tested attribute in a Case."""
    values: tuple[str, ...]
    """The branch keys: the values seen in training, in order of first appearance."""

    keeps_attribute: ClassVar[bool] = False
    """A Discrete Attribute is tested at most once per path (spec R5.2)."""

    @property
    def branch_keys(self) -> tuple[str, ...]:
        return self.values

    def route(self, value: str | float | None) -> str | None:
        """The branch for `value`; None for a Missing Value or a value never
        seen in training. (A number is never a discrete value; the type matches
        `ThresholdTest.route`.)"""
        return value if value in self.values else None


@dataclass(frozen=True)
class ThresholdTest:
    """A test on a Continuous Attribute: `<=` the Threshold, then `>`."""

    attribute: str
    """Name of the tested Conditional Attribute."""
    index: int
    """Column of the tested attribute in a Case."""
    threshold: float

    keeps_attribute: ClassVar[bool] = True
    """A Continuous Attribute may be tested again below, at another Threshold."""

    branch_keys: ClassVar[tuple[str, str]] = ("<=", ">")

    def route(self, value: str | float | None) -> str | None:
        """`"<="` or `">"` for a value (text or already parsed); None for a
        Missing Value."""
        if value is None:
            return None
        return "<=" if float(value) <= self.threshold else ">"


type Test = DiscreteTest | ThresholdTest
"""What a Node asks of a Case's value of one Conditional Attribute."""


@dataclass(frozen=True)
class Node:
    test: Test
    branches: dict[str, Tree]
    """One subtree per branch key of `test`, in the same order."""
    distribution: ClassDistribution

    @property
    def total(self) -> float:
        """Training weight that reached this Node."""
        return sum(self.distribution.values())


type Tree = Node | Leaf


def size(tree: Tree) -> int:
    """Number of Nodes and Leaves, as C4.5 reports a tree's size."""
    if isinstance(tree, Leaf):
        return 1
    return 1 + sum(size(child) for child in tree.branches.values())


def majority(distribution: ClassDistribution) -> str:
    """The class with the most weight (or probability); the first in file order
    wins ties (spec R5.4, R7)."""
    return max(distribution, key=distribution.__getitem__)


def classify(tree: Tree, case: Sequence[str | None]) -> ClassDistribution:
    """Class Distribution for a Case, as probabilities summing to 1 (spec R7).

    `case` holds the Conditional Attribute values in column order; None is a
    Missing Value. A discrete value never seen in training counts as missing.
    At a Node testing a Missing Value the Case goes down every branch (see the
    Missing Value weight rule above the Tests).
    """
    while isinstance(tree, Node):
        branch = tree.test.route(case[tree.test.index])
        if branch is None:
            return _combine_branches(tree, case)
        tree = tree.branches[branch]
    total = tree.total
    if total <= 0:
        return {c: float(c == tree.decision) for c in tree.distribution}
    return {c: w / total for c, w in tree.distribution.items()}


def _combine_branches(node: Node, case: Sequence[str | None]) -> ClassDistribution:
    """Classify `case` down every branch of `node`; sum the distributions, each
    weighted by its branch's share of the training weight."""
    total = sum(child.total for child in node.branches.values())
    combined = dict.fromkeys(node.distribution, 0.0)
    for child in node.branches.values():
        if share := child.total / total:
            for decision, probability in classify(child, case).items():
                combined[decision] += share * probability
    return combined
