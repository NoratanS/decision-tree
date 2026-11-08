"""Node and Leaf types (spec R5.6)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

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


@dataclass(frozen=True)
class Node:
    attribute: str
    """Name of the tested Conditional Attribute."""
    index: int
    """Column of the tested attribute in a Case."""
    branches: dict[str, Tree]
    """Discrete test: one subtree per value, in order of first appearance in
    the Training Set. Threshold test: `"<="` then `">"`."""
    distribution: ClassDistribution
    threshold: float | None = None
    """The Threshold of a test on a Continuous Attribute; None for a discrete test."""

    @property
    def total(self) -> float:
        """Training weight that reached this Node."""
        return sum(self.distribution.values())


type Tree = Node | Leaf


def threshold_branch(number: float, threshold: float) -> Literal["<=", ">"]:
    """The branch key a value takes at a Threshold test."""
    return "<=" if number <= threshold else ">"


def majority(distribution: ClassDistribution) -> str:
    """The class with the most weight (or probability); the first in file order
    wins ties (spec R5.4, R7)."""
    return max(distribution, key=distribution.__getitem__)


def classify(tree: Tree, case: Sequence[str | None]) -> ClassDistribution:
    """Class Distribution for a Case, as probabilities summing to 1 (spec R7).

    `case` holds the Conditional Attribute values in column order; None is a
    Missing Value. A discrete value never seen in training counts as missing.
    At a Node testing a Missing Value the Case goes down every branch, and the
    branches' distributions are combined, each weighted by the branch's share
    of the training weight.
    """
    while isinstance(tree, Node):
        value = case[tree.index]
        if tree.threshold is not None and value is not None:
            tree = tree.branches[threshold_branch(float(value), tree.threshold)]
        elif value in tree.branches:
            tree = tree.branches[value]
        else:
            return _combine_branches(tree, case)
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
