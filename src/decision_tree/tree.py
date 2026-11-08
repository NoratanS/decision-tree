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


type Tree = Node | Leaf


def threshold_branch(number: float, threshold: float) -> Literal["<=", ">"]:
    """The branch key a value takes at a Threshold test."""
    return "<=" if number <= threshold else ">"


def classify(tree: Tree, case: Sequence[str]) -> ClassDistribution:
    """Class Distribution for a Case, as probabilities summing to 1 (spec R7).

    `case` holds the Conditional Attribute values in column order.
    """
    while isinstance(tree, Node):
        value = case[tree.index]
        if tree.threshold is not None:
            tree = tree.branches[threshold_branch(float(value), tree.threshold)]
            continue
        if value not in tree.branches:
            raise ValueError(f"no branch for {tree.attribute} = {value}")
        tree = tree.branches[value]
    total = tree.total
    if total <= 0:
        return {c: float(c == tree.decision) for c in tree.distribution}
    return {c: w / total for c, w in tree.distribution.items()}
