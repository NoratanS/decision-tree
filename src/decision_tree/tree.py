"""Node and Leaf types (spec R5.6)."""

from collections.abc import Sequence
from dataclasses import dataclass

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
    """One subtree per value, in order of first appearance in the Training Set."""
    distribution: ClassDistribution


type Tree = Node | Leaf


def classify(tree: Tree, case: Sequence[str]) -> ClassDistribution:
    """Class Distribution for a Case, as probabilities summing to 1 (spec R7).

    `case` holds the Conditional Attribute values in column order.
    """
    while isinstance(tree, Node):
        value = case[tree.index]
        if value not in tree.branches:
            raise ValueError(f"no branch for {tree.attribute} = {value}")
        tree = tree.branches[value]
    total = tree.total
    if total <= 0:
        return {c: float(c == tree.decision) for c in tree.distribution}
    return {c: w / total for c, w in tree.distribution.items()}
