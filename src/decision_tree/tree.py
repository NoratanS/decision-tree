"""Class Distribution, Tests, Node and Leaf types, and classification (spec
R5.6, R7)."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, eq=False)
class ClassDistribution(Mapping[str, float]):
    """Weight (or probability) of each Decision Class, in file order, zero
    weights included.

    Reads as a mapping from class to weight; it equals any mapping with the
    same classes and weights.
    """

    classes: tuple[str, ...]
    weights: tuple[float, ...]
    """One per class, in the same order."""

    def __post_init__(self) -> None:
        if len(self.classes) != len(self.weights):
            raise ValueError(
                f"{len(self.classes)} classes, {len(self.weights)} weights"
            )

    @classmethod
    def count(
        cls, classes: Sequence[str], pairs: Iterable[tuple[str, float]]
    ) -> ClassDistribution:
        """Sum the Case Weight of each `(decision, weight)` pair per class.

        Raises KeyError for a decision not in `classes`.
        """
        weights = dict.fromkeys(classes, 0.0)
        for decision, weight in pairs:
            weights[decision] += weight
        return cls(tuple(weights), tuple(weights.values()))

    @property
    def total(self) -> float:
        return sum(self.weights)

    def majority(self) -> str:
        """The class with the most weight (or probability); the first in file
        order wins ties, so a distribution with no weight gives the first class
        (spec R5.4, R7)."""
        heaviest = max(range(len(self.weights)), key=self.weights.__getitem__)
        return self.classes[heaviest]

    def errors(self, decision: str) -> float:
        """Weight of the Cases not of `decision`."""
        return self.total - self[decision]

    @property
    def is_pure(self) -> bool:
        """Exactly one class has weight."""
        return sum(1 for w in self.weights if w > 0) == 1

    def probabilities(self) -> ClassDistribution:
        """The same classes with weights summing to 1.

        Raises ValueError when there is no weight to share (see
        `Leaf.probabilities` for an empty Leaf).
        """
        total = self.total
        if total <= 0:
            raise ValueError("a Class Distribution with no weight has no probabilities")
        return ClassDistribution(self.classes, tuple(w / total for w in self.weights))

    def __getitem__(self, decision: str) -> float:
        try:
            return self.weights[self.classes.index(decision)]
        except ValueError:
            raise KeyError(decision) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self.classes)

    def __len__(self) -> int:
        return len(self.classes)


@dataclass(frozen=True)
class Leaf:
    decision: str
    distribution: ClassDistribution

    def probabilities(self) -> ClassDistribution:
        """This Leaf's Class Distribution as probabilities; an empty Leaf (no
        training weight) gives `decision` probability 1 (spec R7)."""
        distribution = self.distribution
        if distribution.total <= 0:
            classes = distribution.classes
            return ClassDistribution(
                classes, tuple(float(c == self.decision) for c in classes)
            )
        return distribution.probabilities()


# A Case with a Missing Value for the tested attribute (or, when classifying, a
# discrete value never seen in training) goes down every branch of the Node,
# weighted by each branch's share of the weight (spec R5.5, R7). As in C4.5:
# growth shares the Case's weight by the known weight of each branch
# (`splitting.CandidateTest.branches`, C4.5's `Group` in `build.c`);
# classification shares the branch's Class Distribution by its training weight,
# `child.distribution.total` (`_combine_branches`, C4.5's `Classify` in
# `classify.c`).


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


type Tree = Node | Leaf


def size(tree: Tree) -> int:
    """Number of Nodes and Leaves, as C4.5 reports a tree's size."""
    if isinstance(tree, Leaf):
        return 1
    return 1 + sum(size(child) for child in tree.branches.values())


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
    return tree.probabilities()


def _combine_branches(node: Node, case: Sequence[str | None]) -> ClassDistribution:
    """Classify `case` down every branch of `node`; sum the distributions, each
    weighted by its branch's share of the training weight."""
    total = sum(child.distribution.total for child in node.branches.values())
    combined = [0.0] * len(node.distribution)
    for child in node.branches.values():
        if share := child.distribution.total / total:
            # Every Class Distribution in a tree lists the same classes in
            # the same order.
            for k, probability in enumerate(classify(child, case).weights):
                combined[k] += share * probability
    return ClassDistribution(node.distribution.classes, tuple(combined))
