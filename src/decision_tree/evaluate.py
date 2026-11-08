"""Error count, error rate and confusion matrix of a tree on a table (spec R8),
and the classified Cases of the `classify` command (spec R10)."""

import random
from dataclasses import dataclass
from functools import cached_property

from decision_tree.table import Cases, DecisionTable
from decision_tree.tree import ClassDistribution, Tree, classify, majority


@dataclass(frozen=True)
class Evaluation:
    classes: tuple[str, ...]
    """Decision Classes in the matrix: the tree's, then any only in the table."""
    confusion: dict[str, dict[str, int]]
    """Actual class → predicted class → number of Cases."""

    @property
    def cases(self) -> int:
        return sum(sum(row.values()) for row in self.confusion.values())

    @property
    def errors(self) -> int:
        """Cases whose predicted class is not their actual class."""
        return self.cases - sum(self.confusion[c][c] for c in self.classes)

    @property
    def error_rate(self) -> float:
        """`errors / cases`; 0 for an empty table."""
        return self.errors / self.cases if self.cases else 0.0


def evaluate(tree: Tree, table: DecisionTable) -> Evaluation:
    """Classify every Case of `table` and compare with its decision.

    The predicted class is the most probable one; on a tie, the one first in
    the Training Set's file order (spec R7).
    """
    classes = tuple(dict.fromkeys([*tree.distribution, *table.classes]))
    confusion = {actual: dict.fromkeys(classes, 0) for actual in classes}
    result = classify_cases(tree, Cases(table.cases, table.decisions))
    for actual, predicted in zip(table.decisions, result.predicted, strict=True):
        confusion[actual][predicted] += 1
    return Evaluation(classes, confusion)


@dataclass(frozen=True)
class Classification:
    classes: tuple[str, ...]
    """The tree's Decision Classes, in file order."""
    distributions: tuple[ClassDistribution, ...]
    """Class Distribution of each Case."""
    decisions: tuple[str, ...] | None
    """Decision Class of each Case from the file; None without decisions."""

    @cached_property
    def predicted(self) -> tuple[str, ...]:
        """Most probable class of each Case; ties go to the class first in file
        order (spec R7)."""
        return tuple(map(majority, self.distributions))

    @property
    def rows(self) -> zip[tuple[ClassDistribution, str, str | None]]:
        """(distribution, predicted, decision or None) of each Case."""
        decisions = self.decisions or (None,) * len(self.distributions)
        return zip(self.distributions, self.predicted, decisions, strict=True)

    @property
    def correct(self) -> int:
        """Cases predicted as their decision; 0 without decisions."""
        if self.decisions is None:
            return 0
        pairs = zip(self.predicted, self.decisions, strict=True)
        return sum(predicted == actual for predicted, actual in pairs)


def classify_cases(tree: Tree, cases: Cases) -> Classification:
    """Classify every Case with `tree`."""
    return Classification(
        classes=tuple(tree.distribution),
        distributions=tuple(classify(tree, case) for case in cases.cases),
        decisions=cases.decisions,
    )


def hold_out(
    table: DecisionTable, ratio: float, *, seed: int
) -> tuple[DecisionTable, DecisionTable]:
    """Split `table` at random into a Training Set and a Test Set.

    `ratio` of the Cases (rounded) go to the Training Set, the rest to the Test
    Set. The same `seed` gives the same split. Both parts keep the file order,
    names and attribute types. Raises ValueError if a part would be empty.
    """
    count = len(table.cases)
    size = round(ratio * count)
    if not 0 < size < count:
        empty = "Training Set" if size <= 0 else "Test Set"
        raise ValueError(f"a {ratio:g} split of {count} cases leaves no {empty}")
    chosen = set(random.Random(seed).sample(range(count), size))
    training = [i for i in range(count) if i in chosen]
    test = [i for i in range(count) if i not in chosen]
    return table.subset(training), table.subset(test)
