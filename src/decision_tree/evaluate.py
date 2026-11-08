"""Error count, error rate and confusion matrix of a tree on a table (spec R8)."""

import random
from dataclasses import dataclass

from decision_tree.table import DecisionTable
from decision_tree.tree import Tree, classify, majority


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
    the Training Set's file order (spec R7). Raises ValueError naming the
    Case (1-based) that the tree cannot classify.
    """
    classes = tuple(dict.fromkeys([*tree.distribution, *table.classes]))
    confusion = {actual: dict.fromkeys(classes, 0) for actual in classes}
    labelled = zip(table.cases, table.decisions, strict=True)
    for number, (case, actual) in enumerate(labelled, start=1):
        try:
            distribution = classify(tree, case)
        except ValueError as error:
            raise ValueError(f"case {number}: {error}") from None
        confusion[actual][majority(distribution)] += 1
    return Evaluation(classes, confusion)


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
