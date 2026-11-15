"""Error count, error rate and confusion matrix of a tree on a table (spec R8),
and the classified Cases of the `classify` command (spec R10)."""

from dataclasses import dataclass
from functools import cached_property

from decision_tree.table import Cases, DecisionTable
from decision_tree.tree import ClassDistribution, Tree, classify


@dataclass(frozen=True)
class Evaluation:
    classes: tuple[str, ...]
    """Decision Classes in the matrix: the tree's, then any only in the table, in
    order of first appearance."""
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

    @property
    def correct(self) -> int:
        """Cases whose predicted class is their actual class."""
        return self.cases - self.errors

    @property
    def accuracy(self) -> float:
        """`1 - error_rate`; 0 for an empty table."""
        return 1 - self.error_rate if self.cases else 0.0


def evaluate(tree: Tree, table: DecisionTable) -> Evaluation:
    """Classify every Case of `table` and compare with its decision.

    The predicted class is the most probable one; on a tie, the one first in
    the Training Set's file order (spec R7).
    """
    evaluation = classify_cases(tree, Cases(table.cases, table.decisions)).evaluation
    assert evaluation is not None  # a DecisionTable always has decisions
    return evaluation


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
        return tuple(d.majority() for d in self.distributions)

    @property
    def rows(self) -> zip[tuple[ClassDistribution, str, str | None]]:
        """(distribution, predicted, decision or None) of each Case."""
        decisions = self.decisions or (None,) * len(self.distributions)
        return zip(self.distributions, self.predicted, decisions, strict=True)

    @cached_property
    def evaluation(self) -> Evaluation | None:
        """Confusion matrix of the predictions against the decisions; None
        without decisions.

        Its classes are the tree's, then any decision the tree does not know,
        in order of first appearance.
        """
        if self.decisions is None:
            return None
        classes = tuple(dict.fromkeys([*self.classes, *self.decisions]))
        confusion = {actual: dict.fromkeys(classes, 0) for actual in classes}
        for predicted, actual in zip(self.predicted, self.decisions, strict=True):
            confusion[actual][predicted] += 1
        return Evaluation(classes, confusion)


def classify_cases(tree: Tree, cases: Cases) -> Classification:
    """Classify every Case with `tree`."""
    return Classification(
        classes=tree.distribution.classes,
        distributions=tuple(classify(tree, case) for case in cases.cases),
        decisions=cases.decisions,
    )
