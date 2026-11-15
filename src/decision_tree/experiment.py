"""One Experiment: build a tree, optionally prune it, and evaluate each tree on
the Training Set and an optional Test Set (spec R8).

Takes loaded tables only and does no I/O; the CLI loads files and renders the
result.
"""

import random
from dataclasses import dataclass

from decision_tree.builder import DEFAULT_BUILD_OPTIONS, BuildOptions, build_tree
from decision_tree.evaluate import Evaluation, evaluate
from decision_tree.pruning import DEFAULT_CONFIDENCE, prune
from decision_tree.table import DecisionTable
from decision_tree.tree import Tree


class SplitError(ValueError):
    """A hold-out ratio is outside (0, 1), or the split would leave the Training
    Set or the Test Set empty."""


@dataclass(frozen=True)
class FileSource:
    """A Test Set loaded from a file."""

    file: str
    """The file name, as the user gave it."""
    table: DecisionTable


@dataclass(frozen=True)
class HoldOut:
    """A random hold-out split of the Training Set (see `hold_out`)."""

    ratio: float
    """Share of the Cases kept for training, between 0 and 1."""
    seed: int | None = None
    """Seed of the shuffle; None picks one at random."""

    def __post_init__(self) -> None:
        if not 0 < self.ratio < 1:
            raise SplitError(f"the ratio must be between 0 and 1, not {self.ratio:g}")


type TestSource = FileSource | HoldOut | None
"""Where the Test Set comes from; None for no Test Set."""


@dataclass(frozen=True)
class TrainingSet:
    """The Training Set the trees were built from."""


type EvaluatedSet = TrainingSet | FileSource | HoldOut
"""A set a tree was evaluated on. A `HoldOut` here always has its seed."""


@dataclass(frozen=True)
class EvaluatedTree:
    """A tree and its evaluation on each set."""

    tree: Tree
    evaluations: tuple[tuple[EvaluatedSet, Evaluation], ...]
    """The Training Set first, then the Test Set if there is one."""


@dataclass(frozen=True)
class PrunedTree(EvaluatedTree):
    """An evaluated pruned tree and the Confidence Level it was pruned at."""

    confidence: float
    """The Confidence Level, a fraction in (0, 1]."""


@dataclass(frozen=True)
class ExperimentResult:
    unpruned: EvaluatedTree
    pruned: PrunedTree | None
    """None when the Experiment does not prune."""


def run_experiment(
    training: DecisionTable,
    test: TestSource = None,
    options: BuildOptions = DEFAULT_BUILD_OPTIONS,
    *,
    confidence: float | None = DEFAULT_CONFIDENCE,
) -> ExperimentResult:
    """Build a tree from `training` with `options`, prune it at `confidence` (a
    fraction in (0, 1], or None for no pruning) and evaluate each tree on each
    set.

    With a `HoldOut`, the tree is built from the training part of the split and
    evaluated on that part and the held-out one, labelled with the seed used.
    Raises SplitError if the split would leave a part empty.
    """
    tests: list[tuple[EvaluatedSet, DecisionTable]] = []
    match test:
        case FileSource(_, table):
            tests.append((test, table))
        case HoldOut(ratio, given_seed):
            seed = random.randrange(2**32) if given_seed is None else given_seed
            training, held_out = hold_out(training, ratio, seed=seed)
            tests.append((HoldOut(ratio, seed), held_out))
    sets = [(TrainingSet(), training), *tests]

    def evaluations(tree: Tree) -> tuple[tuple[EvaluatedSet, Evaluation], ...]:
        return tuple((label, evaluate(tree, table)) for label, table in sets)

    unpruned = build_tree(training, options)
    pruned = None
    if confidence is not None:
        tree = prune(unpruned, confidence)
        pruned = PrunedTree(tree, evaluations(tree), confidence)
    return ExperimentResult(EvaluatedTree(unpruned, evaluations(unpruned)), pruned)


def hold_out(
    table: DecisionTable, ratio: float, *, seed: int
) -> tuple[DecisionTable, DecisionTable]:
    """Split `table` at random into a Training Set and a Test Set.

    `ratio` of the Cases (rounded) go to the Training Set, the rest to the Test
    Set. The same `seed` gives the same split. Both parts keep the file order,
    names and attribute types. Raises SplitError if a part would be empty.
    """
    count = len(table.cases)
    size = round(ratio * count)
    if not 0 < size < count:
        empty = "Training Set" if size <= 0 else "Test Set"
        raise SplitError(f"a {ratio:g} split of {count} cases leaves no {empty}")
    chosen = set(random.Random(seed).sample(range(count), size))
    training = [i for i in range(count) if i in chosen]
    test = [i for i in range(count) if i not in chosen]
    return table.subset(training), table.subset(test)
