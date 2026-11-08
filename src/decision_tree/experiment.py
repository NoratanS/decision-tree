"""One Experiment: build a tree, optionally prune it, and evaluate each tree on
the Training Set and an optional Test Set (spec R8).

Takes loaded tables only and does no I/O; the CLI loads files and renders the
result.
"""

import random
from dataclasses import dataclass

from decision_tree.builder import Criterion, build_tree
from decision_tree.evaluate import Evaluation, evaluate
from decision_tree.pruning import DEFAULT_CONFIDENCE, prune
from decision_tree.table import DecisionTable
from decision_tree.tree import Tree


class SplitError(ValueError):
    """A hold-out split would leave the Training Set or the Test Set empty."""


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


type TestSource = FileSource | HoldOut | None
"""Where the Test Set comes from; None for no Test Set."""


@dataclass(frozen=True)
class TrainingSet:
    """The Training Set the trees were built from."""


@dataclass(frozen=True)
class FileTestSet:
    """A Test Set read from `file`."""

    file: str


@dataclass(frozen=True)
class HoldOutTestSet:
    """The held-out part of a split made with `seed`."""

    seed: int


type EvaluatedSet = TrainingSet | FileTestSet | HoldOutTestSet


@dataclass(frozen=True)
class SetEvaluation:
    """Both trees evaluated on one set."""

    evaluated_set: EvaluatedSet
    unpruned: Evaluation
    pruned: Evaluation | None
    """None when the Experiment does not prune."""


@dataclass(frozen=True)
class ExperimentResult:
    unpruned: Tree
    pruned: Tree | None
    """None when the Experiment does not prune."""
    confidence: float | None
    """The Confidence Level `pruned` was pruned at, a fraction; None without
    pruning."""
    seed: int | None
    """The seed of the hold-out split; None without a split."""
    evaluations: tuple[SetEvaluation, ...]
    """The Training Set first, then the Test Set if there is one."""


def run_experiment(
    training: DecisionTable,
    test: TestSource = None,
    *,
    criterion: Criterion = "gain-ratio",
    quinlan_gain_filter: bool = False,
    min_cases: float = 2,
    confidence: float | None = DEFAULT_CONFIDENCE,
) -> ExperimentResult:
    """Build a tree from `training`, prune it at `confidence` (a fraction in
    (0, 1], or None for no pruning) and evaluate each tree on each set.

    With a `HoldOut`, the tree is built from the training part of the split and
    evaluated on that part and the held-out one. Raises SplitError if the split
    would leave a part empty.
    """
    seed: int | None = None
    tests: list[tuple[EvaluatedSet, DecisionTable]] = []
    match test:
        case FileSource(file, table):
            tests.append((FileTestSet(file), table))
        case HoldOut(ratio, given_seed):
            seed = random.randrange(2**32) if given_seed is None else given_seed
            training, held_out = hold_out(training, ratio, seed=seed)
            tests.append((HoldOutTestSet(seed), held_out))
    sets = [(TrainingSet(), training), *tests]

    unpruned = build_tree(
        training,
        criterion=criterion,
        quinlan_gain_filter=quinlan_gain_filter,
        min_cases=min_cases,
    )
    pruned = None if confidence is None else prune(unpruned, confidence)
    evaluations = tuple(
        SetEvaluation(
            evaluated_set,
            evaluate(unpruned, table),
            None if pruned is None else evaluate(pruned, table),
        )
        for evaluated_set, table in sets
    )
    return ExperimentResult(unpruned, pruned, confidence, seed, evaluations)


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
