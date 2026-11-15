"""The PRD's size target: a 10,000 × 50 Decision Table trains and prunes in
under 30 seconds (spec, "Testing"). Skip with `pytest -m "not slow"`."""

import random
import time

import pytest

from decision_tree.builder import BuildOptions, build_tree
from decision_tree.evaluate import evaluate
from decision_tree.pruning import prune
from decision_tree.table import load_table
from decision_tree.tree import Leaf, Tree

CLASSES = ("A", "B", "C")


def generate_table(
    n_cases: int = 10_000,
    n_attributes: int = 50,
    *,
    seed: int = 0,
    missing: float = 0.05,
    noise: float = 0.1,
) -> str:
    """A seeded Decision Table as text, one Case per line.

    Even Attributes are discrete (2 to 6 values), odd ones continuous (two
    decimals in [0, 100]). The decision follows a rule on the first, second
    and fourth Attributes, with a `noise` share of random labels; each cell is
    a Missing Value (`?`) with probability `missing`.
    """
    rng = random.Random(seed)
    lines = []
    for _ in range(n_cases):
        values = [
            f"v{rng.randrange(2 + a % 5)}"
            if a % 2 == 0
            else f"{rng.uniform(0, 100):.2f}"
            for a in range(n_attributes)
        ]
        if rng.random() < noise:
            decision = rng.choice(CLASSES)
        elif values[0] == "v0":
            decision = "A" if float(values[1]) < 50 else "B"
        else:
            decision = "C" if float(values[3]) > 30 else "A"
        values = ["?" if rng.random() < missing else value for value in values]
        lines.append(" ".join([*values, decision]))
    return "\n".join(lines) + "\n"


def test_generated_table_has_the_target_shape(write_table):
    table = load_table(write_table(generate_table()))

    assert len(table.cases) == 10_000
    assert len(table.attributes) == 50
    assert table.kinds.count("continuous") == 25
    assert any(value is None for case in table.cases for value in case)


@pytest.mark.slow
def test_10000_by_50_table_trains_and_prunes_in_under_30_seconds(write_table):
    path = write_table(generate_table())

    # CPU time of this process, so other programs competing for the CPU do
    # not count; it matches wall-clock time on an idle machine.
    start = time.process_time()
    prune(build_tree(load_table(path)))
    elapsed = time.process_time() - start

    assert elapsed < 30, f"load + train + prune took {elapsed:.1f} s of CPU time"


def many_valued_table(n_cases: int = 20_000, *, seed: int = 0, first: int = 0) -> str:
    """A seeded Decision Table whose first column holds the distinct whole
    numbers `first`, `first + 1`, … in random order, like adult's `fnlwgt`
    (ticket 36).

    Four discrete Attributes (2 to 5 values) follow; the decision follows a
    rule on the first two, with a fifth of random labels. Once a path has used
    them up, only the first column splits the noise left at a Node.
    """
    rng = random.Random(seed)
    numbers = list(range(first, first + n_cases))
    rng.shuffle(numbers)
    lines = []
    for number in numbers:
        values = [f"v{rng.randrange(2 + a)}" for a in range(4)]
        if rng.random() < 0.2:
            decision = rng.choice(CLASSES)
        else:
            decision = "A" if values[0] == "v0" else "B" if values[1] == "v0" else "C"
        lines.append(" ".join([str(number), *values, decision]))
    return "\n".join(lines) + "\n"


def _tested_attributes(tree: Tree) -> set[str]:
    if isinstance(tree, Leaf):
        return set()
    below = (_tested_attributes(child) for child in tree.branches.values())
    return {tree.test.attribute}.union(*below)


@pytest.mark.slow
def test_20000_valued_discrete_attribute_trains_and_evaluates_in_under_60_seconds(
    write_table,
):
    train_path = write_table(many_valued_table(), "train.txt")
    # Every Test Set value of a1 is unseen, so each Case at a Node testing a1
    # goes down all its branches (spec R7). With 20,000 values in 20,000 Cases
    # each branch holds one Case, which only `-m 1` allows.
    test_path = write_table(many_valued_table(seed=1, first=20_000), "test.txt")

    start = time.process_time()
    train = load_table(train_path, discrete=["a1"])
    test = load_table(test_path, like=train)
    unpruned = build_tree(train, BuildOptions(min_cases=1))
    pruned = prune(unpruned)
    evaluate(unpruned, test)
    evaluate(pruned, test)
    elapsed = time.process_time() - start

    assert len(train.values(0)) == 20_000
    assert "a1" in _tested_attributes(unpruned)
    assert elapsed < 60, f"load + train + prune + evaluate took {elapsed:.1f} s"
