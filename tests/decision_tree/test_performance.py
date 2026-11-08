"""The PRD's size target: a 10,000 × 50 Decision Table trains and prunes in
under 30 seconds (spec, "Testing"). Skip with `pytest -m "not slow"`."""

import random
import time

import pytest

from decision_tree.builder import build_tree
from decision_tree.pruning import prune
from decision_tree.table import load_table

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
