from pathlib import Path

import pytest

from decision_tree.builder import build_tree
from decision_tree.table import load_table
from decision_tree.tree import Leaf, Node, classify

DATA = Path(__file__).parent.parent / "data"


@pytest.mark.parametrize("case", range(10))
def test_each_gielda_case_gets_its_training_decision_with_probability_one(case):
    table = load_table(DATA / "gielda.txt")
    tree = build_tree(table)

    distribution = classify(tree, table.cases[case])

    assert distribution[table.decisions[case]] == 1.0
    assert sum(distribution.values()) == 1.0


def test_impure_leaf_gives_its_class_proportions():
    tree = Node(
        "a1", 0, {"x": Leaf("yes", {"yes": 3.0, "no": 1.0})}, {"yes": 3.0, "no": 1.0}
    )

    assert classify(tree, ("x",)) == {"yes": 0.75, "no": 0.25}


def test_empty_leaf_gives_its_class_with_probability_one():
    tree = Leaf("yes", {"yes": 0.0, "no": 0.0})

    assert classify(tree, ()) == {"yes": 1.0, "no": 0.0}


# Interim: ticket 08 replaces this with "unseen value is treated as missing" (R7).
def test_value_not_in_the_tree_is_an_error():
    tree = Node("a1", 0, {"x": Leaf("yes", {"yes": 1.0})}, {"yes": 1.0})

    with pytest.raises(ValueError, match="a1 = z"):
        classify(tree, ("z",))
