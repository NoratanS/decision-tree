from pathlib import Path

import pytest

from decision_tree.builder import build_tree
from decision_tree.table import load_table
from decision_tree.tree import Leaf, Node, classify

DATA = Path(__file__).parent.parent / "data"


@pytest.mark.parametrize("case", range(10))
def test_each_stock_market_case_gets_its_training_decision_with_probability_one(case):
    table = load_table(DATA / "stock_market.txt")
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


def test_golf_outlook_never_seen_in_training_is_classified_as_missing():
    table = load_table(
        DATA / "golf.txt", header=True, continuous=["temperature", "humidity"]
    )
    tree = build_tree(table)

    foggy = classify(tree, ("foggy", "72", "90", "true"))

    assert foggy == classify(tree, (None, "72", "90", "true"))
    # overcast 4/14 Play, sunny 5/14 Don't Play (humidity 90 > 75), rain 5/14
    # Don't Play (windy).
    assert foggy == {
        "Play": pytest.approx(4 / 14),
        "Don't Play": pytest.approx(10 / 14),
    }


def humidity_node():
    return Node(
        "humidity",
        2,
        {
            "<=": Leaf("Play", {"Play": 2.0, "Don't Play": 0.0}),
            ">": Leaf("Don't Play", {"Play": 0.0, "Don't Play": 3.0}),
        },
        {"Play": 2.0, "Don't Play": 3.0},
        threshold=75.0,
    )


@pytest.mark.parametrize(
    ("humidity", "expected"),
    [("70", "Play"), ("75", "Play"), ("75.5", "Don't Play"), ("1e2", "Don't Play")],
)
def test_threshold_node_sends_values_up_to_the_threshold_left(humidity, expected):
    distribution = classify(humidity_node(), ("sunny", "72", humidity, "false"))

    assert distribution[expected] == 1.0


def test_golf_sunny_case_with_missing_humidity_matches_reference_page():
    # Reference page: under sunny, 2 of 5 cases have humidity <= 75 (Play), so
    # the case is Play with 0.4 and Don't Play with 0.6.
    table = load_table(
        DATA / "golf.txt", header=True, continuous=["temperature", "humidity"]
    )
    tree = build_tree(table)

    distribution = classify(tree, ("sunny", "72", None, "false"))

    assert distribution == {
        "Play": pytest.approx(0.4),
        "Don't Play": pytest.approx(0.6),
    }


def test_missing_value_combines_every_branch_weighted_by_its_size():
    tree = Node(
        "a1",
        0,
        {
            "x": Leaf("yes", {"yes": 3.0, "no": 1.0}),
            "y": Leaf("no", {"yes": 0.0, "no": 4.0}),
            "z": Leaf("yes", {"yes": 0.0, "no": 0.0}),
        },
        {"yes": 3.0, "no": 5.0},
    )

    # x: 4/8 * (0.75, 0.25), y: 4/8 * (0, 1), z: empty, no weight.
    assert classify(tree, (None,)) == {"yes": 0.375, "no": 0.625}
