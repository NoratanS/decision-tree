import pytest

from decision_tree.render import render_plain
from decision_tree.tree import Leaf, Node


def test_leaf_without_errors_shows_its_weight():
    assert render_plain(Leaf("up", {"down": 0.0, "up": 3.0})) == "up (3.0)"


def test_leaf_with_errors_shows_weight_and_errors():
    assert render_plain(Leaf("bear", {"bear": 3.0, "bare": 2.0})) == "bear (5.0/2.0)"


def test_fractional_weights_have_one_decimal():
    assert render_plain(Leaf("yes", {"yes": 2.0, "no": 0.375})) == "yes (2.4/0.4)"


def test_nested_nodes_are_indented_with_bars():
    inner = Node(
        "a2",
        1,
        {
            "p": Leaf("yes", {"yes": 2.0, "no": 0.0}),
            "q": Leaf("no", {"yes": 0.0, "no": 1.0}),
        },
        {"yes": 2.0, "no": 1.0},
    )
    tree = Node(
        "a1",
        0,
        {"x": inner, "y": Leaf("no", {"yes": 0.0, "no": 3.0})},
        {"yes": 2.0, "no": 4.0},
    )

    assert render_plain(tree) == (
        "a1 = x:\n|   a2 = p: yes (2.0)\n|   a2 = q: no (1.0)\na1 = y: no (3.0)"
    )


def test_threshold_tests_print_like_c45():
    tree = Node(
        "humidity",
        2,
        {
            "<=": Leaf("Play", {"Play": 2.0, "Don't Play": 0.0}),
            ">": Leaf("Don't Play", {"Play": 0.0, "Don't Play": 3.0}),
        },
        {"Play": 2.0, "Don't Play": 3.0},
        threshold=75.0,
    )

    assert render_plain(tree) == (
        "humidity <= 75 : Play (2.0)\nhumidity > 75 : Don't Play (3.0)"
    )


@pytest.mark.parametrize(
    ("threshold", "text"),
    [(75.0, "75"), (0.25, "0.25"), (-3.0, "-3"), (1e-05, "1e-05")],
)
def test_thresholds_print_without_a_needless_decimal_point(threshold, text):
    tree = Node(
        "a1",
        0,
        {"<=": Leaf("y", {"y": 1.0}), ">": Leaf("y", {"y": 1.0})},
        {"y": 2.0},
        threshold=threshold,
    )

    assert render_plain(tree).startswith(f"a1 <= {text} : ")
