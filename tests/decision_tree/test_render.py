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
