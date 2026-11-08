from pathlib import Path

from decision_tree.builder import build_tree
from decision_tree.render import render_plain
from decision_tree.table import load_table
from decision_tree.tree import Leaf, Node

DATA = Path(__file__).parent.parent / "data"

GIELDA_TREE = """\
a1 = old: down (3.0)
a1 = mid:
|   a2 = yes: down (2.0)
|   a2 = no: up (2.0)
a1 = new: up (3.0)"""


def tree_from(tmp_path, text):
    path = tmp_path / "table.txt"
    path.write_text(text, encoding="utf-8")
    return build_tree(load_table(path))


def test_gielda_tree_matches_reference_page():
    tree = build_tree(load_table(DATA / "gielda.txt"))

    assert render_plain(tree) == GIELDA_TREE


def test_every_node_and_leaf_stores_its_class_distribution():
    tree = build_tree(load_table(DATA / "gielda.txt"))

    assert isinstance(tree, Node)
    assert tree.distribution == {"down": 5.0, "up": 5.0}
    mid = tree.branches["mid"]
    assert isinstance(mid, Node)
    assert mid.distribution == {"down": 2.0, "up": 2.0}
    assert mid.branches["no"] == Leaf("up", {"down": 0.0, "up": 2.0})


def test_pure_table_is_a_single_leaf(tmp_path):
    tree = tree_from(tmp_path, "x yes\ny yes\n")

    assert render_plain(tree) == "yes (2.0)"


def test_no_positive_gain_gives_a_leaf_with_first_class_on_tie(tmp_path):
    tree = tree_from(tmp_path, "p yes\np no\nq yes\nq no\n")

    assert render_plain(tree) == "yes (4.0/2.0)"


def test_discrete_attribute_is_used_once_per_path(tmp_path):
    # Under a1 = x no attribute is left, so the impure branch stays a Leaf.
    # Without Missing Values a reused discrete test has zero Gain anyway;
    # ticket 07 (fractional Case Weights) can test reuse directly.
    tree = tree_from(tmp_path, "x yes\nx no\ny no\n")

    assert render_plain(tree) == "a1 = x: yes (2.0/1.0)\na1 = y: no (1.0)"


def test_empty_branch_gets_parents_most_common_class(tmp_path):
    # a2 = r never occurs under a1 = x; the branch still exists (one per value
    # in the Training Set) and takes x's majority, yes.
    tree = tree_from(tmp_path, "x p yes\nx p yes\nx q no\ny p no\ny q no\ny r no\n")

    assert render_plain(tree) == (
        "a1 = x:\n"
        "|   a2 = p: yes (2.0)\n"
        "|   a2 = q: no (1.0)\n"
        "|   a2 = r: yes (0.0)\n"
        "a1 = y: no (3.0)"
    )
