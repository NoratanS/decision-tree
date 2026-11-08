from pathlib import Path

import pytest

from decision_tree.builder import build_tree
from decision_tree.render import render_plain
from decision_tree.table import load_table
from decision_tree.tree import Leaf, Node, classify

DATA = Path(__file__).parent.parent / "data"
TESTOWA = DATA / "testowaTabDec.txt"
# The reference page's golf table, with its row 7 moved to the top so that
# first-appearance order gives C4.5's branch order (overcast first, true
# before false). C4.5 takes that order from its .names file instead.
GOLF = DATA / "golf.txt"

GOLF_TREE = """\
outlook = overcast: Play (4.0)
outlook = sunny:
|   humidity <= 75 : Play (2.0)
|   humidity > 75 : Don't Play (3.0)
outlook = rain:
|   windy = true: Don't Play (2.0)
|   windy = false: Play (3.0)"""

GIELDA_TREE = """\
a1 = old: down (3.0)
a1 = mid:
|   a2 = yes: down (2.0)
|   a2 = no: up (2.0)
a1 = new: up (3.0)"""


TESTOWA_M1_TREE = """\
a3 = 0:
|   a4 = 1:
|   |   a1 = 0: 0 (1.0)
|   |   a1 = 1: 0 (2.0)
|   |   a1 = 2: 1 (1.0)
|   a4 = 0: 1 (2.0)
a3 = 1:
|   a1 = 0: 1 (1.0)
|   a1 = 1: 2 (0.0)
|   a1 = 2: 2 (2.0)"""

# docs/assignment.md: "-m 2 blocks a1 in two nodes, so the tree differs".
# Empty branches (a1 = 1, a2 = 0) are kept, as spec R4.1 and R5.3 require.
TESTOWA_M2_TREE = """\
a3 = 0:
|   a4 = 1:
|   |   a2 = 1: 0 (2.0)
|   |   a2 = 2: 0 (2.0/1.0)
|   |   a2 = 0: 0 (0.0)
|   a4 = 0: 1 (2.0)
a3 = 1: 2 (3.0/1.0)"""


def tree_from(write_table, text, **options):
    return build_tree(load_table(write_table(text)), **options)


def test_gielda_tree_matches_reference_page():
    tree = build_tree(load_table(DATA / "gielda.txt"))

    assert render_plain(tree) == GIELDA_TREE


def load_golf():
    return load_table(GOLF, header=True, continuous=["temperature", "humidity"])


def test_golf_tree_matches_reference_page():
    tree = build_tree(load_golf())

    assert render_plain(tree) == GOLF_TREE


def test_golf_tree_has_8_nodes_and_no_training_errors():
    # Reference page, "Evaluation on training data": size 8, 0 errors.
    table = load_golf()
    tree = build_tree(table)

    def size(tree):
        if isinstance(tree, Leaf):
            return 1
        return 1 + sum(size(child) for child in tree.branches.values())

    assert size(tree) == 8
    for case, decision in zip(table.cases, table.decisions, strict=True):
        assert classify(tree, case)[decision] == 1.0


def test_continuous_attribute_is_reused_with_another_threshold(write_table):
    # no no yes yes no no: no single Threshold separates the classes, so a1 is
    # tested again below the first Threshold. Thresholds 2 and 4 tie at the root;
    # the lower one wins (spec R4.6).
    table = load_table(write_table("1 n\n2 n\n3 y\n4 y\n5 n\n6 n\n"), continuous=["a1"])

    assert render_plain(build_tree(table)) == (
        "a1 <= 2 : n (2.0)\na1 > 2 :\n|   a1 <= 4 : y (2.0)\n|   a1 > 4 : n (2.0)"
    )


def test_every_node_and_leaf_stores_its_class_distribution():
    tree = build_tree(load_table(DATA / "gielda.txt"))

    assert isinstance(tree, Node)
    assert tree.distribution == {"down": 5.0, "up": 5.0}
    mid = tree.branches["mid"]
    assert isinstance(mid, Node)
    assert mid.distribution == {"down": 2.0, "up": 2.0}
    assert mid.branches["no"] == Leaf("up", {"down": 0.0, "up": 2.0})


def test_pure_table_is_a_single_leaf(write_table):
    tree = tree_from(write_table, "x yes\ny yes\n")

    assert render_plain(tree) == "yes (2.0)"


def test_no_positive_gain_gives_a_leaf_with_first_class_on_tie(write_table):
    tree = tree_from(write_table, "p yes\np no\nq yes\nq no\n")

    assert render_plain(tree) == "yes (4.0/2.0)"


def test_discrete_attribute_is_used_once_per_path(write_table):
    # Under a1 = x no attribute is left, so the impure branch stays a Leaf.
    # Without Missing Values a reused discrete test has zero Gain anyway;
    # ticket 07 (fractional Case Weights) can test reuse directly.
    tree = tree_from(write_table, "x yes\nx no\ny no\n", min_cases=1)

    assert render_plain(tree) == "a1 = x: yes (2.0/1.0)\na1 = y: no (1.0)"


def test_empty_branch_gets_parents_most_common_class(write_table):
    # a2 = r never occurs under a1 = x; the branch still exists (one per value
    # in the Training Set) and takes x's majority, yes.
    tree = tree_from(
        write_table, "x p yes\nx p yes\nx q no\ny p no\ny q no\ny r no\n", min_cases=1
    )

    assert render_plain(tree) == (
        "a1 = x:\n"
        "|   a2 = p: yes (2.0)\n"
        "|   a2 = q: no (1.0)\n"
        "|   a2 = r: yes (0.0)\n"
        "a1 = y: no (3.0)"
    )


def test_min_cases_1_gives_the_course_mode_tree():
    tree = build_tree(load_table(TESTOWA), min_cases=1)

    assert render_plain(tree) == TESTOWA_M1_TREE


def test_default_min_cases_2_blocks_tests_without_two_branches_of_2_cases():
    tree = build_tree(load_table(TESTOWA))

    assert render_plain(tree) == TESTOWA_M2_TREE


def test_min_cases_counts_cases_per_branch(write_table):
    # a1 splits 3 / 3: allowed with -m 3, blocked with -m 4.
    text = "x yes\nx yes\nx yes\ny no\ny no\ny yes\n"

    assert render_plain(tree_from(write_table, text, min_cases=3)) == (
        "a1 = x: yes (3.0)\na1 = y: no (3.0/1.0)"
    )
    assert render_plain(tree_from(write_table, text, min_cases=4)) == "yes (6.0/2.0)"


def test_gain_criterion_picks_the_highest_gain():
    # docs/assignment.md: a1 has the highest Gain (0.78) on testowaTabDec.
    tree = build_tree(load_table(TESTOWA), criterion="gain")

    assert isinstance(tree, Node)
    assert tree.attribute == "a1"


def test_quinlan_gain_filter_drops_tests_below_average_gain():
    # docs/assignment.md, testowaTabDec: Gains 0.78, 0.56, 0.56, 0.41 average
    # 0.58, so only a1 competes on Gain Ratio, though a3 has the best ratio.
    tree = build_tree(load_table(TESTOWA), quinlan_gain_filter=True)

    assert isinstance(tree, Node)
    assert tree.attribute == "a1"


@pytest.mark.parametrize("criterion", ["gain", "gain-ratio"])
def test_tie_between_attributes_goes_to_the_leftmost_column(write_table, criterion):
    # a1 carries no information; a2 and a3 split the cases identically, so
    # they tie under both criteria. Their value names show which one won.
    text = "u q s yes\nv q s yes\nu p t no\nv p t no\n"

    assert render_plain(tree_from(write_table, text, criterion=criterion)) == (
        "a2 = q: yes (2.0)\na2 = p: no (2.0)"
    )


def test_leaf_class_tie_goes_to_the_class_first_in_the_file(write_table):
    # "zebra" sorts last but appears first, so it wins the 1 : 1 tie.
    tree = tree_from(write_table, "p zebra\np apple\n")

    assert render_plain(tree) == "zebra (2.0/1.0)"


def test_quinlan_gain_filter_averages_only_tests_that_pass_min_cases(write_table):
    # a1 has Gain 1.0 but six one-case branches, so -m 2 blocks it. a2 (Gain
    # 0.08) and a3 (Gain 0) remain; their average is 0.04, so a2 competes.
    # Counting a1 in the average (0.36) would leave no test and give a Leaf.
    text = "u p r yes\nv p s yes\nw p r no\nx q t yes\ny q s no\nz q t no\n"

    tree = tree_from(write_table, text, quinlan_gain_filter=True, min_cases=2)

    assert isinstance(tree, Node)
    assert tree.attribute == "a2"


def test_quinlan_gain_filter_keeps_tests_exactly_at_the_average(write_table):
    # a1 and a2 both have Gain 1.0, the average. Both compete; a2 wins on
    # Gain Ratio because it has fewer values.
    text = "x p yes\ny p yes\nz q no\nz q no\n"

    tree = tree_from(write_table, text, quinlan_gain_filter=True, min_cases=1)

    assert isinstance(tree, Node)
    assert tree.attribute == "a2"


def test_mirror_image_tests_tie_and_the_leftmost_wins(write_table):
    # a1 = {p: yes yes no, q: no} and a2 = {r: yes no no, s: yes} are equal up
    # to swapping the classes, so Gain and Gain Ratio are equal in exact
    # arithmetic. Their value names show which one won.
    text = "p s yes\np r yes\np r no\nq r no\n"
    swapped = "s p yes\nr p yes\nr p no\nr q no\n"

    assert render_plain(tree_from(write_table, text, min_cases=1)).startswith("a1 = p:")
    assert render_plain(tree_from(write_table, swapped, min_cases=1)).startswith(
        "a1 = s:"
    )


def test_unknown_criterion_is_rejected():
    with pytest.raises(ValueError, match="criterion"):
        build_tree(load_table(TESTOWA), criterion="entropy")


# Quinlan, "C4.5: Programs for Machine Learning", §2.4: the golf tree when the
# outlook of the case (overcast, 72, 90, true, Play) is unknown. That case goes
# down every outlook branch with weight 5/13, 3/13 and 5/13.
GOLF_MISSING_TREE = """\
outlook = overcast: Play (3.2)
outlook = sunny:
|   humidity <= 75 : Play (2.0)
|   humidity > 75 : Don't Play (3.4/0.4)
outlook = rain:
|   windy = true: Don't Play (2.4/0.4)
|   windy = false: Play (3.0)"""


def test_golf_with_a_missing_outlook_matches_quinlan(write_table):
    text = GOLF.read_text().replace("overcast, 72, 90", "?, 72, 90")
    table = load_table(
        write_table(text), header=True, continuous=["temperature", "humidity"]
    )

    tree = build_tree(table)

    assert render_plain(tree) == GOLF_MISSING_TREE
