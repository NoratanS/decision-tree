from pathlib import Path

import pytest

from decision_tree import BuildOptions, ReadOptions, build_tree, evaluate, load_table
from decision_tree.pruning import estimated_errors, prune
from decision_tree.render import render_plain
from decision_tree.tree import Leaf, Node, size

DATA = Path(__file__).parent.parent / "data"

# Quinlan, "C4.5: Programs for Machine Learning", §4.2: the upper limit of the
# error rate, U_CF(E, N), at the default Confidence Level of 25%.
QUINLAN_RATES = [(6, 0, 0.206), (9, 0, 0.143), (1, 0, 0.750), (16, 1, 0.157)]


@pytest.mark.parametrize(("cases", "errors", "rate"), QUINLAN_RATES)
def test_estimate_matches_quinlans_book(cases, errors, rate):
    estimate = estimated_errors(cases, errors, confidence=0.25)

    assert estimate / cases == pytest.approx(rate, abs=5e-4)


def test_estimate_between_0_and_1_error_is_interpolated():
    # Case Weights give fractional errors. By hand at 25%: 4×(1 − 0.25^¼) =
    # 1.1716 added at 0 errors, 1.1896 at 1 error; halfway, plus the 0.5 error.
    assert estimated_errors(4, 0.5, confidence=0.25) == pytest.approx(
        0.5 + 1.1806, abs=1e-3
    )


def test_estimate_of_a_leaf_that_is_almost_all_errors():
    # C4.5 adds 0.67 × the correct weight when errors + 0.5 >= cases.
    assert estimated_errors(2, 1.5, confidence=0.25) == pytest.approx(1.835)


def test_empty_leaf_has_no_estimated_errors():
    assert estimated_errors(0, 0, confidence=0.25) == 0.0


def _book_subtree():
    # Quinlan §4.2: a test with leaves of 6, 9 and 1 cases and no errors.
    # Subtree estimate 6×0.206 + 9×0.143 + 1×0.750 = 3.273; as a Leaf of 16
    # cases with 1 error, 16×0.157 = 2.512, so the subtree is replaced.
    return Node(
        attribute="education spending",
        index=0,
        branches={
            "n": Leaf("democrat", {"democrat": 6.0, "republican": 0.0}),
            "u": Leaf("democrat", {"democrat": 9.0, "republican": 0.0}),
            "y": Leaf("republican", {"democrat": 0.0, "republican": 1.0}),
        },
        distribution={"democrat": 15.0, "republican": 1.0},
    )


def test_subtree_with_a_higher_estimate_becomes_a_leaf():
    pruned = prune(_book_subtree())

    assert pruned == Leaf("democrat", {"democrat": 15.0, "republican": 1.0})


def test_prune_keeps_the_unpruned_tree_intact():
    tree = _book_subtree()

    prune(tree)

    assert tree == _book_subtree()


def load_golf():
    path = DATA / "golf.txt"
    return load_table(
        path, ReadOptions(header=True), continuous=["temperature", "humidity"]
    )


def course_mode_test_table():
    return build_tree(load_table(DATA / "test_table.txt"), BuildOptions(min_cases=1))


def test_golf_keeps_8_nodes_and_0_errors_after_pruning():
    # Reference page: size 8 and 0 errors both before and after pruning.
    table = load_golf()
    tree = build_tree(table)

    pruned = prune(tree)

    assert pruned == tree
    assert size(pruned) == 8
    assert evaluate(pruned, table).errors == 0


# test_table.txt in course mode (-m 1), pruned at 25%. By hand: the a4 = 1
# subtree estimates 0.75 + 1.0 + 0.75 = 2.5 errors, a Leaf of 4 cases with
# 1 error 2.19, so it becomes a Leaf. At 50% the subtree's 1.59 beats 1.71.
TEST_TABLE_M1_PRUNED = """\
a3 = 0:
|   a4 = 1: 0 (4.0/1.0)
|   a4 = 0: 1 (2.0)
a3 = 1:
|   a1 = 0: 1 (1.0)
|   a1 = 1: 2 (0.0)
|   a1 = 2: 2 (2.0)"""


def test_course_mode_tree_prunes_its_a4_subtree_at_25_percent():
    assert render_plain(prune(course_mode_test_table(), 0.25)) == TEST_TABLE_M1_PRUNED


@pytest.mark.parametrize(
    ("grow", "sizes"),
    [
        (course_mode_test_table, {0.5: 11, 0.25: 8}),
        (lambda: build_tree(load_golf()), {0.25: 8, 0.05: 1}),
    ],
    ids=["test_table -m 1", "golf"],
)
def test_lower_confidence_level_prunes_more(grow, sizes):
    tree = grow()

    assert {level: size(prune(tree, level)) for level in sizes} == sizes


def test_leaves_of_one_class_merge_even_at_100_percent():
    # Default -m 2: both a2 leaves predict 0, so the Leaf makes the same one
    # error and adds no estimate at 100%.
    tree = build_tree(load_table(DATA / "test_table.txt"))

    pruned = prune(tree, 1.0)

    assert render_plain(pruned) == (
        "a3 = 0:\n|   a4 = 1: 0 (4.0/1.0)\n|   a4 = 0: 1 (2.0)\na3 = 1: 2 (3.0/1.0)"
    )


@pytest.mark.parametrize("confidence", [0.0, -0.1, 1.5])
@pytest.mark.parametrize("tree", [_book_subtree(), Leaf("no", {"yes": 0.0, "no": 1.0})])
def test_confidence_level_outside_0_1_is_rejected(tree, confidence):
    with pytest.raises(ValueError, match="confidence"):
        prune(tree, confidence)


@pytest.mark.parametrize("confidence", [0.0, 1.5])
def test_estimate_rejects_a_confidence_level_outside_0_1(confidence):
    with pytest.raises(ValueError, match="confidence"):
        estimated_errors(4, 1, confidence=confidence)
