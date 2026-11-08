from pathlib import Path

import pytest

from decision_tree.evaluate import classify_cases, evaluate, hold_out
from decision_tree.table import Cases, DecisionTable, load_table
from decision_tree.tree import Leaf, Node

DATA = Path(__file__).parent.parent / "data"

# a1 = x → yes, a1 = y → no
TREE = Node(
    "a1",
    0,
    {
        "x": Leaf("yes", {"yes": 2.0, "no": 0.0}),
        "y": Leaf("no", {"yes": 0.0, "no": 2.0}),
    },
    {"yes": 2.0, "no": 2.0},
)


def table(*rows: tuple[str, str]) -> DecisionTable:
    return DecisionTable(
        attributes=("a1",),
        decision="d",
        cases=tuple((value,) for value, _ in rows),
        decisions=tuple(decision for _, decision in rows),
        kinds=("discrete",),
    )


def test_counts_errors_and_fills_the_confusion_matrix():
    result = evaluate(
        TREE, table(("x", "yes"), ("x", "no"), ("y", "no"), ("y", "no"), ("x", "yes"))
    )

    assert result.cases == 5
    assert result.errors == 1
    assert result.error_rate == 0.2
    assert result.classes == ("yes", "no")
    # actual → predicted → count
    assert result.confusion == {"yes": {"yes": 2, "no": 0}, "no": {"yes": 1, "no": 2}}


def test_class_only_in_the_test_set_gets_a_row_and_column():
    result = evaluate(TREE, table(("x", "maybe"), ("y", "no")))

    assert result.classes == ("yes", "no", "maybe")
    assert result.confusion["maybe"] == {"yes": 1, "no": 0, "maybe": 0}
    assert result.errors == 1


def test_most_probable_class_is_predicted_and_ties_go_to_the_first_class():
    tree = Leaf("no", {"yes": 1.0, "no": 1.0})

    result = evaluate(tree, table(("x", "yes")))

    assert result.confusion["yes"]["yes"] == 1


def test_empty_test_set_has_no_errors_and_rate_zero():
    result = evaluate(TREE, table())

    assert (result.cases, result.errors, result.error_rate) == (0, 0, 0.0)


def golf() -> DecisionTable:
    return load_table(
        DATA / "golf.txt", header=True, continuous=["temperature", "humidity"]
    )


def test_hold_out_is_the_same_for_the_same_seed():
    assert hold_out(golf(), 0.7, seed=42) == hold_out(golf(), 0.7, seed=42)


def test_hold_out_differs_for_another_seed():
    assert hold_out(golf(), 0.7, seed=42) != hold_out(golf(), 0.7, seed=7)


def test_hold_out_partitions_the_cases_in_file_order():
    full = golf()

    training, test = hold_out(full, 0.7, seed=42)

    assert (len(training.cases), len(test.cases)) == (10, 4)
    # golf has no repeated cases, so each case's file position is unique.
    position = {case: i for i, case in enumerate(full.cases)}
    for part in (training, test):
        assert part.cases == tuple(sorted(part.cases, key=position.__getitem__))
        assert part.decisions == tuple(full.decisions[position[c]] for c in part.cases)
    assert sorted(training.cases + test.cases) == sorted(full.cases)


def test_hold_out_keeps_names_and_types():
    training, test = hold_out(golf(), 0.7, seed=42)

    for part in (training, test):
        assert part.attributes == ("outlook", "temperature", "humidity", "windy")
        assert part.decision == "play"
        assert part.kinds == ("discrete", "continuous", "continuous", "discrete")


@pytest.mark.parametrize("ratio", [0.01, 0.99])
def test_hold_out_with_an_empty_part_is_an_error(ratio):
    with pytest.raises(ValueError, match="14 cases"):
        hold_out(golf(), ratio, seed=42)


def test_classify_cases_predicts_the_most_probable_class_and_counts_correct_ones():
    tree = Node(
        "a1",
        0,
        {
            "x": Leaf("yes", {"yes": 2.0, "no": 0.0}),
            "y": Leaf("no", {"yes": 1.0, "no": 1.0}),
        },
        {"yes": 3.0, "no": 1.0},
    )

    # y is a tie: yes comes first in file order.
    result = classify_cases(tree, Cases((("x",), ("y",)), decisions=("no", "yes")))

    assert result.classes == ("yes", "no")
    assert result.distributions == ({"yes": 1.0, "no": 0.0}, {"yes": 0.5, "no": 0.5})
    assert result.predicted == ("yes", "yes")
    assert result.correct == 1


def test_classify_cases_without_decisions_has_none_correct():
    result = classify_cases(TREE, Cases((("x",),), decisions=None))

    assert (result.predicted, result.decisions, result.correct) == (("yes",), None, 0)
