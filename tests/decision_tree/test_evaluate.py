from decision_tree.evaluate import classify_cases, evaluate
from decision_tree.table import Cases, DecisionTable
from decision_tree.tree import ClassDistribution, DiscreteTest, Leaf, Node

# a1 = x → yes, a1 = y → no
TREE = Node(
    DiscreteTest("a1", 0, ("x", "y")),
    {
        "x": Leaf("yes", ClassDistribution(("yes", "no"), (2.0, 0.0))),
        "y": Leaf("no", ClassDistribution(("yes", "no"), (0.0, 2.0))),
    },
    ClassDistribution(("yes", "no"), (2.0, 2.0)),
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
    assert result.correct == 4
    assert result.accuracy == 0.8
    assert result.classes == ("yes", "no")
    # actual → predicted → count
    assert result.confusion == {"yes": {"yes": 2, "no": 0}, "no": {"yes": 1, "no": 2}}


def test_class_only_in_the_test_set_gets_a_row_and_column():
    result = evaluate(TREE, table(("x", "maybe"), ("y", "no")))

    assert result.classes == ("yes", "no", "maybe")
    assert result.confusion["maybe"] == {"yes": 1, "no": 0, "maybe": 0}
    assert result.errors == 1


def test_classes_the_tree_does_not_know_follow_its_own_in_order_of_appearance():
    result = evaluate(TREE, table(("y", "rarely"), ("x", "no"), ("x", "maybe")))

    assert result.classes == ("yes", "no", "rarely", "maybe")


def test_most_probable_class_is_predicted_and_ties_go_to_the_first_class():
    tree = Leaf("no", ClassDistribution(("yes", "no"), (1.0, 1.0)))

    result = evaluate(tree, table(("x", "yes")))

    assert result.confusion["yes"]["yes"] == 1


def test_empty_test_set_has_no_errors_and_rate_zero():
    result = evaluate(TREE, table())

    assert (result.cases, result.errors, result.error_rate) == (0, 0, 0.0)
    assert (result.correct, result.accuracy) == (0, 0.0)


def test_classify_cases_predicts_the_most_probable_class_and_evaluates_them():
    tree = Node(
        DiscreteTest("a1", 0, ("x", "y")),
        {
            "x": Leaf("yes", ClassDistribution(("yes", "no"), (2.0, 0.0))),
            "y": Leaf("no", ClassDistribution(("yes", "no"), (1.0, 1.0))),
        },
        ClassDistribution(("yes", "no"), (3.0, 1.0)),
    )

    # y is a tie: yes comes first in file order.
    result = classify_cases(tree, Cases((("x",), ("y",)), decisions=("no", "yes")))

    assert result.classes == ("yes", "no")
    assert result.distributions == (
        ClassDistribution(("yes", "no"), (1.0, 0.0)),
        ClassDistribution(("yes", "no"), (0.5, 0.5)),
    )
    assert result.predicted == ("yes", "yes")
    assert result.evaluation is not None
    assert result.evaluation.classes == ("yes", "no")
    # actual → predicted → count
    assert result.evaluation.confusion == {
        "yes": {"yes": 1, "no": 0},
        "no": {"yes": 1, "no": 0},
    }
    assert result.evaluation.correct == 1


def test_classify_cases_without_decisions_has_no_evaluation():
    result = classify_cases(TREE, Cases((("x",),), decisions=None))

    assert (result.predicted, result.decisions) == (("yes",), None)
    assert result.evaluation is None
