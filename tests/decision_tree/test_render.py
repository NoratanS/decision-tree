import pytest

from decision_tree.evaluate import Classification, Evaluation
from decision_tree.experiment import (
    ExperimentResult,
    FileTestSet,
    HoldOutTestSet,
    SetEvaluation,
    TrainingSet,
)
from decision_tree.render import (
    TableBlock,
    classification_blocks,
    experiment_blocks,
    render_classification_plain,
    render_experiment_plain,
    render_plain,
    set_name,
)
from decision_tree.tree import DiscreteTest, Leaf, Node, ThresholdTest


def test_leaf_without_errors_shows_its_weight():
    assert render_plain(Leaf("up", {"down": 0.0, "up": 3.0})) == "up (3.0)"


def test_leaf_with_errors_shows_weight_and_errors():
    assert render_plain(Leaf("bear", {"bear": 3.0, "bare": 2.0})) == "bear (5.0/2.0)"


def test_fractional_weights_have_one_decimal():
    assert render_plain(Leaf("yes", {"yes": 2.0, "no": 0.375})) == "yes (2.4/0.4)"


def test_nested_nodes_are_indented_with_bars():
    inner = Node(
        DiscreteTest("a2", 1, ("p", "q")),
        {
            "p": Leaf("yes", {"yes": 2.0, "no": 0.0}),
            "q": Leaf("no", {"yes": 0.0, "no": 1.0}),
        },
        {"yes": 2.0, "no": 1.0},
    )
    tree = Node(
        DiscreteTest("a1", 0, ("x", "y")),
        {"x": inner, "y": Leaf("no", {"yes": 0.0, "no": 3.0})},
        {"yes": 2.0, "no": 4.0},
    )

    assert render_plain(tree) == (
        "a1 = x:\n|   a2 = p: yes (2.0)\n|   a2 = q: no (1.0)\na1 = y: no (3.0)"
    )


def test_threshold_tests_print_like_c45():
    tree = Node(
        ThresholdTest("humidity", 2, 75.0),
        {
            "<=": Leaf("Play", {"Play": 2.0, "Don't Play": 0.0}),
            ">": Leaf("Don't Play", {"Play": 0.0, "Don't Play": 3.0}),
        },
        {"Play": 2.0, "Don't Play": 3.0},
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
        ThresholdTest("a1", 0, threshold),
        {"<=": Leaf("y", {"y": 1.0}), ">": Leaf("y", {"y": 1.0})},
        {"y": 2.0},
    )

    assert render_plain(tree).startswith(f"a1 <= {text} : ")


GOLF_CLASSES = ("Play", "Don't Play")


def test_classification_shows_each_case_with_its_class_and_distribution():
    result = Classification(
        GOLF_CLASSES,
        ({"Play": 0.4, "Don't Play": 0.6}, {"Play": 1.0, "Don't Play": 0.0}),
        decisions=None,
    )

    assert render_classification_plain(result) == (
        "case  predicted    Play  Don't Play\n"
        "   1  Don't Play  0.400       0.600\n"
        "   2  Play        1.000       0.000"
    )


def test_classification_with_decisions_marks_wrong_predictions_and_gives_accuracy():
    result = Classification(
        GOLF_CLASSES,
        (
            {"Play": 0.4, "Don't Play": 0.6},
            {"Play": 1.0, "Don't Play": 0.0},
            {"Play": 0.5, "Don't Play": 0.5},
        ),
        decisions=("Play", "Play", "Play"),
    )

    assert render_classification_plain(result) == (
        "case  predicted    Play  Don't Play  actual\n"
        "   1  Don't Play  0.400       0.600  Play    wrong\n"
        "   2  Play        1.000       0.000  Play\n"
        "   3  Play        0.500       0.500  Play\n"
        "\n"
        "Accuracy: 2 of 3 cases (66.7%)"
    )


def roles(block: TableBlock) -> list[list[str | None]]:
    return [[cell.role for cell in row] for row in block.rows]


def test_classification_marks_correct_and_wrong_predictions_by_role():
    result = Classification(
        GOLF_CLASSES,
        ({"Play": 0.4, "Don't Play": 0.6}, {"Play": 1.0, "Don't Play": 0.0}),
        decisions=("Play", "Play"),
    )

    [table] = classification_blocks(result)

    # case, predicted, Play, Don't Play, actual, mark
    assert roles(table) == [
        [None, "wrong", None, None, None, "wrong"],
        [None, "correct", None, None, None, None],
    ]


def test_classification_without_decisions_has_no_roles():
    result = Classification(
        GOLF_CLASSES, ({"Play": 0.4, "Don't Play": 0.6},), decisions=None
    )

    [table] = classification_blocks(result)

    assert roles(table) == [[None, None, None, None]]


@pytest.mark.parametrize(
    ("evaluated_set", "name"),
    [
        (TrainingSet(), "Training Set"),
        (FileTestSet("data/test.txt"), "Test Set (data/test.txt)"),
        (HoldOutTestSet(seed=42), "Test Set (hold-out, --seed 42)"),
    ],
)
def test_set_name_describes_where_the_set_comes_from(evaluated_set, name):
    assert set_name(evaluated_set) == name


# A two-Leaf tree on `a1`, and the single Leaf it prunes to.
SPLIT_TREE = Node(
    DiscreteTest("a1", 0, ("x", "y")),
    {
        "x": Leaf("yes", {"yes": 3.0, "no": 0.0}),
        "y": Leaf("no", {"yes": 1.0, "no": 2.0}),
    },
    {"yes": 4.0, "no": 2.0},
)
ONE_LEAF = Leaf("yes", {"yes": 4.0, "no": 2.0})


def evaluation(yes: tuple[int, int], no: tuple[int, int]) -> Evaluation:
    """Actual `yes` and `no` rows of a confusion matrix: (predicted yes, no)."""
    return Evaluation(
        ("yes", "no"),
        {"yes": {"yes": yes[0], "no": yes[1]}, "no": {"yes": no[0], "no": no[1]}},
    )


def test_experiment_without_pruning_shows_one_tree_and_its_evaluation():
    result = ExperimentResult(
        SPLIT_TREE,
        None,
        None,
        None,
        (SetEvaluation(TrainingSet(), evaluation((3, 0), (1, 2)), None),),
    )

    assert render_experiment_plain(result) == (
        "Unpruned tree (size 3):\n"
        "a1 = x: yes (3.0)\n"
        "a1 = y: no (3.0/1.0)\n"
        "\n"
        "Unpruned tree on Training Set: 6 cases, 1 error (16.7%)\n"
        "  actual \\ predicted  yes  no\n"
        "  yes                   3   0\n"
        "  no                    1   2"
    )


def test_confusion_matrix_cells_are_correct_wrong_or_empty():
    result = ExperimentResult(
        SPLIT_TREE,
        None,
        None,
        None,
        (SetEvaluation(TrainingSet(), evaluation((3, 0), (1, 2)), None),),
    )

    matrix = experiment_blocks(result)[-1]

    assert isinstance(matrix, TableBlock)
    # actual class, predicted yes, predicted no
    assert roles(matrix) == [
        [None, "correct", "empty"],
        [None, "wrong", "correct"],
    ]


def pruned_result(*evaluations: SetEvaluation, confidence=0.25) -> ExperimentResult:
    return ExperimentResult(SPLIT_TREE, ONE_LEAF, confidence, None, evaluations)


ON_TRAINING = SetEvaluation(
    TrainingSet(), evaluation((3, 0), (1, 2)), evaluation((4, 0), (2, 0))
)
PRUNED_ON_TRAINING = (
    "Unpruned tree (size 3):\n"
    "a1 = x: yes (3.0)\n"
    "a1 = y: no (3.0/1.0)\n"
    "\n"
    "Pruned tree (size 1, Confidence Level 25%):\n"
    "yes (6.0/2.0)\n"
    "\n"
    "Unpruned tree on Training Set: 6 cases, 1 error (16.7%)\n"
    "  actual \\ predicted  yes  no\n"
    "  yes                   3   0\n"
    "  no                    1   2\n"
    "\n"
    "Pruned tree on Training Set: 6 cases, 2 errors (33.3%)\n"
    "  actual \\ predicted  yes  no\n"
    "  yes                   4   0\n"
    "  no                    2   0"
)


def test_experiment_with_pruning_shows_both_trees_then_both_evaluations():
    assert render_experiment_plain(pruned_result(ON_TRAINING)) == PRUNED_ON_TRAINING


@pytest.mark.parametrize(("confidence", "text"), [(0.07, "7"), (0.125, "12.5")])
def test_confidence_level_is_shown_in_percent(confidence, text):
    report = render_experiment_plain(pruned_result(ON_TRAINING, confidence=confidence))

    assert f"Pruned tree (size 1, Confidence Level {text}%):\n" in report


def test_experiment_with_a_test_file_evaluates_both_trees_on_it_after_training():
    on_test = SetEvaluation(
        FileTestSet("test.txt"), evaluation((1, 1), (0, 0)), evaluation((2, 0), (0, 0))
    )

    assert render_experiment_plain(pruned_result(ON_TRAINING, on_test)) == (
        PRUNED_ON_TRAINING + "\n"
        "\n"
        "Unpruned tree on Test Set (test.txt): 2 cases, 1 error (50.0%)\n"
        "  actual \\ predicted  yes  no\n"
        "  yes                   1   1\n"
        "  no                    0   0\n"
        "\n"
        "Pruned tree on Test Set (test.txt): 2 cases, 0 errors (0.0%)\n"
        "  actual \\ predicted  yes  no\n"
        "  yes                   2   0\n"
        "  no                    0   0"
    )


def test_experiment_with_a_hold_out_names_the_seed_of_the_split():
    on_test = SetEvaluation(HoldOutTestSet(42), evaluation((0, 1), (0, 1)), None)
    result = ExperimentResult(
        SPLIT_TREE,
        None,
        None,
        42,
        (SetEvaluation(TrainingSet(), evaluation((3, 0), (1, 2)), None), on_test),
    )

    assert render_experiment_plain(result) == (
        "Unpruned tree (size 3):\n"
        "a1 = x: yes (3.0)\n"
        "a1 = y: no (3.0/1.0)\n"
        "\n"
        "Unpruned tree on Training Set: 6 cases, 1 error (16.7%)\n"
        "  actual \\ predicted  yes  no\n"
        "  yes                   3   0\n"
        "  no                    1   2\n"
        "\n"
        "Unpruned tree on Test Set (hold-out, --seed 42): 2 cases, 1 error (50.0%)\n"
        "  actual \\ predicted  yes  no\n"
        "  yes                   0   1\n"
        "  no                    0   1"
    )
