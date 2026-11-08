from pathlib import Path

import pytest

from decision_tree.builder import BuildOptions
from decision_tree.experiment import (
    FileSource,
    FileTestSet,
    HoldOut,
    HoldOutTestSet,
    SplitError,
    TrainingSet,
    hold_out,
    run_experiment,
)
from decision_tree.render import render_plain
from decision_tree.table import DecisionTable, ReadOptions, load_table
from decision_tree.tree import size

DATA = Path(__file__).parent.parent / "data"


def golf() -> DecisionTable:
    return load_table(
        DATA / "golf.txt",
        ReadOptions(header=True),
        continuous=["temperature", "humidity"],
    )


def stock_market() -> DecisionTable:
    return load_table(DATA / "stock_market.txt")


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
    with pytest.raises(SplitError, match="14 cases"):
        hold_out(golf(), ratio, seed=42)


def test_without_a_test_set_evaluates_both_trees_on_the_training_set():
    result = run_experiment(stock_market())

    assert result.pruned is not None
    assert result.confidence == 0.25
    assert result.seed is None
    [training] = result.evaluations
    assert training.evaluated_set == TrainingSet()
    assert (training.unpruned.cases, training.unpruned.errors) == (10, 0)
    assert training.pruned is not None
    assert (training.pruned.cases, training.pruned.errors) == (10, 0)


def test_no_confidence_level_means_no_pruned_tree():
    result = run_experiment(stock_market(), confidence=None)

    assert result.pruned is None
    assert result.confidence is None
    assert render_plain(result.unpruned).startswith("a1 = old: down (3.0)")
    assert [e.pruned for e in result.evaluations] == [None]


def test_confidence_level_sets_how_much_is_pruned():
    # At 5% golf's estimates make the whole tree one Leaf; at 25% it stays.
    at_5 = run_experiment(golf(), confidence=0.05)
    at_25 = run_experiment(golf(), confidence=0.25)

    assert at_5.pruned is not None
    assert render_plain(at_5.pruned) == "Play (14.0/5.0)"
    assert at_5.evaluations[0].pruned is not None
    assert at_5.evaluations[0].pruned.errors == 5
    assert at_25.pruned is not None
    assert size(at_25.pruned) == size(at_25.unpruned) == 8
    assert (at_5.confidence, at_25.confidence) == (0.05, 0.25)


def test_build_options_reach_the_tree():
    result = run_experiment(
        stock_market(), options=BuildOptions(criterion="gain", min_cases=5)
    )

    assert size(result.unpruned) == 1


def test_test_file_is_evaluated_after_the_training_set(write_table):
    # Case 1 is a training case with its decision flipped: one error.
    path = write_table("old yes swr up\nnew no hwr up\n", name="test.txt")
    training = stock_market()
    test = load_table(path, like=training)

    result = run_experiment(training, FileSource("test.txt", test))

    assert [e.evaluated_set for e in result.evaluations] == [
        TrainingSet(),
        FileTestSet("test.txt"),
    ]
    on_test = result.evaluations[1]
    assert on_test.pruned is not None
    for evaluation in (on_test.unpruned, on_test.pruned):
        assert (evaluation.cases, evaluation.errors) == (2, 1)
        assert evaluation.confusion["up"] == {"down": 1, "up": 1}
    assert result.seed is None


def test_hold_out_with_a_seed_gives_the_same_experiment_every_run():
    first = run_experiment(golf(), HoldOut(0.7, seed=42))
    second = run_experiment(golf(), HoldOut(0.7, seed=42))

    assert first == second
    assert first.seed == 42
    training, test = first.evaluations
    assert training.evaluated_set == TrainingSet()
    assert training.unpruned.cases == 10
    assert test.evaluated_set == HoldOutTestSet(seed=42)
    assert test.unpruned.cases == 4


def test_hold_out_without_a_seed_returns_the_seed_it_picked():
    result = run_experiment(golf(), HoldOut(0.7))

    assert result.seed is not None
    assert result.evaluations[1].evaluated_set == HoldOutTestSet(seed=result.seed)
    assert run_experiment(golf(), HoldOut(0.7, seed=result.seed)) == result


def test_hold_out_that_leaves_a_part_empty_is_an_error():
    with pytest.raises(SplitError, match="no Training Set"):
        run_experiment(golf(), HoldOut(0.01, seed=42))
