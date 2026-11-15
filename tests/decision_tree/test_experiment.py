from pathlib import Path

import pytest

from decision_tree.builder import BuildOptions
from decision_tree.experiment import (
    FileSource,
    HoldOut,
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


@pytest.mark.parametrize("ratio", [0, 1, -0.5, 1.5])
def test_hold_out_ratio_outside_0_1_is_an_error(ratio):
    with pytest.raises(SplitError, match="between 0 and 1"):
        HoldOut(ratio)


def test_without_a_test_set_evaluates_both_trees_on_the_training_set():
    result = run_experiment(stock_market())

    assert result.pruned is not None
    assert result.pruned.confidence == 0.25
    for tree in (result.unpruned, result.pruned):
        [(evaluated_set, evaluation)] = tree.evaluations
        assert evaluated_set == TrainingSet()
        assert (evaluation.cases, evaluation.errors) == (10, 0)


def test_no_confidence_level_means_no_pruned_tree():
    result = run_experiment(stock_market(), confidence=None)

    assert result.pruned is None
    assert render_plain(result.unpruned.tree).startswith("a1 = old: down (3.0)")


def test_confidence_level_sets_how_much_is_pruned():
    # At 5% golf's estimates make the whole tree one Leaf; at 25% it stays.
    at_5 = run_experiment(golf(), confidence=0.05)
    at_25 = run_experiment(golf(), confidence=0.25)

    assert at_5.pruned is not None
    assert at_5.pruned.confidence == 0.05
    assert render_plain(at_5.pruned.tree) == "Play (14.0/5.0)"
    [(_, on_training)] = at_5.pruned.evaluations
    assert on_training.errors == 5
    assert at_25.pruned is not None
    assert at_25.pruned.confidence == 0.25
    assert size(at_25.pruned.tree) == size(at_25.unpruned.tree) == 8


def test_build_options_reach_the_tree():
    result = run_experiment(
        stock_market(), options=BuildOptions(criterion="gain", min_cases=5)
    )

    assert size(result.unpruned.tree) == 1


def test_test_file_is_evaluated_after_the_training_set(write_table):
    # Case 1 is a training case with its decision flipped: one error.
    path = write_table("old yes swr up\nnew no hwr up\n", name="test.txt")
    training = stock_market()
    source = FileSource("test.txt", load_table(path, like=training))

    result = run_experiment(training, source)

    assert result.pruned is not None
    for tree in (result.unpruned, result.pruned):
        sets = [evaluated_set for evaluated_set, _ in tree.evaluations]
        assert sets == [TrainingSet(), source]
        _, on_test = tree.evaluations[1]
        assert (on_test.cases, on_test.errors) == (2, 1)
        assert on_test.confusion["up"] == {"down": 1, "up": 1}


def test_hold_out_with_a_seed_gives_the_same_experiment_every_run():
    first = run_experiment(golf(), HoldOut(0.7, seed=42))
    second = run_experiment(golf(), HoldOut(0.7, seed=42))

    assert first == second
    (training, on_training), (test, on_test) = first.unpruned.evaluations
    assert training == TrainingSet()
    assert on_training.cases == 10
    assert test == HoldOut(0.7, seed=42)
    assert on_test.cases == 4


def test_hold_out_without_a_seed_returns_the_seed_it_picked():
    result = run_experiment(golf(), HoldOut(0.7))

    picked, _ = result.unpruned.evaluations[1]
    assert isinstance(picked, HoldOut)
    assert isinstance(picked.seed, int)
    assert picked.ratio == 0.7
    assert run_experiment(golf(), picked) == result


def test_hold_out_that_leaves_a_part_empty_is_an_error():
    with pytest.raises(SplitError, match="no Training Set"):
        run_experiment(golf(), HoldOut(0.01, seed=42))
