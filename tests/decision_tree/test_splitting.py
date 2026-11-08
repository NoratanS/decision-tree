import pytest

from decision_tree.splitting import best_threshold

# Class indexes for the golf examples.
PLAY, DONT = 0, 1
# Every humidity value in the golf Training Set, ascending.
GOLF_HUMIDITY = (65.0, 70.0, 75.0, 78.0, 80.0, 85.0, 90.0, 95.0, 96.0)


def weighted(*pairs, weight=1.0):
    """(value, class index) pairs as (value, class index, Case Weight)."""
    return [(value, k, weight) for value, k in pairs]


def test_golf_sunny_humidity_threshold_is_75_not_the_midpoint():
    # Reference page: under outlook = sunny the test is humidity <= 75. The
    # boundary is between 70 and 85; the midpoint 77.5 is not a training
    # value, and the largest training value at or below it is 75 (from
    # another branch), not this node's 70.
    sunny = weighted((85, DONT), (90, DONT), (70, PLAY), (95, DONT), (70, PLAY))

    test = best_threshold(sunny, 2, GOLF_HUMIDITY, min_cases=2)

    assert test is not None
    assert test.threshold == 75.0
    assert test.partition == ([2.0, 0.0], [0.0, 3.0])
    assert test.gain == pytest.approx(0.971, abs=1e-3)


def test_a_training_value_at_the_midpoint_is_the_threshold():
    test = best_threshold(weighted((1, 0), (3, 1)), 2, (1.0, 2.0, 3.0), min_cases=1)

    assert test is not None
    assert test.threshold == 2.0


def test_tie_between_thresholds_goes_to_the_lower_one():
    # {1} | {2, 3} and {1, 2} | {3} are mirror images with equal Gain.
    cases = weighted((1, 0), (2, 1), (3, 0))

    test = best_threshold(cases, 2, (1.0, 2.0, 3.0), min_cases=1)

    assert test is not None
    assert test.threshold == 1.0


def test_both_branches_need_min_cases():
    # Cutting at 1 separates the classes perfectly but leaves one case on
    # the left, so with -m 2 the best allowed Threshold is 2.
    cases = weighted((1, 0), (2, 1), (3, 1), (4, 1))

    assert best_threshold(cases, 2, (1.0, 2.0, 3.0, 4.0), min_cases=1).threshold == 1.0
    assert best_threshold(cases, 2, (1.0, 2.0, 3.0, 4.0), min_cases=2).threshold == 2.0


def test_counts_are_weighted():
    cases = weighted((1, 0), (2, 1), weight=0.5)

    test = best_threshold(cases, 2, (1.0, 2.0), min_cases=0.5)

    assert test is not None
    assert test.partition == ([0.5, 0.0], [0.0, 0.5])


@pytest.mark.parametrize(
    ("cases", "min_cases"),
    [
        (weighted((5, 0), (5, 1)), 1),  # one distinct value: no boundary
        (weighted((1, 0), (2, 1), (3, 1)), 2),  # every Threshold leaves a side under -m
        ([], 1),
    ],
)
def test_no_allowed_boundary_gives_none(cases, min_cases):
    assert best_threshold(cases, 2, (1.0, 2.0, 3.0, 5.0), min_cases=min_cases) is None
