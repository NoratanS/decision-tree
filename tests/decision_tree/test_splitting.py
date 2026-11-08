import pytest

from decision_tree.splitting import discrete_candidate, threshold_candidate
from decision_tree.tree import DiscreteTest, ThresholdTest

# Class indexes for the golf examples.
PLAY, DONT = 0, 1
# Every humidity value in the golf Training Set, ascending.
GOLF_HUMIDITY = (65.0, 70.0, 75.0, 78.0, 80.0, 85.0, 90.0, 95.0, 96.0)


def weighted(*pairs, weight=1.0):
    """(value, class index) pairs as (value, class index, Case Weight)."""
    return [(value, k, weight) for value, k in pairs]


def threshold(cases, training_values, *, min_cases):
    """The Candidate Test on a two-class Continuous Attribute "x", column 0."""
    return threshold_candidate("x", 0, cases, 2, training_values, min_cases=min_cases)


def test_golf_sunny_humidity_threshold_is_75_not_the_midpoint():
    # Reference page: under outlook = sunny the test is humidity <= 75. The
    # boundary is between 70 and 85; the midpoint 77.5 is not a training
    # value, and the largest training value at or below it is 75 (from
    # another branch), not this node's 70.
    sunny = weighted((85, DONT), (90, DONT), (70, PLAY), (95, DONT), (70, PLAY))

    candidate = threshold_candidate("humidity", 2, sunny, 2, GOLF_HUMIDITY, min_cases=2)

    assert candidate is not None
    assert candidate.test == ThresholdTest("humidity", 2, 75.0)
    assert candidate.partition == ([2.0, 0.0], [0.0, 3.0])
    assert candidate.missing == 0.0
    assert candidate.gain == pytest.approx(0.971, abs=1e-3)


def test_a_training_value_at_the_midpoint_is_the_threshold():
    candidate = threshold(weighted((1, 0), (3, 1)), (1.0, 2.0, 3.0), min_cases=1)

    assert candidate is not None
    assert candidate.test.threshold == 2.0


def test_tie_between_thresholds_goes_to_the_lower_one():
    # {1} | {2, 3} and {1, 2} | {3} are mirror images with equal Gain.
    cases = weighted((1, 0), (2, 1), (3, 0))

    candidate = threshold(cases, (1.0, 2.0, 3.0), min_cases=1)

    assert candidate is not None
    assert candidate.test.threshold == 1.0


def test_both_branches_need_min_cases():
    # Cutting at 1 separates the classes perfectly but leaves one case on
    # the left, so with -m 2 the best allowed Threshold is 2.
    cases = weighted((1, 0), (2, 1), (3, 1), (4, 1))
    values = (1.0, 2.0, 3.0, 4.0)

    assert threshold(cases, values, min_cases=1).test.threshold == 1.0
    assert threshold(cases, values, min_cases=2).test.threshold == 2.0


def test_counts_are_weighted():
    cases = weighted((1, 0), (2, 1), weight=0.5)

    candidate = threshold(cases, (1.0, 2.0), min_cases=0.5)

    assert candidate is not None
    assert candidate.partition == ([0.5, 0.0], [0.0, 0.5])


def test_threshold_candidate_skips_missing_values_and_scales_gain():
    # Known: 1, 2 are class 0 and 3, 4 class 1, a perfect cut at 2 (Gain 1.0
    # on them); the missing Case is 1/5 of the weight.
    cases = weighted((1, 0), (2, 0), (3, 1), (4, 1), (None, 0))

    candidate = threshold(cases, (1.0, 2.0, 3.0, 4.0), min_cases=2)

    assert candidate is not None
    assert candidate.test.threshold == 2.0
    assert candidate.partition == ([2.0, 0.0], [0.0, 2.0])
    assert candidate.missing == 1.0
    assert candidate.gain == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("cases", "min_cases"),
    [
        (weighted((5, 0), (5, 1)), 1),  # one distinct value: no boundary
        (weighted((1, 0), (2, 1), (3, 1)), 2),  # every Threshold leaves a side under -m
        ([], 1),
        (weighted((None, 0), (None, 1)), 1),  # every value missing
    ],
)
def test_no_allowed_boundary_gives_none(cases, min_cases):
    assert threshold(cases, (1.0, 2.0, 3.0, 5.0), min_cases=min_cases) is None


# A test on attribute "a1", column 0, with values p and q.
P_OR_Q = DiscreteTest("a1", 0, ("p", "q"))


def test_discrete_candidate_scores_known_cases_and_scales_gain_by_known_fraction():
    # Known: p = {yes, yes}, q = {no, no}, so Gain on them is 1.0; one of the
    # five Cases is missing, so the Gain is 4/5 of that.
    cases = weighted(("p", 0), ("p", 0), ("q", 1), ("q", 1), (None, 0))

    candidate = discrete_candidate(P_OR_Q, cases, 2, min_cases=2)

    assert candidate is not None
    assert candidate.test == P_OR_Q
    assert candidate.partition == [[2.0, 0.0], [0.0, 2.0]]
    assert candidate.missing == 1.0
    assert candidate.gain == pytest.approx(0.8)


def test_discrete_candidate_needs_two_branches_of_min_cases():
    # p holds 3 Cases and q holds 3: allowed with -m 3, not with -m 4.
    cases = weighted(("p", 0), ("p", 0), ("p", 0), ("q", 1), ("q", 1), ("q", 0))

    assert discrete_candidate(P_OR_Q, cases, 2, min_cases=3) is not None
    assert discrete_candidate(P_OR_Q, cases, 2, min_cases=4) is None


def test_discrete_candidate_counts_only_known_weight_toward_min_cases():
    # q holds one known Case; the missing Case does not lift it to -m 2.
    cases = weighted(("p", 0), ("p", 1), ("q", 1), (None, 1))

    assert discrete_candidate(P_OR_Q, cases, 2, min_cases=2) is None


def test_one_branch_discrete_test_is_no_candidate():
    # Every known Case has value p, so the test cannot separate anything.
    cases = weighted(("p", 0), ("p", 1), ("p", 1), (None, 0))

    assert discrete_candidate(P_OR_Q, cases, 2, min_cases=1) is None
