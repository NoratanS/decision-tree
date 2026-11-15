from pathlib import Path

import pytest

from decision_tree.splitting import columns
from decision_tree.table import ReadOptions, load_table
from decision_tree.tree import DiscreteTest, ThresholdTest

GOLF = Path(__file__).parent.parent / "data" / "golf.txt"
OUTLOOK, HUMIDITY = 0, 2
# Rows of golf.txt (0 is the first Case) with outlook = sunny.
SUNNY = (1, 2, 7, 8, 10)


def load_golf(text=None, write_table=None):
    path = GOLF if text is None else write_table(text)
    return load_table(
        path, ReadOptions(header=True), continuous=["temperature", "humidity"]
    )


def whole(*indices):
    """Cases at `indices`, each with Case Weight 1."""
    return [(i, 1.0) for i in indices]


def test_golf_sunny_humidity_threshold_is_75_not_the_midpoint():
    # Reference page: under outlook = sunny the test is humidity <= 75. The
    # boundary is between 70 and 85; the midpoint 77.5 is not a training
    # value, and the largest training value at or below it is 75 (from
    # another branch), not this node's 70.
    humidity = columns(load_golf())[HUMIDITY]

    candidate = humidity.candidate(whole(*SUNNY), min_cases=2)

    assert candidate is not None
    assert candidate.test == ThresholdTest("humidity", HUMIDITY, 75.0)
    assert candidate.partition == ([2.0, 0.0], [0.0, 3.0])
    assert candidate.missing == 0.0
    assert candidate.gain == pytest.approx(0.971, abs=1e-3)


def column_a1(write_table, text, kind="continuous"):
    """Column a1 of a one-attribute table, forced to `kind`."""
    return columns(load_table(write_table(text), **{kind: ["a1"]}))[0]


def every(column, weight=1.0):
    """Every Case of `column`'s table, each with `weight`."""
    return [(i, weight) for i in range(len(column.values))]


def test_a_training_value_at_the_midpoint_is_the_threshold(write_table):
    # The Node holds 1 and 3; 2 is a training value elsewhere.
    a1 = column_a1(write_table, "1 a\n2 b\n3 b\n")

    candidate = a1.candidate(whole(0, 2), min_cases=1)

    assert candidate is not None
    assert candidate.test.threshold == 2.0


def test_tie_between_thresholds_goes_to_the_lower_one(write_table):
    # {1} | {2, 3} and {1, 2} | {3} are mirror images with equal Gain.
    a1 = column_a1(write_table, "1 a\n2 b\n3 a\n")

    candidate = a1.candidate(every(a1), min_cases=1)

    assert candidate is not None
    assert candidate.test.threshold == 1.0


def test_both_branches_need_min_cases(write_table):
    # Cutting at 1 separates the classes perfectly but leaves one case on
    # the left, so with -m 2 the best allowed Threshold is 2.
    a1 = column_a1(write_table, "1 a\n2 b\n3 b\n4 b\n")

    assert a1.candidate(every(a1), min_cases=1).test.threshold == 1.0
    assert a1.candidate(every(a1), min_cases=2).test.threshold == 2.0


def test_counts_are_weighted(write_table):
    a1 = column_a1(write_table, "1 a\n2 b\n")

    candidate = a1.candidate(every(a1, 0.5), min_cases=0.5)

    assert candidate is not None
    assert candidate.partition == ([0.5, 0.0], [0.0, 0.5])


def test_threshold_candidate_skips_missing_values_and_scales_gain(write_table):
    # Known: 1, 2 are class a and 3, 4 class b, a perfect cut at 2 (Gain 1.0
    # on them); the missing Case is 1/5 of the weight.
    a1 = column_a1(write_table, "1 a\n2 a\n3 b\n4 b\n? a\n")

    candidate = a1.candidate(every(a1), min_cases=2)

    assert candidate is not None
    assert candidate.test.threshold == 2.0
    assert candidate.partition == ([2.0, 0.0], [0.0, 2.0])
    assert candidate.missing == 1.0
    assert candidate.gain == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("text", "n_cases", "min_cases"),
    [
        ("5 a\n5 b\n", 2, 1),  # one distinct value: no boundary
        ("1 a\n2 b\n3 b\n", 3, 2),  # every Threshold leaves a side under -m
        ("1 a\n2 b\n", 0, 1),  # no Cases at the Node
        ("? a\n? b\n1 a\n2 b\n", 2, 1),  # every value at the Node missing
    ],
)
def test_no_allowed_boundary_gives_none(write_table, text, n_cases, min_cases):
    a1 = column_a1(write_table, text)

    assert a1.candidate(whole(*range(n_cases)), min_cases=min_cases) is None


def test_discrete_candidate_scores_known_cases_and_scales_gain_by_known_fraction(
    write_table,
):
    # Known: p = {a, a}, q = {b, b}, so Gain on them is 1.0; one of the five
    # Cases is missing, so the Gain is 4/5 of that.
    a1 = column_a1(write_table, "p a\np a\nq b\nq b\n? a\n", "discrete")

    candidate = a1.candidate(every(a1), min_cases=2)

    assert candidate is not None
    assert candidate.test == DiscreteTest("a1", 0, ("p", "q"))
    assert candidate.partition == [[2.0, 0.0], [0.0, 2.0]]
    assert candidate.missing == 1.0
    assert candidate.gain == pytest.approx(0.8)


def test_discrete_candidate_needs_two_branches_of_min_cases(write_table):
    # p holds 3 Cases and q holds 3: allowed with -m 3, not with -m 4.
    a1 = column_a1(write_table, "p a\np a\np a\nq b\nq b\nq a\n", "discrete")

    assert a1.candidate(every(a1), min_cases=3) is not None
    assert a1.candidate(every(a1), min_cases=4) is None


def test_discrete_candidate_counts_only_known_weight_toward_min_cases(write_table):
    # q holds one known Case; the missing Case does not lift it to -m 2.
    a1 = column_a1(write_table, "p a\np b\nq b\n? b\n", "discrete")

    assert a1.candidate(every(a1), min_cases=2) is None


def test_one_branch_discrete_test_is_no_candidate(write_table):
    # Every known Case at the Node has value p, so the test cannot separate
    # anything; q occurs only in the last Case, which is not at the Node.
    a1 = column_a1(write_table, "p a\np b\np b\n? a\nq a\n", "discrete")

    assert a1.candidate(whole(0, 1, 2, 3), min_cases=1) is None


# Golf with the outlook of row 11 (overcast, 72, 90, true, Play) unknown, as in
# Quinlan's C4.5 book, §2.4.
MISSING_ROW = 11
OVERCAST, RAIN = (0, 3, 12), (4, 5, 6, 9, 13)


def load_golf_missing_outlook(write_table):
    text = GOLF.read_text().replace("overcast, 72, 90", "?, 72, 90")
    return load_golf(text, write_table)


def approx_cases(cases):
    return [(i, pytest.approx(weight)) for i, weight in cases]


def test_branches_spread_a_missing_value_by_known_weight(write_table):
    # 13 Cases have a known outlook: 3 overcast, 5 sunny, 5 rain. The missing
    # Case goes down each branch with weight 3/13, 5/13 and 5/13.
    outlook = columns(load_golf_missing_outlook(write_table))[OUTLOOK]
    cases = whole(*range(14))
    candidate = outlook.candidate(cases, min_cases=2)

    branches = candidate.branches(cases)

    assert list(branches) == ["overcast", "sunny", "rain"]
    assert branches["overcast"] == approx_cases(
        [*whole(*OVERCAST), (MISSING_ROW, 3 / 13)]
    )
    assert branches["sunny"] == approx_cases([*whole(*SUNNY), (MISSING_ROW, 5 / 13)])
    assert branches["rain"] == approx_cases([*whole(*RAIN), (MISSING_ROW, 5 / 13)])


def test_an_empty_discrete_branch_stays_empty(write_table):
    # No known overcast Case is at this Node: the partition leaves overcast
    # out, the branch stays and gets none of the missing Case's weight.
    outlook = columns(load_golf_missing_outlook(write_table))[OUTLOOK]
    cases = whole(*SUNNY, *RAIN, MISSING_ROW)
    candidate = outlook.candidate(cases, min_cases=2)

    branches = candidate.branches(cases)

    assert len(candidate.partition) == 2
    assert branches == {
        "overcast": [],
        "sunny": [*whole(*SUNNY), (MISSING_ROW, 0.5)],
        "rain": [*whole(*RAIN), (MISSING_ROW, 0.5)],
    }


@pytest.mark.parametrize(
    ("attribute", "rows"),
    [(OUTLOOK, range(14)), (1, range(14)), (HUMIDITY, SUNNY), (3, RAIN)],
)
def test_the_split_agrees_with_the_scored_partition(attribute, rows):
    table = load_golf()
    cases = whole(*rows)
    candidate = columns(table)[attribute].candidate(cases, min_cases=2)
    assert candidate is not None

    # The partition leaves out empty branches; none is empty here.
    split = []
    for branch in candidate.branches(cases).values():
        counts = dict.fromkeys(table.classes, 0.0)
        for i, weight in branch:
            counts[table.decisions[i]] += weight
        split.append(list(counts.values()))

    assert split == [list(counts) for counts in candidate.partition]
