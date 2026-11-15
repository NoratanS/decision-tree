import random
from functools import cache
from pathlib import Path

import pytest

from decision_tree.measures import gain, gain_ratio, known_counts
from decision_tree.splitting import columns
from decision_tree.table import ReadOptions, load_table
from decision_tree.tree import DiscreteTest, ThresholdTest

DATA = Path(__file__).parent.parent / "data"
GOLF = DATA / "golf.txt"
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
    assert candidate.branches(whole(*SUNNY)) == {
        "<=": whole(8, 10),
        ">": whole(1, 2, 7),
    }
    # A pure 2 | 3 split: Gain and Split Info are both Info(2, 3).
    assert candidate.gain == pytest.approx(0.971, abs=1e-3)
    assert candidate.gain_ratio == pytest.approx(1.0)


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
    # Each branch holds one Case of weight 0.5: enough for -m 0.5, not 0.6.
    a1 = column_a1(write_table, "1 a\n2 b\n")

    candidate = a1.candidate(every(a1, 0.5), min_cases=0.5)

    assert candidate is not None
    assert candidate.gain == pytest.approx(1.0)
    assert a1.candidate(every(a1, 0.5), min_cases=0.6) is None


# Gain 0.8 over Split Info of branches 2 | 2 plus 1 missing:
# I(0.4, 0.4, 0.2) = 1.5219280948873621.
SCALED_GAIN_RATIO = 0.5256490123859682


def test_threshold_candidate_skips_missing_values_and_scales_gain(write_table):
    # Known: 1, 2 are class a and 3, 4 class b, a perfect cut at 2 (Gain 1.0
    # on them); the missing Case is 1/5 of the weight.
    a1 = column_a1(write_table, "1 a\n2 a\n3 b\n4 b\n? a\n")

    candidate = a1.candidate(every(a1), min_cases=2)

    assert candidate is not None
    assert candidate.test.threshold == 2.0
    assert candidate.gain == pytest.approx(0.8)
    assert candidate.gain_ratio == pytest.approx(SCALED_GAIN_RATIO)


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
    assert candidate.gain == pytest.approx(0.8)
    assert candidate.gain_ratio == pytest.approx(SCALED_GAIN_RATIO)


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
    # No known overcast Case is at this Node: the branch stays and gets none
    # of the missing Case's weight.
    outlook = columns(load_golf_missing_outlook(write_table))[OUTLOOK]
    cases = whole(*SUNNY, *RAIN, MISSING_ROW)
    candidate = outlook.candidate(cases, min_cases=2)

    branches = candidate.branches(cases)

    assert branches == {
        "overcast": [],
        "sunny": [*whole(*SUNNY), (MISSING_ROW, 0.5)],
        "rain": [*whole(*RAIN), (MISSING_ROW, 0.5)],
    }


@pytest.mark.parametrize(
    ("attribute", "rows"),
    [(OUTLOOK, range(14)), (1, range(14)), (HUMIDITY, SUNNY), (3, RAIN)],
)
def test_gain_from_the_branches_equals_the_candidates_gain(attribute, rows):
    table = load_golf()
    cases = whole(*rows)
    candidate = columns(table)[attribute].candidate(cases, min_cases=2)
    assert candidate is not None

    split = class_counts_per_branch(table, candidate.branches(cases))

    assert gain(known_counts(split), split) == pytest.approx(candidate.gain)
    assert gain_ratio(known_counts(split), split) == pytest.approx(candidate.gain_ratio)


def class_counts_per_branch(table, branches):
    counts = []
    for branch in branches.values():
        row = dict.fromkeys(table.classes, 0.0)
        for i, weight in branch:
            row[table.decisions[i]] += weight
        counts.append(list(row.values()))
    return counts


# test_table.txt (script table 1.2), docs/assignment.md: Gain and Gain Ratio
# of each attribute at the root, counted by the columns.
TEST_TABLE = DATA / "test_table.txt"
TEST_TABLE_SCORES = {
    "a1": (0.7799500009615414, 0.5096070168485123),
    "a2": (0.5577277787393192, 0.36441052527276846),
    "a3": (0.5577277787393192, 0.6073508754546141),
    "a4": (0.40828578230082946, 0.41196210749705303),
}


@cache
def root_candidates_of_test_table():
    table = load_table(TEST_TABLE)
    cases = whole(*range(len(table.cases)))
    return {
        name: column.candidate(cases, min_cases=1)
        for name, column in zip(table.attributes, columns(table), strict=True)
    }


@pytest.mark.parametrize(("attribute", "scores"), TEST_TABLE_SCORES.items())
def test_scores_on_test_table_match_the_script(attribute, scores):
    candidate = root_candidates_of_test_table()[attribute]
    expected_gain, expected_gain_ratio = scores

    assert candidate.gain == pytest.approx(expected_gain)
    assert candidate.gain_ratio == pytest.approx(expected_gain_ratio)


def test_a3_beats_a2_on_gain_ratio_though_they_tie_on_gain():
    # Script §5: a3 wins because it has fewer values.
    candidates = root_candidates_of_test_table()

    assert candidates["a2"].gain == pytest.approx(candidates["a3"].gain)
    assert max(candidates, key=lambda a: candidates[a].gain_ratio) == "a3"


def random_continuous_table(rng):
    """Text of a small one-attribute table with ties and Missing Values."""
    rows = [
        f"{rng.choice(['?', *map(str, range(1, 6))])} {rng.choice('abc')}"
        for _ in range(rng.randint(2, 12))
    ]
    return "\n".join(rows) + "\n"


def best_gain_over_every_threshold(column, table, cases, min_cases):
    """Highest `measures.gain` over the allowed Thresholds, each known value
    but the largest tried; None if none is allowed."""
    values = column.values
    known = [(i, w) for i, w in cases if values[i] is not None]
    missing = sum(w for i, w in cases if values[i] is None)
    best = None
    for threshold in sorted({values[i] for i, _ in known})[:-1]:
        branches = {
            "<=": [(i, w) for i, w in known if values[i] <= threshold],
            ">": [(i, w) for i, w in known if values[i] > threshold],
        }
        partition = class_counts_per_branch(table, branches)
        if min(sum(branch) for branch in partition) < min_cases:
            continue
        score = gain(known_counts(partition), partition, missing=missing)
        best = score if best is None else max(best, score)
    return best


@pytest.mark.parametrize("seed", range(200))
def test_threshold_sweep_finds_the_best_gain_of_every_threshold(write_table, seed):
    rng = random.Random(seed)
    table = load_table(write_table(random_continuous_table(rng)), continuous=["a1"])
    a1 = columns(table)[0]
    cases = [(i, rng.uniform(0.1, 1.0)) for i in range(len(table.cases))]
    min_cases = rng.choice([0.2, 0.5, 1.0, 2.0])

    candidate = a1.candidate(cases, min_cases=min_cases)
    expected = best_gain_over_every_threshold(a1, table, cases, min_cases)

    if expected is None:
        assert candidate is None
    else:
        assert candidate is not None
        assert candidate.gain == pytest.approx(expected, abs=1e-9)
