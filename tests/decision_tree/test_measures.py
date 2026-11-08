from collections import Counter
from functools import cache
from pathlib import Path

import pytest

from decision_tree.measures import (
    entropy,
    gain,
    gain_ratio,
    info,
    info_split,
    split_info,
)
from decision_tree.table import DecisionTable, load_table

DATA = Path(__file__).parent.parent / "data"

# Class counts (down, up) per branch on gielda.txt, from the course script table 1.1.
GIELDA = (5, 5)
GIELDA_A1 = ((3, 0), (2, 2), (0, 3))  # old, mid, new
GIELDA_A2 = ((3, 1), (2, 4))  # yes, no
GIELDA_A3 = ((3, 3), (2, 2))  # swr, hwr


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        ((0.67, 0.33), pytest.approx(0.92, abs=0.01)),
        # docs/assignment.md, "Entropy examples (§2.1)".
        ((1.0, 0.0), 0.0),
        ((0.9, 0.1), pytest.approx(0.4689955935892812)),
        ((0.8, 0.2), pytest.approx(0.7219280948873623)),
        ((0.7, 0.3), pytest.approx(0.8812908992306927)),
        ((0.6, 0.4), pytest.approx(0.9709505944546686)),
        ((0.5, 0.5), 1.0),
    ],
)
def test_entropy_matches_course_script(probabilities, expected):
    assert entropy(probabilities) == expected


def test_info_of_gielda_is_one():
    assert info(GIELDA) == pytest.approx(1.0)


def test_info_of_a_pure_set_is_zero():
    assert info((4, 0)) == 0.0


def test_info_takes_fractional_weights():
    assert info((0.5, 0.5)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("partition", "expected"),
    [(GIELDA_A1, 0.4), (GIELDA_A2, 0.8754887502163469), (GIELDA_A3, 1.0)],
)
def test_info_split_on_gielda(partition, expected):
    assert info_split(partition) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("partition", "expected"),
    [(GIELDA_A1, 0.6), (GIELDA_A2, 0.12451124978365313), (GIELDA_A3, 0.0)],
)
def test_gain_on_gielda(partition, expected):
    assert gain(GIELDA, partition) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("partition", "expected"),
    [
        (GIELDA_A1, 1.5709505944546684),
        (GIELDA_A2, 0.9709505944546686),
        (GIELDA_A3, 0.9709505944546686),
    ],
)
def test_split_info_on_gielda(partition, expected):
    assert split_info(partition) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("partition", "expected"),
    [
        (GIELDA_A1, 0.3819343537078458),
        (GIELDA_A2, 0.12823644219877584),
        (GIELDA_A3, 0.0),
    ],
)
def test_gain_ratio_on_gielda(partition, expected):
    assert gain_ratio(GIELDA, partition) == pytest.approx(expected, abs=1e-12)


def test_gain_ratio_is_zero_when_all_cases_fall_in_one_branch():
    assert gain_ratio((2, 3), ((2, 3), (0, 0))) == 0.0


# Quinlan, "C4.5: Programs for Machine Learning", §2.4: golf with one overcast
# case's outlook missing. Class counts (Play, Don't Play) of the 13 known cases.
GOLF_KNOWN = (8, 5)
GOLF_OUTLOOK_KNOWN = ((2, 3), (3, 0), (3, 2))  # sunny, overcast, rain


def test_gain_with_missing_values_is_scaled_by_the_known_fraction():
    assert gain(GOLF_KNOWN, GOLF_OUTLOOK_KNOWN, missing=1) == pytest.approx(
        0.199, abs=5e-4
    )


def test_split_info_counts_missing_values_as_an_extra_branch():
    assert split_info(GOLF_OUTLOOK_KNOWN, missing=1) == pytest.approx(1.809, abs=5e-4)


def test_gain_ratio_with_missing_values():
    assert gain_ratio(GOLF_KNOWN, GOLF_OUTLOOK_KNOWN, missing=1) == pytest.approx(
        0.110, abs=5e-4
    )


# testowaTabDec.txt (script table 1.2). The partitions are counted from the file,
# so these tests check the loader and the measures together.
@cache
def load_testowa() -> DecisionTable:
    return load_table(DATA / "testowaTabDec.txt")


def class_counts_of_testowa() -> list[int]:
    table = load_testowa()
    return [table.decisions.count(c) for c in table.classes]


def partition_on_testowa(attribute: str) -> list[list[int]]:
    """Class counts per value of `attribute`, values in file order."""
    table = load_testowa()
    column = table.attributes.index(attribute)
    branches = Counter(
        (case[column], decision)
        for case, decision in zip(table.cases, table.decisions, strict=True)
    )
    values = dict.fromkeys(case[column] for case in table.cases)
    return [[branches[v, c] for c in table.classes] for v in values]


def _info_split(_counts, partition):
    return info_split(partition)


def _split_info(_counts, partition):
    return split_info(partition)


# docs/assignment.md, testowaTabDec.txt table, one row per measure.
TESTOWA_MEASURES = {
    "Info(X,T)": (
        _info_split,
        {
            "a1": 0.7505430557959409,
            "a2": 0.9727652780181631,
            "a3": 0.9727652780181631,
            "a4": 1.122207274456653,
        },
    ),
    "Gain": (
        gain,
        {
            "a1": 0.7799500009615414,
            "a2": 0.5577277787393192,
            "a3": 0.5577277787393192,
            "a4": 0.40828578230082946,
        },
    ),
    "Split Info": (
        _split_info,
        {
            "a1": 1.5304930567574826,
            "a2": 1.5304930567574824,
            "a3": 0.9182958340544896,
            "a4": 0.9910760598382222,
        },
    ),
    "Gain Ratio": (
        gain_ratio,
        {
            "a1": 0.5096070168485123,
            "a2": 0.36441052527276846,
            "a3": 0.6073508754546141,
            "a4": 0.41196210749705303,
        },
    ),
}


def test_info_of_testowatabdec():
    assert info(class_counts_of_testowa()) == pytest.approx(1.5304930567574824)


@pytest.mark.parametrize(
    ("measure", "attribute", "expected"),
    [
        (name, attribute, expected)
        for name, (_, values) in TESTOWA_MEASURES.items()
        for attribute, expected in values.items()
    ],
)
def test_measures_on_testowatabdec_match_the_script(measure, attribute, expected):
    compute = TESTOWA_MEASURES[measure][0]
    partition = partition_on_testowa(attribute)

    assert compute(class_counts_of_testowa(), partition) == pytest.approx(expected)


def test_a3_beats_a2_on_gain_ratio_though_they_tie_on_gain():
    # Script §5: a3 wins because it has fewer values.
    counts = class_counts_of_testowa()
    a2, a3 = partition_on_testowa("a2"), partition_on_testowa("a3")
    ratios = {
        a: gain_ratio(counts, partition_on_testowa(a))
        for a in load_testowa().attributes
    }

    assert gain(counts, a2) == pytest.approx(gain(counts, a3))
    assert max(ratios, key=ratios.__getitem__) == "a3"
