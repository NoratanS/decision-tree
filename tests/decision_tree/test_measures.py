from pathlib import Path

import pytest

from decision_tree.measures import (
    entropy,
    gain,
    gain_ratio,
    gain_ratio_of,
    info,
    info_split,
    known_counts,
    split_info,
)
from decision_tree.table import load_table

DATA = Path(__file__).parent.parent / "data"

# Class counts (down, up) per branch on stock_market.txt (course script table 1.1).
STOCK_MARKET = (5, 5)
STOCK_MARKET_A1 = ((3, 0), (2, 2), (0, 3))  # old, mid, new
STOCK_MARKET_A2 = ((3, 1), (2, 4))  # yes, no
STOCK_MARKET_A3 = ((3, 3), (2, 2))  # swr, hwr


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


def test_info_of_stock_market_is_one():
    assert info(STOCK_MARKET) == pytest.approx(1.0)


def test_info_of_a_pure_set_is_zero():
    assert info((4, 0)) == 0.0


def test_info_takes_fractional_weights():
    assert info((0.5, 0.5)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("partition", "expected"),
    [
        (STOCK_MARKET_A1, 0.4),
        (STOCK_MARKET_A2, 0.8754887502163469),
        (STOCK_MARKET_A3, 1.0),
    ],
)
def test_info_split_on_stock_market(partition, expected):
    assert info_split(partition) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("partition", "expected"),
    [
        (STOCK_MARKET_A1, 0.6),
        (STOCK_MARKET_A2, 0.12451124978365313),
        (STOCK_MARKET_A3, 0.0),
    ],
)
def test_gain_on_stock_market(partition, expected):
    assert gain(STOCK_MARKET, partition) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("partition", "expected"),
    [
        (STOCK_MARKET_A1, 1.5709505944546684),
        (STOCK_MARKET_A2, 0.9709505944546686),
        (STOCK_MARKET_A3, 0.9709505944546686),
    ],
)
def test_split_info_on_stock_market(partition, expected):
    assert split_info(partition) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("partition", "expected"),
    [
        (STOCK_MARKET_A1, 0.3819343537078458),
        (STOCK_MARKET_A2, 0.12823644219877584),
        (STOCK_MARKET_A3, 0.0),
    ],
)
def test_gain_ratio_on_stock_market(partition, expected):
    assert gain_ratio(STOCK_MARKET, partition) == pytest.approx(expected, abs=1e-12)


def test_gain_ratio_is_zero_when_all_cases_fall_in_one_branch():
    assert gain_ratio((2, 3), ((2, 3), (0, 0))) == 0.0


def test_gain_ratio_of_divides_a_known_gain_by_split_info():
    # Two branches of 2 cases: Split Info is 1 bit; one branch: Split Info 0.
    assert gain_ratio_of(0.5, ((2, 0), (0, 2))) == pytest.approx(0.5)
    assert gain_ratio_of(0.5, ((2, 3), (0, 0))) == 0.0


# Quinlan, "C4.5: Programs for Machine Learning", §2.4: golf with one overcast
# case's outlook missing. Class counts (Play, Don't Play) of the 13 known cases.
GOLF_KNOWN = (8, 5)
GOLF_OUTLOOK_KNOWN = ((2, 3), (3, 0), (3, 2))  # sunny, overcast, rain


def test_known_counts_are_the_class_counts_summed_over_the_branches():
    # Quinlan's golf outlook without the missing Case: 8 Play, 5 Don't Play.
    assert known_counts(GOLF_OUTLOOK_KNOWN) == [8, 5]


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


# test_table.txt (script table 1.2), docs/assignment.md. Gain and Gain Ratio of
# its attributes are checked through the columns in test_splitting.py.
def test_info_of_test_table():
    table = load_table(DATA / "test_table.txt")
    counts = [table.decisions.count(c) for c in table.classes]

    assert info(counts) == pytest.approx(1.5304930567574824)
