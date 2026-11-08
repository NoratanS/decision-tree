import pytest

from decision_tree.measures import (
    entropy,
    gain,
    gain_ratio,
    info,
    info_split,
    split_info,
)

# Class counts (down, up) per branch on gielda.txt, from the course script table 1.1.
GIELDA = (5, 5)
GIELDA_A1 = ((3, 0), (2, 2), (0, 3))  # old, mid, new
GIELDA_A2 = ((3, 1), (2, 4))  # yes, no
GIELDA_A3 = ((3, 3), (2, 2))  # swr, hwr


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        ((0.5, 0.5), 1.0),
        ((0.67, 0.33), pytest.approx(0.92, abs=0.01)),
        ((1.0, 0.0), 0.0),
        ((0.9, 0.1), pytest.approx(0.4689955935892812)),
        ((0.6, 0.4), pytest.approx(0.9709505944546686)),
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
