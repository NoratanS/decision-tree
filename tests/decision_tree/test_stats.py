from pathlib import Path

import pytest

from decision_tree.stats import likely_identifiers, table_stats
from decision_tree.table import ReadOptions, load_table

DATA = Path(__file__).parent.parent / "data"


def test_stock_market_attribute_value_counts_in_file_order():
    stats = table_stats(load_table(DATA / "stock_market.txt"))

    a1 = stats.attributes[0]
    assert a1.name == "a1"
    assert a1.kind == "discrete"
    assert list(a1.counts.items()) == [("old", 3), ("mid", 4), ("new", 3)]
    assert a1.distinct == 3
    assert a1.missing == 0
    assert [a.name for a in stats.attributes] == ["a1", "a2", "a3"]
    assert dict(stats.attributes[1].counts) == {"yes": 4, "no": 6}
    assert dict(stats.attributes[2].counts) == {"swr": 6, "hwr": 4}


def test_stock_market_decision_class_counts():
    stats = table_stats(load_table(DATA / "stock_market.txt"))

    assert stats.decision == "d"
    assert list(stats.class_counts.items()) == [("down", 5), ("up", 5)]


def test_test_table_counts_match_the_table():
    stats = table_stats(load_table(DATA / "test_table.txt"))

    assert [list(a.counts.items()) for a in stats.attributes] == [
        [("0", 3), ("1", 2), ("2", 4)],
        [("1", 3), ("2", 4), ("0", 2)],
        [("0", 6), ("1", 3)],
        [("1", 5), ("0", 4)],
    ]
    assert [a.distinct for a in stats.attributes] == [3, 3, 2, 2]
    assert list(stats.class_counts.items()) == [("0", 3), ("1", 4), ("2", 2)]


def test_attribute_kind_is_the_inferred_type(write_table):
    path = write_table("old 3.7 1 yes\nnew 4 2 no\n")

    stats = table_stats(load_table(path))

    assert [a.kind for a in stats.attributes] == [
        "discrete",
        "continuous",
        "discrete",
    ]


def test_missing_values_are_counted_per_attribute_and_not_as_values(write_table):
    path = write_table("old ? up\n? NA down\nNA yes up\n")

    stats = table_stats(load_table(path))

    assert [a.missing for a in stats.attributes] == [2, 2]
    assert [dict(a.counts) for a in stats.attributes] == [{"old": 1}, {"yes": 1}]


UCI = Path(__file__).parent.parent.parent / "benchmarks" / "data" / "raw"


def test_breast_w_case_id_looks_like_an_identifier():
    table = load_table(UCI / "breast-cancer-wisconsin.data")

    found = likely_identifiers(table_stats(table))

    assert [(a.name, a.distinct, a.known) for a in found] == [("a1", 645, 699)]


def test_glass_case_id_looks_like_an_identifier():
    table = load_table(UCI / "glass.data")

    assert [a.name for a in likely_identifiers(table_stats(table))] == ["a1"]


def test_adult_fnlwgt_does_not_look_like_an_identifier():
    table = load_table(UCI / "adult.data")

    assert likely_identifiers(table_stats(table)) == ()


@pytest.mark.parametrize(
    ("name", "options"),
    [
        ("stock_market.txt", ReadOptions()),
        ("test_table.txt", ReadOptions()),
        ("golf.txt", ReadOptions(header=True)),
    ],
)
def test_course_tables_and_golf_have_no_identifiers(name, options):
    assert likely_identifiers(table_stats(load_table(DATA / name, options))) == ()


def unique_column(values, decision="x"):
    return "".join(f"{value} {decision}\n" for value in values)


@pytest.mark.parametrize(
    ("values", "found"),
    [
        ([f"id{k}" for k in range(20)], True),  # discrete, 20 of 20
        ([f"id{k}" for k in range(19)], False),  # fewer than 20 known
        ([*(f"id{k}" for k in range(18)), "id0", "id1"], True),  # 18 of 20 = 90%
        ([*(f"id{k}" for k in range(17)), "id0", "id1", "id2"], False),  # 85%
        ([*(f"id{k}" for k in range(20)), *["?"] * 5], True),  # missing not counted
    ],
)
def test_identifier_needs_20_known_values_and_90_percent_distinct(
    write_table, values, found
):
    table = load_table(write_table(unique_column(values)))

    assert bool(likely_identifiers(table_stats(table))) is found


def test_integers_forced_continuous_can_be_an_identifier(write_table):
    path = write_table(unique_column(range(20)))

    table = load_table(path, continuous=["a1"])

    assert [a.name for a in likely_identifiers(table_stats(table))] == ["a1"]


def test_unique_fractional_values_are_not_an_identifier(write_table):
    path = write_table(unique_column(f"{k}.5" for k in range(20)))

    assert likely_identifiers(table_stats(load_table(path))) == ()
