from pathlib import Path

from decision_tree.stats import table_stats
from decision_tree.table import load_table

DATA = Path(__file__).parent.parent / "data"


def test_gielda_attribute_value_counts_in_file_order():
    stats = table_stats(load_table(DATA / "gielda.txt"))

    a1 = stats.attributes[0]
    assert a1.name == "a1"
    assert a1.kind == "discrete"
    assert list(a1.counts.items()) == [("old", 3), ("mid", 4), ("new", 3)]
    assert a1.distinct == 3
    assert a1.missing == 0
    assert [a.name for a in stats.attributes] == ["a1", "a2", "a3"]
    assert dict(stats.attributes[1].counts) == {"yes": 4, "no": 6}
    assert dict(stats.attributes[2].counts) == {"swr": 6, "hwr": 4}


def test_gielda_decision_class_counts():
    stats = table_stats(load_table(DATA / "gielda.txt"))

    assert stats.decision == "d"
    assert list(stats.class_counts.items()) == [("down", 5), ("up", 5)]


def test_testowatabdec_counts_match_the_table():
    stats = table_stats(load_table(DATA / "testowaTabDec.txt"))

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
