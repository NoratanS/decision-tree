from pathlib import Path

import pytest

from decision_tree.table import TableError, load_table

DATA = Path(__file__).parent.parent / "data"


def test_gielda_loads_with_default_column_names():
    table = load_table(DATA / "gielda.txt")

    assert table.attributes == ("a1", "a2", "a3")
    assert table.decision == "d"
    assert len(table.cases) == 10
    assert table.cases[0] == ("old", "yes", "swr")
    assert table.decisions[0] == "down"


def test_decision_classes_are_in_file_order():
    assert load_table(DATA / "gielda.txt").classes == ("down", "up")


def test_wrong_column_count_names_the_line(tmp_path):
    path = tmp_path / "ragged.txt"
    path.write_text("a b c\nd e\n", encoding="utf-8")

    with pytest.raises(TableError, match="line 2"):
        load_table(path)


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "blank.txt"
    path.write_text("a b yes\n\n  \nc d no\n", encoding="utf-8")

    assert len(load_table(path).cases) == 2


def test_unreadable_file_is_a_table_error(tmp_path):
    with pytest.raises(TableError, match="missing.txt"):
        load_table(tmp_path / "missing.txt")


def test_empty_file_is_a_table_error(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(TableError, match="no cases"):
        load_table(path)


def test_single_column_is_a_table_error(tmp_path):
    path = tmp_path / "one.txt"
    path.write_text("yes\nno\n", encoding="utf-8")

    with pytest.raises(TableError, match="at least two columns"):
        load_table(path)
