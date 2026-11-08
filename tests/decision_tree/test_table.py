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


def test_wrong_column_count_names_the_line(write_table):
    path = write_table("a b c\nd e\n")

    with pytest.raises(TableError, match="line 2"):
        load_table(path)


def test_blank_lines_are_skipped(write_table):
    path = write_table("a b yes\n\n  \nc d no\n")

    assert len(load_table(path).cases) == 2


def test_unreadable_file_is_a_table_error(tmp_path):
    with pytest.raises(TableError, match="missing.txt"):
        load_table(tmp_path / "missing.txt")


def test_empty_file_is_a_table_error(write_table):
    path = write_table("\n")

    with pytest.raises(TableError, match="no cases"):
        load_table(path)


def test_single_column_is_a_table_error(write_table):
    path = write_table("yes\nno\n")

    with pytest.raises(TableError, match="at least two columns"):
        load_table(path)


@pytest.mark.parametrize("sep", [",", ";", "|", ":"])
def test_separator_is_detected(write_table, sep):
    path = write_table(f"old{sep}yes{sep}down\nnew{sep}no{sep}up\n")

    table = load_table(path)

    assert table.cases == (("old", "yes"), ("new", "no"))
    assert table.decisions == ("down", "up")


def test_runs_of_spaces_and_tabs_are_one_separator(write_table):
    path = write_table("old  \t yes\tdown\n\tnew no    up  \n")

    table = load_table(path)

    assert table.cases == (("old", "yes"), ("new", "no"))
    assert table.decisions == ("down", "up")


def test_comma_with_spaces_after_it_splits_on_the_comma(write_table):
    path = write_table("old, yes, down\nnew, no, up\n")

    assert load_table(path).cases == (("old", "yes"), ("new", "no"))


def test_values_with_spaces_pick_the_separator_that_fits_every_line(write_table):
    path = write_table("New York;old;down\nRome;new;up\n")

    assert load_table(path).cases == (("New York", "old"), ("Rome", "new"))


def test_no_fitting_separator_names_the_line_where_detection_fails(write_table):
    path = write_table("a,b,c\nd,e,f\ng,h\n")

    with pytest.raises(TableError, match=r"line 3: expected 3 columns.*set one"):
        load_table(path)


def test_explicit_separator_can_be_any_single_character(write_table):
    path = write_table("old#yes, really#down\nnew#no#up\n")

    table = load_table(path, sep="#")

    assert table.cases == (("old", "yes, really"), ("new", "no"))


def test_ragged_line_with_explicit_separator_names_the_line(write_table):
    path = write_table("a;b;c\n\nd;e\n")

    with pytest.raises(TableError, match=r"line 3: expected 3 columns"):
        load_table(path, sep=";")


def test_separator_must_be_one_character(write_table):
    path = write_table("a::b\n")

    with pytest.raises(TableError, match="single character"):
        load_table(path, sep="::")


def test_quoted_values_may_contain_the_separator(write_table):
    path = write_table('"New York, NY",old,down\n"Rome",new,up\n')

    assert load_table(path).cases == (("New York, NY", "old"), ("Rome", "new"))


def test_quoted_values_may_contain_whitespace(write_table):
    path = write_table('"New York" old down\nRome   "very new"  up\n')

    assert load_table(path).cases == (("New York", "old"), ("Rome", "very new"))


def test_doubled_quote_inside_quotes_is_a_literal_quote(write_table):
    path = write_table('"say ""hi""";yes\nbye;no\n')

    assert load_table(path).cases == (('say "hi"',), ("bye",))


def test_unclosed_quote_names_the_line(write_table):
    path = write_table('a;b\n"c;d\n')

    with pytest.raises(TableError, match=r"line 2: unclosed quote"):
        load_table(path, sep=";")


def test_header_names_the_columns(write_table):
    path = write_table("\nage,profit,play\nold,yes,down\nnew,no,up\n")

    table = load_table(path, header=True)

    assert table.attributes == ("age", "profit")
    assert table.decision == "play"
    assert table.cases == (("old", "yes"), ("new", "no"))


def test_without_header_the_first_line_is_a_case(write_table):
    path = write_table("age,profit,play\nold,yes,down\n")

    table = load_table(path)

    assert table.attributes == ("a1", "a2")
    assert table.cases[0] == ("age", "profit")


def test_header_without_cases_is_a_table_error(write_table):
    path = write_table("age,profit,play\n")

    with pytest.raises(TableError, match="no cases"):
        load_table(path, header=True)


def test_duplicate_header_names_are_a_table_error(write_table):
    path = write_table("age,age,play\nold,yes,down\n")

    with pytest.raises(TableError, match=r"line 1: duplicate column name 'age'"):
        load_table(path, header=True)


def test_values_are_trimmed_and_case_sensitive(write_table):
    path = write_table(" Old ; yes ;down\nold;Yes; down \n")

    table = load_table(path)

    assert table.cases == (("Old", "yes"), ("old", "Yes"))
    assert table.values(0) == ("Old", "old")
    assert table.decisions == ("down", "down")


def test_encoding_reads_a_cp1250_file_with_polish_characters(write_table):
    text = "wiek;zysk;decyzja\nstary;tak;spadek\nnowy;żółć;wzrost\n"
    path = write_table(text, encoding="cp1250")

    table = load_table(path, header=True, encoding="cp1250")

    assert table.attributes == ("wiek", "zysk")
    assert table.cases[1] == ("nowy", "żółć")


def test_cp1250_file_read_as_utf8_is_a_table_error(write_table):
    path = write_table("stary;żółć;spadek\n", encoding="cp1250")

    with pytest.raises(TableError, match="another encoding"):
        load_table(path)


def test_unknown_encoding_is_a_table_error(write_table):
    path = write_table("a;b\n")

    with pytest.raises(TableError, match="unknown encoding 'klingon'"):
        load_table(path, encoding="klingon")


def test_text_and_whole_numbers_are_discrete_fractions_are_continuous(write_table):
    path = write_table(
        "old 0 3.7 -2 1 yes\nnew 2 4 +5 1e-2 no\nmid 1 -0.5 7 2.0 yes\n",
    )

    table = load_table(path)

    assert table.kinds == (
        "discrete",
        "discrete",
        "continuous",
        "discrete",
        "continuous",
    )


def test_whole_number_written_with_a_point_stays_discrete(write_table):
    path = write_table("1.0 yes\n2.00 no\n3e2 yes\n")

    assert load_table(path).kinds == ("discrete",)


def test_fraction_mixed_with_text_is_discrete(write_table):
    path = write_table("3.7 yes\nhigh no\n")

    assert load_table(path).kinds == ("discrete",)


def test_number_words_are_text(write_table):
    path = write_table("nan yes\ninf no\n0.5 yes\n")

    assert load_table(path).kinds == ("discrete",)


def test_course_tables_are_all_discrete():
    for file in ("gielda.txt", "testowaTabDec.txt"):
        table = load_table(DATA / file)
        assert set(table.kinds) == {"discrete"}


def test_continuous_override_makes_whole_numbers_continuous(write_table):
    path = write_table("temp,hum,play\n85,85,no\n80,90,yes\n", "golf.csv")

    table = load_table(path, header=True, continuous=["temp", "hum"])

    assert table.kinds == ("continuous", "continuous")


def test_discrete_override_makes_fractions_discrete(write_table):
    path = write_table("0.5 1.5 yes\n1.5 2.5 no\n")

    table = load_table(path, discrete=["a2"])

    assert table.kinds == ("continuous", "discrete")


def test_unknown_override_column_is_a_table_error(write_table):
    path = write_table("1 2 yes\n")

    with pytest.raises(TableError, match=r"unknown column 'a9' forced continuous"):
        load_table(path, continuous=["a1", "a9"])
    with pytest.raises(TableError, match=r"unknown column 'age' forced discrete"):
        load_table(path, discrete=["age"])


def test_decision_attribute_cannot_be_overridden(write_table):
    path = write_table("1 2 yes\n")

    with pytest.raises(TableError, match="'d' is the Decision Attribute"):
        load_table(path, discrete=["d"])


def test_column_in_both_overrides_is_a_table_error(write_table):
    path = write_table("1 2 yes\n")

    with pytest.raises(TableError, match="'a1' is forced both"):
        load_table(path, discrete=["a1"], continuous=["a1"])


def test_continuous_override_on_text_names_the_line_and_value(write_table):
    path = write_table("\n1 old yes\n2 new no\n")

    with pytest.raises(TableError, match=r"line 2: a2 is continuous but 'old'"):
        load_table(path, continuous=["a2"])


@pytest.mark.parametrize(
    ("text", "cases"),
    [
        ("1,5;2,3;yes\n0,5;4,1;no\n", (("1,5", "2,3"), ("0,5", "4,1"))),
        ("1,5 2,5 tak\n3,5 1,0 nie\n", (("1,5", "2,5"), ("3,5", "1,0"))),
        ("12:30 sunny yes\n09:15 rain no\n", (("12:30", "sunny"), ("09:15", "rain"))),
    ],
)
def test_tie_on_column_count_avoids_splitting_inside_values(write_table, text, cases):
    assert load_table(write_table(text)).cases == cases


def test_quote_inside_a_value_is_a_literal_character(write_table):
    path = write_table('a 5" yes\nb 7" no\n')

    assert load_table(path).cases == (("a", '5"'), ("b", '7"'))


def test_quoted_values_keep_their_spaces(write_table):
    path = write_table('" a ",yes\n  "b"  ,no\n')

    assert load_table(path).cases == ((" a ",), ("b",))
