from pathlib import Path

import pytest

from decision_tree.table import (
    ReadOptions,
    TableError,
    load_cases,
    load_table,
    parse_columns,
    parse_markers,
)

DATA = Path(__file__).parent.parent / "data"


def test_stock_market_loads_with_default_column_names():
    table = load_table(DATA / "stock_market.txt")

    assert table.attributes == ("a1", "a2", "a3")
    assert table.decision == "d"
    assert len(table.cases) == 10
    assert table.cases[0] == ("old", "yes", "swr")
    assert table.decisions[0] == "down"


def test_decision_classes_are_in_file_order():
    assert load_table(DATA / "stock_market.txt").classes == ("down", "up")


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

    with pytest.raises(TableError, match="at least 2 columns"):
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

    table = load_table(path, ReadOptions(sep="#"))

    assert table.cases == (("old", "yes, really"), ("new", "no"))


def test_ragged_line_with_explicit_separator_names_the_line(write_table):
    path = write_table("a;b;c\n\nd;e\n")

    with pytest.raises(TableError, match=r"line 3: expected 3 columns"):
        load_table(path, ReadOptions(sep=";"))


def test_separator_must_be_one_character(write_table):
    path = write_table("a::b\n")

    with pytest.raises(TableError, match="single character"):
        load_table(path, ReadOptions(sep="::"))


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
        load_table(path, ReadOptions(sep=";"))


def test_header_names_the_columns(write_table):
    path = write_table("\nage,profit,play\nold,yes,down\nnew,no,up\n")

    table = load_table(path, ReadOptions(header=True))

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
        load_table(path, ReadOptions(header=True))


def test_duplicate_header_names_are_a_table_error(write_table):
    path = write_table("age,age,play\nold,yes,down\n")

    with pytest.raises(TableError, match=r"line 1: duplicate column name 'age'"):
        load_table(path, ReadOptions(header=True))


def test_values_are_trimmed_and_case_sensitive(write_table):
    path = write_table(" Old ; yes ;down\nold;Yes; down \n")

    table = load_table(path)

    assert table.cases == (("Old", "yes"), ("old", "Yes"))
    assert table.values(0) == ("Old", "old")
    assert table.decisions == ("down", "down")


def test_encoding_reads_a_cp1250_file_with_polish_characters(write_table):
    text = "wiek;zysk;decyzja\nstary;tak;spadek\nnowy;żółć;wzrost\n"
    path = write_table(text, encoding="cp1250")

    table = load_table(path, ReadOptions(header=True, encoding="cp1250"))

    assert table.attributes == ("wiek", "zysk")
    assert table.cases[1] == ("nowy", "żółć")


def test_cp1250_file_read_as_utf8_is_a_table_error(write_table):
    path = write_table("stary;żółć;spadek\n", encoding="cp1250")

    with pytest.raises(TableError, match="another encoding"):
        load_table(path)


def test_unknown_encoding_is_a_table_error(write_table):
    path = write_table("a;b\n")

    with pytest.raises(TableError, match="unknown encoding 'klingon'"):
        load_table(path, ReadOptions(encoding="klingon"))


def test_utf8_bom_is_not_part_of_the_first_column_name(write_table):
    path = write_table("wiek,płeć,decyzja\n20,K,tak\n30,M,nie\n", encoding="utf-8-sig")

    table = load_table(path, ReadOptions(header=True), continuous=("wiek",))

    assert table.attributes == ("wiek", "płeć")
    assert table.kinds == ("continuous", "discrete")


def test_utf8_bom_is_not_part_of_the_first_value_without_a_header(write_table):
    path = write_table("stary;tak;spadek\nnowy;nie;wzrost\n", encoding="utf-8-sig")

    table = load_table(path)

    assert table.cases[0] == ("stary", "tak")


@pytest.mark.parametrize("encoding", ["UTF8", "utf_8", "U8"])
def test_utf8_bom_is_stripped_for_any_spelling_of_utf8(write_table, encoding):
    path = write_table("wiek;decyzja\n20;tak\n", encoding="utf-8-sig")

    table = load_table(path, ReadOptions(header=True, encoding=encoding))

    assert table.attributes == ("wiek",)


def test_decode_error_in_a_utf8_bom_file_gives_the_file_byte_offset(tmp_path):
    path = tmp_path / "table.txt"
    path.write_bytes(b"\xef\xbb\xbfab;c\n\xff;x\n")

    with pytest.raises(TableError, match="at byte 8;"):
        load_table(path)


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
    for file in ("stock_market.txt", "test_table.txt"):
        table = load_table(DATA / file)
        assert set(table.kinds) == {"discrete"}


def test_continuous_override_makes_whole_numbers_continuous(write_table):
    path = write_table("temp,hum,play\n85,85,no\n80,90,yes\n", "golf.csv")

    table = load_table(path, ReadOptions(header=True), continuous=["temp", "hum"])

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


def golf_training():
    return load_table(
        DATA / "golf.txt",
        ReadOptions(header=True),
        continuous=["temperature", "humidity"],
    )


def test_like_takes_the_attribute_types_of_another_table(write_table):
    # 1 and 2 alone would be inferred discrete.
    path = write_table("sunny 1 2 false Play\n")

    table = load_table(path, like=golf_training())

    assert table.kinds == ("discrete", "continuous", "continuous", "discrete")


def test_like_with_another_column_count_is_an_error(write_table):
    path = write_table("sunny 1 Play\n")

    with pytest.raises(TableError, match="3 columns, expected 5"):
        load_table(path, like=golf_training())


def test_like_checks_continuous_values_and_names_the_line(write_table):
    path = write_table("sunny 1 2 false Play\nsunny 1 high false Play\n")

    with pytest.raises(
        TableError, match="line 2: a3 is continuous but 'high' is not a number"
    ):
        load_table(path, like=golf_training())


@pytest.mark.parametrize("override", ["discrete", "continuous"])
def test_like_cannot_be_combined_with_type_overrides(write_table, override):
    path = write_table("sunny 1 2 false Play\n")

    with pytest.raises(TableError, match="takes that table's types"):
        load_table(path, like=golf_training(), **{override: ["a1"]})


GOLF_HEADER = "outlook temperature humidity windy play\n"


@pytest.mark.parametrize(
    ("header", "column"),
    [
        ("outlook temp humidity windy play\n", "column 2 is named 'temp'"),
        ("outlook humidity temperature windy play\n", "column 2 is named 'humidity'"),
        ("outlook temperature humidity windy result\n", "column 5 is named 'result'"),
    ],
)
def test_like_with_header_names_the_first_column_named_differently(
    write_table, header, column
):
    path = write_table(header + "sunny 1 2 false Play\n")

    with pytest.raises(TableError, match=f"{column}, expected '") as error:
        load_table(path, ReadOptions(header=True), like=golf_training())
    assert "\n" not in str(error.value)


def test_like_with_matching_header_reads_the_table(write_table):
    path = write_table(GOLF_HEADER + "sunny 1 2 false Play\n")

    table = load_table(path, ReadOptions(header=True), like=golf_training())

    assert table.attributes == golf_training().attributes
    assert table.cases == (("sunny", "1", "2", "false"),)


def test_like_without_header_does_not_check_names(write_table):
    training = load_table(write_table("old up\nnew down\n", name="training.txt"))
    path = write_table("x y\nold up\n")

    table = load_table(path, like=training)

    assert table.cases == (("x",), ("old",))


@pytest.mark.parametrize(
    "text",
    [
        "outlook temp humidity windy\nsunny 1 2 false\n",
        "outlook temp humidity windy play\nsunny 1 2 false Play\n",
    ],
)
def test_cases_with_header_name_the_first_column_named_differently(write_table, text):
    path = write_table(text)

    with pytest.raises(TableError, match="column 2 is named 'temp', expected 'temp"):
        load_cases(path, ReadOptions(header=True), like=golf_training())


def test_bom_training_file_matches_a_test_set_header_without_a_bom(write_table):
    training_text = "wiek;płeć;decyzja\n20;K;tak\n30;M;nie\n"
    path = write_table(training_text, name="training.txt", encoding="utf-8-sig")
    training = load_table(path, ReadOptions(header=True))
    test_path = write_table("wiek;płeć;decyzja\n25;K;tak\n", name="test.txt")

    table = load_table(test_path, ReadOptions(header=True), like=training)
    cases = load_cases(test_path, ReadOptions(header=True), like=training)

    assert table.cases == (("25", "K"),)
    assert cases.cases == (("25", "K"),)


def test_cases_without_a_decision_column_have_no_decisions(write_table):
    path = write_table("sunny 85 ? false\nfoggy 70 96 true\n")

    cases = load_cases(path, like=golf_training())

    assert cases.cases == (
        ("sunny", "85", None, "false"),
        ("foggy", "70", "96", "true"),
    )
    assert cases.decisions is None


def test_cases_with_a_decision_column_keep_their_decisions(write_table):
    path = write_table("sunny 85 85 false Play\nrain 70 96 false Don't\n")

    cases = load_cases(path, like=golf_training())

    assert cases.cases == (
        ("sunny", "85", "85", "false"),
        ("rain", "70", "96", "false"),
    )
    assert cases.decisions == ("Play", "Don't")


def test_cases_take_a_header_and_loading_options(write_table):
    text = "outlook;temperature;humidity;windy\nsłońce;85;-;false\n"
    path = write_table(text, encoding="cp1250")

    options = ReadOptions(sep=";", header=True, encoding="cp1250", missing=("-",))

    cases = load_cases(path, options, like=golf_training())

    assert cases.cases == (("słońce", "85", None, "false"),)


def test_cases_of_a_one_attribute_table_may_have_one_column(write_table):
    training = load_table(write_table("old up\nnew down\n", name="training.txt"))

    cases = load_cases(write_table("old\nmid\n"), like=training)

    assert cases.cases == (("old",), ("mid",))
    assert cases.decisions is None


@pytest.mark.parametrize("text", ["sunny 85 Play\n", "sunny 85 85 false Play x\n"])
def test_cases_with_another_column_count_are_an_error(write_table, text):
    with pytest.raises(TableError, match="expected 4 .without a decision. or 5"):
        load_cases(write_table(text), like=golf_training())


def test_cases_check_continuous_values_and_name_the_line(write_table):
    path = write_table("sunny 1 2 false\nsunny 1 high false\n")

    with pytest.raises(
        TableError, match="line 2: humidity is continuous but 'high' is not a number"
    ):
        load_cases(path, like=golf_training())


def test_cases_with_a_missing_decision_name_the_line(write_table):
    path = write_table("sunny 1 2 false Play\nsunny 1 2 false ?\n")

    with pytest.raises(TableError, match="line 2: the decision is missing"):
        load_cases(path, like=golf_training())


def test_default_missing_markers_are_question_mark_na_and_empty_cell(write_table):
    path = write_table("?,x,yes\nNA,,no\nold,y,no\n")

    table = load_table(path)

    assert table.cases == ((None, "x"), (None, None), ("old", "y"))


def test_missing_markers_are_configurable(write_table):
    path = write_table("-,?,yes\nold,,no\n")

    table = load_table(path, ReadOptions(missing=("-",)))

    assert table.cases == ((None, "?"), ("old", ""))


def test_missing_decision_names_the_line(write_table):
    path = write_table("old yes up\n\nnew no ?\n")

    with pytest.raises(TableError, match="line 3: the decision is missing"):
        load_table(path)


def test_type_inference_ignores_missing_values(write_table):
    path = write_table("? 1 yes\n3.5 ? no\n2 NA no\n")

    table = load_table(path)

    assert table.kinds == ("continuous", "discrete")


def test_missing_values_pass_the_continuous_check(write_table):
    path = write_table("1 yes\n? no\n")

    table = load_table(path, continuous=["a1"])

    assert table.cases == (("1",), (None,))


def test_known_values_skip_missing_ones(write_table):
    path = write_table("? yes\nold no\n? no\nnew yes\n")

    assert load_table(path).values(0) == ("old", "new")


def test_override_lists_are_comma_separated():
    assert parse_columns("a1, a3") == ("a1", "a3")
    assert parse_columns(" a1 ,, ") == ("a1",)
    assert parse_columns("") == ()


def test_missing_option_is_a_comma_list_where_an_empty_item_is_the_empty_cell(
    write_table,
):
    path = write_table("old,-,up\n,?,down\nnew,yes,up\n")

    assert parse_markers("-") == ("-",)
    assert parse_markers("?, NA,") == ("?", "NA", "")
    only_dash = load_table(path, ReadOptions(missing=parse_markers("-")))
    dash_and_empty = load_table(path, ReadOptions(missing=parse_markers("-,")))
    assert only_dash.cases == (("old", None), ("", "?"), ("new", "yes"))
    assert dash_and_empty.cases == (("old", None), (None, "?"), ("new", "yes"))
