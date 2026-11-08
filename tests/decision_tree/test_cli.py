from pathlib import Path

import pytest
from typer.testing import CliRunner

from decision_tree.cli import app

DATA = Path(__file__).parent.parent / "data"
STOCK_MARKET = str(DATA / "stock_market.txt")
STOCK_MARKET_TREE = """\
a1 = old: down (3.0)
a1 = mid:
|   a2 = yes: down (2.0)
|   a2 = no: up (2.0)
a1 = new: up (3.0)
"""
STOCK_MARKET_MATRIX = (
    "  actual \\ predicted  down  up\n"
    "  down                   5   0\n"
    "  up                     0   5\n"
)
# Pruning at 25% keeps the whole stock_market tree.
STOCK_MARKET_OUTPUT = (
    "Unpruned tree (size 6):\n" + STOCK_MARKET_TREE + "\n"
    "Pruned tree (size 6, Confidence Level 25%):\n" + STOCK_MARKET_TREE + "\n"
    "Unpruned tree on Training Set: 10 cases, 0 errors (0.0%)\n"
    + STOCK_MARKET_MATRIX
    + "\n"
    "Pruned tree on Training Set: 10 cases, 0 errors (0.0%)\n" + STOCK_MARKET_MATRIX
)
GOLF_ARGS = [str(DATA / "golf.txt"), "--header", "--continuous", "temperature,humidity"]
GOLF_TREE = """\
outlook = overcast: Play (4.0)
outlook = sunny:
|   humidity <= 75 : Play (2.0)
|   humidity > 75 : Don't Play (3.0)
outlook = rain:
|   windy = true: Don't Play (2.0)
|   windy = false: Play (3.0)
"""
STOCK_MARKET_STATS = """\
a1 (discrete): 3 values, 0 missing
  old  3
  mid  4
  new  3

a2 (discrete): 2 values, 0 missing
  yes  4
  no   6

a3 (discrete): 2 values, 0 missing
  swr  6
  hwr  4

d (decision): 2 classes
  down  5
  up    5
"""

runner = CliRunner()
TTY = {"TTY_COMPATIBLE": "1"}


def unpruned_tree(stdout):
    """The text of the first tree `train` prints, without its heading."""
    return stdout.split("\n", 1)[1]


def test_train_prints_plain_text_when_piped():
    result = runner.invoke(app, ["train", STOCK_MARKET])

    assert result.exit_code == 0
    assert result.stdout == STOCK_MARKET_OUTPUT


def test_train_plain_flag_forces_plain_text_in_a_terminal():
    result = runner.invoke(
        app, ["train", STOCK_MARKET, "--plain"], env={"TTY_COMPATIBLE": "1"}
    )

    assert result.exit_code == 0
    assert result.stdout == STOCK_MARKET_OUTPUT


def test_train_draws_a_rich_tree_in_a_terminal():
    result = runner.invoke(app, ["train", STOCK_MARKET], env={"TTY_COMPATIBLE": "1"})

    assert result.exit_code == 0
    assert "├──" in result.stdout
    assert "|   " not in result.stdout


def test_train_golf_prints_thresholds_like_c45():
    result = runner.invoke(app, ["train", *GOLF_ARGS])

    assert result.exit_code == 0
    assert unpruned_tree(result.stdout).startswith(GOLF_TREE + "\n")


def test_rich_tree_shows_thresholds_with_math_signs():
    result = runner.invoke(app, ["train", *GOLF_ARGS], env={"TTY_COMPATIBLE": "1"})

    assert result.exit_code == 0
    assert "humidity ≤ 75" in result.stdout
    assert "humidity > 75" in result.stdout


def test_missing_file_gives_one_line_error():
    result = runner.invoke(app, ["train", "no-such-file.txt"])

    assert result.exit_code == 1
    assert result.stderr.strip().count("\n") == 0
    assert "no-such-file.txt" in result.stderr
    assert "Traceback" not in result.output


def test_wrong_column_count_gives_one_line_error_naming_the_line(write_table):
    path = write_table("a b c\nd e\n")

    result = runner.invoke(app, ["train", str(path)])

    assert result.exit_code == 1
    assert result.stderr.strip().count("\n") == 0
    assert "line 2" in result.stderr


def test_stats_prints_plain_blocks_when_piped():
    result = runner.invoke(app, ["stats", STOCK_MARKET])

    assert result.exit_code == 0
    assert result.stdout == STOCK_MARKET_STATS


def test_stats_plain_flag_forces_plain_text_in_a_terminal():
    result = runner.invoke(
        app, ["stats", STOCK_MARKET, "--plain"], env={"TTY_COMPATIBLE": "1"}
    )

    assert result.exit_code == 0
    assert result.stdout == STOCK_MARKET_STATS


def test_stats_draws_rich_tables_in_a_terminal():
    result = runner.invoke(app, ["stats", STOCK_MARKET], env={"TTY_COMPATIBLE": "1"})

    assert result.exit_code == 0
    assert "│" in result.stdout
    for title in ("a1 (discrete)", "a3 (discrete)", "d (decision)"):
        assert title in result.stdout
    assert "  old  3" not in result.stdout


def test_stats_missing_file_gives_one_line_error():
    result = runner.invoke(app, ["stats", "no-such-file.txt"])

    assert result.exit_code == 1
    assert result.stderr.strip().count("\n") == 0
    assert "no-such-file.txt" in result.stderr


GOLF = "outlook|temp|play\nsłońce|85|nie\nsłońce|80|nie\ndeszcz|70|tak\n"


def test_stats_shows_continuous_attributes(write_table):
    path = write_table("old,3.5,yes\nnew,4,no\n")

    result = runner.invoke(app, ["stats", str(path)])

    assert result.exit_code == 0
    assert "a1 (discrete): 2 values" in result.stdout
    assert "a2 (continuous): 2 values" in result.stdout


def test_stats_takes_every_loading_option(write_table):
    file = str(write_table(GOLF, encoding="cp1250"))

    result = runner.invoke(
        app,
        [
            "stats",
            file,
            "--header",
            "--sep",
            "|",
            "--encoding",
            "cp1250",
            "--continuous",
            "temp",
            "--discrete",
            "outlook",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "outlook (discrete): 2 values" in result.stdout
    assert "temp (continuous): 3 values" in result.stdout
    assert "play (decision): 2 classes" in result.stdout
    assert "słońce" in result.stdout


def test_override_lists_are_comma_separated(write_table):
    path = write_table("1 2 3 yes\n4 5 6 no\n")

    result = runner.invoke(app, ["stats", str(path), "--continuous", "a1, a3"])

    assert result.exit_code == 0, result.output
    assert "a1 (continuous)" in result.stdout
    assert "a2 (discrete)" in result.stdout
    assert "a3 (continuous)" in result.stdout


def test_train_takes_loading_options(write_table):
    # -m 1: the 2 : 1 split would be blocked by the default -m 2.
    result = runner.invoke(
        app, ["train", str(write_table(GOLF)), "--header", "-m", "1"]
    )

    assert result.exit_code == 0, result.output
    assert unpruned_tree(result.stdout).startswith(
        "outlook = słońce: nie (2.0)\noutlook = deszcz: tak (1.0)\n\n"
    )


def test_unknown_override_column_gives_one_line_error(write_table):
    result = runner.invoke(
        app, ["train", str(write_table(GOLF)), "--header", "--continuous", "wind"]
    )

    assert result.exit_code == 1
    assert result.stderr.strip().count("\n") == 0
    assert "unknown column 'wind'" in result.stderr


TEST_TABLE = str(DATA / "test_table.txt")


def test_train_default_uses_gain_ratio_and_m_2():
    result = runner.invoke(app, ["train", TEST_TABLE])

    assert result.exit_code == 0, result.output
    assert unpruned_tree(result.stdout).startswith(
        "a3 = 0:\n|   a4 = 1:\n|   |   a2 = 1:"
    )


def test_train_m_option_sets_the_minimum_branch_size():
    result = runner.invoke(app, ["train", TEST_TABLE, "-m", "1"])

    assert result.exit_code == 0, result.output
    assert unpruned_tree(result.stdout).startswith(
        "a3 = 0:\n|   a4 = 1:\n|   |   a1 = 0:"
    )


def test_train_criterion_gain_ranks_tests_by_gain():
    result = runner.invoke(app, ["train", TEST_TABLE, "--criterion", "gain"])

    assert result.exit_code == 0, result.output
    assert unpruned_tree(result.stdout).startswith("a1 = ")


def test_train_criterion_gain_ratio_can_be_named():
    result = runner.invoke(app, ["train", TEST_TABLE, "--criterion", "gain-ratio"])

    assert result.exit_code == 0, result.output
    assert unpruned_tree(result.stdout).startswith("a3 = ")


def test_train_quinlan_gain_filter_changes_the_root():
    result = runner.invoke(app, ["train", TEST_TABLE, "--quinlan-gain-filter"])

    assert result.exit_code == 0, result.output
    assert unpruned_tree(result.stdout).startswith("a1 = ")


@pytest.mark.parametrize(
    "option", [["--criterion", "entropy"], ["-m", "0"], ["-m", "two"]]
)
def test_train_rejects_bad_split_options(option):
    result = runner.invoke(app, ["train", TEST_TABLE, *option])

    assert result.exit_code == 2


# Case 1 is a training case with its decision flipped: one error.
STOCK_MARKET_TEST = "old yes swr up\nnew no hwr up\n"


def test_train_test_reports_training_and_test_set(write_table):
    test = str(write_table(STOCK_MARKET_TEST, name="test.txt"))

    result = runner.invoke(app, ["train", STOCK_MARKET, "--test", test])

    assert result.exit_code == 0, result.output
    test_matrix = (
        "  actual \\ predicted  down  up\n"
        "  down                   0   0\n"
        "  up                     1   1\n"
    )
    assert result.stdout == (
        STOCK_MARKET_OUTPUT + "\n"
        f"Unpruned tree on Test Set ({test}): 2 cases, 1 error (50.0%)\n"
        + test_matrix
        + "\n"
        f"Pruned tree on Test Set ({test}): 2 cases, 1 error (50.0%)\n" + test_matrix
    )


def test_train_draws_rich_confusion_matrices_in_a_terminal(write_table):
    test = str(write_table(STOCK_MARKET_TEST, name="test.txt"))

    result = runner.invoke(
        app, ["train", STOCK_MARKET, "--test", test], env={"TTY_COMPATIBLE": "1"}
    )

    assert result.exit_code == 0, result.output
    assert "Training Set: 10 cases, 0 errors (0.0%)" in result.stdout
    assert "2 cases, 1 error (50.0%)" in result.stdout
    assert "│" in result.stdout
    assert "  down                   5   0" not in result.stdout


GOLF_SPLIT = ["train", *GOLF_ARGS, "--split", "0.7"]


def test_train_split_with_a_seed_gives_the_same_split_every_run():
    first = runner.invoke(app, [*GOLF_SPLIT, "--seed", "42"])
    second = runner.invoke(app, [*GOLF_SPLIT, "--seed", "42"])

    assert first.exit_code == 0, first.output
    assert first.stdout == second.stdout
    assert "Training Set: 10 cases" in first.stdout
    assert "Test Set (hold-out, --seed 42): 4 cases" in first.stdout


def test_train_split_without_a_seed_prints_the_seed_it_used():
    result = runner.invoke(app, GOLF_SPLIT)

    assert result.exit_code == 0, result.output
    seed = result.stdout.split("Test Set (hold-out, --seed ")[1].split(")")[0]
    again = runner.invoke(app, [*GOLF_SPLIT, "--seed", seed])
    assert again.stdout == result.stdout


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--test", STOCK_MARKET, "--split", "0.7"], "--test and --split"),
        (["--seed", "42"], "--seed needs --split"),
        (["--split", "0.01"], "no Training Set"),
    ],
)
def test_train_bad_evaluation_options_give_one_line_error(options, message):
    result = runner.invoke(app, ["train", *GOLF_ARGS, *options])

    assert result.exit_code == 1
    assert result.stderr.strip().count("\n") == 0
    assert message in result.stderr


@pytest.mark.parametrize("ratio", ["0", "1", "1.5"])
def test_train_rejects_a_split_ratio_outside_0_1(ratio):
    result = runner.invoke(app, ["train", STOCK_MARKET, "--split", ratio])

    assert result.exit_code == 2


def test_test_set_with_another_column_count_gives_one_line_error(write_table):
    test = str(write_table("old yes up\n", name="test.txt"))

    result = runner.invoke(app, ["train", STOCK_MARKET, "--test", test])

    assert result.exit_code == 1
    assert result.stderr.strip().count("\n") == 0
    assert "3 columns, expected 4" in result.stderr


def test_test_case_with_an_unseen_value_is_classified_as_missing(write_table):
    # foggy: old 3/10 down, mid 4/10 and new 3/10 up, so up is right.
    test = str(write_table("old yes swr up\nfoggy no hwr up\n", name="test.txt"))

    result = runner.invoke(app, ["train", STOCK_MARKET, "--test", test])

    assert result.exit_code == 0
    assert "2 cases, 1 error (50.0%)" in result.stdout


def test_test_case_with_text_in_a_continuous_column_gives_a_clear_error(write_table):
    test = str(
        write_table(
            "outlook,temperature,humidity,windy,play\nsunny,85,high,false,Play\n",
            name="test.txt",
        )
    )

    result = runner.invoke(app, ["train", *GOLF_ARGS, "--test", test])

    assert result.exit_code == 1
    assert "line 2: humidity is continuous but 'high' is not a number" in result.stderr


def test_stats_reports_missing_values_with_the_default_markers(write_table):
    path = write_table("old,?,up\n,NA,down\nnew,yes,up\n")

    result = runner.invoke(app, ["stats", str(path)])

    assert result.exit_code == 0, result.output
    assert "a1 (discrete): 2 values, 1 missing" in result.stdout
    assert "a2 (discrete): 1 value, 2 missing" in result.stdout


def test_missing_option_is_a_comma_list_where_an_empty_item_is_the_empty_cell(
    write_table,
):
    path = write_table("old,-,up\n,?,down\nnew,yes,up\n")

    only_dash = runner.invoke(app, ["stats", str(path), "--missing", "-"])
    dash_and_empty = runner.invoke(app, ["stats", str(path), "--missing", "-,"])

    assert only_dash.exit_code == 0, only_dash.output
    assert "a1 (discrete): 3 values, 0 missing" in only_dash.stdout
    assert "a2 (discrete): 2 values, 1 missing" in only_dash.stdout
    assert "a1 (discrete): 2 values, 1 missing" in dash_and_empty.stdout


def test_train_with_missing_values_shows_fractional_leaf_counts(write_table):
    text = (DATA / "golf.txt").read_text().replace("overcast, 72, 90", "?, 72, 90")
    args = [str(write_table(text)), *GOLF_ARGS[1:], "--missing", "?"]

    result = runner.invoke(app, ["train", *args])

    assert result.exit_code == 0, result.output
    assert "outlook = overcast: Play (3.2)\n" in result.stdout
    assert "|   windy = true: Don't Play (2.4/0.4)\n" in result.stdout


def test_missing_decision_gives_one_line_error_naming_the_line(write_table):
    result = runner.invoke(app, ["train", str(write_table("old up\nnew ?\n"))])

    assert result.exit_code == 1
    assert result.stderr.strip().count("\n") == 0
    assert "line 2: the decision is missing" in result.stderr


GOLF_CASES = """\
outlook,temperature,humidity,windy
sunny,85,70,false
rain,70,?,true
foggy,72,90,true
"""
# foggy is classified as if outlook were missing: overcast 4/14 Play, sunny
# and rain 10/14 Don't Play.
GOLF_CLASSIFIED = """\
case  predicted    Play  Don't Play
   1  Play        1.000       0.000
   2  Don't Play  0.000       1.000
   3  Don't Play  0.286       0.714
"""


def test_classify_cases_without_decisions_prints_class_and_distribution(write_table):
    cases = str(write_table(GOLF_CASES, name="cases.txt"))

    result = runner.invoke(app, ["classify", *GOLF_ARGS, cases])

    assert result.exit_code == 0
    assert result.stdout == GOLF_CLASSIFIED


def test_classify_cases_with_decisions_marks_errors_and_reports_accuracy(write_table):
    cases = str(write_table("old yes swr down\nnew no swr down\n", name="cases.txt"))

    result = runner.invoke(app, ["classify", STOCK_MARKET, cases])

    assert result.exit_code == 0
    assert result.stdout == (
        "case  predicted   down     up  actual\n"
        "   1  down       1.000  0.000  down\n"
        "   2  up         0.000  1.000  down    wrong\n"
        "\n"
        "Accuracy: 1 of 2 cases (50.0%)\n"
    )


@pytest.mark.parametrize("text", ["old yes\n", "old yes swr down x\n"])
def test_classify_cases_with_another_column_count_give_one_line_error(
    write_table, text
):
    cases = str(write_table(text, name="cases.txt"))

    result = runner.invoke(app, ["classify", STOCK_MARKET, cases])

    assert result.exit_code == 1
    assert result.stderr.strip().count("\n") == 0
    assert "expected 3 (without a decision) or 4" in result.stderr


def test_classify_plain_flag_forces_plain_text_in_a_terminal(write_table):
    cases = str(write_table(GOLF_CASES, name="cases.txt"))

    result = runner.invoke(
        app,
        ["classify", *GOLF_ARGS, cases, "--plain"],
        env={"TTY_COMPATIBLE": "1"},
    )

    assert result.exit_code == 0
    assert result.stdout == GOLF_CLASSIFIED


def test_classify_draws_a_rich_table_in_a_terminal(write_table):
    cases = str(write_table("old yes swr down\nnew no swr down\n", name="cases.txt"))

    result = runner.invoke(
        app, ["classify", STOCK_MARKET, cases], env={"TTY_COMPATIBLE": "1"}
    )

    assert result.exit_code == 0
    assert "┃" in result.stdout
    assert result.stdout.count("wrong") == 1
    assert "Accuracy: 1 of 2 cases (50.0%)" in result.stdout


def test_classify_takes_the_build_options(write_table):
    # With gain, stock_market's root is still a1, but -m 5 leaves the tree one Leaf:
    # every case gets down 0.5 / up 0.5, predicted down (first in file order).
    cases = str(write_table("new no hwr\n", name="cases.txt"))

    result = runner.invoke(
        app, ["classify", STOCK_MARKET, cases, "--criterion", "gain", "-m", "5"]
    )

    assert result.exit_code == 0
    assert "   1  down       0.500  0.500" in result.stdout


def test_train_no_prune_shows_only_the_unpruned_tree():
    result = runner.invoke(app, ["train", STOCK_MARKET, "--no-prune"])

    assert result.exit_code == 0, result.output
    assert result.stdout == (
        "Unpruned tree (size 6):\n" + STOCK_MARKET_TREE + "\n"
        "Unpruned tree on Training Set: 10 cases, 0 errors (0.0%)\n"
        + STOCK_MARKET_MATRIX
    )


def test_train_c_option_sets_the_confidence_level():
    # At 5% golf's estimates make the whole tree one Leaf; at 25% it stays.
    result = runner.invoke(app, ["train", *GOLF_ARGS, "-c", "5"])

    assert result.exit_code == 0, result.output
    assert "Pruned tree (size 1, Confidence Level 5%):\nPlay (14.0/5.0)\n" in (
        result.stdout
    )
    assert "Pruned tree on Training Set: 14 cases, 5 errors (35.7%)" in result.stdout


def test_train_confidence_level_may_be_fractional():
    result = runner.invoke(app, ["train", STOCK_MARKET, "--confidence", "12.5"])

    assert result.exit_code == 0, result.output
    assert "Pruned tree (size 6, Confidence Level 12.5%):" in result.stdout


@pytest.mark.parametrize("value", ["0", "-5", "101", "high"])
def test_train_rejects_a_confidence_level_outside_0_100(value):
    result = runner.invoke(app, ["train", STOCK_MARKET, "-c", value])

    assert result.exit_code == 2


def test_train_draws_both_rich_trees_in_a_terminal():
    result = runner.invoke(app, ["train", *GOLF_ARGS, "-c", "5"], env=TTY)

    assert result.exit_code == 0, result.output
    assert "Unpruned tree (size 8)" in result.stdout
    assert "Pruned tree (size 1, Confidence Level 5%)" in result.stdout
    assert "(14.0/5.0)" in result.stdout
    assert "Pruned tree on Training Set: 14 cases, 5 errors (35.7%)" in result.stdout


def test_train_rich_no_prune_draws_one_tree():
    result = runner.invoke(app, ["train", STOCK_MARKET, "--no-prune"], env=TTY)

    assert result.exit_code == 0, result.output
    assert "Unpruned tree (size 6)" in result.stdout
    assert "Pruned tree" not in result.stdout


@pytest.mark.parametrize(
    ("options", "tree_file"),
    [(["-m", "1"], "test_table-m1.tree.txt"), ([], "test_table-m2.tree.txt")],
    ids=["course mode", "-m 2"],
)
def test_train_no_prune_prints_the_test_table_reference_tree(options, tree_file):
    # docs/assignment.md's derived trees; see test_builder.py.
    expected = (DATA / tree_file).read_text()

    result = runner.invoke(app, ["train", TEST_TABLE, *options, "--no-prune"])

    assert result.exit_code == 0, result.output
    assert unpruned_tree(result.stdout).startswith(expected + "\n")
