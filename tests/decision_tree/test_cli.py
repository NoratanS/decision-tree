from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from decision_tree.cli import app, print_rich
from decision_tree.render import Cell, Column, TableBlock

DATA = Path(__file__).parent.parent / "data"
STOCK_MARKET = str(DATA / "stock_market.txt")
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


@pytest.mark.parametrize(
    ("options", "env"), [([], {}), (["--plain"], TTY)], ids=["piped", "--plain"]
)
def test_train_prints_the_plain_report(options, env):
    # The report's text is tested in test_render.py.
    result = runner.invoke(app, ["train", STOCK_MARKET, *options], env=env)

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("Unpruned tree (size 6):\na1 = old: down (3.0)\n")
    assert result.stdout.endswith("\n  up                     0   5\n")
    assert "├──" not in result.stdout


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


@pytest.mark.parametrize(
    ("flags", "root"),
    [
        ([], "a3 = 0:\n|   a4 = 1:\n|   |   a2 = 1:"),
        (["-m", "1"], "a3 = 0:\n|   a4 = 1:\n|   |   a1 = 0:"),
        (["--criterion", "gain"], "a1 = "),
        (["--criterion", "gain-ratio"], "a3 = "),
        (["--quinlan-gain-filter"], "a1 = "),
    ],
    ids=["default", "m-1", "gain", "gain-ratio", "quinlan-gain-filter"],
)
def test_train_build_flags_shape_the_tree(flags, root):
    # Default: gain-ratio and -m 2.
    result = runner.invoke(app, ["train", TEST_TABLE, *flags])

    assert result.exit_code == 0, result.output
    assert unpruned_tree(result.stdout).startswith(root)


@pytest.mark.parametrize(
    "option", [["--criterion", "entropy"], ["-m", "0"], ["-m", "two"]]
)
def test_train_rejects_bad_split_options(option):
    result = runner.invoke(app, ["train", TEST_TABLE, *option])

    assert result.exit_code == 2


# Case 1 is a training case with its decision flipped: one error.
STOCK_MARKET_TEST = "old yes swr up\nnew no hwr up\n"


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
def test_train_split_ratio_outside_0_1_gives_one_line_error(ratio):
    result = runner.invoke(app, ["train", STOCK_MARKET, "--split", ratio])

    assert result.exit_code == 1
    assert result.stderr.strip().count("\n") == 0
    assert "--split: the ratio must be between 0 and 1" in result.stderr


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


def test_classify_help_says_the_tree_is_unpruned():
    result = runner.invoke(app, ["classify", "--help"])

    assert result.exit_code == 0
    assert "Build an unpruned tree from TRAIN" in result.stdout


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


def test_rich_adapter_maps_cell_roles_to_colours():
    table = TableBlock(
        "Heading",
        (Column("name"), Column("count", "right")),
        (
            (Cell("plain"), Cell("1", "correct")),
            (Cell("[x]"), Cell("2", "wrong")),
            (Cell("z"), Cell("0", "empty")),
        ),
        footer="Footer",
    )
    out = Console(record=True, force_terminal=True, color_system="standard", width=40)

    print_rich(out, [table])

    text = out.export_text(styles=True)
    assert "\x1b[1mHeading\x1b[0m" in text
    assert "\x1b[1mFooter\x1b[0m" in text
    assert "\x1b[32m1\x1b[0m" in text  # correct: green
    assert "\x1b[31m2\x1b[0m" in text  # wrong: red
    assert "\x1b[2m0\x1b[0m" in text  # empty: dim
    assert "│ plain │" in text and "│ [x]   │" in text  # plain cells, escaped


CLASS_FIRST = "p,x,1\ne,y,1\np,x,2\ne,y,2\n"


def test_stats_prints_a_picked_decision_last_as_d(write_table):
    result = runner.invoke(
        app, ["stats", str(write_table(CLASS_FIRST)), "--decision", "a1"]
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("a2 (discrete): 2 values")
    assert "\na3 (discrete): 2 values" in result.stdout
    assert result.stdout.endswith("d (decision): 2 classes\n  p  2\n  e  2\n")


def test_train_and_classify_use_the_picked_decision(write_table):
    training = str(write_table(CLASS_FIRST, name="training.txt"))
    cases = str(write_table("x,2\ny,1\n", name="cases.txt"))

    trained = runner.invoke(app, ["train", training, "--decision", "a1", "-m", "1"])
    classified = runner.invoke(
        app, ["classify", training, cases, "--decision", "a1", "-m", "1"]
    )

    assert trained.exit_code == 0, trained.output
    assert unpruned_tree(trained.stdout).startswith("a2 = x: p (2.0)\na2 = y: e (2.0)")
    assert classified.exit_code == 0, classified.output
    predicted = [line.split()[1] for line in classified.stdout.splitlines()[1:]]
    assert predicted == ["p", "e"]


def test_unknown_decision_gives_one_line_error_listing_the_columns(write_table):
    result = runner.invoke(
        app, ["stats", str(write_table(CLASS_FIRST)), "--decision", "class"]
    )

    assert result.exit_code == 1
    assert result.stderr.strip().count("\n") == 0
    assert "unknown decision column 'class' (columns: a1, a2, a3)" in result.stderr


def test_forcing_the_picked_decision_gives_one_line_error(write_table):
    path = str(write_table(CLASS_FIRST))
    result = runner.invoke(
        app, ["stats", path, "--decision", "a1", "--continuous", "a1"]
    )

    assert result.exit_code == 1
    assert result.stderr.strip().count("\n") == 0
    assert "'a1' is the Decision Attribute" in result.stderr


def test_train_tests_a_test_set_with_the_picked_decision(write_table):
    training = str(write_table(CLASS_FIRST, name="training.txt"))
    test = str(write_table("p,x,1\ne,x,2\n", name="test.txt"))

    result = runner.invoke(
        app, ["train", training, "--decision", "a1", "-m", "1", "--test", test]
    )

    assert result.exit_code == 0, result.output
    assert "2 cases, 1 error (50.0%)" in result.stdout


def test_header_decision_in_the_middle_is_picked_by_name(write_table):
    training = str(
        write_table("id,Survived,sex\n1,yes,m\n2,no,f\n3,yes,m\n", name="t.csv")
    )
    cases = str(write_table("id,sex\n4,f\n5,m\n", name="cases.csv"))

    stats = runner.invoke(
        app, ["stats", training, "--header", "--decision", "Survived"]
    )
    classified = runner.invoke(
        app,
        ["classify", training, cases, "--header", "--decision", "Survived", "-m", "1"],
    )

    assert stats.exit_code == 0, stats.output
    assert stats.stdout.startswith("id (")
    assert "\nSurvived (decision): 2 classes" in stats.stdout
    assert classified.exit_code == 0, classified.output
    predicted = [line.split()[1] for line in classified.stdout.splitlines()[1:]]
    assert predicted == ["no", "yes"]
