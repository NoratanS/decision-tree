from pathlib import Path

import pytest
from typer.testing import CliRunner

from decision_tree.cli import app

DATA = Path(__file__).parent.parent / "data"
GIELDA = str(DATA / "gielda.txt")
GIELDA_TREE = """\
a1 = old: down (3.0)
a1 = mid:
|   a2 = yes: down (2.0)
|   a2 = no: up (2.0)
a1 = new: up (3.0)
"""
GIELDA_TRAINING = (
    "Training Set: 10 cases, 0 errors (0.0%)\n"
    "  actual \\ predicted  down  up\n"
    "  down                   5   0\n"
    "  up                     0   5\n"
)
GIELDA_OUTPUT = GIELDA_TREE + "\n" + GIELDA_TRAINING
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
GIELDA_STATS = """\
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


def test_train_prints_plain_text_when_piped():
    result = runner.invoke(app, ["train", GIELDA])

    assert result.exit_code == 0
    assert result.stdout == GIELDA_OUTPUT


def test_train_plain_flag_forces_plain_text_in_a_terminal():
    result = runner.invoke(
        app, ["train", GIELDA, "--plain"], env={"TTY_COMPATIBLE": "1"}
    )

    assert result.exit_code == 0
    assert result.stdout == GIELDA_OUTPUT


def test_train_draws_a_rich_tree_in_a_terminal():
    result = runner.invoke(app, ["train", GIELDA], env={"TTY_COMPATIBLE": "1"})

    assert result.exit_code == 0
    assert "├──" in result.stdout
    assert "|   " not in result.stdout


def test_train_golf_prints_thresholds_like_c45():
    result = runner.invoke(app, ["train", *GOLF_ARGS])

    assert result.exit_code == 0
    assert result.stdout.startswith(GOLF_TREE + "\n")


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
    result = runner.invoke(app, ["stats", GIELDA])

    assert result.exit_code == 0
    assert result.stdout == GIELDA_STATS


def test_stats_plain_flag_forces_plain_text_in_a_terminal():
    result = runner.invoke(
        app, ["stats", GIELDA, "--plain"], env={"TTY_COMPATIBLE": "1"}
    )

    assert result.exit_code == 0
    assert result.stdout == GIELDA_STATS


def test_stats_draws_rich_tables_in_a_terminal():
    result = runner.invoke(app, ["stats", GIELDA], env={"TTY_COMPATIBLE": "1"})

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
    assert result.stdout.startswith(
        "outlook = słońce: nie (2.0)\noutlook = deszcz: tak (1.0)\n\n"
    )


def test_unknown_override_column_gives_one_line_error(write_table):
    result = runner.invoke(
        app, ["train", str(write_table(GOLF)), "--header", "--continuous", "wind"]
    )

    assert result.exit_code == 1
    assert result.stderr.strip().count("\n") == 0
    assert "unknown column 'wind'" in result.stderr


TESTOWA = str(DATA / "testowaTabDec.txt")


def test_train_default_uses_gain_ratio_and_m_2():
    result = runner.invoke(app, ["train", TESTOWA])

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("a3 = 0:\n|   a4 = 1:\n|   |   a2 = 1:")


def test_train_m_option_sets_the_minimum_branch_size():
    result = runner.invoke(app, ["train", TESTOWA, "-m", "1"])

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("a3 = 0:\n|   a4 = 1:\n|   |   a1 = 0:")


def test_train_criterion_gain_ranks_tests_by_gain():
    result = runner.invoke(app, ["train", TESTOWA, "--criterion", "gain"])

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("a1 = ")


def test_train_criterion_gain_ratio_can_be_named():
    result = runner.invoke(app, ["train", TESTOWA, "--criterion", "gain-ratio"])

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("a3 = ")


def test_train_quinlan_gain_filter_changes_the_root():
    result = runner.invoke(app, ["train", TESTOWA, "--quinlan-gain-filter"])

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("a1 = ")


@pytest.mark.parametrize(
    "option", [["--criterion", "entropy"], ["-m", "0"], ["-m", "two"]]
)
def test_train_rejects_bad_split_options(option):
    result = runner.invoke(app, ["train", TESTOWA, *option])

    assert result.exit_code == 2


# Case 1 is a training case with its decision flipped: one error.
GIELDA_TEST = "old yes swr up\nnew no hwr up\n"


def test_train_test_reports_training_and_test_set(write_table):
    test = str(write_table(GIELDA_TEST, name="test.txt"))

    result = runner.invoke(app, ["train", GIELDA, "--test", test])

    assert result.exit_code == 0, result.output
    assert result.stdout == (
        GIELDA_OUTPUT + "\n"
        f"Test Set ({test}): 2 cases, 1 error (50.0%)\n"
        "  actual \\ predicted  down  up\n"
        "  down                   0   0\n"
        "  up                     1   1\n"
    )


def test_train_draws_rich_confusion_matrices_in_a_terminal(write_table):
    test = str(write_table(GIELDA_TEST, name="test.txt"))

    result = runner.invoke(
        app, ["train", GIELDA, "--test", test], env={"TTY_COMPATIBLE": "1"}
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
        (["--test", GIELDA, "--split", "0.7"], "--test and --split"),
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
    result = runner.invoke(app, ["train", GIELDA, "--split", ratio])

    assert result.exit_code == 2


def test_test_set_with_another_column_count_gives_one_line_error(write_table):
    test = str(write_table("old yes up\n", name="test.txt"))

    result = runner.invoke(app, ["train", GIELDA, "--test", test])

    assert result.exit_code == 1
    assert result.stderr.strip().count("\n") == 0
    assert "3 columns, expected 4" in result.stderr


# Interim: ticket 08 classifies an unseen value as if it were missing (R7).
def test_test_case_the_tree_cannot_classify_gives_one_line_error(write_table):
    test = str(write_table("old yes swr up\nfoggy no hwr up\n", name="test.txt"))

    result = runner.invoke(app, ["train", GIELDA, "--test", test])

    assert result.exit_code == 1
    assert result.stderr.strip().count("\n") == 0
    assert "case 2" in result.stderr
    assert "a1 = foggy" in result.stderr


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
