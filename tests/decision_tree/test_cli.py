from pathlib import Path

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
    assert result.stdout == GIELDA_TREE


def test_train_plain_flag_forces_plain_text_in_a_terminal():
    result = runner.invoke(
        app, ["train", GIELDA, "--plain"], env={"TTY_COMPATIBLE": "1"}
    )

    assert result.exit_code == 0
    assert result.stdout == GIELDA_TREE


def test_train_draws_a_rich_tree_in_a_terminal():
    result = runner.invoke(app, ["train", GIELDA], env={"TTY_COMPATIBLE": "1"})

    assert result.exit_code == 0
    assert "├──" in result.stdout
    assert "|   " not in result.stdout


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
    result = runner.invoke(app, ["train", str(write_table(GOLF)), "--header"])

    assert result.exit_code == 0, result.output
    assert result.stdout == "outlook = słońce: nie (2.0)\noutlook = deszcz: tak (1.0)\n"


def test_unknown_override_column_gives_one_line_error(write_table):
    result = runner.invoke(
        app, ["train", str(write_table(GOLF)), "--header", "--continuous", "wind"]
    )

    assert result.exit_code == 1
    assert result.stderr.strip().count("\n") == 0
    assert "unknown column 'wind'" in result.stderr
