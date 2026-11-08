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


def test_wrong_column_count_gives_one_line_error_naming_the_line(tmp_path):
    path = tmp_path / "ragged.txt"
    path.write_text("a b c\nd e\n", encoding="utf-8")

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
