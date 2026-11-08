"""Command-line interface (spec R10).

The only module allowed to import Typer and Rich (ADR 0003).
"""

from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.tree import Tree as RichTree

from decision_tree.builder import Criterion, build_tree
from decision_tree.render import (
    attribute_heading,
    branch_text,
    decision_heading,
    leaf_counts,
    render_plain,
    render_stats_plain,
)
from decision_tree.stats import TableStats, table_stats
from decision_tree.table import DecisionTable, TableError, load_table
from decision_tree.tree import Leaf, Node, Tree

app = typer.Typer(no_args_is_help=True, add_completion=False)

_LOADING = "Loading options"
Header = Annotated[
    bool,
    typer.Option(
        "--header",
        help="Take column names from the first line.",
        rich_help_panel=_LOADING,
    ),
]
Sep = Annotated[
    str | None,
    typer.Option(
        "--sep",
        help="Separator, any single character. Default: detect whitespace , ; | :",
        rich_help_panel=_LOADING,
    ),
]
Encoding = Annotated[
    str,
    typer.Option(
        "--encoding", help="File encoding, e.g. cp1250.", rich_help_panel=_LOADING
    ),
]
Discrete = Annotated[
    str,
    typer.Option(
        "--discrete",
        help="Comma-separated columns to treat as discrete.",
        rich_help_panel=_LOADING,
    ),
]
Continuous = Annotated[
    str,
    typer.Option(
        "--continuous",
        help="Comma-separated columns to treat as continuous.",
        rich_help_panel=_LOADING,
    ),
]


_SPLITTING = "Split options"


@app.callback()
def _commands() -> None:
    """Learn C4.5 decision trees from decision tables."""


@app.command()
def train(
    file: Annotated[str, typer.Argument(help="Decision table, one case per line.")],
    plain: Annotated[
        bool, typer.Option("--plain", help="Plain C4.5-style text, even in a terminal.")
    ] = False,
    header: Header = False,
    sep: Sep = None,
    encoding: Encoding = "utf-8",
    discrete: Discrete = "",
    continuous: Continuous = "",
    criterion: Annotated[
        Criterion,
        typer.Option(
            "--criterion",
            help="Rank tests by gain (ID3) or gain-ratio (C4.5).",
            rich_help_panel=_SPLITTING,
        ),
    ] = "gain-ratio",
    quinlan_gain_filter: Annotated[
        bool,
        typer.Option(
            "--quinlan-gain-filter",
            help="Only tests with at least the average Gain compete.",
            rich_help_panel=_SPLITTING,
        ),
    ] = False,
    min_cases: Annotated[
        int,
        typer.Option(
            "-m",
            "--min-cases",
            min=1,
            help="A test needs two branches with at least this many cases.",
            rich_help_panel=_SPLITTING,
        ),
    ] = 2,
) -> None:
    """Build a C4.5 tree and print it."""
    out = Console()
    tree = build_tree(
        _load(
            file,
            header=header,
            sep=sep,
            encoding=encoding,
            discrete=discrete,
            continuous=continuous,
        ),
        criterion=criterion,
        quinlan_gain_filter=quinlan_gain_filter,
        min_cases=min_cases,
    )
    if plain or not out.is_terminal:
        _print_plain(out, render_plain(tree))
    else:
        out.print(_rich_tree(tree, file))


@app.command()
def stats(
    file: Annotated[str, typer.Argument(help="Decision table, one case per line.")],
    plain: Annotated[
        bool, typer.Option("--plain", help="Plain text tables, even in a terminal.")
    ] = False,
    header: Header = False,
    sep: Sep = None,
    encoding: Encoding = "utf-8",
    discrete: Discrete = "",
    continuous: Continuous = "",
) -> None:
    """Show each attribute's type, value counts and missing values."""
    out = Console()
    summary = table_stats(
        _load(
            file,
            header=header,
            sep=sep,
            encoding=encoding,
            discrete=discrete,
            continuous=continuous,
        )
    )
    if plain or not out.is_terminal:
        _print_plain(out, render_stats_plain(summary))
    else:
        _print_rich_stats(out, summary)


def _load(
    file: str,
    *,
    header: bool,
    sep: str | None,
    encoding: str,
    discrete: str,
    continuous: str,
) -> DecisionTable:
    try:
        return load_table(
            file,
            header=header,
            sep=sep,
            encoding=encoding,
            discrete=_column_list(discrete),
            continuous=_column_list(continuous),
        )
    except TableError as error:
        _fail(str(error))


def _column_list(columns: str) -> list[str]:
    """`"a1, a3"` → `["a1", "a3"]`."""
    return [name.strip() for name in columns.split(",") if name.strip()]


def _print_plain(out: Console, text: str) -> None:
    out.print(text, markup=False, highlight=False, soft_wrap=True)


def _print_rich_stats(out: Console, summary: TableStats) -> None:
    for attribute in summary.attributes:
        out.print(f"[bold]{escape(attribute_heading(attribute))}[/]", highlight=False)
        out.print(_count_table("Value", attribute.counts))
    out.print(f"[bold]{escape(decision_heading(summary))}[/]", highlight=False)
    out.print(_count_table("Class", summary.class_counts))


def _count_table(heading: str, counts: dict[str, int]) -> Table:
    table = Table(heading, "Count")
    table.columns[1].justify = "right"
    for value, count in counts.items():
        table.add_row(escape(value), str(count))
    return table


def _rich_tree(tree: Tree, title: str) -> RichTree:
    root = RichTree(f"[bold]{escape(title)}[/]")
    if isinstance(tree, Leaf):
        root.add(_rich_leaf(tree))
    else:
        _add_branches(root, tree)
    return root


def _add_branches(parent: RichTree, node: Node) -> None:
    for value, child in node.branches.items():
        test = f"[cyan]{escape(branch_text(node, value))}[/]"
        if isinstance(child, Leaf):
            parent.add(f"{test} → {_rich_leaf(child)}")
        else:
            _add_branches(parent.add(test), child)


def _rich_leaf(leaf: Leaf) -> str:
    return f"[bold green]{escape(leaf.decision)}[/] [dim]{leaf_counts(leaf)}[/]"


def _fail(message: str) -> NoReturn:
    Console(stderr=True).print(
        f"[bold red]Error:[/] {escape(message)}", highlight=False, soft_wrap=True
    )
    raise typer.Exit(1)


def main() -> None:
    app()
