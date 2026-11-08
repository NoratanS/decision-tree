"""Command-line interface (spec R10).

The only module allowed to import Typer and Rich (ADR 0003).
"""

from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.tree import Tree as RichTree

from decision_tree.builder import build_tree
from decision_tree.render import branch_text, leaf_counts, render_plain
from decision_tree.table import TableError, load_table
from decision_tree.tree import Leaf, Node, Tree

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def _commands() -> None:
    """Learn C4.5 decision trees from decision tables."""


@app.command()
def train(
    file: Annotated[str, typer.Argument(help="Decision table, one case per line.")],
    plain: Annotated[
        bool, typer.Option("--plain", help="Plain C4.5-style text, even in a terminal.")
    ] = False,
) -> None:
    """Build a C4.5 tree with Gain Ratio and print it."""
    out = Console()
    try:
        table = load_table(file)
    except TableError as error:
        _fail(str(error))
    tree = build_tree(table)
    if plain or not out.is_terminal:
        out.print(render_plain(tree), markup=False, highlight=False, soft_wrap=True)
    else:
        out.print(_rich_tree(tree, file))


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
