"""Command-line interface (spec R10).

The only module allowed to import Typer and Rich (ADR 0003).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.tree import Tree as RichTree

from decision_tree.builder import Criterion, build_tree
from decision_tree.evaluate import Classification, Evaluation, classify_cases
from decision_tree.experiment import (
    FileSource,
    HoldOut,
    SplitError,
    TestSource,
    run_experiment,
)
from decision_tree.pruning import DEFAULT_CONFIDENCE
from decision_tree.render import (
    CONFUSION_CORNER,
    Block,
    EvaluationBlock,
    TreeBlock,
    accuracy_text,
    attribute_heading,
    branch_text,
    decision_heading,
    experiment_blocks,
    leaf_counts,
    probability_cells,
    render_classification_plain,
    render_experiment_plain,
    render_stats_plain,
)
from decision_tree.stats import TableStats, table_stats
from decision_tree.table import (
    MISSING_MARKERS,
    DecisionTable,
    ReadOptions,
    TableError,
    load_cases,
    load_table,
    parse_columns,
    parse_markers,
)
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

_DEFAULT_MISSING = ",".join(MISSING_MARKERS)
Missing = Annotated[
    str,
    typer.Option(
        "--missing",
        help="Comma-separated Missing Value markers; an empty item is the empty cell.",
        rich_help_panel=_LOADING,
    ),
]


_SPLITTING = "Split options"
CriterionOption = Annotated[
    Criterion,
    typer.Option(
        "--criterion",
        help="Rank tests by gain (ID3) or gain-ratio (C4.5).",
        rich_help_panel=_SPLITTING,
    ),
]
GainFilter = Annotated[
    bool,
    typer.Option(
        "--quinlan-gain-filter",
        help="Only tests with at least the average Gain compete.",
        rich_help_panel=_SPLITTING,
    ),
]
MinCases = Annotated[
    int,
    typer.Option(
        "-m",
        "--min-cases",
        min=1,
        help="A test needs two branches with at least this many cases.",
        rich_help_panel=_SPLITTING,
    ),
]

_PRUNING = "Pruning options"
ConfidenceOption = Annotated[
    float,
    typer.Option(
        "-c",
        "--confidence",
        help="Confidence Level in percent, above 0 and at most 100. Lower prunes more.",
        rich_help_panel=_PRUNING,
    ),
]
NoPrune = Annotated[
    bool,
    typer.Option(
        "--no-prune", help="Keep the unpruned tree only.", rich_help_panel=_PRUNING
    ),
]

_EVALUATION = "Evaluation options"


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
    missing: Missing = _DEFAULT_MISSING,
    criterion: CriterionOption = "gain-ratio",
    quinlan_gain_filter: GainFilter = False,
    min_cases: MinCases = 2,
    confidence: ConfidenceOption = DEFAULT_CONFIDENCE * 100,
    no_prune: NoPrune = False,
    test: Annotated[
        str | None,
        typer.Option(
            "--test",
            help="Test Set file, same columns and loading options as FILE.",
            rich_help_panel=_EVALUATION,
        ),
    ] = None,
    split: Annotated[
        float | None,
        typer.Option(
            "--split",
            help="Train on this share of FILE (between 0 and 1), test on the rest.",
            rich_help_panel=_EVALUATION,
        ),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help="Seed of the --split shuffle. Default: random, printed.",
            rich_help_panel=_EVALUATION,
        ),
    ] = None,
) -> None:
    """Build a C4.5 tree, prune it, print both and evaluate both."""
    if not 0 < confidence <= 100:
        raise typer.BadParameter("must be above 0 and at most 100", param_hint="-c")
    if split is not None and not 0 < split < 1:
        raise typer.BadParameter("must be between 0 and 1", param_hint="--split")
    if test is not None and split is not None:
        _fail("--test and --split cannot be used together")
    if seed is not None and split is None:
        _fail("--seed needs --split")
    out = Console()
    source: TestSource = None
    with _table_errors():
        options = _read_options(header, sep, encoding, missing)
        training = _load_training(file, options, discrete, continuous)
        if test is not None:
            source = FileSource(test, load_table(test, options, like=training))
    if split is not None:
        source = HoldOut(split, seed)
    try:
        result = run_experiment(
            training,
            source,
            criterion=criterion,
            quinlan_gain_filter=quinlan_gain_filter,
            min_cases=min_cases,
            confidence=None if no_prune else confidence / 100,
        )
    except SplitError as error:
        _fail(f"--split: {error}")

    if plain or not out.is_terminal:
        _print_plain(out, render_experiment_plain(result))
    else:
        _print_rich_blocks(out, experiment_blocks(result))


@app.command()
def classify(
    train_file: Annotated[
        str,
        typer.Argument(metavar="TRAIN", help="Training Set the tree is built from."),
    ],
    cases: Annotated[
        str,
        typer.Argument(
            metavar="CASES",
            help="Cases to classify: TRAIN's columns, with or without the decision.",
        ),
    ],
    plain: Annotated[
        bool, typer.Option("--plain", help="Plain text table, even in a terminal.")
    ] = False,
    header: Header = False,
    sep: Sep = None,
    encoding: Encoding = "utf-8",
    discrete: Discrete = "",
    continuous: Continuous = "",
    missing: Missing = _DEFAULT_MISSING,
    criterion: CriterionOption = "gain-ratio",
    quinlan_gain_filter: GainFilter = False,
    min_cases: MinCases = 2,
) -> None:
    """Build a tree from TRAIN and classify each case of CASES."""
    out = Console()
    with _table_errors():
        options = _read_options(header, sep, encoding, missing)
        training = _load_training(train_file, options, discrete, continuous)
        to_classify = load_cases(cases, options, like=training)
    tree = build_tree(
        training,
        criterion=criterion,
        quinlan_gain_filter=quinlan_gain_filter,
        min_cases=min_cases,
    )
    result = classify_cases(tree, to_classify)
    if plain or not out.is_terminal:
        _print_plain(out, render_classification_plain(result))
    else:
        _print_rich_classification(out, result)


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
    missing: Missing = _DEFAULT_MISSING,
) -> None:
    """Show each attribute's type, value counts and missing values."""
    out = Console()
    with _table_errors():
        options = _read_options(header, sep, encoding, missing)
        table = _load_training(file, options, discrete, continuous)
    summary = table_stats(table)
    if plain or not out.is_terminal:
        _print_plain(out, render_stats_plain(summary))
    else:
        _print_rich_stats(out, summary)


def _read_options(
    header: bool, sep: str | None, encoding: str, missing: str
) -> ReadOptions:
    return ReadOptions(
        sep=sep, header=header, encoding=encoding, missing=parse_markers(missing)
    )


def _load_training(
    file: str, options: ReadOptions, discrete: str, continuous: str
) -> DecisionTable:
    """Load FILE with the --discrete / --continuous overrides."""
    return load_table(
        file,
        options,
        discrete=parse_columns(discrete),
        continuous=parse_columns(continuous),
    )


@contextmanager
def _table_errors() -> Iterator[None]:
    """Report a TableError as a one-line error and exit."""
    try:
        yield
    except TableError as error:
        _fail(str(error))


def _print_plain(out: Console, text: str) -> None:
    out.print(text, markup=False, highlight=False, soft_wrap=True)


def _print_heading(out: Console, text: str) -> None:
    out.print(f"[bold]{escape(text)}[/]", highlight=False, soft_wrap=True)


def _print_rich_stats(out: Console, summary: TableStats) -> None:
    for attribute in summary.attributes:
        _print_heading(out, attribute_heading(attribute))
        out.print(_count_table("Value", attribute.counts))
    _print_heading(out, decision_heading(summary))
    out.print(_count_table("Class", summary.class_counts))


def _print_rich_blocks(out: Console, blocks: list[Block]) -> None:
    """Each block in the given order, one blank line apart."""
    for i, block in enumerate(blocks):
        if i:
            out.print()
        match block:
            case TreeBlock(heading, tree):
                out.print(_rich_tree(tree, heading))
            case EvaluationBlock(heading, evaluation):
                _print_heading(out, heading)
                out.print(_rich_confusion(evaluation))


def _rich_confusion(result: Evaluation) -> Table:
    table = Table(escape(CONFUSION_CORNER))
    for predicted in result.classes:
        table.add_column(escape(predicted), justify="right")
    for actual in result.classes:
        cells = []
        for predicted in result.classes:
            count = result.confusion[actual][predicted]
            cells.append(f"[{_count_style(actual, predicted, count)}]{count}[/]")
        table.add_row(escape(actual), *cells)
    return table


def _print_rich_classification(out: Console, result: Classification) -> None:
    table = Table("Case", "Predicted", *map(escape, result.classes))
    table.columns[0].justify = "right"
    for column in table.columns[2:]:
        column.justify = "right"
    if result.decisions is not None:
        table.add_column("Actual")
        table.add_column("")
    for number, (distribution, predicted, actual) in enumerate(result.rows, start=1):
        cells = probability_cells(result.classes, distribution)
        if actual is None:
            table.add_row(str(number), escape(predicted), *cells)
            continue
        style = "green" if predicted == actual else "red"
        table.add_row(
            str(number),
            f"[{style}]{escape(predicted)}[/]",
            *cells,
            escape(actual),
            "" if predicted == actual else "[red]wrong[/]",
        )
    out.print(table)
    if result.decisions is not None:
        _print_heading(out, accuracy_text(result))


def _count_style(actual: str, predicted: str, count: int) -> str:
    """Green for correct predictions, red for errors, dim for empty cells."""
    if actual == predicted:
        return "green"
    if count:
        return "red"
    return "dim"


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
        test = f"[cyan]{escape(branch_text(node, value, math_signs=True))}[/]"
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
