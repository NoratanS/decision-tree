"""Command-line interface (spec R10).

The only module allowed to import Typer and Rich (ADR 0003).
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.tree import Tree as RichTree

from decision_tree.builder import BuildOptions, Criterion, build_tree
from decision_tree.evaluate import classify_cases
from decision_tree.experiment import (
    FileSource,
    HoldOut,
    SplitError,
    TestSource,
    run_experiment,
)
from decision_tree.pruning import DEFAULT_CONFIDENCE
from decision_tree.render import (
    Block,
    Cell,
    TableBlock,
    TreeBlock,
    TreeRow,
    classification_blocks,
    experiment_blocks,
    render_blocks_plain,
    stats_blocks,
)
from decision_tree.stats import table_stats
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
DecisionOption = Annotated[
    str | None,
    typer.Option(
        "--decision",
        help="Column of the decision: a header name, or aK for column K. "
        "Default: the last column.",
        rich_help_panel=_LOADING,
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
    decision: DecisionOption = None,
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
    if test is not None and split is not None:
        _fail("--test and --split cannot be used together")
    if seed is not None and split is None:
        _fail("--seed needs --split")
    out = Console()
    source: TestSource = None
    with _table_errors():
        options = _read_options(header, sep, encoding, missing, decision)
        training = _load_training(file, options, discrete, continuous)
        if test is not None:
            source = FileSource(test, load_table(test, options, like=training))
    try:
        if split is not None:
            source = HoldOut(split, seed)
        result = run_experiment(
            training,
            source,
            _build_options(criterion, quinlan_gain_filter, min_cases),
            confidence=None if no_prune else confidence / 100,
        )
    except SplitError as error:
        _fail(f"--split: {error}")

    _print(out, experiment_blocks(result), plain=plain)


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
    decision: DecisionOption = None,
    discrete: Discrete = "",
    continuous: Continuous = "",
    missing: Missing = _DEFAULT_MISSING,
    criterion: CriterionOption = "gain-ratio",
    quinlan_gain_filter: GainFilter = False,
    min_cases: MinCases = 2,
) -> None:
    """Build an unpruned tree from TRAIN and classify each case of CASES."""
    out = Console()
    with _table_errors():
        options = _read_options(header, sep, encoding, missing, decision)
        training = _load_training(train_file, options, discrete, continuous)
        to_classify = load_cases(cases, options, like=training)
    tree = build_tree(
        training, _build_options(criterion, quinlan_gain_filter, min_cases)
    )
    _print(out, classification_blocks(classify_cases(tree, to_classify)), plain=plain)


@app.command()
def stats(
    file: Annotated[str, typer.Argument(help="Decision table, one case per line.")],
    plain: Annotated[
        bool, typer.Option("--plain", help="Plain text tables, even in a terminal.")
    ] = False,
    header: Header = False,
    sep: Sep = None,
    encoding: Encoding = "utf-8",
    decision: DecisionOption = None,
    discrete: Discrete = "",
    continuous: Continuous = "",
    missing: Missing = _DEFAULT_MISSING,
) -> None:
    """Show each attribute's type, value counts and missing values."""
    out = Console()
    with _table_errors():
        options = _read_options(header, sep, encoding, missing, decision)
        table = _load_training(file, options, discrete, continuous)
    _print(out, stats_blocks(table_stats(table)), plain=plain, spaced=False)


def _build_options(
    criterion: Criterion, quinlan_gain_filter: bool, min_cases: int
) -> BuildOptions:
    """The build options of `train` and `classify`, made in one place."""
    return BuildOptions(
        criterion=criterion,
        quinlan_gain_filter=quinlan_gain_filter,
        min_cases=min_cases,
    )


def _read_options(
    header: bool, sep: str | None, encoding: str, missing: str, decision: str | None
) -> ReadOptions:
    return ReadOptions(
        sep=sep,
        header=header,
        encoding=encoding,
        missing=parse_markers(missing),
        decision=decision,
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


def _print(
    out: Console, blocks: Sequence[Block], *, plain: bool, spaced: bool = True
) -> None:
    """Plain text with `--plain` or when piped, else Rich."""
    if plain or not out.is_terminal:
        out.print(
            render_blocks_plain(blocks), markup=False, highlight=False, soft_wrap=True
        )
    else:
        print_rich(out, blocks, spaced=spaced)


def print_rich(out: Console, blocks: Sequence[Block], *, spaced: bool = True) -> None:
    """The Rich adapter: each block in order, one blank line apart if `spaced`;
    cell roles become colours."""
    for i, block in enumerate(blocks):
        if i and spaced:
            out.print()
        match block:
            case TreeBlock(heading, rows):
                out.print(_rich_tree(heading, rows))
            case TableBlock(heading, columns, rows, _, footer):
                if heading is not None:
                    _print_heading(out, heading)
                table = Table()
                for column in columns:
                    table.add_column(escape(column.heading), justify=column.justify)
                for row in rows:
                    table.add_row(*map(_rich_cell, row))
                out.print(table)
                if footer is not None:
                    _print_heading(out, footer)


_ROLE_STYLES = {"correct": "green", "wrong": "red", "empty": "dim"}


def _rich_cell(cell: Cell) -> str:
    if cell.role is None:
        return escape(cell.text)
    return f"[{_ROLE_STYLES[cell.role]}]{escape(cell.text)}[/]"


def _print_heading(out: Console, text: str) -> None:
    out.print(f"[bold]{escape(text)}[/]", highlight=False, soft_wrap=True)


def _rich_tree(title: str, rows: Sequence[TreeRow]) -> RichTree:
    """Each row under the last row one level up; `<=` drawn as `≤`, a branch
    and its Leaf joined by `→`."""
    root = RichTree(f"[bold]{escape(title)}[/]")
    parents = [root]
    for row in rows:
        del parents[row.depth + 1 :]
        parents.append(parents[row.depth].add(_rich_row(row)))
    return root


def _rich_row(row: TreeRow) -> str:
    parts = []
    if row.branch is not None:
        sign = "≤" if row.branch.sign == "<=" else row.branch.sign
        test = f"{row.branch.attribute} {sign} {row.branch.value}"
        parts.append(f"[cyan]{escape(test)}[/]")
    if row.leaf is not None:
        decision, counts = escape(row.leaf.decision), escape(row.leaf.counts)
        parts.append(f"[bold green]{decision}[/] [dim]{counts}[/]")
    return " → ".join(parts)


def _fail(message: str) -> NoReturn:
    Console(stderr=True).print(
        f"[bold red]Error:[/] {escape(message)}", highlight=False, soft_wrap=True
    )
    raise typer.Exit(1)


def main() -> None:
    app()
