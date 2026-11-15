"""Each output described once, as blocks, plus the plain-text adapter.

C4.5-style trees (spec R9), table statistics (spec R2), evaluations and set
names (spec R8), the Experiment report and classified Cases (spec R10). A
TreeBlock holds a tree as outline rows, so this is the only output code that
walks a tree. A TableBlock's cells carry a role (correct, wrong, empty) that
plain text ignores and the Rich adapter in `cli.py` colours.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from decision_tree.evaluate import Classification, Evaluation
from decision_tree.experiment import (
    EvaluatedSet,
    ExperimentResult,
    FileTestSet,
    HoldOutTestSet,
    TrainingSet,
)
from decision_tree.stats import AttributeStats, TableStats
from decision_tree.tree import (
    DiscreteTest,
    Leaf,
    Node,
    ThresholdTest,
    Tree,
    size,
)


def render_plain(tree: Tree) -> str:
    """The tree as C4.5 prints it, one test per line, without a final newline."""
    return _outline_plain(tree_outline(tree))


def tree_name(*, pruned: bool) -> str:
    """`Pruned tree` or `Unpruned tree`."""
    return "Pruned tree" if pruned else "Unpruned tree"


def tree_heading(tree: Tree, *, pruned_at: float | None = None) -> str:
    """`Unpruned tree (size 8)`; with `pruned_at` (a fraction), the Confidence
    Level in percent, `Pruned tree (size 5, Confidence Level 25%)`."""
    name = tree_name(pruned=pruned_at is not None)
    if pruned_at is None:
        return f"{name} (size {size(tree)})"
    # Rounded so that 0.07 shows as 7, not 7.000000000000001.
    percent = number_text(round(pruned_at * 100, 10))
    return f"{name} (size {size(tree)}, Confidence Level {percent}%)"


def number_text(number: float) -> str:
    """Shortest exact text for a number: `75`, not `75.0`."""
    if number.is_integer() and abs(number) < 1e16:
        return str(int(number))
    return repr(number)


@dataclass(frozen=True)
class BranchText:
    """The outcome of a Node's test that leads to one branch: `a1 = old`,
    `humidity <= 75` or `humidity > 75`. `sign` is `=`, `<=` or `>`."""

    attribute: str
    sign: Literal["=", "<=", ">"]
    value: str


@dataclass(frozen=True)
class LeafText:
    """A Leaf's class and its counts, `(n)` or `(n/errors)`, one decimal each."""

    decision: str
    counts: str


@dataclass(frozen=True)
class TreeRow:
    """One line of a tree outline, `depth` tests below the root.

    `branch` is None only for a tree that is a single Leaf; `leaf` is None when
    the branch leads to a Node, whose rows follow one level deeper.
    """

    depth: int
    branch: BranchText | None
    leaf: LeafText | None


def tree_outline(tree: Tree) -> tuple[TreeRow, ...]:
    """The tree as outline rows, depth first, branches in the Node's order."""
    if isinstance(tree, Leaf):
        return (TreeRow(0, None, _leaf_text(tree)),)
    rows: list[TreeRow] = []
    _outline_node(tree, 0, rows)
    return tuple(rows)


def _outline_node(node: Node, depth: int, rows: list[TreeRow]) -> None:
    for value, child in node.branches.items():
        branch = _branch_text(node, value)
        if isinstance(child, Leaf):
            rows.append(TreeRow(depth, branch, _leaf_text(child)))
        else:
            rows.append(TreeRow(depth, branch, None))
            _outline_node(child, depth + 1, rows)


def _branch_text(node: Node, value: str) -> BranchText:
    # A Threshold test's branch key is its comparison sign.
    match node.test, value:
        case ThresholdTest(attribute, _, threshold), "<=" | ">" as sign:
            return BranchText(attribute, sign, number_text(threshold))
        case DiscreteTest(attribute), _:
            return BranchText(attribute, "=", value)
    raise ValueError(f"{value!r} is not a branch of {node.test}")


def _leaf_text(leaf: Leaf) -> LeafText:
    total = leaf.distribution.total
    errors = round(leaf.distribution.errors(leaf.decision), 1)
    if errors > 0:
        return LeafText(leaf.decision, f"({total:.1f}/{errors:.1f})")
    return LeafText(leaf.decision, f"({total:.1f})")


def _outline_plain(rows: Sequence[TreeRow]) -> str:
    return "\n".join(map(_row_plain, rows))


def _row_plain(row: TreeRow) -> str:
    """`|   ` per level, then the branch with its colon, then the Leaf if any."""
    parts = []
    if (branch := row.branch) is not None:
        # C4.5 puts a space before the colon on Threshold branches.
        colon = ":" if branch.sign == "=" else " :"
        parts.append(f"{branch.attribute} {branch.sign} {branch.value}{colon}")
    if (leaf := row.leaf) is not None:
        parts.append(f"{leaf.decision} {leaf.counts}")
    return "|   " * row.depth + " ".join(parts)


@dataclass(frozen=True)
class Column:
    """A table column. `plain_heading`, if set, replaces `heading` in plain text."""

    heading: str
    justify: Literal["left", "right"] = "left"
    plain_heading: str | None = None


type Role = Literal["correct", "wrong", "empty"] | None
"""What a cell means: a correct or wrong prediction, or an empty count."""


@dataclass(frozen=True)
class Cell:
    text: str
    role: Role = None


@dataclass(frozen=True)
class TableBlock:
    """An aligned table, under a heading if any, with a closing line if any.

    Plain text indents a table that has a heading and prints the column
    headings as a first row only with `plain_header`.
    """

    heading: str | None
    columns: tuple[Column, ...]
    rows: tuple[tuple[Cell, ...], ...]
    plain_header: bool = True
    footer: str | None = None


def stats_blocks(stats: TableStats) -> list[TableBlock]:
    """One count table per attribute, then the Decision Attribute's."""
    blocks = [
        _counts_table(attribute_heading(attribute), "Value", attribute.counts)
        for attribute in stats.attributes
    ]
    blocks.append(_counts_table(decision_heading(stats), "Class", stats.class_counts))
    return blocks


def attribute_heading(attribute: AttributeStats) -> str:
    """e.g. `a1 (discrete): 3 values, 0 missing`."""
    values = _plural(attribute.distinct, "value", "values")
    return f"{attribute.name} ({attribute.kind}): {values}, {attribute.missing} missing"


def decision_heading(stats: TableStats) -> str:
    """e.g. `d (decision): 2 classes`."""
    classes = _plural(len(stats.class_counts), "class", "classes")
    return f"{stats.decision} (decision): {classes}"


def _counts_table(heading: str, name: str, counts: dict[str, int]) -> TableBlock:
    return TableBlock(
        heading,
        (Column(name), Column("Count", "right")),
        tuple((Cell(value), Cell(str(count))) for value, count in counts.items()),
        plain_header=False,
    )


def _plural(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def set_name(evaluated_set: EvaluatedSet) -> str:
    """`Training Set`, `Test Set (FILE)` or `Test Set (hold-out, --seed N)`."""
    match evaluated_set:
        case TrainingSet():
            return "Training Set"
        case FileTestSet(file):
            return f"Test Set ({file})"
        case HoldOutTestSet(seed):
            return f"Test Set (hold-out, --seed {seed})"


def evaluation_heading(set_name: str, result: Evaluation) -> str:
    """e.g. `Training Set: 10 cases, 1 error (10.0%)` for `set_name` `Training Set`."""
    cases = _plural(result.cases, "case", "cases")
    errors = _plural(result.errors, "error", "errors")
    return f"{set_name}: {cases}, {errors} ({result.error_rate:.1%})"


def _confusion_table(heading: str, result: Evaluation) -> TableBlock:
    """The confusion matrix: one row per actual class, one column per predicted
    class."""
    return TableBlock(
        heading,
        (
            Column("actual \\ predicted"),
            *(Column(predicted, "right") for predicted in result.classes),
        ),
        tuple(
            (
                Cell(actual),
                *(
                    _count_cell(actual, predicted, result.confusion[actual][predicted])
                    for predicted in result.classes
                ),
            )
            for actual in result.classes
        ),
    )


def _count_cell(actual: str, predicted: str, count: int) -> Cell:
    if actual == predicted:
        return Cell(str(count), "correct")
    return Cell(str(count), "wrong" if count else "empty")


@dataclass(frozen=True)
class TreeBlock:
    """A tree's outline rows under its heading, e.g. `Unpruned tree (size 8)`."""

    heading: str
    rows: tuple[TreeRow, ...]


type Block = TreeBlock | TableBlock
"""One part of an output; plain text puts blocks one blank line apart."""


def _tree_block(heading: str, tree: Tree) -> TreeBlock:
    return TreeBlock(heading, tree_outline(tree))


def experiment_blocks(result: ExperimentResult) -> list[Block]:
    """The report of an Experiment in order: the unpruned tree, the pruned tree
    if any, then for each set the unpruned tree's evaluation and the pruned
    tree's if any."""
    blocks: list[Block] = [_tree_block(tree_heading(result.unpruned), result.unpruned)]
    if result.pruned is not None:
        assert result.confidence is not None, "a pruned tree has a Confidence Level"
        heading = tree_heading(result.pruned, pruned_at=result.confidence)
        blocks.append(_tree_block(heading, result.pruned))
    for on_set in result.evaluations:
        name = set_name(on_set.evaluated_set)
        for pruned, evaluation in ((False, on_set.unpruned), (True, on_set.pruned)):
            if evaluation is not None:
                label = f"{tree_name(pruned=pruned)} on {name}"
                heading = evaluation_heading(label, evaluation)
                blocks.append(_confusion_table(heading, evaluation))
    return blocks


def render_experiment_plain(result: ExperimentResult) -> str:
    """The `train` report as plain text; no final newline."""
    return render_blocks_plain(experiment_blocks(result))


def classification_blocks(result: Classification) -> list[TableBlock]:
    """One row per Case: its number, predicted class and the probability of each
    class.

    With decisions, each row also shows the actual class, a `wrong` mark on a
    wrong prediction, and an accuracy line closes the table.
    """
    with_decisions = result.decisions is not None
    columns = [
        Column("Case", "right", plain_heading="case"),
        Column("Predicted", plain_heading="predicted"),
        *(Column(c, "right") for c in result.classes),
    ]
    if with_decisions:
        columns += [Column("Actual", plain_heading="actual"), Column("")]
    rows = []
    for number, (distribution, predicted, actual) in enumerate(result.rows, start=1):
        probabilities = [Cell(f"{distribution[c]:.3f}") for c in result.classes]
        if actual is None:
            rows.append((Cell(str(number)), Cell(predicted), *probabilities))
            continue
        correct = actual == predicted
        rows.append(
            (
                Cell(str(number)),
                Cell(predicted, "correct" if correct else "wrong"),
                *probabilities,
                Cell(actual),
                Cell("") if correct else Cell("wrong", "wrong"),
            )
        )
    footer = _accuracy_text(result) if with_decisions else None
    return [TableBlock(None, tuple(columns), tuple(rows), footer=footer)]


def render_classification_plain(result: Classification) -> str:
    """The `classify` table as plain text; no final newline."""
    return render_blocks_plain(classification_blocks(result))


def _accuracy_text(result: Classification) -> str:
    """e.g. `Accuracy: 2 of 3 cases (66.7%)`."""
    total, correct = len(result.distributions), result.correct
    share = correct / total if total else 0.0
    return f"Accuracy: {correct} of {_plural(total, 'case', 'cases')} ({share:.1%})"


def render_blocks_plain(blocks: Sequence[Block]) -> str:
    """The plain adapter: blocks one blank line apart; cell roles are ignored.
    No final newline."""
    return "\n\n".join(map(_block_plain, blocks))


def _block_plain(block: Block) -> str:
    match block:
        case TreeBlock(heading, rows):
            return f"{heading}:\n{_outline_plain(rows)}"
        case TableBlock(heading, columns, rows, plain_header, footer):
            lines = [] if heading is None else [heading]
            texts = [[cell.text for cell in row] for row in rows]
            if plain_header:
                texts.insert(0, [_plain_heading(column) for column in columns])
            justify = ["<" if c.justify == "left" else ">" for c in columns]
            lines += _aligned_rows(
                texts, justify, indent="" if heading is None else "  "
            )
            if footer is not None:
                lines += ["", footer]
            return "\n".join(lines)


def _plain_heading(column: Column) -> str:
    return column.heading if column.plain_heading is None else column.plain_heading


def _aligned_rows(rows: list[list[str]], justify: list[str], indent: str) -> list[str]:
    """Lines of aligned columns, two spaces apart, after `indent`; `justify` holds
    `"<"` (left) or `">"` (right) per column."""
    widths = [max(map(len, column)) for column in zip(*rows, strict=True)]
    return [
        indent
        + "  ".join(
            f"{cell:{side}{width}}"
            for cell, side, width in zip(row, justify, widths, strict=True)
        ).rstrip()
        for row in rows
    ]
