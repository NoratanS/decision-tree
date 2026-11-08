"""Plain text: C4.5-style trees (spec R9), table statistics (spec R2),
evaluations and set names (spec R8), the Experiment report and classified Cases
(spec R10)."""

from collections.abc import Sequence
from dataclasses import dataclass

from decision_tree.evaluate import Classification, Evaluation
from decision_tree.experiment import (
    EvaluatedSet,
    ExperimentResult,
    FileTestSet,
    HoldOutTestSet,
    TrainingSet,
)
from decision_tree.stats import AttributeStats, TableStats
from decision_tree.tree import ClassDistribution, Leaf, Node, Tree, size


def render_plain(tree: Tree) -> str:
    """The tree as C4.5 prints it, one test per line, without a final newline."""
    if isinstance(tree, Leaf):
        return leaf_text(tree)
    lines: list[str] = []
    _render_node(tree, 0, lines)
    return "\n".join(lines)


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


def branch_text(node: Node, value: str, *, math_signs: bool = False) -> str:
    """The outcome of a Node's test that leads to one branch.

    `a1 = old` for a discrete test; `humidity <= 75` or `humidity > 75` for a
    Threshold test, where `value` is the branch key `"<="` or `">"`. With
    `math_signs`, `<=` is written `≤`.
    """
    if node.threshold is not None:
        sign = "≤" if math_signs and value == "<=" else value
        return f"{node.attribute} {sign} {number_text(node.threshold)}"
    return f"{node.attribute} = {value}"


def number_text(number: float) -> str:
    """Shortest exact text for a number: `75`, not `75.0`."""
    if number.is_integer() and abs(number) < 1e16:
        return str(int(number))
    return repr(number)


def leaf_text(leaf: Leaf) -> str:
    """`class (n)`, or `class (n/errors)` when the leaf misclassifies weight."""
    return f"{leaf.decision} {leaf_counts(leaf)}"


def leaf_counts(leaf: Leaf) -> str:
    """`(n)` or `(n/errors)`, one decimal each."""
    errors = round(leaf.errors, 1)
    if errors > 0:
        return f"({leaf.total:.1f}/{errors:.1f})"
    return f"({leaf.total:.1f})"


def _render_node(node: Node, depth: int, lines: list[str]) -> None:
    indent = "|   " * depth
    # C4.5 puts a space before the colon on Threshold branches.
    colon = " :" if node.threshold is not None else ":"
    for value, child in node.branches.items():
        test = f"{indent}{branch_text(node, value)}{colon}"
        if isinstance(child, Leaf):
            lines.append(f"{test} {leaf_text(child)}")
        else:
            lines.append(test)
            _render_node(child, depth + 1, lines)


def render_stats_plain(stats: TableStats) -> str:
    """One block per attribute, then the Decision Attribute; no final newline."""
    blocks = [
        _counts_block(attribute_heading(attribute), attribute.counts)
        for attribute in stats.attributes
    ]
    blocks.append(_counts_block(decision_heading(stats), stats.class_counts))
    return "\n\n".join(blocks)


def attribute_heading(attribute: AttributeStats) -> str:
    """e.g. `a1 (discrete): 3 values, 0 missing`."""
    values = _plural(attribute.distinct, "value", "values")
    return f"{attribute.name} ({attribute.kind}): {values}, {attribute.missing} missing"


def decision_heading(stats: TableStats) -> str:
    """e.g. `d (decision): 2 classes`."""
    classes = _plural(len(stats.class_counts), "class", "classes")
    return f"{stats.decision} (decision): {classes}"


def _counts_block(heading: str, counts: dict[str, int]) -> str:
    rows = [[value, str(count)] for value, count in counts.items()]
    return "\n".join([heading, *_aligned_rows(rows)])


def _aligned_rows(
    rows: list[list[str]],
    justify: Sequence[str] | None = None,
    *,
    indent: str = "  ",
) -> list[str]:
    """Lines of aligned columns, two spaces apart, after `indent`.

    `justify` holds `"<"` (left) or `">"` (right) per column; by default the
    first column is left-aligned and the others right-aligned.
    """
    widths = [max(map(len, column)) for column in zip(*rows, strict=True)]
    if justify is None:
        justify = ["<", *(">" for _ in widths[1:])]
    return [
        indent
        + "  ".join(
            f"{cell:{side}{width}}"
            for cell, side, width in zip(row, justify, widths, strict=True)
        ).rstrip()
        for row in rows
    ]


def _plural(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


CONFUSION_CORNER = "actual \\ predicted"
"""Heading of the first column of a confusion matrix."""


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


def _confusion_lines(result: Evaluation) -> list[str]:
    """The confusion matrix as aligned lines: one row per actual class, one
    column per predicted class."""
    rows = [[CONFUSION_CORNER, *result.classes]]
    rows.extend(
        [actual, *(str(result.confusion[actual][p]) for p in result.classes)]
        for actual in result.classes
    )
    return _aligned_rows(rows)


@dataclass(frozen=True)
class TreeBlock:
    """A tree under its heading, e.g. `Unpruned tree (size 8)`."""

    heading: str
    tree: Tree


@dataclass(frozen=True)
class EvaluationBlock:
    """An evaluation under its heading, e.g.
    `Pruned tree on Training Set: 14 cases, 5 errors (35.7%)`."""

    heading: str
    evaluation: Evaluation


type Block = TreeBlock | EvaluationBlock
"""One part of the Experiment report; the adapters print blocks blank-line apart."""


def experiment_blocks(result: ExperimentResult) -> list[Block]:
    """The report of an Experiment in order: the unpruned tree, the pruned tree
    if any, then for each set the unpruned tree's evaluation and the pruned
    tree's if any."""
    blocks: list[Block] = [TreeBlock(tree_heading(result.unpruned), result.unpruned)]
    if result.pruned is not None:
        assert result.confidence is not None, "a pruned tree has a Confidence Level"
        heading = tree_heading(result.pruned, pruned_at=result.confidence)
        blocks.append(TreeBlock(heading, result.pruned))
    for on_set in result.evaluations:
        name = set_name(on_set.evaluated_set)
        for pruned, evaluation in ((False, on_set.unpruned), (True, on_set.pruned)):
            if evaluation is not None:
                label = f"{tree_name(pruned=pruned)} on {name}"
                blocks.append(
                    EvaluationBlock(evaluation_heading(label, evaluation), evaluation)
                )
    return blocks


def render_experiment_plain(result: ExperimentResult) -> str:
    """The `train` report as plain text, blocks one blank line apart; no final
    newline."""
    return "\n\n".join(map(_block_plain, experiment_blocks(result)))


def _block_plain(block: Block) -> str:
    match block:
        case TreeBlock(heading, tree):
            return f"{heading}:\n{render_plain(tree)}"
        case EvaluationBlock(heading, evaluation):
            return "\n".join([heading, *_confusion_lines(evaluation)])


def render_classification_plain(result: Classification) -> str:
    """One line per Case: its number, predicted class and the probability of
    each class; no final newline.

    With decisions, each line also shows the actual class, `wrong` marks a
    wrong prediction, and an accuracy line follows.
    """
    with_decisions = result.decisions is not None
    heading = ["case", "predicted", *result.classes]
    if with_decisions:
        heading += ["actual", ""]
    rows = [heading]
    for number, (distribution, predicted, actual) in enumerate(result.rows, start=1):
        row = [str(number), predicted, *probability_cells(result.classes, distribution)]
        if actual is not None:
            row += [actual, "" if actual == predicted else "wrong"]
        rows.append(row)
    # Numbers right-aligned, names left-aligned.
    justify = [">", "<", *(">" for _ in result.classes), "<", "<"][: len(heading)]
    lines = _aligned_rows(rows, justify, indent="")
    if with_decisions:
        lines += ["", accuracy_text(result)]
    return "\n".join(lines)


def probability_cells(
    classes: Sequence[str], distribution: ClassDistribution
) -> list[str]:
    """Each class's probability with three decimals."""
    return [f"{distribution[c]:.3f}" for c in classes]


def accuracy_text(result: Classification) -> str:
    """e.g. `Accuracy: 2 of 3 cases (66.7%)`."""
    total, correct = len(result.distributions), result.correct
    share = correct / total if total else 0.0
    return f"Accuracy: {correct} of {_plural(total, 'case', 'cases')} ({share:.1%})"
