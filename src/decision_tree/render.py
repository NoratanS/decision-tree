"""Plain text: C4.5-style trees (spec R9) and table statistics (spec R2)."""

from decision_tree.stats import AttributeStats, TableStats
from decision_tree.tree import Leaf, Node, Tree


def render_plain(tree: Tree) -> str:
    """The tree as C4.5 prints it, one test per line, without a final newline."""
    if isinstance(tree, Leaf):
        return leaf_text(tree)
    lines: list[str] = []
    _render_node(tree, 0, lines)
    return "\n".join(lines)


def branch_text(node: Node, value: str) -> str:
    """The outcome of a Node's test that leads to one branch, e.g. `a1 = old`."""
    return f"{node.attribute} = {value}"


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
    for value, child in node.branches.items():
        if isinstance(child, Leaf):
            lines.append(f"{indent}{branch_text(node, value)}: {leaf_text(child)}")
        else:
            lines.append(f"{indent}{branch_text(node, value)}:")
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
    value_width = max(map(len, counts))
    count_width = max(len(str(count)) for count in counts.values())
    lines = [
        f"  {value.ljust(value_width)}  {str(count).rjust(count_width)}"
        for value, count in counts.items()
    ]
    return "\n".join([heading, *lines])


def _plural(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"
