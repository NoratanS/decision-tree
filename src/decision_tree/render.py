"""Plain C4.5-style text for a tree (spec R9)."""

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
