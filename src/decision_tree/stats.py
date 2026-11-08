"""Per-attribute statistics of a Decision Table (spec R2, script §1.2)."""

from collections import Counter
from dataclasses import dataclass

from decision_tree.table import DecisionTable


@dataclass(frozen=True)
class AttributeStats:
    name: str
    kind: str
    """`discrete` for now; type inference arrives with flexible loading."""
    counts: dict[str, int]
    """Occurrences of each value, in order of first appearance in the file."""
    missing: int
    """Missing Values; always 0 until the loader recognises missing markers."""

    @property
    def distinct(self) -> int:
        return len(self.counts)


@dataclass(frozen=True)
class TableStats:
    attributes: tuple[AttributeStats, ...]
    """One entry per Conditional Attribute, in column order."""
    decision: str
    """Name of the Decision Attribute."""
    class_counts: dict[str, int]
    """Cases in each Decision Class, in order of first appearance in the file."""


def table_stats(table: DecisionTable) -> TableStats:
    """Value counts of every Conditional Attribute and case counts per class."""
    return TableStats(
        attributes=tuple(
            AttributeStats(
                name=name,
                kind="discrete",
                counts=dict(Counter(case[index] for case in table.cases)),
                missing=0,
            )
            for index, name in enumerate(table.attributes)
        ),
        decision=table.decision,
        class_counts=dict(Counter(table.decisions)),
    )
