"""Per-attribute statistics of a Decision Table (spec R2, script §1.2)."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from decision_tree.table import DecisionTable, Kind


@dataclass(frozen=True)
class AttributeStats:
    name: str
    kind: Kind
    """Discrete or Continuous Attribute, as the loader inferred or was told."""
    counts: dict[str, int]
    """Occurrences of each known value, in order of first appearance in the file."""
    missing: int
    """Number of Missing Values."""

    @property
    def distinct(self) -> int:
        return len(self.counts)

    @property
    def known(self) -> int:
        """Number of Cases with a known value."""
        return sum(self.counts.values())


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
    attributes = []
    for index, name in enumerate(table.attributes):
        counts = Counter(case[index] for case in table.cases)
        missing = counts.pop(None, 0)
        attributes.append(
            AttributeStats(
                name=name,
                kind=table.kinds[index],
                counts=dict(counts),
                missing=missing,
            )
        )
    return TableStats(
        attributes=tuple(attributes),
        decision=table.decision,
        class_counts=dict(Counter(table.decisions)),
    )


IDENTIFIER_MIN_KNOWN = 20
"""Known values an attribute needs before it can look like an identifier."""
IDENTIFIER_MIN_DISTINCT_SHARE = 0.9
"""Share of its known values that must be distinct for an identifier."""


def likely_identifiers(stats: TableStats) -> tuple[AttributeStats, ...]:
    """Attributes that look like case identifiers (e.g. UCI breast-w's ID).

    Such an attribute is discrete or holds whole numbers only, has at least 20
    known values, and at least 90% of them are distinct. A warning, never a
    reason to drop the column: the user decides with `--ignore`.
    """
    return tuple(
        attribute
        for attribute in stats.attributes
        if (attribute.kind == "discrete" or _whole_numbers(attribute.counts))
        and attribute.known >= IDENTIFIER_MIN_KNOWN
        and attribute.distinct >= IDENTIFIER_MIN_DISTINCT_SHARE * attribute.known
    )


def _whole_numbers(values: Iterable[str]) -> bool:
    """Every value is a whole number; continuous values are numbers already."""
    return all(float(value).is_integer() for value in values)
