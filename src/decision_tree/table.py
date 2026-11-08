"""Loading a Decision Table from a text file (spec R1).

Ticket 01 subset: values separated by runs of whitespace, no header, every
Conditional Attribute discrete, the Decision Attribute in the last column.
"""

import os
from dataclasses import dataclass
from functools import cached_property


class TableError(ValueError):
    """The file cannot be read as a Decision Table. The message is one line."""


@dataclass(frozen=True)
class DecisionTable:
    attributes: tuple[str, ...]
    """Names of the Conditional Attributes, in column order."""
    decision: str
    """Name of the Decision Attribute."""
    cases: tuple[tuple[str, ...], ...]
    """Conditional Attribute values of each Case."""
    decisions: tuple[str, ...]
    """Decision Class of each Case, parallel to `cases`."""

    @cached_property
    def classes(self) -> tuple[str, ...]:
        """Decision Classes in order of first appearance in the file."""
        return tuple(dict.fromkeys(self.decisions))

    def values(self, attribute: int) -> tuple[str, ...]:
        """Values of one Conditional Attribute, in order of first appearance."""
        return tuple(dict.fromkeys(case[attribute] for case in self.cases))


def load_table(path: str | os.PathLike[str]) -> DecisionTable:
    """Read a whitespace-separated Decision Table without a header."""
    try:
        with open(path, encoding="utf-8") as file:
            lines = file.read().splitlines()
    except (OSError, UnicodeDecodeError) as error:
        reason = error.strerror if isinstance(error, OSError) else str(error)
        raise TableError(f"cannot read {os.fspath(path)}: {reason}") from None

    parsed: list[list[str]] = []
    width = 0
    for number, line in enumerate(lines, start=1):
        values = line.split()
        if not values:
            continue
        if not parsed:
            width = len(values)
            if width < 2:
                raise TableError(
                    f"{os.fspath(path)}, line {number}: a Decision Table needs "
                    "at least two columns"
                )
        elif len(values) != width:
            raise TableError(
                f"{os.fspath(path)}, line {number}: expected {width} columns, "
                f"found {len(values)}"
            )
        parsed.append(values)

    if not parsed:
        raise TableError(f"{os.fspath(path)}: no cases found")

    return DecisionTable(
        attributes=tuple(f"a{i}" for i in range(1, width)),
        decision="d",
        cases=tuple(tuple(values[:-1]) for values in parsed),
        decisions=tuple(values[-1] for values in parsed),
    )
