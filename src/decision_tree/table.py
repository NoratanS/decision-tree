"""Loading a Decision Table from a text file (spec R1).

One case per line, the Decision Attribute in the last column. The separator is
detected from runs of whitespace, `,` `;` `|` `:` unless one is given.
"""

import math
import os
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Literal

type Kind = Literal["discrete", "continuous"]
type Value = str | None
"""A Conditional Attribute value of a Case; None is a Missing Value."""

MISSING_MARKERS = ("?", "NA", "")
"""Cells read as a Missing Value by default; `""` is the empty cell."""

_WHITESPACE = None
"""Separator meaning "runs of spaces and tabs"."""

_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")

# Detection order, the last resort for breaking a tie on column count. `,` and
# `:` come late because they also appear inside values (`1,5`, `12:30`).
_CANDIDATES: tuple[str | None, ...] = (";", "|", _WHITESPACE, ",", ":")


class TableError(ValueError):
    """The file cannot be read as a Decision Table. The message is one line."""


@dataclass(frozen=True)
class DecisionTable:
    attributes: tuple[str, ...]
    """Names of the Conditional Attributes, in column order."""
    decision: str
    """Name of the Decision Attribute."""
    cases: tuple[tuple[Value, ...], ...]
    """Conditional Attribute values of each Case; None is a Missing Value."""
    decisions: tuple[str, ...]
    """Decision Class of each Case, parallel to `cases`."""
    kinds: tuple[Kind, ...]
    """Type of each Conditional Attribute, parallel to `attributes`."""

    @cached_property
    def classes(self) -> tuple[str, ...]:
        """Decision Classes in order of first appearance in the file."""
        return tuple(dict.fromkeys(self.decisions))

    def values(self, attribute: int) -> tuple[str, ...]:
        """Known values of one Conditional Attribute, in order of first appearance."""
        return tuple(
            dict.fromkeys(
                value for case in self.cases if (value := case[attribute]) is not None
            )
        )

    def subset(self, indices: Sequence[int]) -> DecisionTable:
        """The Cases at `indices`, in that order, with the same names and types."""
        return replace(
            self,
            cases=tuple(self.cases[i] for i in indices),
            decisions=tuple(self.decisions[i] for i in indices),
        )


def load_table(
    path: str | os.PathLike[str],
    *,
    sep: str | None = None,
    header: bool = False,
    encoding: str = "utf-8",
    discrete: Iterable[str] = (),
    continuous: Iterable[str] = (),
    missing: Iterable[str] = MISSING_MARKERS,
    like: DecisionTable | None = None,
) -> DecisionTable:
    """Read a Decision Table.

    `sep` is any single character; `None` detects the separator. With `header`
    the first non-blank line names the columns, otherwise they are `a1 … an`
    and `d`. The file is decoded with `encoding`.

    Attribute types are inferred (ADR 0001); `discrete` and `continuous` name
    columns whose type is forced instead.

    A cell equal to one of the `missing` markers (after trimming) is a Missing
    Value, stored as None; `""` stands for the empty cell. Quoting does not
    change this: `"?"` is missing too. A missing decision is an error.

    With `like` (e.g. the Training Set, when reading a Test Set) the file must
    have that table's column count and takes its attribute types; `discrete`
    and `continuous` are then ignored.
    """
    name, numbers, header_names, rows = _read_rows(
        path, sep=sep, header=header, encoding=encoding, min_columns=2
    )
    if like is not None and len(rows[0]) != len(like.attributes) + 1:
        raise TableError(
            f"{name} has {len(rows[0])} columns, expected "
            f"{len(like.attributes) + 1} like the Training Set"
        )
    names = header_names or [*(f"a{i}" for i in range(1, len(rows[0]))), "d"]
    markers = frozenset(missing)
    decisions = _decisions(name, numbers, rows, markers)
    cases = _cases(rows, len(rows[0]) - 1, markers)
    if like is None:
        kinds = [_infer_kind(column) for column in zip(*cases, strict=True)]
        forced = _overrides(names, discrete, continuous)
    else:
        kinds = list(like.kinds)
        forced = dict(enumerate(like.kinds))
    _check_continuous(name, names, forced, numbers, cases)
    for index, kind in forced.items():
        kinds[index] = kind

    return DecisionTable(
        attributes=tuple(names[:-1]),
        decision=names[-1],
        cases=cases,
        decisions=decisions,
        kinds=tuple(kinds),
    )


@dataclass(frozen=True)
class Cases:
    """Cases to classify, with their decisions if the file has them."""

    cases: tuple[tuple[Value, ...], ...]
    """Conditional Attribute values of each Case; None is a Missing Value."""
    decisions: tuple[str, ...] | None
    """Decision Class of each Case, parallel to `cases`; None if the file has
    no decision column."""


def load_cases(
    path: str | os.PathLike[str],
    *,
    like: DecisionTable,
    sep: str | None = None,
    header: bool = False,
    encoding: str = "utf-8",
    missing: Iterable[str] = MISSING_MARKERS,
) -> Cases:
    """Read Cases to classify with a tree built from `like` (spec R10).

    The file has one column per Conditional Attribute of `like`, optionally
    followed by the decision. The attribute types are `like`'s. The other
    options are as in `load_table`; a header line is skipped.
    """
    name, numbers, _, rows = _read_rows(
        path, sep=sep, header=header, encoding=encoding, min_columns=1
    )
    width = len(like.attributes)
    if len(rows[0]) not in (width, width + 1):
        raise TableError(
            f"{name} has {len(rows[0])} columns, expected {width} (without a "
            f"decision) or {width + 1} like the Training Set"
        )
    markers = frozenset(missing)
    decisions = (
        _decisions(name, numbers, rows, markers) if len(rows[0]) > width else None
    )
    cases = _cases(rows, width, markers)
    _check_continuous(
        name, like.attributes, dict(enumerate(like.kinds)), numbers, cases
    )
    return Cases(cases, decisions)


def _read_rows(
    path: str | os.PathLike[str],
    *,
    sep: str | None,
    header: bool,
    encoding: str,
    min_columns: int,
) -> tuple[str, list[int], list[str] | None, list[list[str]]]:
    """File name, line number of each row, header names (or None), and the rows
    split into trimmed values."""
    name = os.fspath(path)
    if sep is not None and len(sep) != 1:
        raise TableError(f"separator must be a single character, got {sep!r}")
    try:
        with open(path, encoding=encoding) as file:
            text = file.read()
    except LookupError:
        raise TableError(f"unknown encoding {encoding!r}") from None
    except UnicodeDecodeError as error:
        raise TableError(
            f"cannot read {name} as {encoding}: {error.reason} at byte "
            f"{error.start}; try another encoding (e.g. cp1250)"
        ) from None
    except OSError as error:
        raise TableError(f"cannot read {name}: {error.strerror}") from None

    lines = [
        (number, line)
        for number, line in enumerate(text.splitlines(), start=1)
        if line.strip()
    ]
    if len(lines) <= header:
        raise TableError(f"{name}: no cases found")

    candidates = _CANDIDATES if sep is None else (sep,)
    rows = _split_table(name, lines, candidates, min_columns)
    numbers = [number for number, _ in lines]
    names = None
    if header:
        names = rows.pop(0)
        _check_header(name, numbers.pop(0), names)
    return name, numbers, names, rows


def _decisions(
    name: str, numbers: list[int], rows: list[list[str]], markers: frozenset[str]
) -> tuple[str, ...]:
    """The last column; a Missing Value marker there is an error."""
    for number, row in zip(numbers, rows, strict=True):
        if row[-1] in markers:
            raise TableError(
                f"{name}, line {number}: the decision is missing "
                f"({row[-1]!r} is a Missing Value marker)"
            )
    return tuple(row[-1] for row in rows)


def _cases(
    rows: list[list[str]], width: int, markers: frozenset[str]
) -> tuple[tuple[Value, ...], ...]:
    """The first `width` columns, with Missing Value markers as None."""
    return tuple(
        tuple(None if value in markers else value for value in row[:width])
        for row in rows
    )


def _overrides(
    names: list[str], discrete: Iterable[str], continuous: Iterable[str]
) -> dict[int, Kind]:
    """Column index → forced Kind; unknown names are an error."""
    forced: dict[int, Kind] = {}
    requested: tuple[tuple[Kind, Iterable[str]], ...] = (
        ("discrete", discrete),
        ("continuous", continuous),
    )
    for kind, columns in requested:
        for column in columns:
            if column == names[-1]:
                raise TableError(
                    f"{column!r} is the Decision Attribute and cannot be forced {kind}"
                )
            if column not in names:
                known = ", ".join(names[:-1])
                raise TableError(
                    f"unknown column {column!r} forced {kind} (columns: {known})"
                )
            index = names.index(column)
            if forced.get(index, kind) != kind:
                raise TableError(f"{column!r} is forced both discrete and continuous")
            forced[index] = kind
    return forced


def _check_continuous(
    name: str,
    names: Sequence[str],
    kinds: dict[int, Kind],
    numbers: Iterable[int],
    cases: tuple[tuple[Value, ...], ...],
) -> None:
    """Every known value in a column of Kind continuous must be a number."""
    numbers = list(numbers)
    for index, kind in kinds.items():
        if kind == "continuous":
            column = (case[index] for case in cases)
            _check_numeric(name, names[index], numbers, column)


def _check_numeric(
    name: str, column: str, numbers: Iterable[int], values: Iterable[Value]
) -> None:
    for number, value in zip(numbers, values, strict=True):
        if value is not None and not _NUMBER.fullmatch(value):
            raise TableError(
                f"{name}, line {number}: {column} is continuous but {value!r} "
                "is not a number"
            )


def _infer_kind(values: tuple[Value, ...]) -> Kind:
    """Continuous only if every known value is a number and one has a fraction
    (ADR 0001)."""
    known = [v for v in values if v is not None]
    if all(_NUMBER.fullmatch(v) for v in known) and any(map(_fractional, known)):
        return "continuous"
    return "discrete"


def _fractional(value: str) -> bool:
    number = float(value)
    return math.isfinite(number) and not number.is_integer()


def _check_header(name: str, line: int, names: list[str]) -> None:
    if "" in names:
        raise TableError(f"{name}, line {line}: empty column name in the header")
    duplicates = [column for column, count in Counter(names).items() if count > 1]
    if duplicates:
        raise TableError(
            f"{name}, line {line}: duplicate column name {duplicates[0]!r} "
            "in the header"
        )


class _LineError(Exception):
    """One line cannot be split with a given separator."""

    def __init__(self, line: int, message: str) -> None:
        super().__init__(line, message)
        self.line = line
        self.message = message


def _split_table(
    name: str,
    lines: list[tuple[int, str]],
    candidates: tuple[str | None, ...],
    min_columns: int,
) -> list[list[str]]:
    """Split with the candidate that gives the most columns on every line.

    On a tie, prefer a candidate whose values contain none of the other tied
    separators (`1,5;2,3` is `;`-separated), then the earlier candidate.
    """
    fitting: dict[str | None, list[list[str]]] = {}
    errors: list[_LineError] = []
    for sep in candidates:
        try:
            fitting[sep] = _split_all(lines, sep, min_columns)
        except _LineError as error:
            errors.append(error)
    if not fitting:
        furthest = max(errors, key=lambda error: error.line)
        hint = "; no separator fits every line, set one" if len(candidates) > 1 else ""
        raise TableError(f"{name}, line {furthest.line}: {furthest.message}{hint}")

    width = max(len(rows[0]) for rows in fitting.values())
    tied = [sep for sep, rows in fitting.items() if len(rows[0]) == width]

    def splits_inside_values(sep: str | None) -> bool:
        return any(
            _contains(value, other)
            for row in fitting[sep]
            for value in row
            for other in tied
            if other is not sep
        )

    return fitting[min(tied, key=splits_inside_values)]


def _contains(value: str, sep: str | None) -> bool:
    if sep is _WHITESPACE:
        return any(char.isspace() for char in value)
    return sep in value


def _split_all(
    lines: list[tuple[int, str]], sep: str | None, min_columns: int
) -> list[list[str]]:
    rows: list[list[str]] = []
    for number, line in lines:
        try:
            values = _split(line, sep)
        except ValueError as error:
            raise _LineError(number, str(error)) from None
        if not rows:
            if len(values) < min_columns:
                raise _LineError(
                    number, f"a Decision Table needs at least {min_columns} columns"
                )
        elif len(values) != len(rows[0]):
            raise _LineError(
                number,
                f"expected {len(rows[0])} columns separated by "
                f"{_separator_text(sep)}, found {len(values)}",
            )
        rows.append(values)
    return rows


def _split(line: str, sep: str | None) -> list[str]:
    """Split one line on `sep` and trim each value.

    A value starting with `"` is quoted up to the closing `"`: it may contain
    the separator and keeps its spaces; `""` inside quotes is a literal quote.
    A `"` later in a value is an ordinary character. Raises ValueError on an
    unclosed quote.
    """
    if '"' not in line:
        if sep is _WHITESPACE:
            return line.split()
        return [value.strip() for value in line.split(sep)]

    values: list[str] = []
    value: list[str] = []
    quoted_end: int | None = None  # length of the quoted text at `value`'s start
    quoted = False
    chars = iter(line)
    for char in chars:
        if quoted:
            if char != '"':
                value.append(char)
                continue
            following = next(chars, None)
            if following == '"':
                value.append('"')
                continue
            quoted, quoted_end = False, len(value)
            if following is None:
                break
            char = following
        if char == '"' and quoted_end is None and all(c.isspace() for c in value):
            quoted, value = True, []
        elif char == sep or (sep is _WHITESPACE and char.isspace()):
            if sep is not _WHITESPACE or value or quoted_end is not None:
                values.append(_trimmed(value, quoted_end))
            value, quoted_end = [], None
        else:
            value.append(char)
    if quoted:
        raise ValueError("unclosed quote")
    if sep is not _WHITESPACE or value or quoted_end is not None:
        values.append(_trimmed(value, quoted_end))
    return values


def _trimmed(value: list[str], quoted_end: int | None) -> str:
    """Strip outer spaces, but not from the quoted text at the start."""
    text = "".join(value)
    if quoted_end is None:
        return text.strip()
    return text[:quoted_end] + text[quoted_end:].rstrip()


def _separator_text(sep: str | None) -> str:
    return "whitespace" if sep is _WHITESPACE else repr(sep)
