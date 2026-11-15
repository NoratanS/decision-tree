"""Loading a Decision Table from a text file (spec R1).

One case per line, the Decision Attribute in the last column unless another
is named. The separator is detected from runs of whitespace, `,` `;` `|` `:`
unless one is given. Unless the separator is `,`, `1,5` is the number `1.5`.
"""

import codecs
import os
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from functools import cached_property
from typing import Literal

type Kind = Literal["discrete", "continuous"]
type Value = str | None
"""A Conditional Attribute value of a Case; None is a Missing Value."""

MISSING_MARKERS = ("?", "NA", "")
"""Cells read as a Missing Value by default; `""` is the empty cell."""

MAX_DISCRETE_WHOLE_NUMBERS = 10
"""A column of whole numbers with more distinct known values than this is
inferred continuous (ADR 0004)."""

_WHITESPACE = None
"""Separator meaning "runs of spaces and tabs"."""

_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_DECIMAL_COMMA_NUMBER = re.compile(r"-?\d+,\d+")
"""A number with a decimal comma (`1,5`), read when the separator is not `,`."""

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


@dataclass(frozen=True)
class ReadOptions:
    """How to read a Decision Table file; the same for every file of a run.

    `sep` is any single character; `None` detects the separator. With `header`
    the first non-blank line names the columns, otherwise column K is `aK`.
    `decision` names the Decision Attribute's column (`None`: the last one);
    without a header it is then renamed `d`, and the default last column is `d`
    too. The decision always comes last in a loaded table. The file is decoded
    with `encoding`; under any spelling of UTF-8 one leading byte order mark is
    dropped.

    A cell equal to one of the `missing` markers (after trimming) is a Missing
    Value, stored as None; `""` stands for the empty cell. Quoting does not
    change this: `"?"` is missing too.

    `ignore` names columns left out of the table, like `decision` names one:
    header names, or `aK` without a header. The other columns keep their
    names, so without a header they keep their positions (`a2 … an` when `a1`
    is ignored).
    """

    sep: str | None = None
    header: bool = False
    encoding: str = "utf-8"
    missing: tuple[str, ...] = MISSING_MARKERS
    decision: str | None = None
    ignore: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.sep is not None and len(self.sep) != 1:
            raise TableError(f"separator must be a single character, got {self.sep!r}")


DEFAULT_READ_OPTIONS = ReadOptions()
"""No header, detected separator, UTF-8, the default Missing Value markers,
the decision in the last column."""


def parse_columns(text: str) -> tuple[str, ...]:
    """Column names from a comma list: `"a1, a3"` → `("a1", "a3")`."""
    return tuple(name.strip() for name in text.split(",") if name.strip())


def parse_markers(text: str) -> tuple[str, ...]:
    """Missing Value markers from a comma list: `"?, NA,"` → `("?", "NA", "")`.

    An empty item is the empty cell.
    """
    return tuple(marker.strip() for marker in text.split(","))


def load_table(
    path: str | os.PathLike[str],
    options: ReadOptions = DEFAULT_READ_OPTIONS,
    *,
    discrete: Iterable[str] = (),
    continuous: Iterable[str] = (),
    like: DecisionTable | None = None,
) -> DecisionTable:
    """Read a Decision Table as `options` say. A missing decision is an error.

    Attribute types are inferred (ADR 0001, 0004); `discrete` and `continuous` name
    columns whose type is forced instead, never columns `options` ignore.

    With `like` (e.g. the Training Set, when reading a Test Set) the file must
    have that table's column count and takes its attribute types, so `discrete`
    and `continuous` cannot be given. With `header`, its attribute names must
    also be `like`'s, in the same order, and its decision column, wherever it
    is, is found by `like`'s decision name when `options.decision` is given.
    """
    discrete, continuous = tuple(discrete), tuple(continuous)
    if like is not None and (discrete or continuous):
        raise TableError(
            "discrete and continuous cannot be forced on a table read like "
            "another: it takes that table's types"
        )
    for kind, columns in (("discrete", discrete), ("continuous", continuous)):
        for column in columns:
            if column in options.ignore:
                raise TableError(f"{column!r} is ignored and cannot be forced {kind}")
    schema = _Inferred(discrete, continuous) if like is None else _Like(like)
    loaded = _load(path, options, schema=schema, decision="required")
    assert loaded.decisions is not None  # the decision is required
    return DecisionTable(
        attributes=loaded.names[:-1],
        decision=loaded.names[-1],
        cases=loaded.cases,
        decisions=loaded.decisions,
        kinds=loaded.kinds,
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
    options: ReadOptions = DEFAULT_READ_OPTIONS,
    *,
    like: DecisionTable,
) -> Cases:
    """Read Cases to classify with a tree built from `like` (spec R10).

    The file has one column per Conditional Attribute of `like`, in `like`'s
    order, plus optionally the decision column where `options.decision` puts
    it. The attribute types are `like`'s. With `header`, the column names must
    be `like`'s, in the file's order.
    """
    loaded = _load(path, options, schema=_Like(like), decision="optional")
    return Cases(loaded.cases, loaded.decisions)


type _Decision = Literal["required", "optional"]
"""Whether a file must have the decision column or may leave it out."""


@dataclass(frozen=True)
class _Loaded:
    """A file read and checked, before it becomes a DecisionTable or Cases."""

    names: tuple[str, ...]
    """Column names, the decision's last if the file has that column."""
    kinds: tuple[Kind, ...]
    """Type of each Conditional Attribute."""
    cases: tuple[tuple[Value, ...], ...]
    decisions: tuple[str, ...] | None
    """Decision Class of each Case; None if the file has no decision column."""


def _load(
    path: str | os.PathLike[str],
    options: ReadOptions,
    *,
    schema: _Inferred | _Like,
    decision: _Decision,
) -> _Loaded:
    """Read a file, check it against `schema`, split its cases and decisions,
    and check that continuous columns hold numbers, rewritten with a decimal
    point."""
    parsed = _read_rows(
        path,
        options,
        min_columns=2 if decision == "required" else 1,
        widths=schema.widths(options, decision),
    )
    parsed = schema.without_ignored(parsed, options, decision)
    names, width, index = schema.columns(parsed, decision, options.decision)
    if index is not None:
        parsed = _decision_last(parsed, index)
    markers = frozenset(options.missing)
    decisions = _decisions(parsed, markers) if index is not None else None
    cases = _cases(parsed.rows, width, markers)
    decision_names = () if index is None else (names[-1], parsed.names[index])
    decimal_comma = parsed.sep != ","
    kinds = schema.kinds(names, cases, decision_names, decimal_comma=decimal_comma)
    cases = _continuous_numbers(
        parsed.name, names, kinds, parsed.numbers, cases, decimal_comma=decimal_comma
    )
    return _Loaded(names, kinds, cases, decisions)


@dataclass(frozen=True)
class _Inferred:
    """Columns named by the file; types inferred (ADR 0001, 0004) unless forced."""

    discrete: tuple[str, ...]
    continuous: tuple[str, ...]

    def widths(self, options: ReadOptions, decision: _Decision) -> None:
        """Any column count is allowed."""
        return None

    def without_ignored(
        self, parsed: _Rows, options: ReadOptions, decision: _Decision
    ) -> _Rows:
        """`parsed` without the columns `options` ignore; the file has the
        decision column, which cannot be one of them."""
        index = _decision_index(parsed, options.decision)
        return _without_ignored(parsed, options.ignore, parsed.names[index])

    def columns(
        self, parsed: _Rows, decision: _Decision, decision_name: str | None
    ) -> tuple[tuple[str, ...], int, int]:
        """Column names (the decision's last), the number of Conditional
        Attributes and the decision's column index. The file must have the
        decision column, so `decision` must be required."""
        if decision != "required":
            raise ValueError("inferring a table's columns needs its decision column")
        width = len(parsed.rows[0])
        index = _decision_index(parsed, decision_name)
        return _file_names(parsed, index), width - 1, index

    def kinds(
        self,
        names: tuple[str, ...],
        cases: tuple[tuple[Value, ...], ...],
        decision_names: tuple[str, ...],
        *,
        decimal_comma: bool,
    ) -> tuple[Kind, ...]:
        """Each Conditional Attribute's forced or inferred type. The decision,
        by any of `decision_names` (e.g. `d` and its file name `a1`), cannot
        be forced."""
        forced = _overrides(names, decision_names, self.discrete, self.continuous)
        columns = enumerate(zip(*cases, strict=True))
        return tuple(
            forced.get(i) or _infer_kind(column, decimal_comma=decimal_comma)
            for i, column in columns
        )


@dataclass(frozen=True)
class _Like:
    """Columns and types of another table, e.g. the Training Set."""

    table: DecisionTable

    def widths(self, options: ReadOptions, decision: _Decision) -> tuple[int, ...]:
        """The column counts a file may have: the Training Set's attributes
        and the ignored columns, plus the decision column unless it is
        optional."""
        width = len(self.table.attributes) + len(set(options.ignore))
        return (width + 1,) if decision == "required" else (width, width + 1)

    def without_ignored(
        self, parsed: _Rows, options: ReadOptions, decision: _Decision
    ) -> _Rows:
        """`parsed` without the columns `options` ignore, after checking the
        file has the Training Set's column count before they are dropped.

        The file keeps the ignored columns. Without a header, a file without
        the decision column names its columns like the Training Set's file
        does, so `aK` is the same column in both; a file that dropped an
        ignored column but kept its decision cannot be told from one with the
        ignored column and no decision, and is read as the latter.
        """
        allowed = self.widths(options, decision)
        width = len(self.table.attributes) + len(set(options.ignore))
        found = len(parsed.rows[0])
        if decision == "required":
            expected = f"{width + 1}"
        else:
            expected = f"{width} (without a decision) or {width + 1}"
        if found not in allowed:
            kept = f" (with the ignored {', '.join(options.ignore)})"
            raise TableError(
                f"{parsed.name} has {found} columns, expected {expected}"
                f"{kept if options.ignore else ''} like the Training Set"
            )
        if found == width + 1:
            return _without_ignored(
                parsed, options.ignore, self._decision(parsed, options)
            )
        if not parsed.header:
            names = [f"a{k}" for k in range(1, width + 2)]
            names.remove(options.decision if options.decision in names else names[-1])
            parsed = replace(parsed, names=names)
        return _without_ignored(parsed, options.ignore, None)

    def _decision(self, parsed: _Rows, options: ReadOptions) -> str | None:
        """The name of `parsed`'s decision column, before any is ignored: with
        a header the Training Set's decision name (None if no column has it,
        so the name check reports the file), else `options.decision`'s column
        or the last one."""
        if parsed.header:
            return self.table.decision if self.table.decision in parsed.names else None
        return parsed.names[_decision_index(parsed, options.decision)]

    def columns(
        self, parsed: _Rows, decision: _Decision, decision_name: str | None
    ) -> tuple[tuple[str, ...], int, int | None]:
        """Column names (the decision's last), the number of Conditional
        Attributes and the decision's column index (None if the file has no
        decision column), after checking the file has `table`'s column count
        and, with a header, its names in the file's order.

        Without a header, a file that must have the decision is named like any
        other (`aK`, `d`); Cases keep `table`'s names.
        """
        like = self.table
        width = len(like.attributes)
        found = len(parsed.rows[0])
        like_names = (*like.attributes, like.decision)
        if found == width:
            index = None
        elif parsed.header and decision_name is not None:
            index = _header_decision_index(parsed.names, like)
        else:
            index = _decision_index(parsed, decision_name)
        if parsed.header:
            in_file_order = list(like.attributes)
            if index is not None:
                in_file_order.insert(index, like.decision)
            for column, (name, wanted) in enumerate(
                zip(parsed.names, in_file_order, strict=True), start=1
            ):
                if name != wanted:
                    raise TableError(
                        f"{parsed.name}: column {column} is named {name!r}, "
                        f"expected {wanted!r} like the Training Set"
                    )
        if index is None:  # Cases without a decision column
            return like.attributes, width, index
        if not parsed.header and decision == "optional":
            return like_names, width, index
        return _file_names(parsed, index), width, index

    def kinds(
        self,
        names: tuple[str, ...],
        cases: tuple[tuple[Value, ...], ...],
        decision_names: tuple[str, ...],
        *,
        decimal_comma: bool,
    ) -> tuple[Kind, ...]:
        return self.table.kinds


def _without_ignored(
    parsed: _Rows, ignore: Sequence[str], decision_name: str | None
) -> _Rows:
    """`parsed` without the columns named in `ignore`; the others keep their
    names. An unknown name, or the decision column's (`decision_name`, None if
    the file has none), is an error, as is leaving no Conditional Attribute."""
    if not ignore:
        return parsed
    for column in ignore:
        if column == decision_name:
            raise TableError(
                f"{parsed.name}: {column!r} is the Decision Attribute "
                "and cannot be ignored"
            )
        if column not in parsed.names:
            raise TableError(
                f"{parsed.name}: unknown column {column!r} to ignore "
                f"(columns: {', '.join(parsed.names)})"
            )
    keep = [i for i, name in enumerate(parsed.names) if name not in ignore]
    attributes_left = len(keep) - (decision_name is not None)
    if attributes_left == 0:
        raise TableError(
            f"{parsed.name}: ignoring {', '.join(ignore)} leaves no attributes"
        )
    return replace(
        parsed,
        names=[parsed.names[i] for i in keep],
        rows=[[row[i] for i in keep] for row in parsed.rows],
    )


def _file_names(parsed: _Rows, index: int) -> tuple[str, ...]:
    """Column names with the decision's (at `index`) last: the header's, or
    without a header `aK` for column K and `d` for the decision."""
    names = list(parsed.names)
    if not parsed.header:
        names[index] = "d"
    return (*names[:index], *names[index + 1 :], names[index])


def _header_decision_index(header: Sequence[str], like: DecisionTable) -> int:
    """Where `like`'s decision is in `header`, found by its name. If no column
    has that name, the first column that is not the next of `like`'s
    attributes, so the name check reports it (spec R1.10)."""
    if like.decision in header:
        return list(header).index(like.decision)
    attributes = like.attributes
    for column, name in enumerate(header):
        if column >= len(attributes) or name != attributes[column]:
            return column
    return len(header) - 1


def _decision_index(parsed: _Rows, name: str | None) -> int:
    """The column index of the decision named `name`; None is the last column.

    Names are the header's, or `a1 … an` without a header. An unknown name is
    an error listing the valid ones.
    """
    width = len(parsed.rows[0])
    if name is None:
        return width - 1
    names = parsed.names
    if name not in names:
        raise TableError(
            f"{parsed.name}: unknown decision column {name!r} "
            f"(columns: {', '.join(names)})"
        )
    return names.index(name)


def _decision_last(parsed: _Rows, index: int) -> _Rows:
    """`parsed` with the column at `index` moved to the end of every row."""
    if index == len(parsed.rows[0]) - 1:
        return parsed
    rows = [[*row[:index], *row[index + 1 :], row[index]] for row in parsed.rows]
    return replace(parsed, rows=rows)


@dataclass(frozen=True)
class _Rows:
    """A file split into rows of trimmed values."""

    name: str
    """The file name, for error messages."""
    numbers: list[int]
    """Line number of each row."""
    header: bool
    """Whether the names come from the file's header line."""
    sep: str | None
    """The separator the rows were split on; None is whitespace."""
    names: list[str]
    """Column names: the header's, or `a1 … an` without one."""
    rows: list[list[str]]


def _drop_utf8_bom(text: str, encoding: str) -> str:
    """`text` without one leading byte order mark if `encoding` is UTF-8.

    Decoding with plain UTF-8 and dropping the mark afterwards, rather than
    decoding with `utf-8-sig`, keeps decode error offsets counted from the
    start of the file.
    """
    if codecs.lookup(encoding).name == "utf-8":
        return text.removeprefix("\ufeff")
    return text


def _read_rows(
    path: str | os.PathLike[str],
    options: ReadOptions,
    *,
    min_columns: int,
    widths: tuple[int, ...] | None,
) -> _Rows:
    """`path` split into rows; a detected separator prefers giving one of
    `widths` columns (None: any count)."""
    name = os.fspath(path)
    try:
        with open(path, encoding=options.encoding) as file:
            text = _drop_utf8_bom(file.read(), options.encoding)
    except LookupError:
        raise TableError(f"unknown encoding {options.encoding!r}") from None
    except UnicodeDecodeError as error:
        raise TableError(
            f"cannot read {name} as {options.encoding}: {error.reason} at byte "
            f"{error.start}; try another encoding (e.g. cp1250)"
        ) from None
    except OSError as error:
        raise TableError(f"cannot read {name}: {error.strerror}") from None

    lines = [
        (number, line)
        for number, line in enumerate(text.splitlines(), start=1)
        if line.strip()
    ]
    if len(lines) <= options.header:
        raise TableError(f"{name}: no cases found")

    candidates = _CANDIDATES if options.sep is None else (options.sep,)
    # Without a header, an empty last column may be a real attribute when the
    # decision is elsewhere; with one, its empty name marks a trailing separator.
    trailing = options.header or options.decision is None
    sep, rows = _split_table(name, lines, candidates, min_columns, widths, trailing)
    numbers = [number for number, _ in lines]
    if options.header:
        names = rows.pop(0)
        _check_header(name, numbers.pop(0), names)
    else:
        names = [f"a{k}" for k in range(1, len(rows[0]) + 1)]
    return _Rows(name, numbers, options.header, sep, names, rows)


def _decisions(parsed: _Rows, markers: frozenset[str]) -> tuple[str, ...]:
    """The last column; a Missing Value marker there is an error."""
    for number, row in zip(parsed.numbers, parsed.rows, strict=True):
        if row[-1] in markers:
            raise TableError(
                f"{parsed.name}, line {number}: the decision is missing "
                f"({row[-1]!r} is a Missing Value marker)"
            )
    return tuple(row[-1] for row in parsed.rows)


def _cases(
    rows: list[list[str]], width: int, markers: frozenset[str]
) -> tuple[tuple[Value, ...], ...]:
    """The first `width` columns, with Missing Value markers as None."""
    return tuple(
        tuple(None if value in markers else value for value in row[:width])
        for row in rows
    )


def _overrides(
    names: Sequence[str],
    decision_names: Sequence[str],
    discrete: Iterable[str],
    continuous: Iterable[str],
) -> dict[int, Kind]:
    """Column index → forced Kind; unknown names and any of `decision_names`
    (the caller passes every name the decision goes by) are an error."""
    forced: dict[int, Kind] = {}
    requested: tuple[tuple[Kind, Iterable[str]], ...] = (
        ("discrete", discrete),
        ("continuous", continuous),
    )
    for kind, columns in requested:
        for column in columns:
            if column in decision_names:
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


def _continuous_numbers(
    name: str,
    names: Sequence[str],
    kinds: Sequence[Kind],
    lines: Sequence[int],
    cases: tuple[tuple[Value, ...], ...],
    *,
    decimal_comma: bool,
) -> tuple[tuple[Value, ...], ...]:
    """`cases` with every known value of a continuous column checked to be a
    number and written with a decimal point; `lines` is each case's line
    number. Only cases with a value to rewrite are copied."""
    continuous = [i for i, kind in enumerate(kinds) if kind == "continuous"]
    if not continuous:
        return cases
    checked: list[tuple[Value, ...]] = []
    for line, case in zip(lines, cases, strict=True):
        rewritten: list[Value] | None = None
        for index in continuous:
            value = case[index]
            if value is None:
                continue
            number = _number(value, decimal_comma=decimal_comma)
            if number is None:
                raise TableError(
                    f"{name}, line {line}: {names[index]} is continuous "
                    f"but {value!r} is not a number"
                )
            if number != value:
                rewritten = rewritten or list(case)
                rewritten[index] = number
        checked.append(case if rewritten is None else tuple(rewritten))
    return tuple(checked)


def _number(value: str, *, decimal_comma: bool) -> str | None:
    """`value` as a number with a decimal point, or None if it is not a
    number. With `decimal_comma`, `1,5` is the number `1.5`."""
    if _NUMBER.fullmatch(value):
        return value
    if decimal_comma and _DECIMAL_COMMA_NUMBER.fullmatch(value):
        return value.replace(",", ".")
    return None


def _infer_kind(values: tuple[Value, ...], *, decimal_comma: bool) -> Kind:
    """Continuous only if every known value is a number and one has a fraction
    (ADR 0001), or they are whole numbers with more than
    MAX_DISCRETE_WHOLE_NUMBERS distinct values (ADR 0004)."""
    known = [_number(v, decimal_comma=decimal_comma) for v in values if v is not None]
    # Decimal is exact: 1e400 stays whole and 2**53 + 1 stays apart from 2**53.
    numbers = [Decimal(v) for v in known if v is not None]
    if len(numbers) < len(known):
        return "discrete"
    if any(n != n.to_integral_value() for n in numbers):
        return "continuous"
    if len(set(numbers)) > MAX_DISCRETE_WHOLE_NUMBERS:
        return "continuous"
    return "discrete"


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
    widths: tuple[int, ...] | None,
    trailing: bool,
) -> tuple[str | None, list[list[str]]]:
    """The separator, and the rows split with it, that gives the most columns
    on every line; with `trailing`, a separator ending every line is dropped
    (see `_split_all`).

    Candidates giving one of `widths` columns come first, if any does. `,` is
    left out if another candidate fits and every value it leaves holding a
    `,` is a decimal comma number (`1,5;0,5` is two numbers, not three
    columns). On a tie, prefer a candidate whose values contain none of the
    other tied separators (`1,5;2,3` is `;`-separated), then the earlier one.
    """
    fitting: dict[str | None, list[list[str]]] = {}
    errors: list[_LineError] = []
    for sep in candidates:
        try:
            rows = _split_all(lines, sep, min_columns)
            if trailing and _ends_with_separator(lines, rows, sep, min_columns, widths):
                rows = [row[:-1] for row in rows]
            fitting[sep] = rows
        except _LineError as error:
            errors.append(error)
    if not fitting:
        furthest = max(errors, key=lambda error: error.line)
        hint = "; no separator fits every line, set one" if len(candidates) > 1 else ""
        raise TableError(f"{name}, line {furthest.line}: {furthest.message}{hint}")
    if widths is not None:
        wanted = {sep: rows for sep, rows in fitting.items() if len(rows[0]) in widths}
        fitting = wanted or fitting
    if "," in fitting and any(
        sep != "," and _decimal_commas_only(rows) for sep, rows in fitting.items()
    ):
        del fitting[","]

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

    sep = min(tied, key=splits_inside_values)
    return sep, fitting[sep]


def _decimal_commas_only(rows: list[list[str]]) -> bool:
    """Some value holds a `,` and every such value is a decimal comma number."""
    with_comma = [value for row in rows for value in row if "," in value]
    return bool(with_comma) and all(
        _DECIMAL_COMMA_NUMBER.fullmatch(value) for value in with_comma
    )


def _contains(value: str, sep: str | None) -> bool:
    if sep is _WHITESPACE:
        return any(char.isspace() for char in value)
    return sep in value


def _split_all(
    lines: list[tuple[int, str]], sep: str | None, min_columns: int
) -> list[list[str]]:
    """Every line split on `sep`, all with one column count."""
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


def _ends_with_separator(
    lines: list[tuple[int, str]],
    rows: list[list[str]],
    sep: str | None,
    min_columns: int,
    widths: tuple[int, ...] | None,
) -> bool:
    """Every line ends in `sep` and an unquoted empty cell (`1,2,yes,`), and
    dropping that column leaves at least `min_columns`, one of `widths` if
    given. A quoted `""` is a value, not a trailing separator."""
    if sep is _WHITESPACE or len(rows[0]) - 1 < min_columns:
        return False
    if widths is not None and len(rows[0]) - 1 not in widths:
        return False
    return all(
        row[-1] == "" and not line.rsplit(sep, 1)[-1].strip()
        for (_, line), row in zip(lines, rows, strict=True)
    )


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
