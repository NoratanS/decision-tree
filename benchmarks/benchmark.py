"""Benchmark the C4.5 tree on UCI datasets against published C4.5 error rates.

Usage:
    uv run python benchmarks/benchmark.py                 # every dataset
    uv run python benchmarks/benchmark.py iris glass      # some of them
    uv run python benchmarks/benchmark.py --repeats 1     # quick run
    uv run python benchmarks/benchmark.py --download      # refresh the raw files

Protocol (Quinlan 1996, "Improved Use of Continuous Attributes in C4.5",
JAIR 4, Table 1): ten complete stratified ten-fold cross-validations; the error
rate is the mean of the ten, with its standard error. Each fold trains on nine
blocks and tests the pruned tree on the tenth. adult uses its own train/test
files instead, as in its `adult.names`.

The reference error is C4.5 Release 7 (Quinlan's table), the release without
the MDL threshold penalty, like this implementation. Release 7 uses the
average-Gain filter, so the default here is `--quinlan-gain-filter -m 2 -c 25`;
`--course` turns the filter off.

Raw UCI files live in `benchmarks/data/raw/`. Each raw file is rewritten into a
temporary directory with the decision last and ID columns dropped: the loader
has no option to drop columns (ticket 35), and a decision-last copy keeps one
read path for every dataset, though `--decision` (ticket 31) now reads a
class-first file directly. Integer columns
are read as discrete by default (ADR 0001, ticket 32), so each dataset names its
numeric columns as continuous.

Standard library and `decision_tree` only (CLAUDE.md ground rules).
"""

import argparse
import csv
import math
import random
import statistics
import sys
import tempfile
import time
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from decision_tree import BuildOptions, ReadOptions, build_tree, evaluate, prune
from decision_tree.table import DecisionTable, load_table
from decision_tree.tree import size

UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases"
RAW = Path(__file__).parent / "data" / "raw"
CONFIDENCE = 0.25
OK_POINTS = 3.0
"""A result within this many percentage points of the reference is `ok`."""

type Row = list[str]


@dataclass(frozen=True)
class Dataset:
    name: str
    files: dict[str, str]
    """Raw file name → download URL."""
    prepare: Callable[[Path], list[Row]]
    """Raw file → rows with the decision last."""
    continuous: Sequence[int] = ()
    """1-based attribute numbers (`a1` …) of the prepared rows to force continuous."""
    reference: float | None = None
    """Published C4.5 error rate in percent; None when there is none."""
    source: str = ""
    test_file: str | None = None
    """For a train/test dataset: the raw Test Set file, prepared like the first."""


# --- preparing raw files ----------------------------------------------------


def _read_csv(
    path: Path, *, skip: Callable[[str], bool] = lambda _: False
) -> list[Row]:
    with path.open(newline="", encoding="utf-8") as file:
        lines = [line for line in file if line.strip() and not skip(line)]
    return [[cell.strip() for cell in row] for row in csv.reader(lines)]


def _read_whitespace(path: Path) -> list[Row]:
    with path.open(encoding="utf-8") as file:
        return [line.split() for line in file if line.strip()]


def _as_is(path: Path) -> list[Row]:
    return _read_csv(path)


def _drop_first(path: Path) -> list[Row]:
    return [row[1:] for row in _read_csv(path)]


def _class_first(path: Path) -> list[Row]:
    return [[*row[1:], row[0]] for row in _read_csv(path)]


def _heart(path: Path) -> list[Row]:
    # num 0 = no heart disease; 1-4 = presence (Quinlan's heart-c has 2 classes)
    rows = _read_csv(path)
    return [[*row[:-1], "absent" if row[-1] == "0" else "present"] for row in rows]


def _adult(path: Path) -> list[Row]:
    # adult.test starts with "|1x3 Cross validator" and ends classes with "."
    rows = _read_csv(path, skip=lambda line: line.startswith("|"))
    return [[*row[:-1], row[-1].rstrip(".")] for row in rows]


def _cols(*numbers: int) -> tuple[int, ...]:
    return numbers


DATASETS = [
    Dataset(
        "iris",
        {"iris.data": f"{UCI}/iris/iris.data"},
        _as_is,
        continuous=_cols(1, 2, 3, 4),
        reference=4.87,
        source="Quinlan 1996, Table 1, Rel 7",
    ),
    Dataset(
        "breast-w",
        {
            "breast-cancer-wisconsin.data": (
                f"{UCI}/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
            )
        },
        _drop_first,
        continuous=range(1, 10),
        reference=5.29,
        source="Quinlan 1996, Table 1, Rel 7",
    ),
    Dataset(
        "glass",
        {"glass.data": f"{UCI}/glass/glass.data"},
        _drop_first,
        continuous=range(1, 10),
        reference=32.1,
        source="Quinlan 1996, Table 1, Rel 7",
    ),
    Dataset(
        "credit-a",
        {"crx.data": f"{UCI}/credit-screening/crx.data"},
        _as_is,
        continuous=_cols(2, 3, 8, 11, 14, 15),
        reference=15.8,
        source="Quinlan 1996, Table 1, Rel 7",
    ),
    Dataset(
        "credit-g",
        {"german.data": f"{UCI}/statlog/german/german.data"},
        _read_whitespace,
        continuous=_cols(2, 5, 8, 11, 13, 16, 18),
        reference=28.9,
        source="Quinlan 1996, Table 1, Rel 7",
    ),
    Dataset(
        "heart-c",
        {"processed.cleveland.data": f"{UCI}/heart-disease/processed.cleveland.data"},
        _heart,
        continuous=_cols(1, 4, 5, 8, 10, 12),
        reference=24.9,
        source="Quinlan 1996, Table 1, Rel 7",
    ),
    Dataset(
        "hepatitis",
        {"hepatitis.data": f"{UCI}/hepatitis/hepatitis.data"},
        _class_first,
        continuous=_cols(1, 14, 15, 16, 17, 18),
        reference=21.7,
        source="Quinlan 1996, Table 1, Rel 7",
    ),
    Dataset(
        "sonar",
        {
            "sonar.all-data": (
                f"{UCI}/undocumented/connectionist-bench/sonar/sonar.all-data"
            )
        },
        _as_is,
        continuous=range(1, 61),
        reference=28.4,
        source="Quinlan 1996, Table 1, Rel 7",
    ),
    Dataset("car", {"car.data": f"{UCI}/car/car.data"}, _as_is),
    Dataset(
        "mushroom",
        {"agaricus-lepiota.data": f"{UCI}/mushroom/agaricus-lepiota.data"},
        _class_first,
    ),
    Dataset(
        "vote",
        {"house-votes-84.data": f"{UCI}/voting-records/house-votes-84.data"},
        _class_first,
    ),
    Dataset(
        "adult",
        {
            "adult.data": f"{UCI}/adult/adult.data",
            "adult.test": f"{UCI}/adult/adult.test",
        },
        _adult,
        continuous=_cols(1, 3, 5, 11, 12, 13),
        reference=15.54,
        source="adult.names (C4.5, train/test)",
        test_file="adult.test",
    ),
]


# --- downloading ------------------------------------------------------------


def download(datasets: Sequence[Dataset], *, force: bool) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        for name, url in dataset.files.items():
            target = RAW / name
            if target.exists() and not force:
                continue
            print(f"downloading {url}", file=sys.stderr)
            request = urllib.request.Request(
                url, headers={"User-Agent": "decision-tree-benchmark"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                target.write_bytes(response.read())


def _load(dataset: Dataset, raw: str, workdir: Path, like: DecisionTable | None):
    rows = dataset.prepare(RAW / raw)
    path = workdir / f"{raw}.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows(rows)
    options = ReadOptions(sep=",")
    if like is not None:
        return load_table(path, options, like=like)
    continuous = [f"a{n}" for n in dataset.continuous]
    return load_table(path, options, continuous=continuous)


# --- cross-validation -------------------------------------------------------


def stratified_folds(
    table: DecisionTable, k: int, rng: random.Random
) -> list[list[int]]:
    """`k` blocks of similar size and class distribution (Quinlan 1996, note 3)."""
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, decision in enumerate(table.decisions):
        by_class[decision].append(index)
    folds: list[list[int]] = [[] for _ in range(k)]
    position = 0
    for decision in table.classes:
        indices = by_class[decision]
        rng.shuffle(indices)
        for index in indices:
            folds[position % k].append(index)
            position += 1
    return folds


@dataclass(frozen=True)
class Run:
    """One complete cross-validation (or one train/test run)."""

    pruned_error: float
    unpruned_error: float
    pruned_size: float


def _fit(train: DecisionTable, test: DecisionTable, options: BuildOptions):
    unpruned = build_tree(train, options)
    pruned = prune(unpruned, CONFIDENCE)
    return evaluate(unpruned, test), evaluate(pruned, test), size(pruned)


def cross_validate(
    table: DecisionTable, options: BuildOptions, k: int, rng: random.Random
) -> Run:
    pruned_errors = unpruned_errors = 0
    sizes = []
    folds = stratified_folds(table, k, rng)
    for held in range(k):
        train = [i for f, fold in enumerate(folds) if f != held for i in fold]
        unpruned, pruned, pruned_size = _fit(
            table.subset(train), table.subset(folds[held]), options
        )
        unpruned_errors += unpruned.errors
        pruned_errors += pruned.errors
        sizes.append(pruned_size)
    n = len(table.cases)
    return Run(pruned_errors / n, unpruned_errors / n, statistics.fmean(sizes))


# --- reporting --------------------------------------------------------------


@dataclass(frozen=True)
class Result:
    dataset: Dataset
    cases: int
    attributes: int
    classes: int
    runs: list[Run]
    seconds: float

    def mean(self, field: str) -> float:
        return statistics.fmean(getattr(run, field) for run in self.runs) * 100

    @property
    def standard_error(self) -> float:
        errors = [run.pruned_error * 100 for run in self.runs]
        if len(errors) < 2:
            return math.nan
        return statistics.stdev(errors) / math.sqrt(len(errors))

    @property
    def verdict(self) -> str:
        reference = self.dataset.reference
        if reference is None:
            return "-"
        return (
            "ok" if abs(self.mean("pruned_error") - reference) <= OK_POINTS else "check"
        )


def run(dataset: Dataset, options: BuildOptions, args: argparse.Namespace) -> Result:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        first = next(iter(dataset.files))
        table = _load(dataset, first, workdir, None)
        if dataset.test_file is not None:
            test = _load(dataset, dataset.test_file, workdir, table)
            unpruned, pruned, pruned_size = _fit(table, test, options)
            runs = [Run(pruned.error_rate, unpruned.error_rate, pruned_size)]
        else:
            runs = [
                cross_validate(table, options, args.folds, random.Random(args.seed + r))
                for r in range(args.repeats)
            ]
    return Result(
        dataset,
        cases=len(table.cases),
        attributes=len(table.attributes),
        classes=len(table.classes),
        runs=runs,
        seconds=time.perf_counter() - started,
    )


HEADER = (
    "dataset",
    "cases",
    "attrs",
    "classes",
    "pruned err %",
    "± se",
    "C4.5 ref %",
    "diff",
    "verdict",
    "unpruned err %",
    "pruned size",
    "time s",
)


def _row(result: Result) -> tuple[str, ...]:
    error = result.mean("pruned_error")
    reference = result.dataset.reference
    se = result.standard_error
    return (
        result.dataset.name,
        str(result.cases),
        str(result.attributes),
        str(result.classes),
        f"{error:.2f}",
        "-" if math.isnan(se) else f"{se:.2f}",
        "-" if reference is None else f"{reference:.2f}",
        "-" if reference is None else f"{error - reference:+.2f}",
        result.verdict,
        f"{result.mean('unpruned_error'):.2f}",
        f"{statistics.fmean(r.pruned_size for r in result.runs):.1f}",
        f"{result.seconds:.1f}",
    )


def print_table(rows: list[tuple[str, ...]]) -> None:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for number, row in enumerate(rows):
        cells = [
            cell.ljust(width) if i == 0 else cell.rjust(width)
            for i, (cell, width) in enumerate(zip(row, widths, strict=True))
        ]
        print("  ".join(cells))
        if number == 0:
            print("  ".join("-" * width for width in widths))


def main(argv: Sequence[str] | None = None) -> int:
    names = [dataset.name for dataset in DATASETS]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "datasets", nargs="*", metavar="DATASET", help=f"one of: {', '.join(names)}"
    )
    parser.add_argument("--repeats", type=int, default=10, help="complete CVs (10)")
    parser.add_argument("--folds", type=int, default=10, help="folds per CV (10)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the first CV (0)")
    parser.add_argument(
        "--course",
        action="store_true",
        help="no average-Gain filter (the CLI default); Quinlan's C4.5 uses it",
    )
    parser.add_argument(
        "--download", action="store_true", help="download the raw files again"
    )
    args = parser.parse_args(argv)
    unknown = sorted(set(args.datasets) - set(names))
    if unknown:
        parser.error(f"unknown dataset {unknown[0]!r}; choose from {', '.join(names)}")
    chosen = [d for d in DATASETS if not args.datasets or d.name in args.datasets]

    download(chosen, force=args.download)
    options = BuildOptions(quinlan_gain_filter=not args.course, min_cases=2)
    print(
        f"{args.repeats} x {args.folds}-fold stratified CV, seed {args.seed}; "
        f"gain-ratio, quinlan-gain-filter={not args.course}, -m 2, -c 25\n"
    )
    rows = [HEADER]
    for dataset in chosen:
        print(f"running {dataset.name} ...", file=sys.stderr, flush=True)
        rows.append(_row(run(dataset, options, args)))
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
