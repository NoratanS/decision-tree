# decision-tree

A C4.5 decision tree written from scratch in Python, for the course *Systemy uczące się* (Jan Kozak, UE Katowice). This is the **extended version**: attribute values can be integers, real numbers or text, separated by whitespace, `,` `;` `|` or `:`.

All the algorithm code (loading, Entropy, Gain, Gain Ratio, thresholds, missing values, tree building, pruning, classification, evaluation) uses only the Python standard library. No numpy, pandas or scikit-learn. `typer` and `rich` are used **only** in `src/decision_tree/cli.py`, for argument parsing and coloured output. The instructor approved this on 2026-11-08. Deleting `cli.py` leaves a complete package with no dependencies.

## Install

Needs Python 3.15 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run decision-tree --help
```

## Usage

The CLI has three commands:

| Command | What it does |
|---|---|
| `stats FILE` | Each attribute's type, distinct values, value counts and missing values |
| `train FILE` | Build a tree, prune it, print both trees and evaluate both |
| `classify TRAIN CASES` | Build a tree from `TRAIN`, then print the predicted class and Class Distribution for each case in `CASES` |

### Course mode (the script's tree)

The course script builds the tree for knowledge representation: Gain Ratio, no minimum branch size and no pruning.

```bash
uv run decision-tree train tests/data/test_table.txt -m 1 --no-prune
```

```
a3 = 0:
|   a4 = 1:
|   |   a1 = 0: 0 (1.0)
|   |   a1 = 1: 0 (2.0)
|   |   a1 = 2: 1 (1.0)
|   a4 = 0: 1 (2.0)
a3 = 1:
|   a1 = 0: 1 (1.0)
|   a1 = 1: 2 (0.0)
|   a1 = 2: 2 (2.0)
```

The root is `a3`, as in the script's hand calculation: `a2` and `a3` have equal Gain, and `a3` wins on Gain Ratio.

### Full C4.5 (continuous attributes, pruning)

```bash
uv run decision-tree train tests/data/golf.txt --header --continuous humidity
```

```
outlook = overcast: Play (4.0)
outlook = sunny:
|   humidity <= 75 : Play (2.0)
|   humidity > 75 : Don't Play (3.0)
outlook = rain:
|   windy = true: Don't Play (2.0)
|   windy = false: Play (3.0)
```

This is the golf tree from the reference page. Leaf counts read `(cases)` or `(cases/errors)`.

### Evaluate on unseen data

```bash
uv run decision-tree train data.csv --test test.csv        # a separate Test Set
uv run decision-tree train data.csv --split 0.7 --seed 42  # 70/30 hold-out
```

Both print the error rate and confusion matrix of the unpruned and pruned tree on the Training Set and the Test Set.

### Classify new cases

```bash
uv run decision-tree classify tests/data/golf.txt new.txt --header --continuous humidity
```

```
case  predicted    Play  Don't Play
   1  Don't Play  0.400       0.600
```

A case with a missing value (`sunny, 70, ?, false`) goes down every branch. The result, 0.4 / 0.6, matches the reference page.

### Loading options

These work with every command.

| Option | Meaning |
|---|---|
| `--header` | Take column names from the first line. Without it, columns are `a1 … an` and the decision is `d` |
| `--sep X` | Separator. Default: detected |
| `--decision NAME` | Decision column. Default: the last column |
| `--ignore a1,a7` | Leave columns out, e.g. a case ID (`stats` warns when a column looks like one) |
| `--discrete` / `--continuous` | Override the inferred attribute type |
| `--missing "?,NA,"` | Missing Value markers. Default: `?`, `NA` and the empty cell |
| `--encoding cp1250` | File encoding. Default: UTF-8, with Excel's BOM removed |

Type inference: text is discrete. Whole numbers are discrete when a column has at most 10 distinct values, continuous otherwise. Any value with a fractional part (`3.7`, or `3,7` when the separator is not `,`) makes the column continuous.

### Build and pruning options (`train`, `classify`)

| Option | Default | Meaning |
|---|---|---|
| `--criterion gain\|gain-ratio` | `gain-ratio` | ID3 (Gain) or C4.5 (Gain Ratio) |
| `-m N` | 2 | A test needs at least two branches with ≥ N cases |
| `--quinlan-gain-filter` | off | Only tests with at least the average Gain compete, as in Quinlan's C4.5 |
| `-c CF` | 25 | Confidence Level for pruning, in percent; lower prunes more |
| `--no-prune` | | Keep only the unpruned tree |
| `--plain` | | Plain text instead of Rich output (automatic when piped) |

## Python API

```python
from decision_tree import ReadOptions, build_tree, prune, classify, load_table

table = load_table(
    "tests/data/golf.txt", ReadOptions(header=True), continuous=["humidity"]
)
tree = prune(build_tree(table))
dist = classify(tree, ("sunny", "70", None, "false"))  # None = Missing Value
dist["Play"], dist["Don't Play"]  # (0.4, 0.6)
```

Also exported: `BuildOptions`, `load_cases`, `evaluate`, `run_experiment`.

## How it is built

```
src/decision_tree/
  table.py       file → Decision Table (separator, header, types, missing values)
  stats.py       per-attribute value counts, identifier warning
  measures.py    Entropy, Info, Gain, Split Info, Gain Ratio, on weighted counts
  splitting.py   Candidate Tests per attribute, threshold search, the split
  builder.py     recursive tree growth with Case Weights
  pruning.py     C4.5 pessimistic error pruning
  tree.py        Tests, Node / Leaf, classification
  evaluate.py    error rate, confusion matrix
  experiment.py  build → prune → evaluate, hold-out split
  render.py      C4.5-style text output
  cli.py         Typer + Rich (the only module with dependencies)
```

The terms (Case, Gain Ratio, Case Weight, Threshold, ...) are defined in [`GLOSSARY.md`](GLOSSARY.md).

## Tests

```bash
uv run pytest                 # 663 tests, about 18 s
uv run pytest -m "not slow"   # skip the performance tests
```

The tests check the script's hand calculations (Info, Gain, Split Info and Gain Ratio for every attribute of `test_table.txt` and `stock_market.txt`), the reference page's golf tree and missing-value example, Quinlan's pruning example, and the file-format edge cases. Performance tests: a 10,000 × 50 table trains in under 30 s, and a table with a 20,000-value discrete attribute in under 60 s.

## Benchmark against published C4.5

`benchmarks/benchmark.py` runs Quinlan's protocol (ten stratified 10-fold cross-validations; adult uses its own train/test split) on 12 UCI datasets, then compares the pruned-tree error with C4.5 Release 7 (Quinlan 1996, *JAIR* 4, Table 1).

```bash
uv run python benchmarks/benchmark.py              # all datasets, about 30 s
uv run python benchmarks/benchmark.py iris glass   # some of them
```

Results from 2026-11-15 (`-m 2 -c 25 --quinlan-gain-filter`):

| Dataset | Cases | Pruned error % | C4.5 ref % | Difference |
|---|---:|---:|---:|---:|
| iris | 150 | 5.47 ± 0.26 | 4.87 | +0.60 |
| breast-w | 699 | 5.55 ± 0.16 | 5.29 | +0.26 |
| glass | 214 | 29.11 ± 0.58 | 32.10 | −2.99 |
| credit-a | 690 | 14.62 ± 0.36 | 15.80 | −1.18 |
| credit-g | 1000 | 28.89 ± 0.31 | 28.90 | −0.01 |
| heart-c | 303 | 27.29 ± 0.48 | 24.90 | +2.39 |
| hepatitis | 155 | 22.32 ± 0.61 | 21.70 | +0.62 |
| sonar | 208 | 30.19 ± 0.76 | 28.40 | +1.79 |
| adult | 32561 | 14.58 | 15.54 | −0.96 |
| car | 1728 | 7.30 ± 0.11 | – | |
| mushroom | 8124 | 0.00 | – | |
| vote | 435 | 3.52 ± 0.08 | – | |

All 9 datasets with a published reference are within 3 percentage points of it. Exact equality is not expected, because the fold assignment differs from Quinlan's.

## Sources

- Jan Kozak, *Systemy uczące się (drzewa decyzyjne ID3 / C4.5)*, UE Katowice 2020 (course script).
- Herman-Iżycka & Strzelczak, [C4.5 description](https://www.mimuw.edu.pl/~awojna/SID/referaty/strzelczak/c4_5Main.html), MIMUW (reference page).
- J. R. Quinlan, *C4.5: Programs for Machine Learning*, Morgan Kaufmann 1993; "Improved Use of Continuous Attributes in C4.5", *JAIR* 4, 1996.
- UCI Machine Learning Repository (benchmark data, CC BY 4.0).
