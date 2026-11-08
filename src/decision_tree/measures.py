"""Split measures from the course script (§2–5), on weighted class counts.

A *class counts* argument is a sequence of weights, one per Decision Class.
A *partition* is one class counts sequence per branch of a test.
Logarithms are base 2 and zero-weight terms contribute 0.

With Missing Values (Quinlan's method, ADR 0002), `counts` and `partition`
cover only the cases whose tested value is known; `missing` is the weight of
the rest.
"""

import math
from collections.abc import Sequence

type ClassCounts = Sequence[float]
type Partition = Sequence[ClassCounts]

# Gains, and differences between scores, this small are rounding noise.
EPSILON = 1e-12


def entropy(probabilities: Sequence[float]) -> float:
    """I(P) = -sum(p * log2 p)."""
    # "+ 0.0" turns the -0.0 of a pure distribution into 0.0.
    return -sum(p * math.log2(p) for p in probabilities if p > 0) + 0.0


def info(counts: ClassCounts) -> float:
    """Info(T): Entropy of the Decision Classes in a weighted set of cases."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    return entropy([c / total for c in counts])


def info_split(partition: Partition) -> float:
    """Info(X,T): weighted average of Info over the branches of a test."""
    total = sum(sum(branch) for branch in partition)
    if total <= 0:
        return 0.0
    return sum(sum(branch) / total * info(branch) for branch in partition)


def gain(counts: ClassCounts, partition: Partition, *, missing: float = 0.0) -> float:
    """Gain(X,T) = F * (Info(T) - Info(X,T)), F the known fraction of the weight."""
    gained = info(counts) - info_split(partition)
    if missing <= 0:
        return gained
    known = sum(counts)
    return known / (known + missing) * gained if known > 0 else 0.0


def split_info(partition: Partition, *, missing: float = 0.0) -> float:
    """Split Info(X,T): Entropy of the branch sizes, ignoring Decision Classes.

    The cases with a Missing Value count as one more branch, as in C4.5.
    """
    return info([*(sum(branch) for branch in partition), missing])


def gain_ratio(
    counts: ClassCounts, partition: Partition, *, missing: float = 0.0
) -> float:
    """Gain Ratio = Gain / Split Info; 0 when Split Info is 0 (a one-branch test)."""
    split = split_info(partition, missing=missing)
    if split <= 0:
        return 0.0
    return gain(counts, partition, missing=missing) / split
