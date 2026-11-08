"""Split measures from the course script (§2–5), on weighted class counts.

A *class counts* argument is a sequence of weights, one per Decision Class.
A *partition* is one class counts sequence per branch of a test.
Logarithms are base 2 and zero-weight terms contribute 0.
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


def gain(counts: ClassCounts, partition: Partition) -> float:
    """Gain(X,T) = Info(T) - Info(X,T)."""
    return info(counts) - info_split(partition)


def split_info(partition: Partition) -> float:
    """Split Info(X,T): Entropy of the branch sizes, ignoring Decision Classes."""
    return info([sum(branch) for branch in partition])


def gain_ratio(counts: ClassCounts, partition: Partition) -> float:
    """Gain Ratio = Gain / Split Info; 0 when Split Info is 0 (a one-branch test)."""
    split = split_info(partition)
    if split <= 0:
        return 0.0
    return gain(counts, partition) / split
