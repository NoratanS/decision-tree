"""C4.5 pessimistic error pruning (spec R6).

The estimate follows Quinlan's C4.5 (`AddErrs` in `prune.c`): the upper limit
of the binomial confidence interval for the error rate, at the Confidence
Level. Like C4.5, the normal deviate comes from a small table, interpolated
linearly; this is what reproduces the values in Quinlan's book.
"""

import math

from decision_tree.measures import EPSILON
from decision_tree.tree import Leaf, Node, Tree, majority

# Confidence Level → normal deviate, from C4.5's `prune.c`.
_LEVELS = (0.0, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.40, 1.00)
_DEVIATES = (4.0, 3.09, 2.58, 2.33, 1.65, 1.28, 0.84, 0.25, 0.00)


DEFAULT_CONFIDENCE = 0.25
"""C4.5's default Confidence Level (25%)."""


def prune(tree: Tree, confidence: float = DEFAULT_CONFIDENCE) -> Tree:
    """A pruned copy of `tree`; `tree` itself is left as it is.

    Bottom-up: after its branches are pruned, a Node becomes a Leaf of its
    most common Decision Class when that Leaf's estimated errors are no higher
    than the summed estimates of the branches. A lower `confidence` (the
    Confidence Level, in (0, 1]) gives higher estimates and prunes more.
    Raises ValueError for a `confidence` outside (0, 1].
    """
    _check_confidence(confidence)
    return _prune(tree, confidence)[0]


def _prune(tree: Tree, confidence: float) -> tuple[Tree, float]:
    """The pruned subtree and its estimated errors."""
    if isinstance(tree, Leaf):
        return tree, _estimated_errors(tree.total, tree.errors, confidence)
    branches: dict[str, Tree] = {}
    subtree_errors = 0.0
    for value, child in tree.branches.items():
        branches[value], child_errors = _prune(child, confidence)
        subtree_errors += child_errors
    leaf = Leaf(majority(tree.distribution), tree.distribution)
    leaf_errors = _estimated_errors(leaf.total, leaf.errors, confidence)
    if leaf_errors <= subtree_errors + EPSILON:
        return leaf, leaf_errors
    return Node(
        attribute=tree.attribute,
        index=tree.index,
        branches=branches,
        distribution=tree.distribution,
        threshold=tree.threshold,
    ), subtree_errors


def estimated_errors(weight: float, errors: float, *, confidence: float) -> float:
    """Pessimistic number of errors of a Leaf holding `weight` of training
    Cases, `errors` of it misclassified: `weight × U_CF(errors, weight)`.

    `confidence` is the Confidence Level, between 0 (exclusive) and 1.
    """
    _check_confidence(confidence)
    return _estimated_errors(weight, errors, confidence)


def _check_confidence(confidence: float) -> None:
    if not 0 < confidence <= 1:
        raise ValueError(f"confidence must be in (0, 1], got {confidence}")


def _estimated_errors(weight: float, errors: float, confidence: float) -> float:
    """`estimated_errors` without the range check."""
    if weight <= 0:
        return 0.0
    return errors + _added_errors(weight, errors, confidence)


def _added_errors(weight: float, errors: float, confidence: float) -> float:
    """How many errors to add to the observed ones (C4.5's `AddErrs`).

    The constants 1e-6, 0.9999 and 0.67 are C4.5's own.
    """
    if errors < 1e-6:
        # No errors: the exact binomial bound, 1 − CF^(1/N).
        return weight * (1 - math.exp(math.log(confidence) / weight))
    if errors < 0.9999:
        # Between 0 and 1 error (Case Weights): interpolate linearly.
        at_zero = _added_errors(weight, 0.0, confidence)
        at_one = _added_errors(weight, 1.0, confidence)
        return at_zero + errors * (at_one - at_zero)
    if errors + 0.5 >= weight:
        # Nearly every Case wrong: C4.5's fixed share of the correct weight.
        return 0.67 * (weight - errors)
    # The normal approximation, with a continuity correction of 0.5.
    z2 = _deviate(confidence) ** 2
    observed = errors + 0.5
    upper = (
        observed
        + z2 / 2
        + math.sqrt(z2 * (observed * (1 - observed / weight) + z2 / 4))
    ) / (weight + z2)
    return weight * upper - errors


def _deviate(confidence: float) -> float:
    """Normal deviate for `confidence`, interpolated from C4.5's table."""
    i = next(k for k, level in enumerate(_LEVELS) if confidence <= level)
    low, high = _LEVELS[i - 1], _LEVELS[i]
    return _DEVIATES[i - 1] + (_DEVIATES[i] - _DEVIATES[i - 1]) * (confidence - low) / (
        high - low
    )
