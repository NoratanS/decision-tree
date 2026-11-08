"""C4.5 decision trees from scratch, standard library only.

The command-line interface lives in `decision_tree.cli` and is not imported here.
"""

from decision_tree.builder import build_tree
from decision_tree.evaluate import evaluate
from decision_tree.experiment import run_experiment
from decision_tree.pruning import prune
from decision_tree.table import ReadOptions, load_cases, load_table
from decision_tree.tree import classify

__all__ = [
    "ReadOptions",
    "build_tree",
    "classify",
    "evaluate",
    "load_cases",
    "load_table",
    "prune",
    "run_experiment",
]
