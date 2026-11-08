import ast
import subprocess
import sys
from pathlib import Path

import decision_tree

PACKAGE = Path(decision_tree.__file__).parent
CLI_ONLY = {"typer", "rich"}


def test_public_api_is_exported():
    from decision_tree import build_tree, classify, evaluate, load_table

    for function in (load_table, build_tree, classify, evaluate):
        assert callable(function)


def _imported_top_level_modules(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module.split(".")[0]


def test_only_the_cli_imports_typer_or_rich():
    offenders = [
        path.name
        for path in PACKAGE.rglob("*.py")
        if path.name != "cli.py" and CLI_ONLY & set(_imported_top_level_modules(path))
    ]

    assert offenders == []


def test_importing_the_package_does_not_load_typer_or_rich():
    code = (
        "import sys, decision_tree; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in {'typer', 'rich'}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"
