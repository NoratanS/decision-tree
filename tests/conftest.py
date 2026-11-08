from collections.abc import Callable
from pathlib import Path

import pytest

type WriteTable = Callable[..., Path]


@pytest.fixture
def write_table(tmp_path: Path) -> WriteTable:
    """Write `text` to a file in `tmp_path` and return its path."""

    def write(text: str, name: str = "table.txt", encoding: str = "utf-8") -> Path:
        path = tmp_path / name
        path.write_text(text, encoding=encoding)
        return path

    return write
