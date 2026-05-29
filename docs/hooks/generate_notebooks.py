"""Generate documentation notebooks from lightweight source files."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE_CELL_MARKER = "<!-- code-cell -->"
PY_CELL_MARKER = re.compile(r"^# %%")

NOTEBOOKS = (
    {
        "markdown": ROOT / "example" / "KB0371.md",
        "code": ROOT / "example" / "KB0371.py",
        "output": ROOT / "example" / "KB0371.ipynb",
    },
)


def _read_code_cells(path: Path) -> list[str]:
    cells: list[str] = []
    current: list[str] = []
    in_cell = False

    for line in path.read_text(encoding="utf-8").splitlines():
        if PY_CELL_MARKER.match(line):
            if in_cell:
                cells.append("\n".join(current).rstrip())
            current = []
            in_cell = True
            continue

        if in_cell:
            current.append(line)

    if in_cell:
        cells.append("\n".join(current).rstrip())

    return cells


def _build_cells(markdown_path: Path, code_path: Path) -> list[dict[str, object]]:
    code_cells = _read_code_cells(code_path)
    code_index = 0
    markdown_lines: list[str] = []
    cells: list[dict[str, object]] = []

    def flush_markdown() -> None:
        source = "\n".join(markdown_lines).strip()
        markdown_lines.clear()
        if source:
            cells.append(
                {
                    "cell_type": "markdown",
                    "id": f"markdown-{len(cells) + 1}",
                    "metadata": {},
                    "source": source,
                }
            )

    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        if line.strip() == CODE_CELL_MARKER:
            flush_markdown()
            if code_index >= len(code_cells):
                raise ValueError(f"{markdown_path} has more code-cell markers than {code_path}")

            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": f"code-{code_index + 1}",
                    "metadata": {},
                    "outputs": [],
                    "source": code_cells[code_index],
                }
            )
            code_index += 1
            continue

        markdown_lines.append(line)

    flush_markdown()

    if code_index != len(code_cells):
        raise ValueError(f"{code_path} has more code cells than {markdown_path} markers")

    return cells


def generate_notebooks() -> None:
    for notebook in NOTEBOOKS:
        output_path = notebook["output"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cells": _build_cells(notebook["markdown"], notebook["code"]),
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "pygments_lexer": "ipython3",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        output_path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def on_pre_build(config) -> None:  # noqa: ANN001
    generate_notebooks()


if __name__ == "__main__":
    generate_notebooks()
