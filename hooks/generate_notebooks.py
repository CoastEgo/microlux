"""Generate documentation notebooks from Jupytext source files."""

from pathlib import Path

import jupytext


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = (
    {
        "source": ROOT / "example" / "KB0371.py",
        "output": ROOT / "example" / "KB0371.ipynb",
    },
)


def generate_notebooks() -> None:
    for notebook in NOTEBOOKS:
        output_path = notebook["output"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        jupytext.write(jupytext.read(notebook["source"]), output_path, fmt="ipynb")


def on_pre_build(config) -> None:  # noqa: ANN001
    generate_notebooks()


if __name__ == "__main__":
    generate_notebooks()
