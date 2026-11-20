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
        content = jupytext.read(notebook["source"])
        # Stable IDs make repeated exports identical and avoid live-reload loops.
        for index, cell in enumerate(content.cells):
            cell["id"] = f"cell-{index}"
        serialized = jupytext.writes(content, fmt="ipynb")
        if (
            not output_path.exists()
            or output_path.read_text(encoding="utf-8") != serialized
        ):
            output_path.write_text(serialized, encoding="utf-8")


def on_pre_build(config) -> None:  # noqa: ANN001
    generate_notebooks()


if __name__ == "__main__":
    generate_notebooks()
