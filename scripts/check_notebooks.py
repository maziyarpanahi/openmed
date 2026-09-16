#!/usr/bin/env python3
"""Execute the curated gallery notebooks offline and verify committed outputs.

The gallery in ``examples/notebooks/`` is designed to run end to end with no
network access, no model downloads, and no real PHI. This checker is the
dependency-free form of the ``notebooks-execute`` CI gate: it runs each gallery
notebook top to bottom with a small stdlib kernel stand-in and fails when

* any code cell raises,
* a gallery notebook uses IPython magics (``%``/``!``) or environment-specific
  shell commands, or
* a code cell's freshly produced output does not match the output committed in
  the ``.ipynb`` file (stale or missing executed output).

Only notebooks listed in ``GALLERY`` are checked; they are the notebooks that
promise deterministic, offline, synthetic-data execution. Run it directly or
through the ``test_notebook_gallery_executes`` pytest gate:

    python scripts/check_notebooks.py
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "examples" / "notebooks"

GALLERY = (
    "01_quickstart_redaction.ipynb",
    "02_batch_dataset.ipynb",
    "03_fhir_export.ipynb",
    "04_eval_walkthrough.ipynb",
)

_EXPR_SENTINEL = "__openmed_last_expr__"


class NotebookCheckError(RuntimeError):
    """Raised when a gallery notebook fails the executed-output check."""


def _cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def _committed_outputs(cell: dict[str, Any]) -> list[tuple[str, str]]:
    """Normalize a code cell's committed outputs to (kind, text) pairs."""
    normalized: list[tuple[str, str]] = []
    for output in cell.get("outputs") or []:
        kind = output.get("output_type")
        if kind == "stream":
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            normalized.append(("stream", text))
        elif kind in {"execute_result", "display_data"}:
            data = output.get("data") or {}
            for mime_type, value in sorted(data.items()):
                if isinstance(value, list):
                    text = "".join(str(part) for part in value)
                elif isinstance(value, str):
                    text = value
                else:
                    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
                normalized.append((mime_type, text))
        else:
            normalized.append((kind, ""))
    return normalized


def _uses_ipython_magic(source: str) -> bool:
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("%") or stripped.startswith("!"):
            return True
    return False


def _exec_with_display(source: str, namespace: dict[str, Any]) -> Any:
    """Execute a cell, returning the last expression value as IPython would.

    The final expression of a cell is displayed by IPython unless its value is
    ``None``. To mirror that without running side effects twice, the last
    expression statement is rewritten to an assignment to a sentinel name.
    """
    tree = ast.parse(source)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        assign = ast.Assign(
            targets=[ast.Name(id=_EXPR_SENTINEL, ctx=ast.Store())],
            value=tree.body[-1].value,
        )
        ast.copy_location(assign, tree.body[-1])
        tree.body[-1] = assign
        source = ast.unparse(tree)
    exec(compile(source, "<notebook-cell>", "exec"), namespace)
    return namespace.pop(_EXPR_SENTINEL, None)


def _fresh_outputs(source: str, namespace: dict[str, Any]) -> list[tuple[str, str]]:
    if _uses_ipython_magic(source):
        raise NotebookCheckError(
            "gallery notebooks must not use IPython magics or shell commands"
        )
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        last_value = _exec_with_display(source, namespace)
    outputs: list[tuple[str, str]] = []
    text = buffer.getvalue()
    if text:
        outputs.append(("stream", text))
    if last_value is not None:
        outputs.append(("text/plain", repr(last_value) + "\n"))
    return outputs


def check_notebook(path: str | Path) -> dict[str, Any]:
    """Execute one gallery notebook and verify its committed outputs."""
    notebook_path = Path(path)
    if not notebook_path.is_file():
        raise NotebookCheckError(f"gallery notebook not found: {notebook_path}")

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise NotebookCheckError(f"{notebook_path.name}: expected nbformat 4")

    namespace: dict[str, Any] = {"__name__": "__main__"}
    code_cells = 0
    executed_output_cells = 0
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        code_cells += 1
        source = _cell_source(cell)
        try:
            fresh = _fresh_outputs(source, namespace)
        except NotebookCheckError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced with cell context
            raise NotebookCheckError(
                f"{notebook_path.name}: code cell {index + 1} raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        committed = _committed_outputs(cell)
        if fresh or committed:
            executed_output_cells += 1
        if fresh != committed:
            raise NotebookCheckError(
                f"{notebook_path.name}: stale or missing committed output in "
                f"code cell {index + 1}\n"
                f"  fresh: {fresh!r}\n"
                f"  committed: {committed!r}"
            )

    if code_cells == 0:
        raise NotebookCheckError(f"{notebook_path.name}: no code cells found")
    if executed_output_cells == 0:
        raise NotebookCheckError(
            f"{notebook_path.name}: no executed output committed; run the "
            "notebook and commit its outputs"
        )
    return {
        "notebook": notebook_path.name,
        "code_cells": code_cells,
        "executed_output_cells": executed_output_cells,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        action="append",
        default=[],
        metavar="PATH",
        help="check a specific notebook path (repeatable; default: the gallery)",
    )
    args = parser.parse_args(argv)

    notebooks = [Path(p) for p in args.notebook]
    if not notebooks:
        notebooks = [NOTEBOOKS_DIR / name for name in GALLERY]

    failed = False
    for notebook_path in notebooks:
        try:
            report = check_notebook(notebook_path)
        except NotebookCheckError as exc:
            failed = True
            print(f"FAIL  {notebook_path.name}: {exc}", file=sys.stderr)
            continue
        print(
            f"OK    {report['notebook']}: {report['code_cells']} code cells, "
            f"{report['executed_output_cells']} with committed output"
        )

    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
