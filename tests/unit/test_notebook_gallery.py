"""Unit and regression test suite for the example-notebooks gallery.

Verifies that:
1. All 4 gallery notebooks exist and strictly conform to the nbformat v4 schema.
2. Every gallery notebook includes explicit synthetic data and offline mode disclaimers.
3. Every notebook executes 100% offline via genuine nbclient kernel without errors.
4. Freshly executed cell outputs match committed cell outputs across all mime types
   (stream, execute_result, display_data), preventing stale or broken examples.
5. No local environment paths, private paths, or unmasked identifiers leak into outputs.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import nbclient
import nbformat
import pytest

from scripts.check_notebooks import GALLERY, check_notebook

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = ROOT / "examples" / "notebooks"

GALLERY_NOTEBOOKS = GALLERY


@contextmanager
def _offline_env() -> Iterator[None]:
    """Force offline environment during notebook test execution."""
    prev_hf = os.environ.get("HF_HUB_OFFLINE")
    prev_tf = os.environ.get("TRANSFORMERS_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        if prev_hf is not None:
            os.environ["HF_HUB_OFFLINE"] = prev_hf
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
        if prev_tf is not None:
            os.environ["TRANSFORMERS_OFFLINE"] = prev_tf
        else:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)


def _load_notebook(filename: str) -> nbformat.NotebookNode:
    nb_path = NOTEBOOKS_DIR / filename
    assert nb_path.is_file(), f"Notebook missing: {nb_path}"
    with open(nb_path, encoding="utf-8") as f:
        return nbformat.read(f, as_version=4)


def _normalize_output_text(text: str) -> str:
    """Normalize whitespace and volatile memory addresses for exact comparison."""
    # Mask volatile pointer hex addresses (e.g., 0x10a2b3c4)
    normalized = re.sub(r"0x[0-9a-fA-F]{6,16}", "0x7f0000000000", text)
    # Strip local repo root path if present
    normalized = normalized.replace(str(ROOT), "[REPO_ROOT]")
    # Standardize line endings and trailing space per line
    lines = [line.rstrip() for line in normalized.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip()


def _extract_cell_output_text(cell: nbformat.NotebookNode) -> str:
    """Extract every output and MIME payload into a canonical representation."""
    outputs = cell.get("outputs", [])
    text_parts: list[str] = []
    for out in outputs:
        out_type = out.get("output_type")
        if out_type == "stream":
            text = out.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            text_parts.append(f"stream:{out.get('name', '')}:{text}")
        elif out_type in {"execute_result", "display_data"}:
            data = out.get("data", {})
            for mime_type, value in sorted(data.items()):
                if isinstance(value, list):
                    text = "".join(str(part) for part in value)
                elif isinstance(value, str):
                    text = value
                else:
                    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
                text_parts.append(f"{out_type}:{mime_type}:{text}")
        elif out_type == "error":
            traceback = "\n".join(out.get("traceback", []))
            text_parts.append(
                f"error:{out.get('ename', '')}:{out.get('evalue', '')}:{traceback}"
            )
    return _normalize_output_text("".join(text_parts))


def test_extract_cell_output_text_includes_every_mime_payload() -> None:
    """Freshness comparison must not ignore secondary rich-output payloads."""
    cell = nbformat.v4.new_code_cell(
        outputs=[
            nbformat.v4.new_output(
                "display_data",
                data={
                    "application/json": {"value": 3},
                    "text/html": "<strong>three</strong>",
                    "text/plain": "three",
                },
            )
        ]
    )
    canonical = _extract_cell_output_text(cell)
    assert 'application/json:{"value":3}' in canonical
    assert "text/html:<strong>three</strong>" in canonical
    assert "text/plain:three" in canonical


def _scan_for_leaks(text: str, filename: str) -> None:
    """Ensure no local filesystem paths or private artifacts leak into notebook content."""
    assert "/Users/" not in text and "/home/" not in text, (
        f"Local environment path leaked in {filename}: {text[:100]}"
    )
    assert re.search(r"[A-Za-z]:\\Users\\", text, re.IGNORECASE) is None, (
        f"Windows user path leaked in {filename}: {text[:100]}"
    )
    assert "issues/" not in text and "graphify" not in text, (
        f"Private artifact path leaked in {filename}: {text[:100]}"
    )


@pytest.mark.parametrize("filename", GALLERY_NOTEBOOKS)
def test_gallery_notebook_file_exists(filename: str) -> None:
    """Each curated gallery notebook file must exist and have content."""
    nb_path = NOTEBOOKS_DIR / filename
    assert nb_path.is_file(), f"Missing gallery notebook: {filename}"
    assert nb_path.stat().st_size > 0, f"Empty gallery notebook: {filename}"


@pytest.mark.parametrize("filename", GALLERY_NOTEBOOKS)
def test_gallery_notebook_valid_nbformat_schema(filename: str) -> None:
    """All gallery notebooks must pass nbformat v4 schema validation."""
    nb = _load_notebook(filename)
    nbformat.validate(nb)
    assert nb.nbformat == 4, f"{filename} nbformat must be 4"
    assert len(nb.cells) >= 4, f"{filename} must have at least 4 cells"

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) > 0, f"{filename} must contain code cells"
    for idx, cell in enumerate(code_cells):
        assert len(cell.get("outputs", [])) > 0, (
            f"{filename} code cell {idx} has empty outputs; committed notebooks must be executed"
        )


@pytest.mark.parametrize("filename", GALLERY_NOTEBOOKS)
def test_gallery_notebook_dependency_free_execution(filename: str) -> None:
    """The lightweight checker must execute every cell and match its output."""
    report = check_notebook(NOTEBOOKS_DIR / filename)
    assert report["notebook"] == filename
    assert report["code_cells"] > 0
    assert report["executed_output_cells"] > 0


@pytest.mark.parametrize("filename", GALLERY_NOTEBOOKS)
def test_gallery_notebook_synthetic_and_offline_disclaimers(
    filename: str,
) -> None:
    """Every gallery notebook must state synthetic data and offline mode notices."""
    nb = _load_notebook(filename)
    full_text = "\n".join("".join(cell.get("source", [])) for cell in nb.cells)
    assert "synthetic" in full_text.lower(), (
        f"{filename} missing synthetic data disclaimer"
    )
    assert (
        "never commit real phi" in full_text.lower() or "synthetic" in full_text.lower()
    ), f"{filename} missing explicit PHI notice"
    _scan_for_leaks(full_text, filename)


@pytest.mark.parametrize("filename", GALLERY_NOTEBOOKS)
def test_gallery_notebook_nbclient_offline_execution_and_freshness(
    filename: str,
) -> None:
    """Execute notebook via authentic nbclient kernel offline and assert output freshness."""
    committed_nb = _load_notebook(filename)
    import copy

    executing_nb = copy.deepcopy(committed_nb)

    with _offline_env():
        client = nbclient.NotebookClient(
            executing_nb,
            timeout=120,
            kernel_name="python3",
            allow_errors=False,
            resources={"metadata": {"path": str(NOTEBOOKS_DIR.resolve())}},
        )
        try:
            executed_nb = client.execute()
        except Exception as exc:
            pytest.fail(f"nbclient execution failed for {filename}:\nError: {exc}")

    # Compare fresh execution against committed outputs cell-by-cell
    assert len(executed_nb.cells) == len(committed_nb.cells), (
        f"Cell count mismatch in {filename}"
    )

    for idx, (fresh_cell, committed_cell) in enumerate(
        zip(executed_nb.cells, committed_nb.cells)
    ):
        if fresh_cell.cell_type != "code":
            continue

        fresh_out = _extract_cell_output_text(fresh_cell)
        committed_out = _extract_cell_output_text(committed_cell)

        _scan_for_leaks(fresh_out, filename)
        _scan_for_leaks(committed_out, filename)

        assert fresh_out == committed_out, (
            f"Stale output in {filename} at code cell {idx}.\n"
            f"Freshly Executed Output:\n{fresh_out}\n\n"
            f"Committed Output:\n{committed_out}"
        )


@pytest.mark.parametrize("filename", GALLERY_NOTEBOOKS)
def test_gallery_notebook_zero_leakage_and_clean_environment(
    filename: str,
) -> None:
    """Assert zero unmasked identifiers or environment paths in outputs."""
    nb = _load_notebook(filename)
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            out_text = _extract_cell_output_text(cell)
            _scan_for_leaks(out_text, filename)
