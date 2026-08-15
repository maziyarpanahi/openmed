"""Tests for notebook cell redaction and PHI-safe reporting."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

from openmed.interop.notebook_redaction import (
    NotebookCellRecord,
    NotebookCellSummary,
    NotebookRedactionPolicy,
    NotebookRedactionResult,
    redact_notebook,
)


def _build_synthetic_notebook() -> nbformat.NotebookNode:
    """Construct a synthetic notebook with 3 markdown, 4 code, and 1 raw cell."""

    nb = nbformat.v4.new_notebook()

    # 3 markdown cells
    nb.cells.append(nbformat.v4.new_markdown_cell("Introduction"))
    nb.cells.append(
        nbformat.v4.new_markdown_cell("Patient John Doe, contact john@example.com")
    )
    nb.cells.append(nbformat.v4.new_markdown_cell("Results section"))

    # Cell 3: code with stream output
    cell3 = nbformat.v4.new_code_cell("print('Patient John Doe')")
    cell3.execution_count = 1
    cell3.outputs.append(nbformat.v4.new_output("stream", text="Patient John Doe\n"))
    nb.cells.append(cell3)

    # Cell 4: code with text/html display_data
    cell4 = nbformat.v4.new_code_cell("display(HTML('<p>Patient Jane Smith</p>'))")
    cell4.execution_count = 2
    cell4.outputs.append(
        nbformat.v4.new_output(
            "display_data", data={"text/html": "<p>Patient Jane Smith</p>"}
        )
    )
    nb.cells.append(cell4)

    # Cell 5: code with image/png and text/plain
    cell5 = nbformat.v4.new_code_cell("plt.show()")
    cell5.execution_count = 3
    cell5.outputs.append(
        nbformat.v4.new_output(
            "display_data",
            data={
                "image/png": "iVBORw0KGgo=",
                "text/plain": "Patient Bob Johnson",
            },
        )
    )
    nb.cells.append(cell5)

    # Cell 6: code with application/json execute_result
    cell6 = nbformat.v4.new_code_cell("print(json.dumps({'name': 'Alice Williams'}))")
    cell6.execution_count = 4
    output6 = nbformat.v4.new_output(
        "execute_result",
        data={"application/json": {"name": "Alice Williams"}},
    )
    output6["execution_count"] = 4
    cell6.outputs.append(output6)
    nb.cells.append(cell6)

    # Cell 7: raw cell (not redacted)
    nb.cells.append(nbformat.v4.new_raw_cell("Some raw content"))

    return nb


@pytest.fixture
def synthetic_notebook() -> nbformat.NotebookNode:
    """Return a fresh synthetic notebook with synthetic PHI placeholders."""

    return _build_synthetic_notebook()


# ---------------------------------------------------------------------------
# Per-action tests
# ---------------------------------------------------------------------------


def test_mask_action(synthetic_notebook: nbformat.NotebookNode) -> None:
    policy = NotebookRedactionPolicy(
        dry_run=True, action_overrides={"markdown": "mask"}
    )
    result = redact_notebook(synthetic_notebook, policy=policy)

    assert result.notebook.cells[1].source == "***"


def test_replace_action(synthetic_notebook: nbformat.NotebookNode) -> None:
    policy = NotebookRedactionPolicy(
        dry_run=True, action_overrides={"markdown": "replace"}
    )
    result = redact_notebook(synthetic_notebook, policy=policy)

    assert result.notebook.cells[1].source == "[REDACTED]"


def test_redact_action(synthetic_notebook: nbformat.NotebookNode) -> None:
    policy = NotebookRedactionPolicy(
        dry_run=True, action_overrides={"markdown": "redact"}
    )
    result = redact_notebook(synthetic_notebook, policy=policy)

    assert result.notebook.cells[1].source == ""


def test_keep_action(synthetic_notebook: nbformat.NotebookNode) -> None:
    original_source = synthetic_notebook.cells[1].source
    policy = NotebookRedactionPolicy(
        dry_run=True, action_overrides={"markdown": "keep"}
    )
    result = redact_notebook(synthetic_notebook, policy=policy)

    assert result.notebook.cells[1].source == original_source


# ---------------------------------------------------------------------------
# Per-MIME-type tests
# ---------------------------------------------------------------------------


def test_text_plain_redacted(synthetic_notebook: nbformat.NotebookNode) -> None:
    policy = NotebookRedactionPolicy(dry_run=True)
    result = redact_notebook(synthetic_notebook, policy=policy)

    # Cell 5 has text/plain "Patient Bob Johnson" alongside image/png
    data = result.notebook.cells[5].outputs[0]["data"]
    assert "text/plain" in data
    assert data["text/plain"] == "***"


def test_text_html_redacted(synthetic_notebook: nbformat.NotebookNode) -> None:
    policy = NotebookRedactionPolicy(dry_run=True)
    result = redact_notebook(synthetic_notebook, policy=policy)

    data = result.notebook.cells[4].outputs[0]["data"]
    assert data["text/html"] == "***"


def test_image_png_removed(synthetic_notebook: nbformat.NotebookNode) -> None:
    policy = NotebookRedactionPolicy(dry_run=True)
    result = redact_notebook(synthetic_notebook, policy=policy)

    data = result.notebook.cells[5].outputs[0]["data"]
    assert "image/png" not in data
    assert "text/plain" in data
    assert data["text/plain"] == "***"


def test_application_json_redacted(
    synthetic_notebook: nbformat.NotebookNode,
) -> None:
    policy = NotebookRedactionPolicy(dry_run=True)
    result = redact_notebook(synthetic_notebook, policy=policy)

    # application/json content is a dict in nbformat; the entire value is
    # replaced with the mask token (whole-content replacement, not recursive
    # dict traversal — out of scope for Size S).
    json_data = result.notebook.cells[6].outputs[0]["data"]["application/json"]
    assert json_data == "***"


def test_stream_output_redacted(
    synthetic_notebook: nbformat.NotebookNode,
) -> None:
    policy = NotebookRedactionPolicy(dry_run=True)
    result = redact_notebook(synthetic_notebook, policy=policy)

    stream_output = result.notebook.cells[3].outputs[0]
    assert stream_output["output_type"] == "stream"
    assert stream_output["text"] == "***"


def test_svg_removed() -> None:
    nb = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell("display(SVG(svg_string))")
    cell.execution_count = 1
    cell.outputs.append(
        nbformat.v4.new_output(
            "display_data",
            data={"image/svg+xml": "<svg><text>Patient Eve</text></svg>"},
        )
    )
    nb.cells.append(cell)

    policy = NotebookRedactionPolicy(dry_run=True)
    result = redact_notebook(nb, policy=policy)

    data = result.notebook.cells[0].outputs[0]["data"]
    assert "image/svg+xml" not in data


# ---------------------------------------------------------------------------
# Dry-run and determinism
# ---------------------------------------------------------------------------


def test_dry_run_no_write(
    synthetic_notebook: nbformat.NotebookNode,
    tmp_path: Path,
) -> None:
    output = tmp_path / "redacted.ipynb"
    policy = NotebookRedactionPolicy(dry_run=True)
    redact_notebook(synthetic_notebook, policy=policy, output_path=str(output))

    assert not output.exists()


def test_dry_run_deterministic(
    synthetic_notebook: nbformat.NotebookNode,
) -> None:
    policy = NotebookRedactionPolicy(dry_run=True)
    result1 = redact_notebook(copy.deepcopy(synthetic_notebook), policy=policy)
    result2 = redact_notebook(copy.deepcopy(synthetic_notebook), policy=policy)

    assert result1.notebook.cells == result2.notebook.cells
    assert result1.summary == result2.summary


# ---------------------------------------------------------------------------
# PHI-safety
# ---------------------------------------------------------------------------


def test_no_raw_phi_in_summary(
    synthetic_notebook: nbformat.NotebookNode,
) -> None:
    policy = NotebookRedactionPolicy(dry_run=True)
    result = redact_notebook(synthetic_notebook, policy=policy)

    serialized = json.dumps(result.summary.to_dict())
    assert "John" not in serialized
    assert "Jane" not in serialized
    assert "Bob" not in serialized
    assert "Alice" not in serialized
    assert "john@example.com" not in serialized


# ---------------------------------------------------------------------------
# Metadata shape preservation
# ---------------------------------------------------------------------------


def test_metadata_shape_preserved(
    synthetic_notebook: nbformat.NotebookNode,
) -> None:
    original_outputs = []
    for cell in synthetic_notebook.cells:
        if cell.cell_type == "code":
            for output in cell.outputs:
                original_outputs.append(
                    {
                        "output_type": output.get("output_type"),
                        "metadata": copy.deepcopy(output.get("metadata", {})),
                        "execution_count": output.get("execution_count"),
                    }
                )

    policy = NotebookRedactionPolicy(dry_run=True)
    result = redact_notebook(synthetic_notebook, policy=policy)

    redacted_outputs = []
    for cell in result.notebook.cells:
        if cell.cell_type == "code":
            for output in cell.outputs:
                redacted_outputs.append(
                    {
                        "output_type": output.get("output_type"),
                        "metadata": copy.deepcopy(output.get("metadata", {})),
                        "execution_count": output.get("execution_count"),
                    }
                )

    assert len(redacted_outputs) == len(original_outputs)
    for original, redacted in zip(original_outputs, redacted_outputs):
        assert redacted["output_type"] == original["output_type"]
        assert redacted["metadata"] == original["metadata"]
        assert redacted["execution_count"] == original["execution_count"]


# ---------------------------------------------------------------------------
# Execution order preservation
# ---------------------------------------------------------------------------


def test_execution_order_preserved(
    synthetic_notebook: nbformat.NotebookNode,
) -> None:
    original_types = [cell.cell_type for cell in synthetic_notebook.cells]
    original_execution_counts = [
        cell.execution_count
        for cell in synthetic_notebook.cells
        if cell.cell_type == "code"
    ]

    policy = NotebookRedactionPolicy(dry_run=True)
    result = redact_notebook(synthetic_notebook, policy=policy)

    assert len(result.notebook.cells) == len(synthetic_notebook.cells)
    redacted_types = [cell.cell_type for cell in result.notebook.cells]
    assert redacted_types == original_types

    redacted_execution_counts = [
        cell.execution_count
        for cell in result.notebook.cells
        if cell.cell_type == "code"
    ]
    assert redacted_execution_counts == original_execution_counts
    assert redacted_execution_counts == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Ordering invariant
# ---------------------------------------------------------------------------


def test_cell_records_notebook_order() -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_markdown_cell("Introduction"))
    nb.cells.append(nbformat.v4.new_markdown_cell("Patient John Doe"))
    nb.cells.append(nbformat.v4.new_markdown_cell("Methods"))
    cell3 = nbformat.v4.new_code_cell("x = 1")
    cell3.execution_count = 1
    nb.cells.append(cell3)
    cell4 = nbformat.v4.new_code_cell("print('Patient Jane Smith')")
    cell4.execution_count = 2
    cell4.outputs.append(nbformat.v4.new_output("stream", text="Patient Jane Smith\n"))
    nb.cells.append(cell4)
    cell5 = nbformat.v4.new_code_cell("y = 2")
    cell5.execution_count = 3
    nb.cells.append(cell5)
    nb.cells.append(nbformat.v4.new_markdown_cell("Discussion"))
    nb.cells.append(nbformat.v4.new_markdown_cell("Patient Bob Johnson"))

    policy = NotebookRedactionPolicy(dry_run=True)
    result = redact_notebook(nb, policy=policy)

    indices = [record.index for record in result.summary.cell_records]
    assert indices == [0, 1, 2, 3, 4, 5, 6, 7]


# ---------------------------------------------------------------------------
# Policy flag combinations
# ---------------------------------------------------------------------------


def test_redact_markdown_only(
    synthetic_notebook: nbformat.NotebookNode,
) -> None:
    original_code_outputs = [
        copy.deepcopy(cell.outputs)
        for cell in synthetic_notebook.cells
        if cell.cell_type == "code"
    ]

    policy = NotebookRedactionPolicy(
        dry_run=True, redact_markdown=True, redact_outputs=False
    )
    result = redact_notebook(synthetic_notebook, policy=policy)

    assert result.notebook.cells[1].source == "***"
    assert result.notebook.cells[0].source == "***"

    redacted_code_outputs = [
        cell.outputs for cell in result.notebook.cells if cell.cell_type == "code"
    ]
    assert redacted_code_outputs == original_code_outputs


def test_redact_outputs_only(
    synthetic_notebook: nbformat.NotebookNode,
) -> None:
    original_markdown_sources = [
        cell.source for cell in synthetic_notebook.cells if cell.cell_type == "markdown"
    ]

    policy = NotebookRedactionPolicy(
        dry_run=True, redact_markdown=False, redact_outputs=True
    )
    result = redact_notebook(synthetic_notebook, policy=policy)

    redacted_markdown_sources = [
        cell.source for cell in result.notebook.cells if cell.cell_type == "markdown"
    ]
    assert redacted_markdown_sources == original_markdown_sources

    stream_output = result.notebook.cells[3].outputs[0]
    assert stream_output["text"] == "***"


def test_redact_both(synthetic_notebook: nbformat.NotebookNode) -> None:
    policy = NotebookRedactionPolicy(
        dry_run=True, redact_markdown=True, redact_outputs=True
    )
    result = redact_notebook(synthetic_notebook, policy=policy)

    assert result.notebook.cells[1].source == "***"
    stream_output = result.notebook.cells[3].outputs[0]
    assert stream_output["text"] == "***"


def test_redact_neither(
    synthetic_notebook: nbformat.NotebookNode,
) -> None:
    original_sources = [cell.source for cell in synthetic_notebook.cells]
    original_outputs = [
        copy.deepcopy(cell.outputs)
        for cell in synthetic_notebook.cells
        if cell.cell_type == "code"
    ]

    policy = NotebookRedactionPolicy(
        dry_run=True, redact_markdown=False, redact_outputs=False
    )
    result = redact_notebook(synthetic_notebook, policy=policy)

    redacted_sources = [cell.source for cell in result.notebook.cells]
    assert redacted_sources == original_sources

    redacted_outputs = [
        cell.outputs for cell in result.notebook.cells if cell.cell_type == "code"
    ]
    assert redacted_outputs == original_outputs

    assert result.summary.redacted_cells == 0


# ---------------------------------------------------------------------------
# Action validation
# ---------------------------------------------------------------------------


def test_invalid_action_raises() -> None:
    with pytest.raises(ValueError):
        NotebookRedactionPolicy(action_overrides={"markdown": "invalid_action"})
