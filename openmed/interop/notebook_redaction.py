"""Cell-level Jupyter notebook redaction with PHI-safe summary reporting.

The adapter preserves notebook structure by editing only markdown cell sources
and code cell outputs. Code cell sources (the executable code itself) are never
modified, so redaction never silently alters notebook semantics. Binary output
MIME types are dropped entirely while text outputs are masked, replaced, or
cleared according to the configured action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from openmed.core.schemas.span import ACTION_VALUES
from openmed.multimodal.exceptions import MissingDependencyError

__all__ = [
    "NotebookRedactionPolicy",
    "NotebookCellRecord",
    "NotebookCellSummary",
    "NotebookRedactionResult",
    "redact_notebook",
]

_NOTEBOOK_HINT = 'Install with: pip install "openmed[notebook]".'

_TEXT_MIME_TYPES = frozenset(
    {
        "text/plain",
        "text/html",
        "application/json",
    }
)

_BINARY_MIME_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/svg+xml",
        "application/pdf",
    }
)


@dataclass(frozen=True)
class NotebookRedactionPolicy:
    """Configuration controlling which notebook fields are redacted and how.

    Attributes:
        redact_markdown: When ``True``, markdown cell sources are redacted.
        redact_outputs: When ``True``, code cell outputs are redacted.
        action_overrides: Per-cell-type action overrides keyed by cell type
            (e.g. ``{"markdown": "replace"}``). Each value must be one of
            :data:`openmed.core.schemas.span.ACTION_VALUES`.
        dry_run: When ``True``, the notebook is processed in memory and
            returned without writing to disk.
    """

    redact_markdown: bool = True
    redact_outputs: bool = True
    action_overrides: Mapping[str, str] = field(default_factory=dict)
    dry_run: bool = False

    def __post_init__(self) -> None:
        for value in self.action_overrides.values():
            if value not in ACTION_VALUES:
                raise ValueError(
                    f"Invalid action {value!r}; expected one of {ACTION_VALUES}."
                )


@dataclass(frozen=True)
class NotebookCellRecord:
    """PHI-safe record describing the redaction outcome for one cell.

    Attributes:
        index: Zero-based position of the cell in ``notebook.cells``.
        cell_type: The notebook cell type (``"markdown"``, ``"code"``,
            ``"raw"``).
        action: The redaction action applied to the cell.
        redacted: ``True`` if any field of the cell was modified.
    """

    index: int
    cell_type: str
    action: str
    redacted: bool


@dataclass(frozen=True)
class NotebookCellSummary:
    """Aggregate, PHI-safe summary of a notebook redaction pass.

    Attributes:
        total_cells: Total number of cells in the notebook.
        redacted_cells: Number of cells where ``redacted`` is ``True``.
        cell_type_counts: Dict of cell type to occurrence count.
        cell_records: Tuple of per-cell records in natural cell index order.
    """

    total_cells: int
    redacted_cells: int
    cell_type_counts: dict[str, int]
    cell_records: tuple[NotebookCellRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict with no raw cell content."""

        return {
            "total_cells": self.total_cells,
            "redacted_cells": self.redacted_cells,
            "cell_type_counts": dict(self.cell_type_counts),
            "cell_records": [record.__dict__ for record in self.cell_records],
        }


@dataclass(frozen=True)
class NotebookRedactionResult:
    """Result of a notebook redaction pass.

    Attributes:
        notebook: The processed ``nbformat.NotebookNode`` (typed as ``Any`` to
            avoid a hard ``nbformat`` dependency at import time).
        summary: The PHI-safe aggregate summary of the redaction pass.
    """

    notebook: Any  # nbformat.NotebookNode at runtime
    summary: NotebookCellSummary


def _resolve_action(
    cell_type: str,
    policy: NotebookRedactionPolicy,
) -> str:
    """Return the redaction action for a cell type, honoring overrides."""

    return policy.action_overrides.get(cell_type, "mask")


def _apply_action(action: str) -> str | None:
    """Map an action to its replacement text.

    Returns ``None`` for ``"keep"`` to signal that no change should be made.
    ``"hash"`` and ``"format_preserve"`` are not meaningful for notebook
    redaction (we do not hash cell content here), so both fall back to the
    ``"mask"`` replacement.
    """

    if action == "keep":
        return None
    if action == "replace":
        return "[REDACTED]"
    if action == "redact":
        return ""
    # "mask", "hash", and "format_preserve" all collapse to the mask token.
    # "hash" is accepted (it passes ACTION_VALUES validation) but we do not
    # compute hashes for notebook content, so it masks like "mask".
    # "format_preserve" has no format to preserve for whole-cell replacement,
    # so it also masks.
    return "***"


def _redact_text(value: Any, replacement: str | None) -> tuple[str, bool]:
    """Return ``(new_value, changed)`` for a single text field."""

    if replacement is None:
        return value, False
    return replacement, value != replacement


def _redact_output(output: Any, action: str) -> bool:
    """Redact one code cell output in place. Return ``True`` if changed."""

    output_type = output.get("output_type")
    changed = False

    if output_type == "stream":
        text = output.get("text", "")
        if isinstance(text, list):
            text = "".join(text)
        replacement = _apply_action(action)
        new_text, text_changed = _redact_text(text, replacement)
        if text_changed:
            output["text"] = new_text
            changed = True
    elif output_type in ("execute_result", "display_data"):
        data = output.get("data", {})
        # Iterate over a snapshot of keys so we can mutate the dict safely.
        for mime_type in list(data.keys()):
            if mime_type in _TEXT_MIME_TYPES:
                content = data[mime_type]
                if isinstance(content, list):
                    content = "".join(content)
                replacement = _apply_action(action)
                new_content, content_changed = _redact_text(content, replacement)
                if content_changed:
                    data[mime_type] = new_content
                    changed = True
            else:
                # Binary MIME types and unknown MIME types are removed
                # conservatively to avoid leaking embedded PHI.
                del data[mime_type]
                changed = True

    return changed


def redact_notebook(
    source: str | Any,
    *,
    policy: NotebookRedactionPolicy | None = None,
    output_path: str | None = None,
) -> NotebookRedactionResult:
    """Redact markdown sources and code outputs in a Jupyter notebook.

    Code cell sources (the executable code) are never modified. Markdown cell
    sources and code cell outputs are redacted according to ``policy``. Binary
    outputs are dropped; text outputs are masked, replaced, or cleared.

    Args:
        source: Either a file path string to read with ``nbformat`` or an
            existing ``nbformat.NotebookNode``.
        policy: Redaction policy. When ``None``, a default
            :class:`NotebookRedactionPolicy` is used.
        output_path: Destination path to write the redacted notebook. Required
            when ``policy.dry_run`` is ``False``; ignored when ``dry_run`` is
            ``True``.

    Returns:
        A :class:`NotebookRedactionResult` containing the processed notebook
        and a PHI-safe summary.

    Raises:
        MissingDependencyError: If ``nbformat`` is not installed.
        ValueError: If ``output_path`` is ``None`` while ``dry_run`` is
            ``False``.
    """

    if policy is None:
        policy = NotebookRedactionPolicy()

    try:
        import nbformat
    except ImportError as exc:
        raise MissingDependencyError(
            dependency="nbformat",
            instruction=_NOTEBOOK_HINT,
        ) from exc

    if isinstance(source, str):
        notebook = nbformat.read(source, as_version=4)
    else:
        notebook = source

    cell_records: list[NotebookCellRecord] = []
    redacted_cells = 0
    cell_type_counts: dict[str, int] = {}

    for index, cell in enumerate(notebook.cells):
        cell_type = cell.get("cell_type", "raw")
        cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + 1
        action = _resolve_action(cell_type, policy)
        cell_changed = False

        if cell_type == "markdown" and policy.redact_markdown:
            replacement = _apply_action(action)
            new_source, source_changed = _redact_text(
                cell.get("source", ""), replacement
            )
            if source_changed:
                cell["source"] = new_source
                cell_changed = True
        elif cell_type == "code" and policy.redact_outputs:
            for output in cell.get("outputs", []):
                if _redact_output(output, action):
                    cell_changed = True

        if cell_changed:
            redacted_cells += 1

        cell_records.append(
            NotebookCellRecord(
                index=index,
                cell_type=cell_type,
                action=action,
                redacted=cell_changed,
            )
        )

    summary = NotebookCellSummary(
        total_cells=len(notebook.cells),
        redacted_cells=redacted_cells,
        cell_type_counts=cell_type_counts,
        cell_records=tuple(cell_records),
    )
    result = NotebookRedactionResult(notebook=notebook, summary=summary)

    if policy.dry_run:
        return result

    if output_path is None:
        raise ValueError("output_path is required when dry_run=False")

    nbformat.write(notebook, output_path)
    return result
