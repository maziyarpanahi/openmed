#!/usr/bin/env python3
"""Interactive Gradio demo for OpenMed de-identification.

Paste a *synthetic* clinical note, choose a redaction method, and see the
de-identified text next to the PII entities OpenMed detected — no code
required. This is the lowest-friction way to try ``deidentify`` before
wiring it into a pipeline.

Run::

    pip install gradio
    python examples/gradio_deid_app.py

Only synthetic, non-PHI sample text is used here. Never paste real patient
data into a demo UI (see ``SECURITY.md``). ``gradio`` is an optional,
example-local dependency — it is not part of the core package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openmed import deidentify

GRADIO_INSTALL_HINT = (
    "Gradio is required for this demo. Install it with: pip install gradio"
)
DEIDENTIFICATION_METHODS = ("mask", "replace", "hash")
ENTITY_TABLE_HEADERS = ("Label", "Text", "Start", "End", "Confidence")

SYNTHETIC_CLINICAL_TEXT = (
    "Synthetic note: John Doe (MRN 123456) was seen on 01/15/2023 by "
    "Dr. Alice Smith. Reach the patient at john.doe@example.com or "
    "(415) 555-0142."
)


@dataclass(frozen=True)
class DeidentificationView:
    """UI-ready view of one de-identification run.

    Attributes:
        deidentified_text: The redacted text returned by ``deidentify``.
        entity_rows: Detected PII entities flattened into table rows.
    """

    deidentified_text: str
    entity_rows: list[list[str]]


def entities_to_rows(entities: list[Any]) -> list[list[str]]:
    """Flatten detected PII entities into stringified table rows.

    Args:
        entities: ``PIIEntity`` objects from a de-identification result.

    Returns:
        One ``[label, text, start, end, confidence]`` row per entity, with
        every cell coerced to ``str`` so the UI can render them directly.
    """
    rows: list[list[str]] = []
    for entity in entities:
        confidence = getattr(entity, "confidence", None)
        rows.append(
            [
                str(getattr(entity, "label", "") or ""),
                str(getattr(entity, "text", "") or ""),
                str(getattr(entity, "start", "")),
                str(getattr(entity, "end", "")),
                "" if confidence is None else f"{float(confidence):.2f}",
            ]
        )
    return rows


def run_deidentification(text: str, method: str) -> DeidentificationView:
    """De-identify *text* with *method* and adapt the result for the UI.

    This wraps the public :func:`openmed.deidentify` API. Model access (and
    any one-time download) happens here when invoked, never at import time,
    which keeps the module import-safe for tests.

    Args:
        text: Input text to de-identify.
        method: One of :data:`DEIDENTIFICATION_METHODS`
            (``"mask"``, ``"replace"``, or ``"hash"``).

    Returns:
        A :class:`DeidentificationView` with the redacted text and a flat
        entity table.

    Raises:
        ValueError: If *method* is not a supported method.
    """
    if method not in DEIDENTIFICATION_METHODS:
        raise ValueError(
            f"Unsupported method {method!r}; choose one of {DEIDENTIFICATION_METHODS}."
        )
    result = deidentify(text, method=method)
    return DeidentificationView(
        deidentified_text=result.deidentified_text,
        entity_rows=entities_to_rows(result.pii_entities),
    )


def build_demo() -> Any:
    """Build the Gradio Blocks UI without launching a server.

    Returns:
        A ``gradio.Blocks`` instance ready for ``.launch()``.

    Raises:
        SystemExit: With an actionable install hint when the optional
            ``gradio`` dependency is not available.
    """
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - covered via monkeypatch
        raise SystemExit(GRADIO_INSTALL_HINT) from exc

    def _on_submit(text: str, method: str) -> tuple[str, list[list[str]]]:
        view = run_deidentification(text, method)
        return view.deidentified_text, view.entity_rows

    with gr.Blocks(title="OpenMed de-identification") as demo:
        gr.Markdown(
            "# OpenMed de-identification demo\n"
            "Paste a **synthetic** clinical note (never real PHI), pick a "
            "method, and inspect both the redacted text and the detected "
            "entities."
        )
        with gr.Row():
            text_in = gr.Textbox(
                label="Input text",
                value=SYNTHETIC_CLINICAL_TEXT,
                lines=6,
            )
            method_in = gr.Radio(
                choices=list(DEIDENTIFICATION_METHODS),
                value="mask",
                label="Method",
            )
        submit = gr.Button("De-identify", variant="primary")
        text_out = gr.Textbox(label="De-identified text", lines=6)
        entities_out = gr.Dataframe(
            headers=list(ENTITY_TABLE_HEADERS),
            label="Detected PII entities",
            wrap=True,
        )
        submit.click(_on_submit, [text_in, method_in], [text_out, entities_out])
    return demo


def main() -> None:
    """Launch the de-identification demo locally."""
    build_demo().launch()


if __name__ == "__main__":
    main()
