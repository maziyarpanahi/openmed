"""Rich Jupyter/IPython display helpers for de-identification and NER results.

This module renders a displaCy-style colored highlight view of detected spans
directly inside a notebook cell, without pulling in spaCy. It accepts the
public result surfaces returned by OpenMed helpers:

- :class:`openmed.core.results.AnalyzeResult` (from :func:`openmed.analyze_text`)
- :class:`openmed.core.pii.DeidentificationResult` (from :func:`openmed.deidentify`)
- lists of :class:`openmed.processing.outputs.EntityPrediction` /
  :class:`openmed.core.pii.PIIEntity`
- lists of :class:`openmed.core.schemas.span.OpenMedSpan`
- plain ``list[dict]`` span payloads (the legacy ``entities`` list)

IPython is an *optional* dependency imported lazily: rendering the HTML string
never requires IPython, and :func:`show` returns the HTML string unless an
active IPython shell is available to display it inline.
"""

from __future__ import annotations

import html as html_mod
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Set

from .outputs import OutputFormatter

__all__ = ["render_spans_html", "show", "NormalizedSpan"]

# A single shared formatter instance drives the per-label color palette so the
# notebook widget matches ``analyze_text(output_format="html")`` exactly.
_COLOR_FORMATTER = OutputFormatter()


@dataclass(frozen=True)
class NormalizedSpan:
    """A display-ready span normalized from any supported input shape."""

    start: int
    end: int
    label: str
    score: Optional[float] = None


def _coerce_float(value: Any) -> Optional[float]:
    """Return a built-in float for numeric-like values, else ``None``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    """Return a built-in int for integer-like values, else ``None``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if hasattr(value, "item"):
        try:
            return int(value.item())
        except (TypeError, ValueError):
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _span_from_object(obj: Any) -> Optional[NormalizedSpan]:
    """Normalize one span-like object (dataclass or dict) to a NormalizedSpan.

    Recognizes the ``start``/``end`` offsets plus a label field that may be
    named ``label``, ``entity_type``, ``canonical_label``, ``entity_group``, or
    ``entity``, and a score field named ``score`` or ``confidence``. Spans
    without usable integer offsets are dropped (they cannot be highlighted).
    """
    if isinstance(obj, dict):
        getter = obj.get
    else:

        def getter(key: str, default: Any = None) -> Any:
            return getattr(obj, key, default)

    start = _coerce_int(getter("start"))
    end = _coerce_int(getter("end"))
    if start is None or end is None or end <= start:
        return None

    label = (
        getter("label")
        or getter("entity_type")
        or getter("canonical_label")
        or getter("entity_group")
        or getter("entity")
        or "ENTITY"
    )
    label = str(label).strip() or "ENTITY"
    # Strip BIO/BILOU prefixes so legends and colors stay consistent.
    for prefix in ("B-", "I-", "L-", "U-", "E-", "S-"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
            break

    score = _coerce_float(getter("score"))
    if score is None:
        score = _coerce_float(getter("confidence"))

    return NormalizedSpan(start=start, end=end, label=label, score=score)


def _resolve_text_and_spans(
    text_or_result: Any,
    spans: Optional[Sequence[Any]],
) -> tuple[str, List[Any]]:
    """Resolve the ``(text, raw_spans)`` pair from the supported call shapes.

    Supports:
    - ``render_spans_html(text, spans)`` — explicit text + span sequence.
    - ``render_spans_html(result)`` — a typed result object exposing ``text``
      (``AnalyzeResult``) or ``original_text`` (``DeidentificationResult``)
      plus an ``entities`` / ``pii_entities`` sequence.
    """
    if spans is not None:
        return str(text_or_result), list(spans)

    result = text_or_result

    # AnalyzeResult / PredictionResult expose ``.text`` + ``.entities``.
    text = getattr(result, "text", None)
    if text is not None and hasattr(result, "entities"):
        return str(text), list(getattr(result, "entities") or [])

    # DeidentificationResult exposes ``.original_text`` + ``.pii_entities``.
    original = getattr(result, "original_text", None)
    if original is not None and hasattr(result, "pii_entities"):
        return str(original), list(getattr(result, "pii_entities") or [])

    # Mapping-style payloads (e.g. AnalyzeResult.to_dict()).
    if isinstance(result, dict):
        text = result.get("text") or result.get("original_text") or ""
        raw = result.get("entities") or result.get("pii_entities") or []
        return str(text), list(raw)

    raise TypeError(
        "render_spans_html expects (text, spans) or a result object exposing "
        "text/entities (AnalyzeResult) or original_text/pii_entities "
        f"(DeidentificationResult); got {type(result).__name__}"
    )


def _label_color(label: str) -> str:
    """Return the shared CSS background color for ``label``."""
    return _COLOR_FORMATTER._get_entity_color(label)


def _clamp_spans(
    text_len: int,
    spans: Sequence[NormalizedSpan],
) -> List[NormalizedSpan]:
    """Clamp spans to ``text_len`` and discard spans with no visible text."""
    clamped: List[NormalizedSpan] = []
    for span in spans:
        start = max(0, min(span.start, text_len))
        end = max(0, min(span.end, text_len))
        if end > start:
            clamped.append(
                NormalizedSpan(start=start, end=end, label=span.label, score=span.score)
            )
    return clamped


def _span_layers(
    spans: Sequence[NormalizedSpan],
) -> List[List[NormalizedSpan]]:
    """Assign spans to deterministic non-overlapping annotation layers.

    Each input span appears exactly once. Nested, crossing, and duplicate spans
    move to additional layers instead of being split, clipped, or discarded.
    """
    if not spans:
        return [[]]

    indexed = list(enumerate(spans))
    ordered = sorted(
        indexed,
        key=lambda item: (item[1].start, -item[1].end, item[0]),
    )
    layers: List[List[NormalizedSpan]] = []
    layer_ends: List[int] = []

    for _, span in ordered:
        for index, occupied_end in enumerate(layer_ends):
            if span.start >= occupied_end:
                layers[index].append(span)
                layer_ends[index] = span.end
                break
        else:
            layers.append([span])
            layer_ends.append(span.end)

    return layers


def _score_suffix(score: Optional[float]) -> str:
    """Return a compact score annotation, or an empty string when absent."""
    if score is None:
        return ""
    return f" {score:.2f}"


def _render_legend(labels: Sequence[str]) -> str:
    """Render the color legend for the distinct ``labels`` (ordered)."""
    if not labels:
        return ""
    chips: List[str] = []
    for label in labels:
        color = _label_color(label)
        chips.append(
            '<span class="openmed-legend-item" '
            'style="display:inline-flex;align-items:center;margin:0 8px 4px 0;'
            'font-size:0.85em;">'
            f'<span style="display:inline-block;width:12px;height:12px;'
            f"background:{color};border:1px solid rgba(0,0,0,0.15);"
            'border-radius:3px;margin-right:4px;"></span>'
            f"{html_mod.escape(label)}</span>"
        )
    return (
        '<div class="openmed-legend" '
        'style="margin-bottom:8px;line-height:1.6;">' + "".join(chips) + "</div>"
    )


def _render_text_layer(
    text: str,
    spans: Sequence[NormalizedSpan],
    *,
    layer_index: int,
    show_confidence: bool,
) -> str:
    """Render one non-overlapping annotation layer without losing source text."""
    parts = [
        '<div class="openmed-display-text openmed-display-layer" '
        f'data-layer="{layer_index}" aria-label="Annotation layer '
        f'{layer_index + 1}">'
    ]
    cursor = 0

    for span in spans:
        parts.append(html_mod.escape(text[cursor : span.start]))

        chunk = html_mod.escape(text[span.start : span.end])
        color = _label_color(span.label)
        label_esc = html_mod.escape(span.label)
        label_attr = html_mod.escape(span.label, quote=True)
        score_suffix = _score_suffix(span.score) if show_confidence else ""
        tooltip = label_attr
        if show_confidence and span.score is not None:
            tooltip = f"{label_attr}: {span.score:.3f}"

        parts.append(
            '<mark class="openmed-entity openmed-entity-'
            f'{html_mod.escape(span.label.lower(), quote=True)}" '
            f'data-start="{span.start}" data-end="{span.end}" '
            f'data-label="{label_attr}" data-layer="{layer_index}" '
            f'style="background:{color};padding:0.2em 0.35em;margin:0 0.15em;'
            'border-radius:0.35em;line-height:1;" '
            f'title="{tooltip}">'
            f"{chunk}"
            '<span class="openmed-entity-label" '
            'style="font-size:0.7em;font-weight:700;line-height:1;'
            "border-radius:0.35em;text-transform:uppercase;vertical-align:middle;"
            'margin-left:0.35em;">'
            f"{label_esc}{html_mod.escape(score_suffix)}</span>"
            "</mark>"
        )
        cursor = span.end

    parts.append(html_mod.escape(text[cursor:]))
    parts.append("</div>")
    return "".join(parts)


def render_spans_html(
    text_or_result: Any,
    spans: Optional[Sequence[Any]] = None,
    *,
    title: Optional[str] = None,
    show_legend: bool = True,
    show_confidence: bool = True,
) -> str:
    """Render highlighted entity/PHI spans as a self-contained HTML string.

    The output is a displaCy-style view: the source text with each detected
    span wrapped in a colored ``<mark>`` element carrying the label, an
    optional confidence score, and a hover tooltip. A color legend for the
    labels present is prepended by default.

    Args:
        text_or_result: Either the source text (when ``spans`` is given) or a
            result object — :class:`~openmed.core.results.AnalyzeResult`,
            :class:`~openmed.core.pii.DeidentificationResult`, a
            ``PredictionResult``, or a mapping with ``text``/``entities``.
        spans: An optional sequence of spans. Each item may be an
            :class:`~openmed.processing.outputs.EntityPrediction`,
            :class:`~openmed.core.pii.PIIEntity`,
            :class:`~openmed.core.schemas.span.OpenMedSpan`, or a plain ``dict``
            with ``start``/``end`` offsets and a label field. When omitted, the
            spans are read from ``text_or_result``.
        title: Optional heading rendered above the highlighted text.
        show_legend: Include the per-label color legend (default ``True``).
        show_confidence: Annotate each highlight with its confidence score when
            available (default ``True``).

    Returns:
        A standalone HTML string. Rendering never requires IPython.

    Example:
        >>> html = render_spans_html(
        ...     "Contact John Doe.",
        ...     [{"start": 8, "end": 16, "label": "PERSON", "score": 0.98}],
        ... )
        >>> "PERSON" in html and "John Doe" in html
        True
    """
    text, raw_spans = _resolve_text_and_spans(text_or_result, spans)

    normalized: List[NormalizedSpan] = []
    for raw in raw_spans:
        span = _span_from_object(raw)
        if span is not None:
            normalized.append(span)

    normalized = _clamp_spans(len(text), normalized)
    layers = _span_layers(normalized)

    # Distinct labels in first-seen order for a stable legend.
    seen: Set[str] = set()
    ordered_labels: List[str] = []
    for span in normalized:
        if span.label not in seen:
            seen.add(span.label)
            ordered_labels.append(span.label)

    parts: List[str] = []
    parts.append('<div class="openmed-display" style="line-height:2.2;')
    parts.append(
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;\">"
    )

    if title:
        parts.append(
            '<div class="openmed-display-title" '
            'style="font-weight:600;margin-bottom:6px;">'
            f"{html_mod.escape(str(title))}</div>"
        )

    if show_legend:
        parts.append(_render_legend(ordered_labels))

    parts.append(
        '<div class="openmed-display-layers" style="display:grid;gap:0.35em;">'
    )
    for layer_index, layer in enumerate(layers):
        parts.append(
            _render_text_layer(
                text,
                layer,
                layer_index=layer_index,
                show_confidence=show_confidence,
            )
        )
    parts.append("</div></div>")

    return "".join(parts)


def show(
    text_or_result: Any,
    spans: Optional[Sequence[Any]] = None,
    **kwargs: Any,
) -> Optional[str]:
    """Display highlighted spans in a notebook, or return the HTML string.

    In an active IPython/Jupyter shell this renders the widget inline via
    ``IPython.display.display`` and returns ``None`` so the cell does not emit
    a second raw-string representation. Outside an active IPython shell — or
    when IPython is not installed — it returns the HTML string without raising.

    Args:
        text_or_result: Source text or a supported result object (see
            :func:`render_spans_html`).
        spans: Optional span sequence (see :func:`render_spans_html`).
        **kwargs: Forwarded to :func:`render_spans_html` (``title``,
            ``show_legend``, ``show_confidence``).

    Returns:
        ``None`` after successful inline display; otherwise the rendered HTML
        string so non-notebook callers can inspect or embed it.
    """
    html = render_spans_html(text_or_result, spans, **kwargs)

    try:  # IPython is an optional, lazily imported dependency.
        import IPython
        from IPython.display import HTML, display
    except Exception:
        return html

    try:
        if IPython.get_ipython() is None:
            return html
    except Exception:
        return html

    try:
        display(HTML(html))
    except Exception:
        # Any display failure must degrade gracefully to the raw string.
        return html
    return None
