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
never requires IPython, and :func:`show` degrades to returning the HTML string
when IPython is absent.
"""

from __future__ import annotations

import html as html_mod
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Set, Tuple

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
) -> Tuple[str, List[Any]]:
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


def _build_segments(
    text_len: int,
    spans: Sequence[NormalizedSpan],
) -> List[Tuple[int, int, Optional[NormalizedSpan]]]:
    """Partition ``[0, text_len)`` into non-overlapping (start, end, span) parts.

    The text is cut at every span boundary, producing a flat sequence of
    segments that tile the whole string with no gaps and no dropped characters
    — even when the input spans overlap or touch. Each covered segment carries
    the highest-scoring span that covers it (ties resolved by the earlier
    span), so overlaps render deterministically instead of corrupting offsets.
    """
    if not spans:
        return [(0, text_len, None)]

    # Clamp to the text bounds and drop empties defensively.
    clamped: List[NormalizedSpan] = []
    for span in spans:
        start = max(0, min(span.start, text_len))
        end = max(0, min(span.end, text_len))
        if end > start:
            clamped.append(
                NormalizedSpan(start=start, end=end, label=span.label, score=span.score)
            )

    if not clamped:
        return [(0, text_len, None)]

    boundaries = {0, text_len}
    for span in clamped:
        boundaries.add(span.start)
        boundaries.add(span.end)
    ordered = sorted(boundaries)

    segments: List[Tuple[int, int, Optional[NormalizedSpan]]] = []
    for seg_start, seg_end in zip(ordered, ordered[1:]):
        if seg_end <= seg_start:
            continue
        covering: Optional[NormalizedSpan] = None
        best_score = float("-inf")
        for span in clamped:
            if span.start <= seg_start and span.end >= seg_end:
                score = span.score if span.score is not None else 0.0
                if score > best_score:
                    best_score = score
                    covering = span
        segments.append((seg_start, seg_end, covering))
    return segments


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

    text_len = len(text)
    segments = _build_segments(text_len, normalized)

    # Distinct labels in first-seen order for a stable legend.
    seen: Set[str] = set()
    ordered_labels: List[str] = []
    for _, _, span in segments:
        if span is not None and span.label not in seen:
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

    parts.append('<div class="openmed-display-text">')
    for seg_start, seg_end, span in segments:
        chunk = html_mod.escape(text[seg_start:seg_end])
        if span is None:
            parts.append(chunk)
            continue

        color = _label_color(span.label)
        label_esc = html_mod.escape(span.label)
        score_suffix = _score_suffix(span.score) if show_confidence else ""
        tooltip = label_esc
        if span.score is not None:
            tooltip = f"{label_esc}: {span.score:.3f}"

        parts.append(
            '<mark class="openmed-entity openmed-entity-'
            f'{html_mod.escape(span.label.lower())}" '
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
    parts.append("</div></div>")

    return "".join(parts)


def show(
    text_or_result: Any,
    spans: Optional[Sequence[Any]] = None,
    **kwargs: Any,
) -> str:
    """Display highlighted spans in a notebook, or return the HTML string.

    In an IPython/Jupyter environment this renders the widget inline via
    ``IPython.display.display``. Outside IPython — or when IPython is not
    installed — it simply returns the HTML string without raising.

    Args:
        text_or_result: Source text or a supported result object (see
            :func:`render_spans_html`).
        spans: Optional span sequence (see :func:`render_spans_html`).
        **kwargs: Forwarded to :func:`render_spans_html` (``title``,
            ``show_legend``, ``show_confidence``).

    Returns:
        The rendered HTML string in every case, so callers and tests can
        inspect the output regardless of the runtime environment.
    """
    html = render_spans_html(text_or_result, spans, **kwargs)

    try:  # IPython is an optional, lazily imported dependency.
        from IPython.display import HTML, display
    except Exception:
        return html

    try:
        display(HTML(html))
    except Exception:
        # Any display failure must degrade gracefully to the raw string.
        return html
    return html
