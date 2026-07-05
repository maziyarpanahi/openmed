"""Unit tests for the Jupyter/IPython rich display widget (OM-239).

These tests assert the HTML rendering of de-identification / NER spans:
- one highlight element per span with the right label text and color,
- HTML escaping of the source text,
- graceful degradation when IPython is not installed,
- character-lossless rendering across overlapping and adjacent spans.

All fixtures are synthetic; no real PHI is used.
"""

from __future__ import annotations

import builtins
import re
from html.parser import HTMLParser

import pytest

from openmed.processing import render_spans_html, show
from openmed.processing.display import NormalizedSpan
from openmed.processing.outputs import EntityPrediction, OutputFormatter


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _MarkCollector(HTMLParser):
    """Collect ``<mark>`` highlight elements and their visible text."""

    def __init__(self) -> None:
        super().__init__()
        self.marks: list[dict[str, str]] = []
        self._depth = 0
        self._current: dict[str, str] | None = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "mark":
            self._depth = 1
            self._current = {k: (v or "") for k, v in attrs}
            self._text_parts = []
        elif self._current is not None:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._current is None:
            return
        self._depth -= 1
        if self._depth == 0:
            self._current["_text"] = "".join(self._text_parts)
            self.marks.append(self._current)
            self._current = None

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            self._text_parts.append(data)


def _marks(html: str) -> list[dict[str, str]]:
    parser = _MarkCollector()
    parser.feed(html)
    return parser.marks


def _visible_text(html: str) -> str:
    """Strip tags and unescape entities to recover the visible characters."""
    import html as html_mod

    return html_mod.unescape(re.sub(r"<[^>]+>", "", html))


# --------------------------------------------------------------------------- #
# render_spans_html — highlight elements, labels, colors
# --------------------------------------------------------------------------- #
def test_render_produces_one_highlight_per_span_with_label_and_color() -> None:
    text = "Contact John Doe and Jane Roe today."
    spans = [
        {"start": 8, "end": 16, "label": "PERSON", "score": 0.98},
        {"start": 21, "end": 29, "label": "PERSON", "score": 0.87},
    ]

    html = render_spans_html(text, spans)
    marks = _marks(html)

    assert len(marks) == 2
    assert html.count("<mark") == 2

    # Each highlight wraps the correct source substring and carries the label.
    assert "John Doe" in marks[0]["_text"]
    assert "Jane Roe" in marks[1]["_text"]
    for mark in marks:
        assert "PERSON" in mark["_text"]
        # Color comes from the shared OutputFormatter palette.
        expected_color = OutputFormatter()._get_entity_color("PERSON")
        assert expected_color in mark["style"]


def test_legend_lists_each_distinct_label_once() -> None:
    text = "Email a@b.co, phone 555-0100, name Sam Jones."
    spans = [
        {"start": 6, "end": 12, "label": "EMAIL", "score": 0.9},
        {"start": 20, "end": 28, "label": "PHONE", "score": 0.8},
        {"start": 35, "end": 44, "label": "PERSON", "score": 0.95},
    ]

    html = render_spans_html(text, spans)

    assert "openmed-legend" in html
    legend = html.split("openmed-legend", 1)[1].split("openmed-display-text", 1)[0]
    for label in ("EMAIL", "PHONE", "PERSON"):
        assert legend.count(label) == 1


def test_confidence_score_appears_in_output_and_tooltip() -> None:
    html = render_spans_html(
        "Patient Ann Lee.",
        [{"start": 8, "end": 15, "label": "PERSON", "score": 0.876}],
    )
    # Compact score annotation on the chip.
    assert "0.88" in html
    # Full-precision tooltip on the mark.
    assert 'title="PERSON: 0.876"' in html


def test_show_confidence_false_omits_score_chip() -> None:
    html = render_spans_html(
        "Patient Ann Lee.",
        [{"start": 8, "end": 15, "label": "PERSON", "score": 0.876}],
        show_confidence=False,
    )
    assert "0.88" not in html
    assert "PERSON" in html


def test_show_legend_false_omits_legend() -> None:
    html = render_spans_html(
        "x John Doe",
        [{"start": 2, "end": 10, "label": "PERSON", "score": 0.9}],
        show_legend=False,
    )
    assert "openmed-legend" not in html


def test_title_is_rendered_and_escaped() -> None:
    html = render_spans_html(
        "hello",
        [{"start": 0, "end": 5, "label": "X"}],
        title="A & B <report>",
    )
    assert "openmed-display-title" in html
    assert "A &amp; B &lt;report&gt;" in html


# --------------------------------------------------------------------------- #
# HTML escaping
# --------------------------------------------------------------------------- #
def test_source_text_is_html_escaped_and_cannot_break_rendering() -> None:
    # Angle brackets and ampersands both inside and outside a span.
    text = "Tag <b>bold</b> & value a<c for Dr. <X>."
    spans = [{"start": 25, "end": 31, "label": "PERSON", "score": 0.9}]  # "Dr. <X"

    html = render_spans_html(text, spans)

    # Raw source markup must not survive verbatim.
    assert "<b>bold</b>" not in html
    assert "&lt;b&gt;bold&lt;/b&gt;" in html
    assert "&amp;" in html
    # The visible text (tags stripped, entities unescaped) equals the source.
    assert _visible_text(html.split("openmed-display-text", 1)[1]).count("<b>bold</b>")


def test_escaped_source_roundtrips_to_original_characters() -> None:
    text = "a<b>&c\"d'e"
    spans = [{"start": 0, "end": 3, "label": "T"}]
    body = render_spans_html(text, spans).split("openmed-display-text", 1)[1]
    # Everything after the legend/title: visible text minus the injected label.
    visible = _visible_text(body)
    for ch in text:
        assert ch in visible


# --------------------------------------------------------------------------- #
# Overlapping and adjacent spans — no dropped characters
# --------------------------------------------------------------------------- #
def _source_chars(html: str) -> str:
    """Recover only the source characters from the highlighted text body.

    Disables the score chip and uses labels that do not collide with the
    source alphabet so the injected label text can be filtered out cleanly,
    leaving exactly the original characters the renderer emitted.
    """
    body = html.split('class="openmed-display-text">', 1)[1]
    visible = _visible_text(body)
    # Injected label chips are uppercase ASCII letters; strip them out.
    return "".join(ch for ch in visible if not ("A" <= ch <= "Z"))


def test_adjacent_spans_preserve_all_characters() -> None:
    text = "abcdefgh"
    spans = [
        {"start": 0, "end": 4, "label": "AAA"},
        {"start": 4, "end": 8, "label": "BBB"},
    ]
    html = render_spans_html(text, spans, show_legend=False, show_confidence=False)
    # Every source character survives, in order, ignoring injected labels.
    assert _source_chars(html) == text


def test_overlapping_spans_preserve_all_characters() -> None:
    text = "0123456789"
    spans = [
        {"start": 0, "end": 6, "label": "LOW", "score": 0.30},
        {"start": 3, "end": 10, "label": "HIGH", "score": 0.99},
    ]
    html = render_spans_html(text, spans, show_legend=False, show_confidence=False)
    # Source is digits only; no character dropped or duplicated.
    digits = "".join(ch for ch in _source_chars(html) if ch.isdigit())
    assert digits == text

    # The higher-scoring span wins the contested [3,6) region deterministically.
    assert "HIGH" in html


def test_span_offsets_out_of_range_are_clamped_without_crash() -> None:
    text = "short"
    spans = [{"start": 2, "end": 999, "label": "OOR", "score": 0.7}]
    html = render_spans_html(text, spans, show_legend=False, show_confidence=False)
    assert _source_chars(html) == text


def test_empty_span_list_renders_plain_text() -> None:
    html = render_spans_html("just text", [])
    assert "<mark" not in html
    assert "just text" in html


# --------------------------------------------------------------------------- #
# Input-shape flexibility (dict, dataclass, typed objects)
# --------------------------------------------------------------------------- #
def test_accepts_entity_prediction_dataclasses() -> None:
    text = "Patient John Doe."
    entities = [
        EntityPrediction(
            text="John Doe", label="PERSON", confidence=0.9, start=8, end=16
        )
    ]
    html = render_spans_html(text, entities)
    assert html.count("<mark") == 1
    assert "PERSON" in html


def test_accepts_openmed_span_objects() -> None:
    span_mod = pytest.importorskip("openmed.core.schemas.span")
    from openmed.core.schemas.span import OpenMedSpan, hmac_text_hash

    text = "Patient with hypertension."
    span = OpenMedSpan(
        doc_id="doc-1",
        start=13,
        end=25,
        text_hash=hmac_text_hash("hypertension", "k"),
        entity_type="CONDITION",
        canonical_label="CONDITION",
        score=0.92,
    )
    html = render_spans_html(text, [span])
    assert html.count("<mark") == 1
    assert "CONDITION" in html
    assert "hypertension" in _visible_text(html)


def test_accepts_analyze_result_object_directly() -> None:
    from openmed.core.results import AnalyzeResult

    result = AnalyzeResult(
        text="Patient Jane Roe today.",
        entities=[
            EntityPrediction(
                text="Jane Roe", label="PERSON", confidence=0.95, start=8, end=16
            )
        ],
        model="unit-test-model",
        timestamp="2026-01-01T00:00:00",
    )
    html = render_spans_html(result)
    assert "Jane Roe" in _visible_text(html)
    assert "PERSON" in html


def test_analyze_result_repr_html_matches_render() -> None:
    from openmed.core.results import AnalyzeResult

    result = AnalyzeResult(
        text="Patient Jane Roe today.",
        entities=[
            EntityPrediction(
                text="Jane Roe", label="PERSON", confidence=0.95, start=8, end=16
            )
        ],
        model="unit-test-model",
        timestamp="2026-01-01T00:00:00",
    )
    html = result._repr_html_()
    assert "<mark" in html
    assert "PERSON" in html
    assert "unit-test-model" in html


def test_deidentification_result_repr_html_highlights_original_text() -> None:
    from datetime import datetime

    from openmed.core.pii import DeidentificationResult, PIIEntity

    entity = PIIEntity(
        text="John Doe",
        label="PERSON",
        confidence=0.97,
        start=8,
        end=16,
    )
    result = DeidentificationResult(
        original_text="Patient John Doe was seen.",
        deidentified_text="Patient [PERSON] was seen.",
        pii_entities=[entity],
        method="mask",
        timestamp=datetime(2026, 1, 1),
    )
    html = result._repr_html_()
    assert "<mark" in html
    assert "PERSON" in html
    assert "John Doe" in _visible_text(html)


def test_accepts_mapping_payload_from_to_dict() -> None:
    payload = {
        "text": "Patient John Doe.",
        "entities": [
            {"start": 8, "end": 16, "label": "PERSON", "score": 0.9},
        ],
    }
    html = render_spans_html(payload)
    assert html.count("<mark") == 1
    assert "John Doe" in _visible_text(html)


def test_bio_prefixes_are_stripped_from_labels() -> None:
    html = render_spans_html(
        "John Doe",
        [{"start": 0, "end": 8, "entity_group": "B-PERSON", "score": 0.9}],
        show_legend=True,
    )
    assert "PERSON" in html
    assert "B-PERSON" not in html


def test_invalid_input_raises_type_error() -> None:
    with pytest.raises(TypeError):
        render_spans_html(object())


def test_normalized_span_is_public() -> None:
    span = NormalizedSpan(start=0, end=1, label="X", score=0.5)
    assert span.label == "X"


# --------------------------------------------------------------------------- #
# show() — IPython optional / lazy
# --------------------------------------------------------------------------- #
def test_show_returns_html_string_when_ipython_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``show`` must return the HTML string without raising if IPython is gone."""
    real_import = builtins.__import__

    def _blocked_import(name: str, *args: object, **kwargs: object):
        if name == "IPython" or name.startswith("IPython."):
            raise ImportError("No module named 'IPython'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    result = show(
        "Patient John Doe.",
        [{"start": 8, "end": 16, "label": "PERSON", "score": 0.9}],
    )
    assert isinstance(result, str)
    assert "PERSON" in result
    assert "John Doe" in result


def test_show_uses_ipython_display_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When IPython is importable, ``show`` displays and still returns the HTML."""
    import sys
    import types

    displayed: list[object] = []

    fake_ipython = types.ModuleType("IPython")
    fake_display = types.ModuleType("IPython.display")

    class _HTML:
        def __init__(self, data: str) -> None:
            self.data = data

    def _display(obj: object) -> None:
        displayed.append(obj)

    fake_display.HTML = _HTML  # type: ignore[attr-defined]
    fake_display.display = _display  # type: ignore[attr-defined]
    fake_ipython.display = fake_display  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", fake_display)

    html = show(
        "Patient John Doe.",
        [{"start": 8, "end": 16, "label": "PERSON", "score": 0.9}],
    )
    assert isinstance(html, str)
    assert len(displayed) == 1
    assert isinstance(displayed[0], _HTML)
    assert "PERSON" in displayed[0].data
