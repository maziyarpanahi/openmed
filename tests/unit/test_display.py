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


class _TagCollector(HTMLParser):
    """Collect parsed tags and attribute names for active-content checks."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, set[str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, {name for name, _ in attrs}))


def _marks(html: str) -> list[dict[str, str]]:
    parser = _MarkCollector()
    parser.feed(html)
    return parser.marks


def _visible_text(html: str) -> str:
    """Strip tags and unescape entities to recover the visible characters."""
    import html as html_mod

    return html_mod.unescape(re.sub(r"<[^>]+>", "", html))


def _layer_bodies(html: str) -> list[str]:
    """Return each complete annotation layer body in rendered order."""
    return re.findall(
        r'<pre class="openmed-display-text openmed-display-layer"[^>]*>'
        r"(.*?)</pre>",
        html,
        flags=re.DOTALL,
    )


def _source_text_from_layer(body: str) -> str:
    """Recover source text while excluding the renderer's annotation chips."""
    without_labels = re.sub(
        r'<span class="openmed-entity-label"[^>]*>.*?</span>',
        "",
        body,
        flags=re.DOTALL,
    )
    return _visible_text(without_labels)


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
    assert "0.876" not in html
    assert 'title="PERSON"' in html
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
    assert _visible_text(_layer_bodies(html)[0]).count("<b>bold</b>")


def test_escaped_source_roundtrips_to_original_characters() -> None:
    text = "a<b>&c\"d'e"
    spans = [{"start": 0, "end": 3, "label": "T"}]
    body = _layer_bodies(render_spans_html(text, spans))[0]
    # Everything after the legend/title: visible text minus the injected label.
    visible = _visible_text(body)
    for ch in text:
        assert ch in visible


def test_render_preserves_newlines_and_indentation_semantically() -> None:
    text = "Line one\n  Patient Jane Doe\nLine three"
    spans = [{"start": 11, "end": 27, "label": "PERSON", "score": 0.9}]

    html = render_spans_html(text, spans, show_legend=False)
    bodies = _layer_bodies(html)

    assert '<pre class="openmed-display-text openmed-display-layer"' in html
    assert "white-space:pre-wrap" in html
    assert len(bodies) == 1
    assert _source_text_from_layer(bodies[0]) == text


def test_labels_are_inert_and_output_has_no_active_or_remote_content() -> None:
    label = '"><script src="https://invalid.test/x.js">bad()</script>'

    html = render_spans_html(
        "safe text",
        [{"start": 0, "end": 4, "label": label, "score": 0.5}],
        title="<img src=x onerror=bad()>",
    )
    marks = _marks(html)
    tags = _TagCollector()
    tags.feed(html)

    assert len(marks) == 1
    assert marks[0]["data-label"] == label
    assert marks[0]["class"].startswith("openmed-entity openmed-entity-")
    assert "<script" not in html.lower()
    assert "<img" not in html.lower()
    assert "<link" not in html.lower()
    assert all(tag not in {"script", "img", "link"} for tag, _ in tags.tags)
    assert all(not ({"src", "href"} & attrs) for _, attrs in tags.tags)
    assert "url(" not in html.lower()
    assert "&lt;script" in html


# --------------------------------------------------------------------------- #
# Overlapping and adjacent spans — no dropped characters
# --------------------------------------------------------------------------- #
def _source_chars(html: str) -> str:
    """Recover only the source characters from the highlighted text body.

    Disables the score chip and uses labels that do not collide with the
    source alphabet so the injected label text can be filtered out cleanly,
    leaving exactly the original characters the renderer emitted.
    """
    bodies = _layer_bodies(html)
    assert len(bodies) == 1
    return _source_text_from_layer(bodies[0])


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
    # Each complete source line survives in its own deterministic layer.
    layer_bodies = _layer_bodies(html)
    assert len(layer_bodies) == 2
    for body in layer_bodies:
        assert _source_text_from_layer(body) == text

    # Every input annotation remains one complete, independently inspectable mark.
    marks = _marks(html)
    assert len(marks) == len(spans)
    assert [
        (mark["data-start"], mark["data-end"], mark["data-label"], mark["_text"])
        for mark in marks
    ] == [
        ("0", "6", "LOW", "012345 LOW"),
        ("3", "10", "HIGH", "3456789 HIGH"),
    ]


def test_duplicate_spans_render_once_each_on_deterministic_layers() -> None:
    text = "Jane Roe"
    spans = [
        {"start": 0, "end": 8, "label": "PERSON", "score": 0.9},
        {"start": 0, "end": 8, "label": "DUPLICATE", "score": 0.8},
    ]

    first = render_spans_html(text, spans, show_legend=False)
    second = render_spans_html(text, spans, show_legend=False)

    assert first == second
    assert len(_marks(first)) == 2
    assert len(_layer_bodies(first)) == 2
    assert all(_source_text_from_layer(body) == text for body in _layer_bodies(first))


def test_span_offsets_out_of_range_are_clamped_without_crash() -> None:
    text = "short"
    spans = [{"start": 2, "end": 999, "label": "OOR", "score": 0.7}]
    html = render_spans_html(text, spans, show_legend=False, show_confidence=False)
    assert _source_chars(html) == text


def test_fractional_offsets_are_rejected_instead_of_silently_truncated() -> None:
    html = render_spans_html(
        "abcdef",
        [{"start": 1.5, "end": 4, "label": "INVALID"}],
        show_legend=False,
    )

    assert "<mark" not in html
    assert _source_chars(html) == "abcdef"


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
    assert "0.95" not in html


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
    assert "0.97" not in html


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
    """``show`` renders a typed result without raising if IPython is gone."""
    from openmed.core.results import AnalyzeResult

    real_import = builtins.__import__

    def _blocked_import(name: str, *args: object, **kwargs: object):
        if name == "IPython" or name.startswith("IPython."):
            raise ImportError("No module named 'IPython'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    result = AnalyzeResult(
        text="Patient John Doe.",
        entities=[
            EntityPrediction(
                text="John Doe",
                label="PERSON",
                confidence=0.9,
                start=8,
                end=16,
            )
        ],
        model="unit-test-model",
        timestamp="2026-01-01T00:00:00",
    )
    expected = render_spans_html(result)

    assert show(result) == expected


def test_show_returns_html_when_ipython_is_installed_but_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importable IPython alone must not trigger notebook display side effects."""
    import sys
    import types

    displayed: list[object] = []
    fake_ipython = types.ModuleType("IPython")
    fake_display = types.ModuleType("IPython.display")
    fake_ipython.get_ipython = lambda: None  # type: ignore[attr-defined]
    fake_display.HTML = lambda data: data  # type: ignore[attr-defined]
    fake_display.display = displayed.append  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", fake_display)

    text = "Patient John Doe."
    spans = [{"start": 8, "end": 16, "label": "PERSON", "score": 0.9}]
    expected = render_spans_html(text, spans)

    assert show(text, spans) == expected
    assert displayed == []


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
    fake_ipython.get_ipython = lambda: object()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", fake_display)

    text = "Patient John Doe."
    spans = [{"start": 8, "end": 16, "label": "PERSON", "score": 0.9}]
    expected = render_spans_html(text, spans)

    result = show(
        text,
        spans,
    )
    assert result is None
    assert len(displayed) == 1
    assert isinstance(displayed[0], _HTML)
    assert displayed[0].data == expected
