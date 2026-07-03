"""Tests for self-contained risk dashboard rendering."""

from __future__ import annotations

import re
from html.parser import HTMLParser

from openmed.risk import (
    enforce_kanon,
    kanon_report,
    render_risk_dashboard,
    risk_report,
    write_risk_dashboard,
)


class _BalancedHTMLParser(HTMLParser):
    _VOID_TAGS = {"br", "hr", "img", "input", "link", "meta"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in self._VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        return None

    def handle_endtag(self, tag: str) -> None:
        assert self.stack, f"unexpected closing tag: {tag}"
        assert self.stack[-1] == tag, f"expected </{self.stack[-1]}>, got </{tag}>"
        self.stack.pop()


def _assert_balanced_html(document: str) -> None:
    parser = _BalancedHTMLParser()
    parser.feed(document)
    parser.close()
    assert parser.stack == []


def _sample_risk() -> dict:
    return risk_report(
        [
            {
                "record_id": "a",
                "age": 73,
                "city": "Riverton",
                "visit_date": "2024-01-05",
            },
            {
                "record_id": "b",
                "age": 73,
                "city": "Riverton",
                "visit_date": "2024-01-05",
            },
            {
                "record_id": "unique",
                "age": 94,
                "city": "Smallville",
                "visit_date": "2024-01-05",
            },
        ]
    )


def test_render_risk_dashboard_returns_self_contained_html_document():
    html = render_risk_dashboard(_sample_risk(), title="Risk Review")

    assert html.count("<html") == 1
    assert html.startswith("<!doctype html>\n<html")
    assert html.rstrip().endswith("</html>")
    assert "Headline Metrics" in html
    assert "Singleton Records" in html
    assert "Top Quasi-identifiers" in html
    assert not re.search(r"\b(?:src|href)=[\"']https?://", html)
    _assert_balanced_html(html)


def test_render_risk_dashboard_escapes_record_content():
    risk = {
        "leakage_rate": 0.5,
        "reid_rate": 0.25,
        "k_min": 1,
        "singleton_records": [
            {
                "record_id": 'note-<>&"',
                "record_index": 0,
                "effective_k": 1,
                "quasi_identifier_key": [
                    {"category": "city", "values": ['Paris & <Rome> "Milan"']}
                ],
            }
        ],
        "quasi_identifiers": [
            {
                "record_id": 'note-<>&"',
                "record_index": 0,
                "category": 'city"',
                "value": 'Paris & <Rome> "Milan"',
                "source": "field",
            }
        ],
    }

    html = render_risk_dashboard(risk, title='Risk <Dashboard> "Q"')

    assert "Risk &lt;Dashboard&gt; &quot;Q&quot;" in html
    assert "note-&lt;&gt;&amp;&quot;" in html
    assert "Paris &amp; &lt;Rome&gt; &quot;Milan&quot;" in html
    assert 'note-<>&"' not in html
    assert "<Rome>" not in html


def test_render_risk_dashboard_includes_kanon_section_when_supplied():
    records = [
        {"age": 30, "zip": "1000", "disease": "flu"},
        {"age": 30, "zip": "1000", "disease": "cold"},
        {"age": 41, "zip": "2000", "disease": "flu"},
    ]
    risk = risk_report(records)
    kanon = kanon_report(
        records,
        quasi_identifiers=["age", "zip"],
        sensitive_attributes=["disease"],
    )

    html = render_risk_dashboard(risk, kanon=kanon)

    assert "K-Anonymity Equivalence Classes" in html
    assert "Class Size Distribution" in html
    assert "Equivalence Classes" in html
    assert "l-diversity" in html


def test_render_risk_dashboard_includes_enforcement_section_when_supplied():
    records = [
        {"age": 30, "zip": "10001", "visit_date": "2024-01-01", "disease": "flu"},
        {"age": 31, "zip": "10002", "visit_date": "2024-01-02", "disease": "cold"},
    ]
    risk = risk_report(records)
    enforced = enforce_kanon(
        records,
        quasi_identifiers=["age", "zip", "visit_date"],
        sensitive_attributes=["disease"],
        target_k=2,
    )

    html = render_risk_dashboard(risk, kanon=enforced)

    assert "K-Anonymity Enforcement" in html
    assert "Selected Generalization" in html
    assert "Max re-id bound" in html
    assert "Bound check" in html


def test_write_risk_dashboard_writes_balanced_html_and_returns_path(tmp_path):
    path = tmp_path / "risk-dashboard.html"

    returned = write_risk_dashboard(_sample_risk(), path, title="Risk Review")

    assert returned == path
    html = path.read_text(encoding="utf-8")
    assert html.count("<html") == 1
    _assert_balanced_html(html)


def test_render_risk_dashboard_is_deterministic_for_fixed_input():
    risk = _sample_risk()
    first = render_risk_dashboard(risk)
    second = render_risk_dashboard(risk)

    assert first == second
