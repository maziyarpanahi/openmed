"""Tests for self-contained risk dashboard rendering."""

from __future__ import annotations

import re
from copy import deepcopy
from html.parser import HTMLParser

import pytest

from openmed.risk import (
    AnonymityPolicy,
    anonymize_release,
    assess_release,
    enforce_kanon,
    kanon_report,
    render_release_assessment_dashboard,
    render_risk_dashboard,
    risk_report,
    write_release_assessment_dashboard,
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


def _release_rows() -> list[dict[str, object]]:
    return [
        {
            "patient_id": "patient-id-canary-a",
            "age": 30,
            "zip": "raw-qi-value-canary",
            "condition": "sensitive-value-canary-a",
        },
        {
            "patient_id": "patient-id-canary-b",
            "age": 30,
            "zip": "raw-qi-value-canary",
            "condition": "sensitive-value-canary-b",
        },
    ]


def _release_policy() -> AnonymityPolicy:
    return AnonymityPolicy(
        quasi_identifiers=("age", "zip"),
        sensitive_attributes=("condition",),
        privacy_unit="patient_id",
        target_k=2,
    )


def test_render_risk_dashboard_returns_self_contained_html_document():
    html = render_risk_dashboard(_sample_risk(), title="Risk Review")

    assert html.count("<html") == 1
    assert html.startswith("<!doctype html>\n<html")
    assert html.rstrip().endswith("</html>")
    assert "Headline Metrics" in html
    assert "Singleton Records" in html
    assert "Top Quasi-identifiers" in html
    assert "Local-sensitive diagnostic" in html
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


def test_legacy_dashboard_remains_explicitly_local_sensitive_and_detailed():
    html = render_risk_dashboard(
        {
            "leakage_rate": 1.0,
            "reid_rate": 1.0,
            "k_min": 1,
            "singleton_records": [
                {
                    "record_id": "local-record-canary",
                    "record_index": 0,
                    "effective_k": 1,
                    "quasi_identifier_key": [
                        {"category": "city", "values": ["local-qi-canary"]}
                    ],
                }
            ],
            "quasi_identifiers": [
                {
                    "record_id": "local-record-canary",
                    "category": "city",
                    "value": "local-qi-canary",
                }
            ],
        }
    )

    assert "Local-sensitive diagnostic" in html
    assert "local-record-canary" in html
    assert "local-qi-canary" in html


def test_release_assessment_dashboard_renders_only_allowlisted_aggregates():
    assessment = assess_release(_release_rows(), _release_policy())
    payload = assessment.to_dict()
    payload["records"] = [{"zip": "injected-record-canary"}]
    payload["source_path"] = "/private/injected-path-canary.csv"
    payload["warnings"] = ["warning-text-canary"]
    payload["record_id"] = "injected-record-id-canary"
    payload["k_anonymity"]["equivalence_classes"] = [
        {
            "key": ["injected-class-key-canary"],
            "members": [999],
        }
    ]

    html = render_release_assessment_dashboard(payload, title="Release Evidence")

    assert "Release Evidence" in html
    assert "Aggregate evidence only" in html
    assert "Qualified expert review is required" in html
    assert "meets declared policy" in html
    assert "Achieved k" in html
    assert "Class Size Distribution" in html
    for canary in (
        "patient-id-canary-a",
        "raw-qi-value-canary",
        "sensitive-value-canary-a",
        "injected-record-canary",
        "/private/injected-path-canary.csv",
        "warning-text-canary",
        "injected-record-id-canary",
        "injected-class-key-canary",
    ):
        assert canary not in html
    assert not re.search(r"\b(?:src|href)=[\"']https?://", html)
    _assert_balanced_html(html)


def test_anonymization_dashboard_never_traverses_sensitive_records():
    result = anonymize_release(_release_rows(), _release_policy())
    assert any(
        "raw-qi-value-canary" in str(value) or "sensitive-value-canary" in str(value)
        for record in result.records
        for value in record.values()
    )

    html = render_release_assessment_dashboard(result)

    assert "Before Anonymization" in html
    assert "After Anonymization" in html
    assert "Transformation and Utility" in html
    assert "Selected Generalization" in html
    for canary in (
        "patient-id-canary-a",
        "patient-id-canary-b",
        "raw-qi-value-canary",
        "sensitive-value-canary-a",
        "sensitive-value-canary-b",
    ):
        assert canary not in html
    _assert_balanced_html(html)


def test_release_assessment_dashboard_rejects_detailed_risk_mappings():
    with pytest.raises(TypeError, match="safe release assessment"):
        render_release_assessment_dashboard(_sample_risk())


def test_release_dashboard_preserves_valid_literal_schema_labels_with_escaping():
    rows = [
        {
            "患者 ID": "patient-a",
            "Patient Age": 40,
            "Région, cohort | `v1`": "north",
            "Diagnostic <group>": "alpha",
        },
        {
            "患者 ID": "patient-b",
            "Patient Age": 40,
            "Région, cohort | `v1`": "north",
            "Diagnostic <group>": "beta",
        },
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("Patient Age", "Région, cohort | `v1`"),
        sensitive_attributes=("Diagnostic <group>",),
        privacy_unit="患者 ID",
        target_k=2,
    )

    assessment = assess_release(rows, policy).to_dict()
    payload = {
        "artifact": "deidentification_anonymization_summary",
        "before": assessment,
        "after": assessment,
        "generalization": {
            "information_loss": 0.0,
            "levels": [
                {"attribute": attribute, "level": 0, "loss": 0.0}
                for attribute in policy.quasi_identifiers
            ],
        },
        "utility": {
            "row_suppression_rate": 0.0,
            "privacy_unit_suppression_rate": 0.0,
            "quasi_identifier_cell_change_rate": 0.0,
            "released_rows": 2,
        },
    }

    html = render_release_assessment_dashboard(payload)

    assert "Patient Age" in html
    assert "Région, cohort | `v1`" in html
    assert "Diagnostic &lt;group&gt;" in html
    assert "Diagnostic <group>" not in html
    assert "<script" not in html
    _assert_balanced_html(html)


def test_release_dashboard_drops_unsafe_control_and_bidi_labels():
    payload = assess_release(_release_rows(), _release_policy()).to_dict()
    payload["quasi_identifiers"].append("Alice Canary\u202e")
    payload["attribute_disclosure"].append(
        {
            "attribute": "Sensitive\u0000Canary Name",
            "l_diversity": {"achieved": 1, "violating_classes": 0},
            "t_closeness": {"achieved": 0, "violating_classes": 0},
        }
    )

    html = render_release_assessment_dashboard(payload)

    assert "Alice Canary" not in html
    assert "Canary Name" not in html


@pytest.mark.parametrize(
    "path",
    [
        ("row_count",),
        ("privacy_unit_count",),
        ("policy", "target_k"),
        ("policy", "target_l"),
        ("policy", "target_t"),
        ("k_anonymity", "achieved_k"),
        ("sample_identity_risk", "max"),
        ("warning_count",),
        ("attribute_disclosure", 0, "l_diversity", "achieved"),
        ("attribute_disclosure", 0, "l_diversity", "violating_classes"),
        ("attribute_disclosure", 0, "t_closeness", "achieved"),
        ("attribute_disclosure", 0, "t_closeness", "violating_classes"),
    ],
)
def test_release_dashboard_never_echoes_strings_from_numeric_fields(
    path: tuple[str | int, ...],
) -> None:
    payload = deepcopy(assess_release(_release_rows(), _release_policy()).to_dict())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = "numeric-field-canary"

    html = render_release_assessment_dashboard(payload)

    assert "numeric-field-canary" not in html


@pytest.mark.parametrize(
    "path",
    [
        ("generalization", "information_loss"),
        ("utility", "row_suppression_rate"),
        ("utility", "privacy_unit_suppression_rate"),
        ("utility", "quasi_identifier_cell_change_rate"),
        ("utility", "released_rows"),
        ("generalization", "levels", 0, "level"),
        ("generalization", "levels", 0, "loss"),
    ],
)
def test_anonymization_dashboard_never_echoes_strings_from_numeric_fields(
    path: tuple[str | int, ...],
) -> None:
    payload = deepcopy(
        anonymize_release(_release_rows(), _release_policy()).to_safe_dict()
    )
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = "numeric-field-canary"

    html = render_release_assessment_dashboard(payload)

    assert "numeric-field-canary" not in html


def test_write_release_assessment_dashboard_writes_safe_balanced_html(tmp_path):
    result = anonymize_release(_release_rows(), _release_policy())
    path = tmp_path / "release-assessment.html"

    returned = write_release_assessment_dashboard(result, path)

    assert returned == path
    html = path.read_text(encoding="utf-8")
    assert "Aggregate evidence only" in html
    assert "raw-qi-value-canary" not in html
    _assert_balanced_html(html)
