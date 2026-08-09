"""Focused tests for deterministic, PHI-safe human-review packets."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import (
    ReviewCitation,
    ReviewFinding,
    ReviewGateResult,
    build_review_packet,
    render_review_packet,
)

SYNTHETIC_PROTECTED_VALUE = "SYNTHETIC_PROTECTED_VALUE_42"


def test_typed_packet_is_deterministic_and_reports_gate_status():
    findings = [
        ReviewFinding(
            finding_id="finding-2",
            label="medication_review",
            confidence=0.91,
            citation_ids=("citation-local",),
            protected_text="SYNTHETIC_PROTECTED_VALUE_2",
        ),
        ReviewFinding(
            finding_id="finding-1",
            label="renal_function_measure",
            confidence=0.78,
            uncertainty="uncertain",
            source_start=12,
            source_end=24,
            protected_text=SYNTHETIC_PROTECTED_VALUE,
            citation_ids=("citation-local",),
        ),
    ]
    citations = [
        ReviewCitation(
            citation_id="citation-local",
            source="synthetic-guidance",
            locator="section-4",
            title="Synthetic local guidance",
        )
    ]
    gates = [
        ReviewGateResult(
            gate_id="uncertainty-policy",
            passed=False,
            reason="requires_review",
            severity="warning",
            blocking=True,
        ),
        ReviewGateResult(gate_id="schema-check", passed=True),
    ]

    first = build_review_packet(findings, citations, gates)
    second = build_review_packet(
        tuple(reversed(findings)),
        tuple(reversed(citations)),
        tuple(reversed(gates)),
    )

    assert first.to_json() == second.to_json()
    payload = json.loads(first.to_json())
    assert payload["review_status"] == "blocked"
    assert payload["summary"] == {
        "citation_count": 1,
        "failed_gate_count": 1,
        "finding_count": 2,
        "gate_count": 2,
        "review_required": True,
    }
    assert payload["findings"][0]["finding_id"] == "finding-1"
    assert SYNTHETIC_PROTECTED_VALUE not in first.to_json()
    assert payload["findings"][0]["protected_text_available"] is True
    assert payload["findings"][0]["source_hash"].startswith("sha256:")


def test_mapping_records_drop_raw_values_from_reports_and_gate_details():
    packet = build_review_packet(
        findings=[
            {
                "id": "finding-mapped",
                "label": "synthetic_finding",
                "text": SYNTHETIC_PROTECTED_VALUE,
                "start": 3,
                "end": 11,
                "metadata": {
                    "priority": 2,
                    "text": SYNTHETIC_PROTECTED_VALUE,
                },
            }
        ],
        citations=[
            {
                "id": "citation-mapped",
                "source": "synthetic-source",
                "excerpt": SYNTHETIC_PROTECTED_VALUE,
            }
        ],
        gates=[
            {
                "gate": "privacy-check",
                "passed": False,
                "reason": SYNTHETIC_PROTECTED_VALUE,
                "details": {
                    "count": 1,
                    "message": SYNTHETIC_PROTECTED_VALUE,
                },
            }
        ],
    )

    safe_json = render_review_packet(packet)
    payload = json.loads(safe_json)

    assert SYNTHETIC_PROTECTED_VALUE not in safe_json
    assert payload["findings"][0]["source_offset"] == {"start": 3, "end": 11}
    assert "text" not in payload["findings"][0].get("attributes", {})
    assert "message" not in payload["gate_results"][0].get("details", {})
    assert payload["gate_results"][0]["reason"] == "protected"


def test_protected_text_requires_explicit_render_opt_in():
    finding = ReviewFinding(
        finding_id="finding-opt-in",
        label="synthetic_finding",
        text=SYNTHETIC_PROTECTED_VALUE,
    )
    citation = ReviewCitation(
        citation_id="citation-opt-in",
        source="synthetic-source",
        quote=SYNTHETIC_PROTECTED_VALUE,
    )
    packet = build_review_packet([finding], [citation])

    safe_payload = packet.to_dict()
    opted_in_payload = packet.to_dict(include_protected_text=True)
    alias_payload = render_review_packet(
        packet,
        format="dict",
        allow_protected_text=True,
    )

    assert "protected_text" not in safe_payload["findings"][0]
    assert "protected_text" not in safe_payload["citations"][0]
    assert (
        opted_in_payload["findings"][0]["protected_text"] == SYNTHETIC_PROTECTED_VALUE
    )
    assert (
        opted_in_payload["citations"][0]["protected_text"] == SYNTHETIC_PROTECTED_VALUE
    )
    assert alias_payload == opted_in_payload


def test_markdown_renderer_is_safe_by_default_and_can_be_opted_in():
    packet = build_review_packet(
        [
            ReviewFinding(
                finding_id="finding-markdown",
                label="synthetic_finding",
                protected_text=SYNTHETIC_PROTECTED_VALUE,
            )
        ]
    )

    safe_markdown = render_review_packet(packet, format="markdown")
    local_markdown = render_review_packet(
        packet,
        format="markdown",
        include_protected_text=True,
    )

    assert "# Human review packet" in safe_markdown
    assert SYNTHETIC_PROTECTED_VALUE not in safe_markdown
    assert SYNTHETIC_PROTECTED_VALUE in local_markdown


def test_gate_report_like_objects_are_accepted_without_network_access():
    class LocalGateReport:
        gate_results = (
            {
                "gate": "local-check",
                "passed": True,
                "reason": "ok",
                "details": {"metric": 0.9},
            },
        )

    packet = build_review_packet(
        findings=(),
        citations=(),
        gate_results=LocalGateReport(),
    )

    assert packet.review_status == "ready_for_review"
    assert packet.gate_results[0].gate_id == "local-check"
    assert packet.gate_results[0].details["metric"] == 0.9


def test_invalid_records_raise_without_echoing_input_values():
    with pytest.raises(ValueError, match="finding_id") as error:
        ReviewFinding(finding_id="", label="synthetic_finding")

    assert SYNTHETIC_PROTECTED_VALUE not in str(error.value)


def test_typed_records_do_not_echo_protected_values_in_repr_or_gate_reason():
    finding = ReviewFinding(
        finding_id="finding-private",
        label=SYNTHETIC_PROTECTED_VALUE,
        protected_text=SYNTHETIC_PROTECTED_VALUE,
    )
    gate = ReviewGateResult(
        gate_id="private-check",
        passed=False,
        reason=SYNTHETIC_PROTECTED_VALUE,
        details={"detail": SYNTHETIC_PROTECTED_VALUE},
    )

    assert SYNTHETIC_PROTECTED_VALUE not in repr(finding)
    assert SYNTHETIC_PROTECTED_VALUE not in repr(gate)
    assert SYNTHETIC_PROTECTED_VALUE not in gate.to_dict().__repr__()
    assert gate.reason == "provided"
    assert gate.reason_hash is not None
