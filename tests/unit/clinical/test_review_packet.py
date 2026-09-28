"""Focused tests for deterministic, PHI-safe human-review packets."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import (
    ReviewCitation,
    ReviewFinding,
    ReviewGateResult,
    ReviewPacket,
    build_review_packet,
    render_review_packet,
)
from openmed.core.audit import hash_text

SYNTHETIC_PROTECTED_VALUE = "SYNTHETIC_PROTECTED_VALUE_42"


@pytest.mark.parametrize("flag", ["false", "true", 0, 1, None, []])
def test_non_boolean_protected_render_flags_are_rejected(flag):
    finding = ReviewFinding("synthetic-id", "finding", text=SYNTHETIC_PROTECTED_VALUE)
    citation = ReviewCitation(
        "synthetic-citation", "source", text=SYNTHETIC_PROTECTED_VALUE
    )
    packet = build_review_packet([finding], [citation])
    for render in (
        finding.to_dict,
        citation.to_dict,
        packet.to_dict,
        packet.to_json,
        packet.to_markdown,
    ):
        with pytest.raises(ValueError, match="boolean"):
            render(include_protected_text=flag)
    for keyword in ("include_protected_text", "allow_protected_text"):
        with pytest.raises(ValueError, match="boolean"):
            render_review_packet(packet, **{keyword: flag})


def test_lowercase_identifiers_labels_and_metadata_are_not_plaintext():
    marker = "synthetic-private-marker"
    packet = build_review_packet(
        [ReviewFinding(marker, marker, attributes={"name": marker, "id": 123456})],
        [
            ReviewCitation(
                marker, marker, locator="section-" + marker, metadata={"source": marker}
            )
        ],
        [ReviewGateResult(marker, True, details={"identifier": marker})],
    )
    for rendered in (packet.to_json(), packet.to_markdown(), repr(packet)):
        assert marker not in rendered
        assert "123456" not in rendered


def test_review_status_cannot_override_a_blocking_gate():
    with pytest.raises(ValueError, match="gate"):
        build_review_packet(
            gates=[ReviewGateResult("privacy", False, blocking=True)],
            review_status="ready_for_review",
        )


def test_nested_metadata_is_frozen_and_rendered_as_an_independent_copy():
    finding = ReviewFinding(
        "synthetic-id", "finding", attributes={"source": {"count": 1}}
    )
    packet = build_review_packet([finding])
    original = packet.to_json()
    with pytest.raises(TypeError):
        finding.attributes["source"]["count"] = 2
    rendered = packet.to_dict()
    rendered["findings"][0]["attributes"]["source"]["count"] = 3
    assert packet.to_json() == original


def test_opaque_identifiers_preserve_citation_links_when_reconstructed():
    citation = ReviewCitation("synthetic-citation", "source")
    finding = ReviewFinding(
        "synthetic-finding", "finding", citation_ids=(citation.citation_id,)
    )
    assert finding.citation_ids == (citation.citation_id,)
    assert (
        ReviewCitation(citation.citation_id, citation.source).citation_id
        == citation.citation_id
    )


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
    assert {item["finding_id"] for item in payload["findings"]} == {
        "identifier:" + hash_text("finding-1"),
        "identifier:" + hash_text("finding-2"),
    }
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
    assert packet.gate_results[0].gate_id == "identifier:" + hash_text("local-check")
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


def test_unstructured_metadata_cannot_bypass_default_protected_text_boundary():
    private_value = "SYNTHETIC_PRIVATE_SOURCE_VALUE"
    packet = build_review_packet(
        findings=[
            ReviewFinding(
                finding_id=private_value,
                label=private_value,
                status=private_value,
                uncertainty=private_value,
                citation_ids=(private_value,),
                source_hash=private_value,
                attributes={private_value: 1},
            )
        ],
        citations=[
            ReviewCitation(
                citation_id=private_value,
                source=private_value,
                locator=private_value,
                title=private_value,
                published=private_value,
                source_hash=private_value,
                metadata={private_value: 2},
            )
        ],
        gates=[
            ReviewGateResult(
                gate_id=private_value,
                passed=False,
                severity=private_value,
                reason=f"review_{private_value}",
                reason_hash=private_value,
                citation_ids=(private_value,),
                details={private_value: 3},
            )
        ],
    )

    assert private_value.casefold() not in packet.to_json().casefold()
    assert private_value.casefold() not in packet.to_markdown().casefold()
    assert private_value.casefold() not in repr(packet).casefold()

    with pytest.raises(ValueError, match="schema version"):
        ReviewPacket(schema_version=private_value)
    with pytest.raises(ValueError, match="advisory"):
        ReviewPacket(advisory=private_value)
