"""Synthetic document-subset, critical-leakage and span-integrity controls."""

import json
import traceback
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from openmed.clinical.brief import ClinicalBrief, _digest, build_clinical_brief
from openmed.clinical.exporters.fhir import (
    BriefDocumentError,
    export_brief_document,
    import_brief_document,
)
from openmed.interop.fhir.reference_integrity import check_bundle_reference_integrity
from tests.unit.clinical.test_brief import fixture_context

DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_brief(sentences=None):
    value, context = (
        fixture_context() if sentences is None else fixture_context(sentences)
    )
    return build_clinical_brief(value, model="extractive", context=context)


def export(brief=None, **kwargs):
    return export_brief_document(
        brief or make_brief(), recorded_at=DATE, privacy_detector=lambda _: [], **kwargs
    )


def test_roundtrip_preserves_provenance_offsets_order_and_review_without_writes():
    brief = make_brief()
    result = export(brief)
    payload = result.to_response()
    bundle = payload["bundle"]
    assert check_bundle_reference_integrity(bundle).valid
    assert bundle["type"] == "document"
    assert all("request" not in entry for entry in bundle["entry"])
    composition = bundle["entry"][0]["resource"]
    assert composition["resourceType"] == "Composition"
    assert composition["status"] == "preliminary"
    assert payload["document_reference"]["docStatus"] == "preliminary"
    assert (
        payload["document_reference"]["content"][0]["attachment"]["url"]
        == payload["document_url"]
    )
    assert "attester" not in composition and "subject" not in composition
    assert [section["title"] for section in composition["section"]] == [
        "Claim 1",
        "Claim 2",
        "Claim 3",
        "Limitations",
    ]
    metadata = json.loads(composition["extension"][0]["valueString"])
    assert metadata["brief_digest"] == brief.digest
    assert metadata["provenance_digest"] == brief.to_dict()["provenance"]["record_hash"]
    for index, citation in enumerate(metadata["citations"]):
        assert {
            key: citation[key] for key in brief.citations[index]
        } == brief.citations[index]
        evidence = bundle["entry"][index + 2]
        assert composition["section"][index]["entry"] == [
            {"reference": evidence["fullUrl"]}
        ]
        assert evidence["resource"]["identifier"][0]["value"] == citation["evidence_id"]
        assert (
            evidence["resource"]["content"][0]["attachment"]["url"]
            == "urn:sha256:" + citation["evidence_hash"][7:]
        )
    imported = import_brief_document(
        json.loads(json.dumps(payload)), privacy_detector=lambda _: []
    )
    assert imported.to_response() == payload
    assert imported.to_dict() == result.to_dict()
    assert "source_payloads_omitted" in result.to_dict()["conversion_loss"]
    assert brief.summary not in repr(result)
    assert "dehydration" not in json.dumps(result.to_dict())
    payload["bundle"]["entry"].clear()
    assert result.to_response()["bundle"]["entry"]


@pytest.mark.parametrize(
    "status", ["final", "amended", "entered-in-error", "PRIVATE_SENTINEL"]
)
def test_no_brief_can_be_finalized(status):
    with pytest.raises(BriefDocumentError, match="^unsupported_finalization$"):
        export(status=status)


def test_refusal_and_missing_provenance_fail_closed():
    value, _ = fixture_context()
    refused = build_clinical_brief(value, model="extractive")
    with pytest.raises(BriefDocumentError, match="^refused_brief$"):
        export(refused)
    brief = make_brief()
    audit = brief.to_dict()
    audit.pop("digest")
    audit["provenance"] = {}
    with pytest.raises(BriefDocumentError, match="^missing_provenance$"):
        export(replace(brief, _audit_json=json.dumps(audit)))


@pytest.mark.parametrize(
    "mutation", ["summary", "record_hash", "evidence_hash", "review"]
)
def test_provenance_drift_and_fabricated_review_are_rejected(mutation):
    brief = make_brief()
    audit = brief.to_dict()
    audit.pop("digest")
    if mutation == "summary":
        brief = replace(brief, summary=brief.summary + " Changed.")
        audit["summary_digest"] = _digest(brief.summary)
        audit["summary_characters"] = len(brief.summary)
    elif mutation == "review":
        audit["provenance"]["review_status"] = "approved"
    elif mutation == "record_hash":
        audit["provenance"]["record_hash"] = "sha256:" + "0" * 64
    else:
        audit["provenance"]["evidence"][0]["evidence_hash"] = "sha256:" + "0" * 64
    with pytest.raises(BriefDocumentError):
        export(replace(brief, _audit_json=json.dumps(audit)))


@pytest.mark.parametrize(
    "mutation",
    [
        "status",
        "attachment",
        "extension",
        "narrative",
        "reference",
        "section_order",
        "request",
        "timestamp",
        "offset",
        "boolean_offset",
    ],
)
def test_closed_subset_rejects_unsafe_or_lossy_mutations(mutation):
    payload = export().to_response()
    composition = payload["bundle"]["entry"][0]["resource"]
    if mutation == "status":
        composition["status"] = "final"
    elif mutation == "attachment":
        payload["document_reference"]["content"][0]["attachment"]["title"] = (
            "/private/PRIVATE_SENTINEL"
        )
    elif mutation == "extension":
        composition["extension"].append(
            {"url": "https://example.invalid", "valueString": "PRIVATE_SENTINEL"}
        )
    elif mutation == "narrative":
        composition["text"]["div"] += "PRIVATE_SENTINEL"
    elif mutation == "reference":
        composition["section"][0]["entry"][0]["reference"] = (
            "https://ehr.invalid/Patient/PRIVATE_SENTINEL"
        )
    elif mutation == "section_order":
        composition["section"].reverse()
    elif mutation == "request":
        payload["bundle"]["entry"][0]["request"] = {
            "method": "POST",
            "url": "Composition",
        }
    elif mutation == "timestamp":
        payload["bundle"]["timestamp"] = "2026-01-01T00:00:00"
    else:
        metadata = json.loads(composition["extension"][0]["valueString"])
        metadata["citations"][0]["output_start"] = (
            True if mutation == "boolean_offset" else 1
        )
        composition["extension"][0]["valueString"] = json.dumps(metadata)
    with pytest.raises(BriefDocumentError) as caught:
        import_brief_document(payload, privacy_detector=lambda _: [])
    assert "PRIVATE_SENTINEL" not in str(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "identifier", ["dehydration", "DEHYDRATION", "dehydration@example.invalid"]
)
def test_original_identifier_tokens_never_pass_the_boundary(identifier):
    with pytest.raises(BriefDocumentError, match="^privacy$"):
        export(original_identifiers=(identifier,))


@pytest.mark.parametrize(
    "target", ["narrative", "extensions", "attachment", "noncritical"]
)
def test_detector_scans_every_rendered_surface_and_blocks_all_findings(target):
    seen = []

    def detector(text):
        seen.append(text)
        needle = {
            "narrative": "dehydration",
            "extensions": "provenance_digest",
            "attachment": "application/fhir+json",
            "noncritical": "dehydration",
        }[target]
        start = text.index(needle)
        return [
            {
                "label": "DATE" if target == "noncritical" else "NAME",
                "start": start,
                "end": start + len(needle),
            }
        ]

    with pytest.raises(BriefDocumentError, match="^privacy$"):
        export_brief_document(make_brief(), recorded_at=DATE, privacy_detector=detector)
    assert seen


def test_callback_errors_are_not_chained_or_rendered_and_missing_detector_blocks():
    def broken(_):
        raise RuntimeError("PRIVATE_SENTINEL /private/secret 123456789")

    def public_error(_):
        raise BriefDocumentError("PRIVATE_SENTINEL")

    for detector in (
        broken,
        public_error,
        None,
        lambda _: {"invalid": "PRIVATE_SENTINEL"},
    ):
        with pytest.raises(BriefDocumentError) as caught:
            export_brief_document(
                make_brief(), recorded_at=DATE, privacy_detector=detector
            )
        assert caught.value.__context__ is None
        assert "PRIVATE_SENTINEL" not in "".join(
            traceback.format_exception(caught.value)
        )


def test_explicit_export_time_is_required_and_normalized():
    with pytest.raises(BriefDocumentError, match="^invalid_export_time$"):
        export_brief_document(
            make_brief(),
            recorded_at=datetime(2026, 1, 1),
            privacy_detector=lambda _: [],
        )
    assert export().to_response()["bundle"]["timestamp"] == "2026-01-01T00:00:00+00:00"


def test_unicode_and_html_narrative_roundtrip_preserves_scalar_offsets(monkeypatch):
    import tests.unit.clinical.test_brief as fixtures

    monkeypatch.setattr(
        fixtures,
        "SENTENCES",
        (
            "The admission problem was nausea & dehydration.",
            "The discharge diagnosis was dehydration.",
            "Symptoms improved after café fluids.",
        ),
    )
    brief = make_brief(fixtures.SENTENCES)
    assert brief.refusal_reason is None
    result = export(brief)
    assert (
        "&amp;" in result.to_response()["bundle"]["entry"][0]["resource"]["text"]["div"]
    )
    assert (
        import_brief_document(
            result.to_response(), privacy_detector=lambda _: []
        ).to_response()
        == result.to_response()
    )
    with pytest.raises(BriefDocumentError, match="^privacy$"):
        import_brief_document(
            result.to_response(),
            privacy_detector=lambda _: [],
            original_identifiers=("CAFÉ",),
        )


def test_malformed_private_input_yields_only_a_controlled_error():
    with pytest.raises(BriefDocumentError, match="^invalid_document$") as caught:
        import_brief_document({"PRIVATE_SENTINEL": []}, privacy_detector=lambda _: [])
    assert caught.value.__context__ is None


@pytest.mark.parametrize("identifier", ["李雷", "Li", "123456789", "अनिल"])
def test_short_multilingual_and_numeric_identifiers_never_reach_output(
    monkeypatch, identifier
):
    import tests.unit.clinical.test_brief as fixtures

    monkeypatch.setattr(
        fixtures,
        "SENTENCES",
        (
            "The admission problem was dehydration.",
            "The discharge diagnosis was dehydration.",
            f"Symptoms improved after {identifier} fluids.",
        ),
    )
    brief = make_brief(fixtures.SENTENCES)
    assert brief.refusal_reason is None
    assert identifier in brief.summary
    with pytest.raises(BriefDocumentError, match="^privacy$"):
        export(brief, original_identifiers=(identifier,))
