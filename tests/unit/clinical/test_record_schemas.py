"""Contract tests for the bundled clinical record JSON Schemas.

The fixtures are synthetic and PHI-free: controlled codes, counts, offsets,
opaque references, and digests only.
"""

from __future__ import annotations

import builtins
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from openmed.clinical.brief import (
    BriefContext,
    BriefFact,
    BriefRefusal,
    _compose,
    _digest,
    brief_policy_fingerprint,
)
from openmed.clinical.evidence_packet import (
    EVIDENCE_PACKET_KIND,
    EVIDENCE_PACKET_SCHEMA_VERSION,
    REJECTION_CATEGORIES,
    build_evidence_packet,
    fingerprint_evidence_review,
)
from openmed.clinical.nli import NLI_LABELS, verify
from openmed.clinical.nli_gate import NLIThresholds
from openmed.clinical.record_schemas import (
    CLINICAL_RECORD_SCHEMA_NAMES,
    ClinicalRecordSchemaError,
    clinical_record_schema_fingerprint,
    clinical_record_schema_id,
    clinical_record_schema_json,
    clinical_record_schema_snapshot,
    compare_all_clinical_record_drift,
    compare_clinical_record_drift,
    load_clinical_record_schema,
    load_clinical_record_snapshot,
    validate_clinical_record,
    write_clinical_record_snapshot,
)
from openmed.clinical.review_packet import (
    PROTECTED_TEXT_POLICY,
    REVIEW_PACKET_SCHEMA_VERSION,
)
from openmed.clinical.review_state_machine import (
    ReviewState,
    ReviewStateMachine,
    make_opaque_event_id,
)
from openmed.clinical.sdoh_evidence import (
    ASSERTION_STATUSES,
    EVIDENCE_TYPES,
    REVIEW_STATUSES,
    SDOH_DETERMINANTS,
    SDOH_EVIDENCE_SCHEMA_VERSION,
    SOURCE_SECTIONS,
    SDOHEvidence,
    SDOHEvidenceReport,
)
from openmed.core.capabilities import MissingOptionalDependencyError
from openmed.core.pii import DeidentificationResult

SENTENCES = (
    "The admission problem was dehydration.",
    "The discharge diagnosis was dehydration.",
    "Symptoms improved after fluids.",
)

FORBIDDEN_TEXT_KEYS = frozenset(
    {
        "claim",
        "content",
        "excerpt",
        "raw_text",
        "source_text",
        "surface",
        "text",
        "value",
    }
)


def _reviewed_brief_records() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the audit and response records of one reviewed synthetic brief."""

    text = " ".join(SENTENCES)
    result = DeidentificationResult(text, text, [], "mask", datetime(2026, 1, 1))
    facts = [
        BriefFact(
            "synthetic:ref-" + str(i), field, "affirmed", "certain", "recent", "patient"
        )
        for i, field in enumerate(
            ("admission_reason", "discharge_diagnoses", "hospital_course")
        )
    ]
    policy = brief_policy_fingerprint(text, tuple(facts))
    rows = []
    for i, sentence in enumerate(SENTENCES):
        ref = "synthetic:ref-" + str(i)
        start = text.index(sentence)
        row = dict(
            reference_id=ref,
            source_id="synthetic:document",
            start=start,
            end=start + len(sentence),
            policy_fingerprint=policy,
            review_state="approved",
            synthetic=True,
            verified=True,
        )
        fingerprint = fingerprint_evidence_review(
            **{
                key: row[key]
                for key in (
                    "reference_id",
                    "source_id",
                    "start",
                    "end",
                    "policy_fingerprint",
                )
            }
        )
        machine = ReviewStateMachine()
        for state in (ReviewState.IN_REVIEW, ReviewState.APPROVED):
            machine.transition(
                state, make_opaque_event_id((ref, state.value)), fingerprint
            )
        row["review_transitions"] = machine.transitions
        rows.append(row)
    thresholds = NLIThresholds(
        calibration_id="synthetic-test-only", calibration_method="synthetic-fixture"
    )
    context = BriefContext(
        build_evidence_packet(rows, policy_fingerprint=policy),
        _digest(text),
        tuple(facts),
        lambda _premise, _hypothesis: {
            "entailment": 1.0,
            "contradiction": 0.0,
            "neutral": 0.0,
            "calibration_id": "synthetic-test-only",
        },
        thresholds,
        lambda _text: [],
    )
    brief = _compose(result, "extractive", "bhc", context, [])
    assert brief.refusal_reason is None
    return brief.to_dict(), brief.to_response()


def _record_payloads() -> dict[str, Any]:
    """Return one real serialization for every clinical record schema."""

    audit, response = _reviewed_brief_records()
    policy = brief_policy_fingerprint(" ".join(SENTENCES), ())
    evidence_packet = build_evidence_packet([], policy_fingerprint=policy)
    finding = SDOHEvidence(
        evidence_type="self_report",
        assertion="present",
        source_section="social_history",
        source_span=(12, 40),
        determinant="housing",
    )
    return {
        "brief_audit": audit,
        "brief_response": response,
        "evidence_packet": evidence_packet.to_dict(),
        "nli_verification": verify(
            ["The discharge diagnosis was dehydration."],
            " ".join(SENTENCES),
            backend="heuristic",
        ),
        "sdoh_evidence_report": SDOHEvidenceReport(findings=(finding,)).to_dict(),
    }


def _property_names(node: Any) -> set[str]:
    """Collect every declared property name in a JSON Schema document."""

    names: set[str] = set()
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict):
            names.update(str(key) for key in properties)
        for value in node.values():
            names |= _property_names(value)
    elif isinstance(node, list):
        for entry in node:
            names |= _property_names(entry)
    return names


def _resolve(schema: dict[str, Any], node: Any) -> Any:
    """Follow local ``$ref`` pointers so nested declarations can be asserted."""

    while isinstance(node, dict) and "$ref" in node:
        target: Any = schema
        for part in str(node["$ref"]).removeprefix("#/").split("/"):
            target = target[part]
        node = target
    return node


def test_every_bundled_record_schema_is_valid_and_identified() -> None:
    for name in CLINICAL_RECORD_SCHEMA_NAMES:
        schema = load_clinical_record_schema(name)
        Draft202012Validator.check_schema(schema)
        assert schema["$id"] == clinical_record_schema_id(name)
        assert isinstance(schema["schema_version"], int)
        assert json.loads(clinical_record_schema_json(name)) == schema


def test_serialized_records_validate_against_bundled_schemas() -> None:
    for name, record in _record_payloads().items():
        validate_clinical_record(name, record)
    audit, response = _reviewed_brief_records()
    assert "summary" not in audit
    assert response["summary"] == " ".join(SENTENCES) and "summary" in response


def test_records_with_extra_text_fields_are_rejected() -> None:
    payloads = _record_payloads()
    payloads["brief_audit"]["summary"] = "synthetic free text"
    payloads["brief_response"]["source_text"] = "synthetic free text"
    payloads["evidence_packet"]["excerpt"] = "synthetic free text"
    payloads["nli_verification"][0]["text"] = "synthetic free text"
    payloads["sdoh_evidence_report"]["findings"][0]["source_text"] = "synthetic text"
    for name, record in payloads.items():
        with pytest.raises(ClinicalRecordSchemaError):
            validate_clinical_record(name, record)


@pytest.mark.parametrize(
    "name",
    ["brief_audit", "evidence_packet", "nli_verification", "sdoh_evidence_report"],
)
def test_value_free_record_schemas_never_declare_free_text_fields(name: str) -> None:
    assert not (
        _property_names(load_clinical_record_schema(name)) & FORBIDDEN_TEXT_KEYS
    )


def test_generated_summary_text_is_declared_only_where_it_is_produced() -> None:
    audit = load_clinical_record_schema("brief_audit")
    assert "summary" not in audit["properties"]
    envelope = _resolve(audit, audit["properties"]["envelope"])
    envelope_summary = _resolve(audit, envelope["properties"]["summary"]["anyOf"][0])
    assert envelope_summary["additionalProperties"] is False
    assert envelope_summary["required"] == ["summary_digest"]
    assert "summary_digest" in envelope_summary["properties"]
    response = load_clinical_record_schema("brief_response")
    assert response["properties"]["summary"] == {"type": "string"}
    assert set(response["properties"]) - set(audit["properties"]) == {"summary"}


def test_committed_snapshot_matches_bundled_schemas() -> None:
    snapshot = clinical_record_schema_snapshot()
    assert snapshot == load_clinical_record_snapshot()
    assert sorted(snapshot) == sorted(CLINICAL_RECORD_SCHEMA_NAMES)
    for name, state in snapshot.items():
        assert state["fingerprint"] == clinical_record_schema_fingerprint(name)
        assert state["properties"] == sorted(state["properties"])
        assert state["required"] == sorted(state["required"])
        assert state["required"] == [
            key for key in state["required"] if key in state["properties"]
        ]


def test_removing_a_declared_field_is_reported_as_breaking_drift() -> None:
    schema = json.loads(json.dumps(load_clinical_record_schema("evidence_packet")))
    del schema["properties"]["rejection_report"]
    drift = compare_clinical_record_drift("evidence_packet", schema)
    assert drift.breaking_change is True
    assert drift.removed_properties == ("rejection_report",)
    assert drift.fingerprint != drift.snapshot_fingerprint

    schema["schema_version"] = EVIDENCE_PACKET_SCHEMA_VERSION + 1
    bumped = compare_clinical_record_drift("evidence_packet", schema)
    assert bumped.version_bumped is True
    assert bumped.breaking_change is False


def test_any_schema_edit_changes_the_record_fingerprint() -> None:
    schema = json.loads(json.dumps(load_clinical_record_schema("nli_verification")))
    schema["title"] = "OpenMed clinical NLI verification record (renamed)"
    drift = compare_clinical_record_drift("nli_verification", schema)
    assert drift.fingerprint != drift.snapshot_fingerprint
    assert drift.breaking_change is False
    assert compare_all_clinical_record_drift()["nli_verification"] == (
        compare_clinical_record_drift("nli_verification")
    )


def test_unknown_record_schema_names_are_rejected() -> None:
    with pytest.raises(KeyError):
        clinical_record_schema_fingerprint("brief_audit_v2")
    with pytest.raises(KeyError):
        validate_clinical_record("brief_audit_v2", {})


def test_inverted_offsets_are_rejected_even_when_the_schema_passes() -> None:
    payloads = _record_payloads()
    payloads["evidence_packet"]["references"] = [
        {
            "reference_id": "synthetic:reference",
            "source_id": "synthetic:document",
            "start": 40,
            "end": 12,
            "review_state": "approved",
            "policy_fingerprint": "sha256:" + "0" * 64,
            "review_transitions": [],
            "synthetic": True,
            "verified": True,
        }
    ]
    payloads["sdoh_evidence_report"]["findings"][0]["source_span"] = [40, 12]
    payloads["brief_audit"]["citations"][0]["output_start"] = 9
    payloads["brief_audit"]["citations"][0]["output_end"] = 2
    with pytest.raises(ClinicalRecordSchemaError, match="references\\[0\\]"):
        validate_clinical_record("evidence_packet", payloads["evidence_packet"])
    with pytest.raises(ClinicalRecordSchemaError, match="findings\\[0\\].source_span"):
        validate_clinical_record(
            "sdoh_evidence_report", payloads["sdoh_evidence_report"]
        )
    with pytest.raises(ClinicalRecordSchemaError, match="citations\\[0\\].output"):
        validate_clinical_record("brief_audit", payloads["brief_audit"])


def test_validation_reports_the_failing_location() -> None:
    payloads = _record_payloads()
    payloads["evidence_packet"]["references"] = [
        {
            "reference_id": "synthetic:reference",
            "source_id": "synthetic:document",
            "start": -1,
            "end": 12,
            "review_state": "approved",
            "policy_fingerprint": "sha256:" + "0" * 64,
            "review_transitions": [],
            "synthetic": True,
            "verified": True,
        }
    ]
    with pytest.raises(ClinicalRecordSchemaError) as excinfo:
        validate_clinical_record("evidence_packet", payloads["evidence_packet"])
    assert "references/0/start" in str(excinfo.value)


def test_schema_enums_track_the_runtime_contracts() -> None:
    audit = load_clinical_record_schema("brief_audit")
    response = load_clinical_record_schema("brief_response")
    refusal_enum = audit["$defs"]["refusal_reason"]["enum"]
    assert refusal_enum == sorted(member.value for member in BriefRefusal)
    assert (
        response["properties"]["refusal_reason"]
        == (audit["properties"]["refusal_reason"])
    )
    review_packet = _resolve(audit, audit["properties"]["review_packet"])
    assert review_packet["properties"]["schema_version"]["const"] == (
        REVIEW_PACKET_SCHEMA_VERSION
    )
    privacy = _resolve(audit, review_packet["properties"]["privacy"])
    assert privacy["properties"]["protected_text_policy"]["const"] == (
        PROTECTED_TEXT_POLICY
    )
    assert audit["properties"]["schema_version"]["const"] == 1
    assert response["properties"]["schema_version"]["const"] == 1

    packet = load_clinical_record_schema("evidence_packet")
    assert packet["properties"]["kind"]["const"] == EVIDENCE_PACKET_KIND
    assert packet["properties"]["schema_version"]["const"] == (
        EVIDENCE_PACKET_SCHEMA_VERSION
    )
    assert packet["$defs"]["rejection_counts"]["propertyNames"]["enum"] == sorted(
        REJECTION_CATEGORIES
    )
    reference = packet["$defs"]["reference"]
    assert _resolve(packet, reference["properties"]["review_state"]) == {
        "enum": ["approved"]
    }

    nli = load_clinical_record_schema("nli_verification")
    assert nli["$defs"]["verification"]["properties"]["label"]["enum"] == sorted(
        NLI_LABELS
    )

    sdoh = load_clinical_record_schema("sdoh_evidence_report")
    finding = sdoh["$defs"]["finding"]["properties"]
    assert sdoh["properties"]["schema_version"]["const"] == (
        SDOH_EVIDENCE_SCHEMA_VERSION
    )
    assert finding["assertion"]["enum"] == sorted(ASSERTION_STATUSES)
    assert finding["evidence_type"]["enum"] == sorted(EVIDENCE_TYPES)
    assert finding["review_status"]["enum"] == sorted(REVIEW_STATUSES)
    assert finding["source_section"]["enum"] == sorted(SOURCE_SECTIONS)
    assert finding["determinant"]["enum"] == sorted(SDOH_DETERMINANTS)


def test_snapshot_writer_round_trips(tmp_path: Path) -> None:
    path = write_clinical_record_snapshot(
        tmp_path / "clinical-record-fingerprints.json"
    )
    assert path.read_bytes() == (
        json.dumps(clinical_record_schema_snapshot(), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    assert load_clinical_record_snapshot(path) == clinical_record_schema_snapshot()


def test_missing_jsonschema_reports_the_dev_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "jsonschema":
            raise ImportError("jsonschema is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(MissingOptionalDependencyError) as excinfo:
        validate_clinical_record("sdoh_evidence_report", {})
    assert "jsonschema" in str(excinfo.value)
    assert "dev" in str(excinfo.value)
