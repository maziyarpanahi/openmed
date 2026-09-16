"""Standalone MCP clinical handler tests with synthetic offline artifacts."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from openmed.core.schemas import OpenMedSpan, hmac_text_hash
from openmed.mcp import server as mcp_server
from openmed.mcp.tool_registry import validate_registered_tool_output

_SYNTHETIC_SURFACE = "aster syndrome alpha"
_SYNTHETIC_PRIVATE_MARKER = "synthetic-person-marker-alpha"


def _clinical_span() -> dict[str, object]:
    return OpenMedSpan(
        doc_id="synthetic-note-1301",
        start=12,
        end=12 + len(_SYNTHETIC_SURFACE),
        text_hash=hmac_text_hash(_SYNTHETIC_SURFACE, "synthetic-test-key"),
        entity_type="synthetic_condition",
        canonical_label="CONDITION",
        score=0.99,
        detector="synthetic-unit-detector",
        evidence={"synthetic": True},
        metadata={
            "grounding_surface": _SYNTHETIC_SURFACE,
            "nested": {"text": _SYNTHETIC_PRIVATE_MARKER},
            "synthetic": True,
        },
    ).to_dict()


def _install_synthetic_vocabulary(cache_root: Path) -> None:
    vocabulary = cache_root / "grounding" / "icd10cm" / "concepts.jsonl"
    vocabulary.parent.mkdir(parents=True)
    vocabulary.write_text(
        json.dumps(
            {
                "aliases": [_SYNTHETIC_SURFACE],
                "canonical_term": "Aster syndrome",
                "concept_id": "SYN-COND-1301",
                "system": "icd10cm",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_standalone_ground_and_fhir_handlers_are_deterministic_and_phi_safe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_synthetic_vocabulary(tmp_path)
    monkeypatch.setenv("OPENMED_CACHE_DIR", str(tmp_path))
    span = _clinical_span()

    first = mcp_server.openmed_ground(
        spans=[deepcopy(span)],
        vocabularies=["icd10cm"],
        max_candidates=1,
    )
    second = mcp_server.openmed_ground(
        spans=[deepcopy(span)],
        vocabularies=["icd10cm"],
        max_candidates=1,
    )

    assert first == second
    assert first["status"] == "completed"
    assert first["error"] is None
    assert first["grounded_concepts"][0]["code"] == "SYN-COND-1301"
    assert first["grounded_concepts"][0]["display"] == "Aster syndrome"
    assert validate_registered_tool_output("openmed_ground", first) == first
    grounding_json = json.dumps(first, sort_keys=True)
    assert _SYNTHETIC_SURFACE not in grounding_json
    assert _SYNTHETIC_PRIVATE_MARKER not in grounding_json

    exported = mcp_server.openmed_export_fhir(
        spans=first["spans"],
        doc_id="synthetic-note-1301",
        bundle_type="collection",
    )

    assert exported["status"] == "completed"
    assert exported["error"] is None
    assert exported["resource_count"] == 1
    resource = exported["bundle"]["entry"][0]["resource"]
    assert resource["resourceType"] == "Condition"
    assert resource["code"]["coding"][0]["code"] == "SYN-COND-1301"
    assert validate_registered_tool_output("openmed_export_fhir", exported) == exported
    export_json = json.dumps(exported, sort_keys=True)
    assert _SYNTHETIC_SURFACE not in export_json
    assert _SYNTHETIC_PRIVATE_MARKER not in export_json


def test_fhir_handler_removes_direct_patient_identifier_fields() -> None:
    response = mcp_server.openmed_export_fhir(
        spans=[_clinical_span()],
        resources=[
            {
                "resourceType": "Patient",
                "id": "synthetic-patient-1301",
                "name": [{"text": _SYNTHETIC_PRIVATE_MARKER}],
                "identifier": [{"value": _SYNTHETIC_PRIVATE_MARKER}],
            }
        ],
    )

    assert response["status"] == "completed"
    patient = response["bundle"]["entry"][0]["resource"]
    assert "name" not in patient
    assert "identifier" not in patient
    assert _SYNTHETIC_PRIVATE_MARKER not in json.dumps(response, sort_keys=True)


def test_risk_handler_returns_only_aggregate_phi_safe_output() -> None:
    records = [
        {
            "age": 93,
            "city": "Synthetic Harbor Alpha",
            "condition": "Aster syndrome",
        }
    ]

    first = mcp_server.openmed_risk_score(
        spans=[_clinical_span()],
        records=deepcopy(records),
        quasi_identifiers=["age", "city"],
    )
    second = mcp_server.openmed_risk_score(
        spans=[_clinical_span()],
        records=deepcopy(records),
        quasi_identifiers=["age", "city"],
    )

    assert first == second
    assert first["status"] == "completed"
    assert first["error"] is None
    assert first["risk_report"]["detail_level"] == "aggregate_phi_safe"
    assert first["risk_report"]["record_count"] == 1
    assert first["risk_report"]["minimum_k"] == 1
    assert first["risk_report"]["singleton_record_count"] == 1
    assert validate_registered_tool_output("openmed_risk_score", first) == first
    serialized = json.dumps(first, sort_keys=True)
    assert "Synthetic Harbor Alpha" not in serialized
    assert _SYNTHETIC_PRIVATE_MARKER not in serialized
    assert "quasi_identifier_key" not in serialized
