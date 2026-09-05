"""Regression tests for the canonical grounded-span FHIR facade."""

from __future__ import annotations

import inspect
import json

import pytest

from openmed.clinical.exporters.fhir import (
    FHIRBundle,
    FHIRExportSummary,
    to_bundle,
    to_fhir,
)
from openmed.clinical.exporters.fhir.facade import to_fhir as facade_to_fhir
from openmed.clinical.exporters.fhir.grounded import to_fhir as grounded_to_fhir
from openmed.clinical.grounding import Candidate, GroundedSpan


def _span(
    label: str | None,
    *,
    start: int,
    system: str,
    code: str,
) -> GroundedSpan:
    text = f"synthetic-{code}"
    return GroundedSpan(
        text=text,
        start=start,
        end=start + len(text),
        canonical_label=label,
        candidates=(
            Candidate(
                system=system,
                code=code,
                display=text,
                score=0.99,
                source="synthetic",
                matched_alias=text,
                match_kind="exact",
                vocab_version="synthetic-v1",
            ),
        ),
    )


def _mixed_spans() -> tuple[GroundedSpan, ...]:
    return (
        _span("CONDITION", start=0, system="ICD10CM", code="C-1"),
        _span("LAB_TEST", start=20, system="LOINC", code="L-1"),
        _span("MEDICATION", start=40, system="RXNORM", code="M-1"),
        _span("PROCEDURE", start=60, system="SNOMED", code="P-1"),
    )


def test_public_facade_dispatches_mixed_canonical_labels_to_one_bundle() -> None:
    bundle = to_fhir(_mixed_spans(), doc_id="synthetic-facade")

    assert isinstance(bundle, FHIRBundle)
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "transaction"
    assert set(bundle) == {"resourceType", "type", "entry"}
    assert [entry["resource"]["resourceType"] for entry in bundle["entry"]] == [
        "Condition",
        "Observation",
        "MedicationStatement",
        "Procedure",
    ]
    assert all(
        entry["resource"]["resourceType"] != "Patient" for entry in bundle["entry"]
    )
    assert bundle.summary == FHIRExportSummary(
        exported_by_label={
            "CONDITION": 1,
            "LAB_TEST": 1,
            "MEDICATION": 1,
            "PROCEDURE": 1,
        },
        unmapped_by_label={},
    )
    assert "summary" not in json.dumps(bundle, sort_keys=True)


def test_unmapped_canonical_label_is_skipped_and_counted() -> None:
    mapped = _span("CONDITION", start=0, system="ICD10CM", code="C-1")
    unmapped = _span("BODY_SITE", start=20, system="SNOMED", code="B-1")

    bundle = to_fhir((mapped, unmapped), doc_id="synthetic-unmapped")

    assert [entry["resource"]["resourceType"] for entry in bundle["entry"]] == [
        "Condition"
    ]
    assert bundle.summary.to_dict() == {
        "exported_by_label": {"CONDITION": 1},
        "unmapped_by_label": {"BODY_SITE": 1},
        "resource_count": 1,
        "unmapped_count": 1,
    }


def test_facade_output_is_byte_stable_and_honors_bundle_type() -> None:
    first = to_fhir(
        _mixed_spans(),
        doc_id="synthetic-stable",
        bundle_type="batch",
    )
    second = to_fhir(
        _mixed_spans(),
        doc_id="synthetic-stable",
        bundle_type="batch",
    )

    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert first.summary.to_dict() == second.summary.to_dict()
    assert first["type"] == "batch"


def test_doc_id_alias_and_public_imports_share_one_implementation() -> None:
    by_doc_id = to_fhir(_mixed_spans(), doc_id="synthetic-alias")
    by_legacy_name = to_fhir(
        _mixed_spans(),
        document_id="synthetic-alias",
    )

    assert by_doc_id == by_legacy_name
    assert facade_to_fhir is grounded_to_fhir is to_fhir
    assert inspect.signature(to_fhir).parameters["doc_id"].default == (
        "openmed-document"
    )
    assert callable(to_bundle)

    with pytest.raises(ValueError, match="must match"):
        to_fhir(
            _mixed_spans(),
            doc_id="synthetic-one",
            document_id="synthetic-two",
        )


def test_system_routes_are_explicit_and_only_apply_to_unlabeled_spans() -> None:
    unlabeled = _span(None, start=0, system="LOINC", code="L-1")
    unknown_label = _span("OTHER", start=20, system="LOINC", code="L-2")

    bundle = to_fhir(
        (unlabeled, unknown_label),
        doc_id="synthetic-system-route",
        systems={"loinc": "Observation"},
    )

    assert [entry["resource"]["resourceType"] for entry in bundle["entry"]] == [
        "Observation"
    ]
    assert bundle.summary.exported_by_label == {"UNLABELED": 1}
    assert bundle.summary.unmapped_by_label == {"OTHER": 1}
