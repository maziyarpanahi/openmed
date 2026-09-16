"""Synthetic offline tests for note-type routing and extraction scoping."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.clinical.routing import (
    GENERIC_PROFILE,
    PATHOLOGY_PROFILE,
    RADIOLOGY_PROFILE,
    ROUTING_PROVENANCE_KEY,
    attach_routing_provenance,
    build_extraction_plan,
    classify_and_select_profile,
    extract_scoped_medication_candidates,
    resolve_profile,
    route_analysis,
    select_profile,
)
from openmed.clinical.sections import classify_document, detect_sections

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "note_type_routing.jsonl"
)


def _fixture_rows() -> tuple[dict, ...]:
    return tuple(
        json.loads(line)
        for line in _FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _entity_spans(text: str, entities: list[dict]) -> list[dict]:
    spans = []
    for entity in entities:
        start = text.index(entity["surface"])
        spans.append(
            {
                "label": entity["label"],
                "score": 0.99,
                "start": start,
                "end": start + len(entity["surface"]),
                "surface": entity["surface"],
            }
        )
    return spans


def test_radiology_and_pathology_select_profiles_and_expected_sections() -> None:
    rows = _fixture_rows()

    for row in rows:
        classification = classify_document(row["text"])
        selection = resolve_profile(classification)
        sections = detect_sections(row["text"])

        assert selection.profile.name == row["note_type"].removesuffix("_report")
        assert selection.provenance.to_dict() == {
            "profile": selection.profile.name,
            "confidence": classification["confidence"],
            "fallback_reason": None,
        }
        assert [
            section.label for section in selection.profile.scope_sections(sections)
        ] == (row["expected_sections"])
        assert selection.profile.entity_priorities
        assert selection.profile.section_scoped_stage_config
        assert row["metadata"] == {"synthetic": True, "restricted_data": False}


def test_profile_specific_scoping_preserves_target_sections_and_drops_leaks() -> None:
    for row in _fixture_rows():
        text = row["text"]
        classification = classify_document(text)
        spans = _entity_spans(text, row["entities"])
        plan = build_extraction_plan(
            text,
            classification,
            medication_entities=spans,
            problem_mentions=spans,
            lab_value_mentions=spans,
        )

        kept_sections = {
            row["entities"][index]["section"]
            for index, entity in enumerate(row["entities"])
            if entity["surface"] in {span["surface"] for span in plan.problem_mentions}
        }
        assert "medications" not in kept_sections
        assert set(row["expected_sections"]) <= {
            section.label for section in plan.sections
        }

        if row["note_type"] == "pathology_report":
            assert "staging" in {
                section.label for section in plan.sections_for_stage("problem-list")
            }
            assert any(span["surface"] == "pT1 N0" for span in plan.problems)


def test_scoped_medication_filter_uses_detected_profile_sections() -> None:
    row = _fixture_rows()[0]
    text = row["text"]
    entities = _entity_spans(text, row["entities"])

    candidates = extract_scoped_medication_candidates(
        text,
        entities,
        classify_document(text),
    )

    assert [candidate.text for candidate in candidates] == ["contrast agent"]


def test_unknown_and_low_confidence_routes_are_pass_through_and_provenanced() -> None:
    entities = [
        {"label": "CONDITION", "start": 0, "end": 4},
        {"label": "MEDICATION", "start": 80, "end": 90},
    ]
    regression_text = "Synthetic note with unchanged generic extraction."

    for classification, reason in (
        ({"type": "unknown", "confidence": 0.0}, "unknown_document_type"),
        ({"type": "radiology_report", "confidence": 0.49}, "low_confidence"),
    ):
        selection = resolve_profile(classification)
        assert selection.profile is GENERIC_PROFILE
        assert selection.provenance.fallback_reason == reason

        plan = build_extraction_plan(
            regression_text,
            classification,
            medication_entities=entities,
        )
        assert list(plan.medication_entities) == entities
        assert plan.routing_provenance.to_dict() == {
            "profile": "generic",
            "confidence": classification["confidence"],
            "fallback_reason": reason,
        }


def test_analysis_results_carry_only_phi_free_routing_provenance() -> None:
    classification = {
        "type": "pathology_report",
        "confidence": 0.91,
    }
    legacy_result = {"entities": [{"label": "CONDITION"}], "metadata": {"kept": True}}

    annotated = attach_routing_provenance(legacy_result, classification)
    assert annotated["entities"] == legacy_result["entities"]
    assert annotated["metadata"]["kept"] is True
    assert annotated["metadata"][ROUTING_PROVENANCE_KEY] == {
        "profile": "pathology",
        "confidence": 0.91,
        "fallback_reason": None,
    }

    routed = route_analysis(
        "PATHOLOGY REPORT\nDIAGNOSIS: Synthetic finding.",
        legacy_result,
        classify_document_result=classification,
    )
    assert routed.profile is PATHOLOGY_PROFILE
    assert routed.routing_provenance["profile"] == "pathology"
    assert routed[ROUTING_PROVENANCE_KEY]["fallback_reason"] is None


def test_profile_constants_and_classifier_helper_are_stable() -> None:
    radiology_text = "RADIOLOGY REPORT\nFINDINGS: Synthetic finding."

    assert select_profile(classify_document(radiology_text)) is RADIOLOGY_PROFILE
    assert classify_and_select_profile(radiology_text).profile is RADIOLOGY_PROFILE
    assert GENERIC_PROFILE.pass_through is True
    assert RADIOLOGY_PROFILE.cue_terms["laterality"]
    assert PATHOLOGY_PROFILE.cue_terms["staging"]


def test_committed_routing_fixture_is_synthetic_and_unrestricted() -> None:
    raw = _FIXTURE.read_text(encoding="utf-8")

    assert "DUA" not in raw
    assert all(row["metadata"]["synthetic"] for row in _fixture_rows())
    assert all(not row["metadata"]["restricted_data"] for row in _fixture_rows())
