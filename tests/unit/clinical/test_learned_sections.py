"""Synthetic offline tests for learned clinical section refinement."""

from __future__ import annotations

from openmed.clinical.sections import (
    SECTION_LOINC_MAP,
    SectionHead,
    detect_sections,
    validate_sections,
)
from openmed.eval import run_section_messy_eval
from openmed.processing.text import resolve_sections

MESSY_NOTE = (
    "Patient reports worsening cough for three days. "
    "Past medical history notable for childhood asthma. "
    "Current medications include a synthetic inhaler. "
    "Assessment is a viral URI and plan is supportive care."
)


def test_learned_refinement_reconciles_messy_note_into_coded_spans() -> None:
    rules = detect_sections(MESSY_NOTE)
    refined = detect_sections(MESSY_NOTE, use_learned=True)

    assert [section["label"] for section in rules] == ["unsectioned"]
    assert [section["label"] for section in refined] == [
        "history_of_present_illness",
        "past_medical_history",
        "medications",
        "assessment",
        "plan",
    ]
    assert all(section["source"] == "learned" for section in refined)
    assert all(0.0 <= section["confidence"] <= 1.0 for section in refined)
    assert all(section["codes"] for section in refined)
    assert all(
        section["loinc_code"] == SECTION_LOINC_MAP[section["label"]]
        for section in refined
    )
    validate_sections(MESSY_NOTE, refined)
    assert "Current medications" in MESSY_NOTE[refined[2]["start"] :]


def test_rule_spans_carry_source_confidence_and_loinc_codings() -> None:
    text = "HPI: Synthetic cough.\nAssessment and Plan: Continue care."

    sections = detect_sections(text)

    assert [section["source"] for section in sections] == ["rule", "rule"]
    assert all(section["confidence"] == 0.9 for section in sections)
    assert [section["loinc_code"] for section in sections] == [
        "10164-2",
        "51847-2",
    ]
    assert all(
        section["coding"][0]["system"] == "http://loinc.org" for section in sections
    )


def test_injected_head_is_used_only_for_an_unsectioned_gap() -> None:
    calls: list[str] = []
    text = "Synthetic narrative. Current medications include a synthetic tablet."
    start = text.index("Current medications")

    def predictor(source: str, *, language: str | None = None):
        calls.append(language or "default")
        return [
            {"label": "medications", "start": start, "end": len(source), "score": 0.99}
        ]

    sections = detect_sections(text, language="en", learned_head=predictor)

    assert calls == ["en"]
    assert [section["label"] for section in sections] == [
        "unsectioned",
        "medications",
    ]
    assert sections[-1]["source"] == "learned"
    validate_sections(text, sections)


def test_section_head_is_lazy_and_accepts_injected_predictions() -> None:
    head = SectionHead(predictor=lambda text, **_: [("medications", 0, len(text))])

    assert not head.loaded
    result = head("Medications include a synthetic tablet.")

    assert not head.loaded
    assert result[0]["label"] == "medications"
    assert result[0]["source"] == "learned"


def test_resolve_sections_preserves_precomputed_coded_sequence() -> None:
    supplied = detect_sections("MEDICATIONS: Synthetic tablet.")

    resolved = resolve_sections("ignored because supplied spans win", supplied)

    assert resolved == supplied
    assert resolved[0] is supplied[0]


def test_public_messy_section_report_beats_rules_baseline() -> None:
    report = run_section_messy_eval()

    assert report.metrics["section_gate_passed"] is True
    assert report.metrics["boundary_f1"] >= 0.80
    assert report.metrics["label_accuracy"] >= 0.85
    assert report.metrics["boundary_f1"] > report.metrics["baseline_boundary_f1"]
    assert report.metadata["leakage_check_passed"] is True
