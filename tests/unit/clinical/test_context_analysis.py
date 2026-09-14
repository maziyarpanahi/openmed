"""Clinical context composition with synthetic source and offset evidence."""

import copy
import json
from types import SimpleNamespace

import pytest

from openmed.clinical import (
    ClinicalAnalysisError,
    analyze_clinical_context,
    validated_clinical_entities,
)


def span(text, surface, label="Disease", **extra):
    start = text.index(surface)
    return {
        "start": start,
        "end": start + len(surface),
        "label": label,
        "score": 0.95,
        **extra,
    }


def test_german_sections_assertions_and_evidence_preserve_original_unicode_offsets():
    text = "🩺 Patientin: Anna Beispiel.\r\nFamilienanamnese: Mutter mit Diabetes.\r\nMedikation: Metformin 500 mg zweimal täglich.\r\nBefund: Keine Dyspnoe."
    spans = [
        span(text, "Diabetes"),
        span(text, "Metformin", "Drug"),
        span(text, "Dyspnoe", "Symptom"),
    ]
    original = copy.deepcopy(spans)
    result = analyze_clinical_context(text, spans, language="DE")
    assert result["complete"] and result["status"] == "needs_review"
    assert result["qualification"]["qualified_languages"] == []
    assert result["language"]["language"] == "de"
    sections = result["tasks"]["sections"]["records"]
    assert [s["label"] for s in sections] == [
        "unsectioned",
        "family_history",
        "medications",
        "findings",
    ]
    assert "".join(text[s["start"] : s["end"]] for s in sections) == text
    assertions = result["tasks"]["assertions"]["records"]
    assert [(a["experiencer"], a["negation"]) for a in assertions] == [
        ("family", "affirmed"),
        ("patient", "affirmed"),
        ("patient", "negated"),
    ]
    assert any(
        text[e["start"] : e["end"]] == "Mutter" for e in assertions[0]["evidence"]
    )
    assert any(
        text[e["start"] : e["end"]] == "Keine" for e in assertions[2]["evidence"]
    )
    encoded = json.dumps(result)
    assert all(
        value not in encoded
        for value in [
            "Anna Beispiel",
            "Diabetes",
            "Metformin",
            "Dyspnoe",
            "Mutter",
            "Keine",
        ]
    )
    assert spans == original


def test_section_prior_is_not_reported_as_an_explicit_subject_cue():
    text = "Familienanamnese: Diabetes."
    result = analyze_clinical_context(text, [span(text, "Diabetes")], language="de")
    assertion = result["tasks"]["assertions"]["records"][0]
    assert assertion["experiencer"] == "family"
    assert assertion["context_sources"]["experiencer"] == "section"
    assert [e["source"] for e in assertion["evidence"]] == ["section_header"]


def test_incomplete_model_coverage_retains_only_independent_section_result():
    text = "Family History: Diabetes."
    result = analyze_clinical_context(
        text, [span(text, "Diabetes")], language="en", entity_coverage_complete=False
    )
    assert result["status"] == "partial" and not result["complete"]
    assert result["tasks"]["sections"]["complete"]
    for task in ("entities", "assertions"):
        assert (
            not result["tasks"][task]["records"]
            and not result["tasks"][task]["complete"]
        )


def test_unimplemented_assertion_language_never_falls_back_to_english():
    text = "Antécédents familiaux: diabète."
    result = analyze_clinical_context(text, [span(text, "diabète")], language="fr")
    assert result["tasks"]["assertions"]["status"] == "unsupported"
    assert result["tasks"]["assertions"]["records"] == []
    assert result["tasks"]["sections"]["complete"]
    assert result["tasks"]["entities"]["complete"]
    assert result["status"] == "partial"


@pytest.mark.parametrize("mixed", [False, True])
def test_uncertain_or_mixed_language_requires_an_explicit_context_override(
    monkeypatch, mixed
):
    monkeypatch.setattr(
        "openmed.clinical.analysis.resolve_clinical_language",
        lambda *a, **kw: SimpleNamespace(
            language="en",
            locale="en_US",
            source="auto",
            confidence=0.5,
            needs_review=True,
            mixed=mixed,
        ),
    )
    result = analyze_clinical_context(
        "Diabetes.", [{"start": 0, "end": 8, "label": "Disease"}]
    )
    assert result["tasks"]["assertions"]["status"] == "unsupported"
    assert result["language"]["needs_review"]


@pytest.mark.parametrize(
    "changes",
    [
        {"start": True},
        {"end": 99},
        {"label": "private-label-marker"},
        {"score": float("nan")},
        {"score": float("inf")},
        {"score": True},
        {"score": 1.1},
        {"text": "private-text-marker"},
    ],
)
def test_invalid_entities_fail_without_raw_error_echo(changes):
    with pytest.raises(ClinicalAnalysisError) as failure:
        analyze_clinical_context(
            "Diabetes.",
            [dict({"start": 0, "end": 8, "label": "Disease"}, **changes)],
            language="en",
        )
    assert "private" not in str(failure.value)


def test_deduplication_prefers_measured_confidence_and_keeps_stable_source_ids():
    text = "Diabetes and metformin."
    output = validated_clinical_entities(
        text,
        [
            span(text, "metformin", "Medication"),
            span(text, "Diabetes", score=None),
            span(text, "Diabetes", score=0.8),
            span(text, "Diabetes", score=0.7),
        ],
    )
    assert [(r["id"], r["label"]) for r in output] == [
        ("e1", "Disease"),
        ("e2", "Drug"),
    ]
    assert output[0]["score"] == 0.8
    unknown = validated_clinical_entities(
        "Diabetes", [{"start": 0, "end": 8, "label": "Disease"}]
    )[0]
    assert unknown["score"] is None


def test_entity_limit_stops_consuming_an_unbounded_iterator(monkeypatch):
    monkeypatch.setattr("openmed.clinical.analysis.MAX_ANALYSIS_ENTITIES", 2)
    from itertools import repeat

    with pytest.raises(ClinicalAnalysisError, match="clinical_entity_limit"):
        analyze_clinical_context(
            "Diabetes.",
            repeat({"start": 0, "end": 8, "label": "Disease"}),
            language="en",
        )


def test_cancellation_and_postprocessing_deadline_cannot_return_partial_success(
    monkeypatch,
):
    with pytest.raises(ClinicalAnalysisError, match="clinical_cancelled"):
        analyze_clinical_context(
            "Diabetes.", [], language="en", cancel_check=lambda: True
        )
    from openmed.core.clinical_language import resolve_clinical_language

    now = [0.0]
    monkeypatch.setattr("openmed.clinical.analysis.time.monotonic", lambda: now[0])

    def delayed(*args, **kwargs):
        now[0] = 31.0
        return resolve_clinical_language(*args, **kwargs)

    monkeypatch.setattr("openmed.clinical.analysis.resolve_clinical_language", delayed)
    with pytest.raises(ClinicalAnalysisError, match="clinical_timeout"):
        analyze_clinical_context("Diabetes.", [], language="en")
