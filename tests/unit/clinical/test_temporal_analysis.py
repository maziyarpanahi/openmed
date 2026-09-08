"""Source-backed clinical event/timeline API and conservative anchoring."""

import json

import pytest

from openmed.clinical.analysis import ClinicalAnalysisError, analyze_clinical_context


def analyze(text, mentions, **kwargs):
    entities = [
        {
            "start": text.index(value),
            "end": text.index(value) + len(value),
            "label": label,
            "score": 0.99,
        }
        for value, label in mentions
    ]
    return analyze_clinical_context(
        text,
        entities,
        language=kwargs.pop("language", "de"),
        tasks=kwargs.pop("tasks", ["events", "timeline"]),
        **kwargs,
    )


def test_german_event_roles_and_chronological_order_are_source_backed():
    text = "Am 03.02.2026 erfolgte Kardioversion.\nAm 01.02.2026 wurde Metoprolol von 25 mg auf 50 mg erhöht."
    result = analyze(
        text,
        [
            ("Kardioversion", "Procedure"),
            ("Metoprolol", "Drug"),
            ("25 mg", "Dose"),
            ("50 mg", "Dose"),
        ],
    )
    assert result["complete"] and result["status"] == "needs_review"
    events = result["tasks"]["events"]["records"]
    change = next(e for e in events if e["type"] == "medication_change")
    assert change["trigger"]["value"] == "increased"
    assert text[change["trigger"]["start"] : change["trigger"]["end"]] == "erhöht"
    assert {a["role"]: a["normalized"]["value"] for a in change["attributes"]} == {
        "old_dose": 25,
        "new_dose": 50,
    }
    timeline = result["tasks"]["timeline"]["records"]
    assert [e["anchor"]["value"] for e in timeline] == ["2026-02-01", "2026-02-03"]
    assert all(e["coding_eligible"] is False for e in timeline)
    assert all(
        term not in json.dumps(result)
        for term in ("Kardioversion", "Metoprolol", "25 mg", "50 mg", "erhöht")
    )


@pytest.mark.parametrize(
    "text",
    [
        "Kardioversion gestern.",
        "Kardioversion am 01.02.26.",
        "Kardioversion am 31.02.2026.",
        "Geburtsdatum 01.02.2026. Kardioversion.",
        "Kardioversion am Morgen.",
        "Am 01.02.2026 und 03.02.2026 erfolgte Kardioversion.",
        "01.02.2026.\nKardioversion.",
    ],
)
def test_unanchored_ambiguous_invalid_or_identifier_dates_cannot_become_event_dates(
    text,
):
    result = analyze(text, [("Kardioversion", "Procedure")])
    assert result["complete"]
    event = result["tasks"]["timeline"]["records"][0]
    assert event["anchor"] is None and event["ordering_group"] == "unanchored"


def test_relative_date_requires_explicit_reference_and_never_uses_dct_fallback():
    text = "Kardioversion gestern.\nMetoprolol begonnen."
    result = analyze(
        text,
        [("Kardioversion", "Procedure"), ("Metoprolol", "Drug")],
        reference_date="2026-09-08",
    )
    timeline = result["tasks"]["timeline"]["records"]
    assert timeline[0]["anchor"]["value"] == "2026-09-07"
    assert timeline[0]["anchor"]["reference_date"] == "2026-09-08"
    assert timeline[1]["anchor"] is None


def test_family_and_negated_mentions_retain_context_without_becoming_coding_facts():
    text = "Familienanamnese: Mutter mit Diabetes am 01.02.2026.\nBefund: Keine Dyspnoe am 02.02.2026."
    result = analyze(text, [("Diabetes", "Disease"), ("Dyspnoe", "Symptom")])
    events = result["tasks"]["timeline"]["records"]
    assert events[0]["context"]["experiencer"] == "family"
    assert events[1]["context"]["negation"] == "negated"
    assert all(e["coding_eligible"] is False for e in events)


def test_negated_action_retains_trigger_context():
    text = "Metoprolol wurde nicht abgesetzt."
    result = analyze(text, [("Metoprolol", "Drug")])
    event = result["tasks"]["events"]["records"][0]
    assert event["trigger"]["value"] == "stopped"
    assert event["trigger_context"]["negation"] == "negated"
    assert event["coding_eligible"] is False


def test_conflicting_numeric_fragments_cannot_be_event_doses():
    text = "Metoprolol 47,5 mg wurde erhöht."
    result = analyze(
        text, [("Metoprolol", "Drug"), ("47", "Strength"), ("5 mg", "Dose")]
    )
    event = result["tasks"]["events"]["records"][0]
    assert event["attributes"] == []
    assert "5 mg" not in json.dumps(result)


def test_elevated_lab_beside_medication_is_not_a_dose_increase():
    text = "Metoprolol 25 mg, CRP erhöht."
    result = analyze(
        text, [("Metoprolol", "Drug"), ("25 mg", "Dose"), ("CRP", "Lab Test")]
    )
    assert all(
        e["type"] != "medication_change" for e in result["tasks"]["events"]["records"]
    )


def test_event_actions_do_not_link_across_sentences_or_competing_heads():
    text = "Metoprolol. Ramipril und Amlodipin wurden begonnen."
    result = analyze(
        text, [("Metoprolol", "Drug"), ("Ramipril", "Drug"), ("Amlodipin", "Drug")]
    )
    events = result["tasks"]["events"]["records"]
    assert len(events) == 3 and all(e["type"] == "clinical_mention" for e in events)


def test_english_ambiguous_slash_dates_remain_unanchored():
    text = "Cardioversion on 01/02/2026."
    result = analyze(text, [("Cardioversion", "Procedure")], language="en")
    assert result["tasks"]["timeline"]["records"][0]["anchor"] is None


@pytest.mark.parametrize(
    "reference", ["20260908", "2026-02-29", "private source marker", True]
)
def test_invalid_reference_dates_fail_without_source_echo(reference):
    with pytest.raises(ClinicalAnalysisError, match="invalid_clinical_reference_date"):
        analyze("Diabetes", [("Diabetes", "Disease")], reference_date=reference)


def test_reference_date_cannot_be_silently_ignored_by_non_timeline_tasks():
    with pytest.raises(ClinicalAnalysisError):
        analyze(
            "Diabetes",
            [("Diabetes", "Disease")],
            tasks=["events"],
            reference_date="2026-09-08",
        )


def test_temporal_tasks_require_full_coverage_and_supported_language():
    for kwargs in (dict(entity_coverage_complete=False), dict(language="fr")):
        result = analyze(
            "Diabetes",
            [("Diabetes", "Disease")],
            tasks=["sections", "events", "timeline"],
            **kwargs,
        )
        assert result["status"] == "partial"
        assert result["tasks"]["sections"]["complete"]
        assert all(
            not result["tasks"][t]["complete"] and result["tasks"][t]["records"] == []
            for t in ("events", "timeline")
        )


def test_scope_trigger_limit_fails_only_dependent_tasks(monkeypatch):
    monkeypatch.setattr("openmed.clinical.temporal_analysis.MAX_SCOPE_TRIGGERS", 1)
    result = analyze(
        "Metoprolol begonnen und abgesetzt.",
        [("Metoprolol", "Drug")],
        tasks=["entities", "events", "timeline"],
    )
    assert result["tasks"]["entities"]["complete"]
    for task in ("events", "timeline"):
        assert result["tasks"][task]["error"] == "clinical_event_trigger_limit"
        assert not result["tasks"][task]["records"]
