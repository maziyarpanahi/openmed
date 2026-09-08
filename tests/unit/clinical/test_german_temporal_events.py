"""Explicit German cues, original offsets and no implicit temporal anchors."""

import pytest

from openmed.clinical.events import (
    extract_lab_trend_events,
    extract_medication_change_events,
)
from openmed.clinical.temporal_german import GERMAN_TIMEX_RE
from openmed.clinical.temporal_normalizer import normalize_temporal


def mention(text, value, label):
    start = text.index(value)
    return dict(start=start, end=start + len(value), label=label, score=0.99)


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("01.02.2026", "2026-02-01"),
        ("01/02/2026", "2026-02-01"),
        ("2026-02-01", "2026-02-01"),
        ("3. März 2026", "2026-03-03"),
        ("heute", "2026-03-31"),
        ("gestern", "2026-03-30"),
        ("übermorgen", "2026-04-02"),
        ("vor drei Tagen", "2026-03-28"),
        ("in zwei Wochen", "2026-04-14"),
        ("vor einem Monat", "2026-02-28"),
        ("seit 2 Tagen", "2026-03-29/2026-03-31"),
    ],
)
def test_german_dates_and_relative_spans_preserve_offsets(phrase, expected):
    text = "🫀 Befund: " + phrase + "."
    match = next(GERMAN_TIMEX_RE.finditer(text))
    assert text[match.start() : match.end()] == phrase
    result = normalize_temporal(
        text, [(match.start(), match.end())], "2026-03-31", language="de"
    )[0]
    assert result.value == expected and result.text == phrase
    assert result.span == (match.start(), match.end())


@pytest.mark.parametrize(
    "phrase,flag",
    [
        ("01.02.26", "ambiguous"),
        ("31.02.2026", "invalid"),
        ("2026-02-30", "invalid"),
        ("vor 3 Tagen", "unanchored"),
        ("gestern", "unanchored"),
        ("last week", "unsupported"),
    ],
)
def test_invalid_ambiguous_unanchored_or_wrong_language_never_invent_a_date(
    phrase, flag
):
    result = normalize_temporal(phrase, [(0, len(phrase))], None, language="de")[0]
    assert result.value is None and flag in result.granularity_flags


def test_language_changes_numeric_date_contract_without_changing_english_default():
    phrase = "01/02/2026"
    assert normalize_temporal(phrase, [(0, len(phrase))], None)[0].value is None
    assert (
        normalize_temporal(phrase, [(0, len(phrase))], None, language="de")[0].value
        == "2026-02-01"
    )
    with pytest.raises(ValueError):
        normalize_temporal(phrase, [(0, len(phrase))], None, language="fr")


@pytest.mark.parametrize(
    "cue,action",
    [
        ("begonnen", "started"),
        ("wieder begonnen", "restarted"),
        ("abgesetzt", "stopped"),
        ("pausiert", "held"),
        ("reduziert", "decreased"),
    ],
)
def test_german_medication_action_is_one_original_trigger(cue, action):
    text = f"Metoprolol wurde {cue}."
    frames = extract_medication_change_events(
        text, [mention(text, "Metoprolol", "Drug")], language="de"
    )
    assert len(frames) == 1
    trigger = frames[0].role_slots("action")[0]
    assert trigger.value == action and text[trigger.start : trigger.end] == cue


def test_german_increase_keeps_explicit_old_and_new_dose_roles():
    text = "Metoprolol wurde von 25 mg auf 50 mg erhöht."
    frames = extract_medication_change_events(
        text,
        [
            mention(text, "Metoprolol", "Drug"),
            mention(text, "25 mg", "Dose"),
            mention(text, "50 mg", "Dose"),
        ],
        language="de",
    )
    assert len(frames) == 1
    assert frames[0].role_slots("old_dose")[0].value == "25 mg"
    assert frames[0].role_slots("new_dose")[0].value == "50 mg"


def test_german_lab_trend_has_analyte_and_trigger_evidence():
    text = "CRP ist gestiegen."
    frames = extract_lab_trend_events(
        text, [mention(text, "CRP", "lab_name")], language="de"
    )
    assert len(frames) == 1
    assert frames[0].role_slots("direction")[0].value == "rising"
    assert frames[0].role_slots("analyte")[0].value == "CRP"


@pytest.mark.parametrize(
    "language,text,surface,expected_role",
    [
        ("en", "Metoprolol increased from 25 mg to 50 mg.", "25 mg", "old_dose"),
        ("de", "Metoprolol von 25 mg auf 50 mg erhöht.", "50 mg", "new_dose"),
        ("en", "Metoprolol was reduced to 25 mg.", "25 mg", "new_dose"),
        ("de", "Metoprolol von 50 mg reduziert.", "50 mg", "old_dose"),
    ],
)
def test_explicit_dose_role_cannot_fill_its_missing_counterpart(
    language, text, surface, expected_role
):
    frame = extract_medication_change_events(
        text,
        [mention(text, "Metoprolol", "Drug"), mention(text, surface, "Dose")],
        language=language,
    )[0]
    assert frame.role_slots(expected_role)[0].value == surface
    other_role = "old_dose" if expected_role == "new_dose" else "new_dose"
    assert not frame.role_slots(other_role)


def test_continuation_and_other_language_are_not_silently_restarts():
    for cue in ("fortgesetzt", "started"):
        text = f"Metoprolol {cue}."
        assert not extract_medication_change_events(
            text, [mention(text, "Metoprolol", "Drug")], language="de"
        )
    with pytest.raises(ValueError):
        extract_medication_change_events("", language="fr")


def test_trigger_limit_rejects_before_role_graph_construction(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail(
            "role graph should not be constructed after exceeding the trigger limit"
        )

    monkeypatch.setattr(
        "openmed.clinical.events.extract._build_event_frame", unexpected
    )
    with pytest.raises(ValueError, match="clinical_event_trigger_limit"):
        extract_medication_change_events(
            "begonnen abgesetzt", language="de", max_events=1
        )
    with pytest.raises(ValueError, match="clinical_event_trigger_limit"):
        extract_lab_trend_events("gestiegen gesunken", language="de", max_events=1)
