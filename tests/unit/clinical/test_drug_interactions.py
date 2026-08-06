"""Tests for offline drug-drug interaction flagging with synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.clinical import find_interactions
from openmed.clinical.decision_support import (
    CLINICAL_DECISION_SUPPORT_DISCLAIMER,
    GuardedSuggestion,
)
from openmed.clinical.drug_interactions import (
    DRUG_INTERACTION_ADVISORY,
    INTERACTION_DATA_NOTICE,
    InteractionTableError,
)
from openmed.clinical.grounding import Candidate, GroundedSpan

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "synthetic_ddi_table.json"
)


def _normalized_medications() -> list[dict[str, object]]:
    return [
        {
            "rxcui": "900000000001",
            "name": "Synthetic Drug Alpha",
            "start": 0,
            "end": 20,
        },
        {
            "rxcui": "900000000002",
            "name": "Synthetic Drug Beta",
            "start": 25,
            "end": 44,
        },
    ]


def test_synthetic_fixture_is_explicitly_non_clinical() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert payload["metadata"]["synthetic"] is True
    assert payload["metadata"]["clinical_use"] is False
    assert "Not valid for patient care" in payload["metadata"]["notice"]
    assert "license" in INTERACTION_DATA_NOTICE


def test_matching_pair_returns_one_guarded_flag_with_required_fields() -> None:
    flags = find_interactions(_normalized_medications(), FIXTURE)

    assert len(flags) == 1
    flag = flags[0]
    assert isinstance(flag, GuardedSuggestion)
    assert flag.disclaimer == CLINICAL_DECISION_SUPPORT_DISCLAIMER
    assert flag.requires_clinician_review is True
    assert flag.autonomous_decision is False
    assert [(span.start, span.end) for span in flag.source_spans] == [
        (0, 20),
        (25, 44),
    ]

    suggestion = flag.suggestion
    assert suggestion["kind"] == "drug_drug_interaction"
    assert suggestion["severity"] == "synthetic-review-priority"
    assert suggestion["source_citation"]
    assert suggestion["advisory"] == DRUG_INTERACTION_ADVISORY
    assert flag.provenance["interaction_data"] == "caller_supplied"


@pytest.mark.parametrize("empty_table", [None, {}, {"interactions": []}])
def test_empty_user_supplied_table_returns_no_interaction_flags(empty_table) -> None:
    assert find_interactions(_normalized_medications(), empty_table) == []


def test_unnormalizable_medication_surfaces_not_checked_note() -> None:
    suggestions = find_interactions(
        [
            {
                "name": "Synthetic Unknown",
                "start": 0,
                "end": 17,
            },
            _normalized_medications()[0],
        ],
        FIXTURE,
    )

    assert len(suggestions) == 1
    note = suggestions[0]
    assert note.suggestion["kind"] == "normalization_note"
    assert note.suggestion["status"] == "not_checked"
    assert "not checked" in note.suggestion["note"]
    assert "non-interacting" not in note.suggestion["note"]
    assert note.requires_clinician_review is True


def test_offline_mode_never_invokes_optional_rxnorm_lookup() -> None:
    calls: list[str] = []

    def lookup(name: str) -> str:
        calls.append(name)
        return "900000000002"

    suggestions = find_interactions(
        [{"name": "Synthetic Unknown", "start": 0, "end": 17}],
        FIXTURE,
        offline=True,
        rxnorm_lookup=lookup,
    )

    assert calls == []
    assert suggestions[0].suggestion["status"] == "not_checked"


def test_opted_in_lookup_only_normalizes_and_local_table_decides_verdict() -> None:
    calls: list[str] = []

    def lookup(name: str) -> str:
        calls.append(name)
        return "900000000002"

    suggestions = find_interactions(
        [
            _normalized_medications()[0],
            {"name": "Synthetic Drug Beta", "start": 25, "end": 44},
        ],
        FIXTURE,
        offline=False,
        rxnorm_lookup=lookup,
    )

    assert calls == ["Synthetic Drug Beta"]
    assert [item.suggestion["kind"] for item in suggestions] == [
        "drug_drug_interaction"
    ]
    assert (
        find_interactions(
            [
                _normalized_medications()[0],
                {"name": "Synthetic Drug Beta", "start": 25, "end": 44},
            ],
            {},
            offline=False,
            rxnorm_lookup=lookup,
        )
        == []
    )


def test_existing_grounded_span_rxnorm_output_is_accepted() -> None:
    grounded = [
        GroundedSpan(
            text="Synthetic Drug Alpha",
            start=0,
            end=20,
            candidates=(
                Candidate(
                    system="RXNORM",
                    code="900000000001",
                    display="Synthetic Drug Alpha",
                    score=1.0,
                ),
            ),
        ),
        GroundedSpan(
            text="Synthetic Drug Beta",
            start=25,
            end=44,
            candidates=(
                Candidate(
                    system="RXNORM",
                    code="900000000002",
                    display="Synthetic Drug Beta",
                    score=1.0,
                ),
            ),
        ),
    ]

    assert len(find_interactions(grounded, FIXTURE)) == 1


@pytest.mark.parametrize(
    "system",
    [
        "RXNORM",
        "http://www.nlm.nih.gov/research/umls/rxnorm",
        "http://purl.bioontology.org/ontology/RXNORM",
    ],
)
def test_rxnorm_system_code_records_are_accepted(system: str) -> None:
    medications = [
        {
            "name": "Synthetic Drug Alpha",
            "codes": [{"system": system, "code": "900000000001"}],
            "start": 0,
            "end": 20,
        },
        {
            "name": "Synthetic Drug Beta",
            "system": system,
            "code": "900000000002",
            "start": 25,
            "end": 44,
        },
    ]

    assert len(find_interactions(medications, FIXTURE)) == 1


def test_bare_rxcuis_are_order_independent_and_duplicate_safe() -> None:
    forward = find_interactions(["900000000001", "900000000002"], FIXTURE)
    reverse = find_interactions(["900000000002", "900000000001"], FIXTURE)
    duplicated = find_interactions(
        ["900000000001", "900000000001", "900000000002"], FIXTURE
    )

    assert len(forward) == len(reverse) == len(duplicated) == 1
    assert forward[0].suggestion["severity"] == reverse[0].suggestion["severity"]
    assert [
        medication["rxcui"] for medication in reverse[0].suggestion["medications"]
    ] == ["900000000001", "900000000002"]


def test_reversed_duplicate_table_rows_are_one_unordered_interaction() -> None:
    row = {
        "severity": "synthetic-review-priority",
        "description": "Synthetic pair Alpha-Beta is listed for test-only review.",
        "source_citation": "Synthetic source, schema version 1",
    }
    table = {
        "interactions": [
            {**row, "rxcui_1": "900000000001", "rxcui_2": "900000000002"},
            {**row, "rxcui_1": "900000000002", "rxcui_2": "900000000001"},
        ]
    }

    assert len(find_interactions(_normalized_medications(), table)) == 1


def test_flags_are_review_advisories_and_never_auto_action_verdicts() -> None:
    serialized = json.dumps(
        [
            flag.to_dict()
            for flag in find_interactions(_normalized_medications(), FIXTURE)
        ]
    ).casefold()

    assert "clinician_review_required" in serialized
    assert "do not prescribe" not in serialized
    assert 'autonomous_decision": true' not in serialized


def test_interaction_table_must_be_local_and_cited() -> None:
    with pytest.raises(InteractionTableError, match="local file path"):
        find_interactions(_normalized_medications(), "https://example.test/ddi.json")

    with pytest.raises(InteractionTableError, match="source citation"):
        find_interactions(
            _normalized_medications(),
            {
                "interactions": [
                    {
                        "rxcui_1": "900000000001",
                        "rxcui_2": "900000000002",
                        "severity": "synthetic",
                        "description": "Synthetic test row.",
                    }
                ]
            },
        )

    with pytest.raises(InteractionTableError, match="schema_version"):
        find_interactions(
            _normalized_medications(),
            {"schema_version": 2, "interactions": []},
        )
