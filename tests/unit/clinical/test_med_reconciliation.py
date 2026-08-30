"""Golden and safety tests for document-local medication reconciliation."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from openmed.clinical import (
    MEDICATION_RECONCILIATION_ADVISORY,
    CoreferenceChain,
    MedicationMention,
    normalize_medication_route,
    reconcile_medications,
)
from openmed.core.schemas import OpenMedSpan, hmac_text_hash

FIXTURE_PATH = (
    Path(__file__).parents[2] / "fixtures/clinical/medication_reconciliation_gold.json"
)


def _load_cases() -> list[dict[str, object]]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert payload["synthetic"] is True
    return payload["cases"]


@pytest.mark.parametrize("case", _load_cases(), ids=lambda case: case["id"])
def test_golden_multi_mention_fixture_reconciles_current_state(case):
    result = reconcile_medications(case["mentions"], document_id="synthetic-document")

    expected = case["expected"]
    matching = next(
        medication
        for medication in result
        if medication.ingredient == expected["ingredient"]
    )
    assert matching.current_status == expected["current_status"]
    assert matching.current_dose == expected["current_dose"]
    assert matching.current_route == expected["current_route"]
    assert [entry.status for entry in matching.history] == expected["history_statuses"]
    assert [list(offset) for offset in matching.source_offsets] == expected[
        "source_offsets"
    ]
    if "system" in expected:
        assert matching.system == expected["system"]
        assert matching.code == expected["code"]


def test_timestamp_order_is_independent_of_input_order():
    result = reconcile_medications(
        [
            {
                "text": "first synthetic mention",
                "ingredient": "synthetic-drug",
                "status": "stopped",
                "effective_time": "2026-03-01",
                "offset": (50, 63),
            },
            {
                "text": "second synthetic mention",
                "ingredient": "synthetic-drug",
                "status": "started",
                "effective_time": "2026-01-01",
                "offset": (5, 18),
            },
            {
                "ingredient": "synthetic-drug",
                "status": "held",
                "effective_time": "2026-02-01",
                "offset": (28, 41),
            },
        ]
    )

    assert [entry.status for entry in result[0].history] == [
        "started",
        "held",
        "stopped",
    ]
    assert result[0].current_status == "stopped"


def test_coreference_identity_merges_different_surfaces():
    result = reconcile_medications(
        [
            MedicationMention(
                text="metformin",
                ingredient="metformin",
                coref_entity_id="synthetic-medication-chain",
                dose="500 mg",
                offset=(0, 9),
            ),
            {
                "text": "the medication",
                "coref_entity_id": "synthetic-medication-chain",
                "dose": "500 mg",
                "offset": (25, 39),
            },
        ]
    )

    assert len(result) == 1
    assert result[0].ingredient == "metformin"
    assert result[0].coref_entity_id == "synthetic-medication-chain"
    assert result[0].mention_count == 2
    assert result[0].source_offsets == ((0, 9), (25, 39))


def test_span_coreference_chain_supplies_identity_for_anaphoric_mention():
    first = OpenMedSpan(
        doc_id="synthetic-document",
        start=0,
        end=9,
        text_hash=hmac_text_hash("metformin", "synthetic-secret"),
        entity_type="MEDICATION",
        canonical_label="MEDICATION",
    )
    second = OpenMedSpan(
        doc_id="synthetic-document",
        start=20,
        end=22,
        text_hash=hmac_text_hash("it", "synthetic-secret"),
        entity_type="MEDICATION",
        canonical_label="MEDICATION",
    )
    chain = CoreferenceChain(
        chain_id="span-medication-chain",
        members=(first, second),
        representative=first,
        confidence=1.0,
    )

    result = reconcile_medications(
        [
            {"text": "metformin", "offset": (0, 9)},
            {"text": "it", "offset": (20, 22)},
        ],
        document_id="synthetic-document",
        coreference_chains=(chain,),
    )

    assert len(result) == 1
    assert result[0].ingredient == "metformin"
    assert result[0].coref_entity_id == "span-medication-chain"
    assert result[0].source_offsets == ((0, 9), (20, 22))


def test_local_ingredient_grounder_can_supply_normalized_identity():
    result = reconcile_medications(
        [{"text": "synthetic brand", "dose": "5 mg", "offset": (0, 15)}],
        ingredient_grounder=lambda _surface: {"ingredient": "Synthetic Drug"},
    )

    assert len(result) == 1
    assert result[0].ingredient == "synthetic drug"


def test_grounding_candidates_join_to_one_ingredient_without_network():
    result = reconcile_medications(
        [
            {
                "text": "synthetic brand",
                "candidates": [
                    {
                        "system": "RXNORM",
                        "code": "111",
                        "display": "Metformin 500 mg Tablet",
                        "score": 0.95,
                    }
                ],
                "dose": "500 mg",
                "route": "PO",
                "offset": (1, 16),
            },
            {
                "ingredient": "metformin",
                "system": "rxnorm",
                "code": "111",
                "dose": "500 mg",
                "route": "oral",
                "offset": (30, 39),
            },
        ],
        ingredient_grounder=lambda _surface: (_ for _ in ()).throw(
            AssertionError("grounder should not be needed for coded mentions")
        ),
    )

    assert len(result) == 1
    assert result[0].ingredient == "metformin"
    assert result[0].system == "RXNORM"
    assert result[0].code == "111"
    assert result[0].current_route == "oral"


def test_same_time_dose_and_route_conflicts_are_not_silently_merged():
    result = reconcile_medications(
        [
            {
                "text": "first dose mention",
                "ingredient": "synthetic-drug",
                "dose": "5 mg",
                "route": "oral",
                "effective_time": "2026-04-01",
                "offset": (4, 17),
            },
            {
                "text": "second dose mention",
                "ingredient": "synthetic-drug",
                "dose": "10 mg",
                "route": "intravenous",
                "effective_time": "2026-04-01",
                "offset": (40, 53),
            },
        ]
    )

    medication = result[0]
    assert medication.current_dose is None
    assert medication.current_route is None
    assert {conflict.field for conflict in medication.conflicts} == {"dose", "route"}
    assert medication.conflicts[0].source_offsets == ((4, 17), (40, 53))
    encoded = json.dumps(medication.to_dict(), sort_keys=True)
    assert "first dose mention" not in encoded
    assert "second dose mention" not in encoded
    assert "text" not in medication.to_dict()


def test_section_precedence_resolves_untimestamped_attribute_values():
    result = reconcile_medications(
        [
            {
                "ingredient": "synthetic-drug",
                "dose": "5 mg",
                "section": "history",
                "offset": (2, 15),
            },
            {
                "ingredient": "synthetic-drug",
                "dose": "10 mg",
                "section": "plan",
                "offset": (30, 43),
            },
        ]
    )

    assert result[0].current_dose == "10 mg"
    assert result[0].conflicts == ()


def test_offsets_and_normalized_values_are_serialized_without_source_text():
    result = reconcile_medications(
        [
            {
                "text": "Patient-specific synthetic medication",
                "ingredient": "Synthetic Medication",
                "dose": " 500  MG ",
                "route": "P.O.",
                "offset": (10, 49),
            }
        ]
    )

    payload = result[0].to_dict()
    assert payload["ingredient"] == "synthetic medication"
    assert payload["current_dose"] == "500 mg"
    assert payload["current_route"] == "oral"
    assert payload["source_offsets"] == [[10, 49]]
    assert "Patient-specific synthetic medication" not in json.dumps(payload)
    assert "heuristic" not in MEDICATION_RECONCILIATION_ADVISORY


def test_reconciliation_is_offline_and_rejects_mixed_documents(
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    with pytest.raises(ValueError, match="one document"):
        reconcile_medications(
            [
                {"ingredient": "drug-a", "document_id": "doc-a"},
                {"ingredient": "drug-a", "document_id": "doc-b"},
            ]
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [("PO", "oral"), ("IV", "intravenous"), ("by mouth", "oral")],
)
def test_route_normalization(value, expected):
    assert normalize_medication_route(value) == expected


def test_invalid_absolute_timestamp_is_rejected():
    with pytest.raises(ValueError, match="absolute normalized"):
        reconcile_medications(
            [{"ingredient": "synthetic-drug", "effective_time": "yesterday"}]
        )
