from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from jsonschema import Draft202012Validator

from openmed.clinical.journey_contracts import EvidenceLocator
from openmed.structured.facts import (
    FACT_PROFILE_SPECS,
    ClinicalFactNormalizer,
    ComponentOutputEnvelope,
    FactFragment,
    FactNormalizationError,
    FactNormalizationRequest,
    MappingFactAdapter,
    RelationParticipant,
    load_clinical_fact_normalization_schema,
)
from openmed.structured.store import StoreState

SUBJECT_ID = "subject_aaaaaaaaaaaaaaaa"
ENCOUNTER_ID = "encounter_aaaaaaaaaaaaaaaa"
EVIDENCE_ID = "evidence_aaaaaaaaaaaaaaaa"
EVIDENCE_TWO_ID = "evidence_bbbbbbbbbbbbbbbb"
PARENT_FACT_ID = "fact_aaaaaaaaaaaaaaaa"
CANARY = "SYNTHETIC-PRIVATE-COMPONENT-CANARY"
GOLDEN_PATH = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "structured"
    / "clinical_fact_profiles_golden.json"
)


def _evidence(
    evidence_id: str = EVIDENCE_ID, *, start: int = 4, end: int = 18
) -> EvidenceLocator:
    return EvidenceLocator(
        locator_id=evidence_id,
        artifact_id="artifact_aaaaaaaaaaaaaaaa",
        location_type="text_span",
        location={"start": start, "end": end},
    )


def _output(
    payload: object | None = None,
    *,
    component: str = "clinical.extractor",
    output_schema: str = "component.output",
) -> ComponentOutputEnvelope:
    return ComponentOutputEnvelope(
        component=component,
        component_version="1.0.0",
        output_schema=output_schema,
        output_schema_version="1.0.0",
        output=payload if payload is not None else {"candidate": "synthetic"},
    )


def _condition_fields() -> dict[str, object]:
    return {
        "assertion": "affirmed",
        "certainty": "certain",
        "experiencer": "patient",
        "status": "active",
        "value": {"code": "synthetic-condition"},
        "confidence": 0.92,
    }


def _fragment(
    *,
    fields: dict[str, object] | None = None,
    component: str = "clinical.extractor",
    kind: str = "extraction",
    evidence_ids: tuple[str, ...] = (EVIDENCE_ID,),
    field_states: dict[str, str] | None = None,
    payload: object | None = None,
) -> FactFragment:
    return FactFragment(
        kind=kind,
        component_output=_output(payload, component=component),
        fields=fields if fields is not None else _condition_fields(),
        evidence_ids=evidence_ids,
        field_states=field_states or {},
    )


def _request(
    *fragments: FactFragment,
    profile: str = "condition",
    parent_fact_ids: tuple[str, ...] = (),
) -> FactNormalizationRequest:
    return FactNormalizationRequest(
        subject_id=SUBJECT_ID,
        encounter_id=ENCOUNTER_ID,
        profile=profile,
        fragments=fragments or (_fragment(),),
        parent_fact_ids=parent_fact_ids,
    )


def _normalize(request: FactNormalizationRequest, *, second: bool = False):
    evidence = {EVIDENCE_ID: _evidence()}
    if second:
        evidence[EVIDENCE_TWO_ID] = _evidence(EVIDENCE_TWO_ID, start=20, end=32)
    return ClinicalFactNormalizer().normalize(request, evidence_by_id=evidence)


def test_condition_fact_preserves_negation_uncertainty_and_family_history() -> None:
    fields = _condition_fields() | {
        "assertion": "negated",
        "certainty": "uncertain",
        "experiencer": "family",
        "status": "history",
        "effective_time": {"precision": "year", "start": "2024"},
    }

    result = _normalize(_request(_fragment(fields=fields)))

    assert result.ok
    assert result.value is not None
    fact = result.value.fact
    assert fact.attributes["assertion"] == "negated"
    assert fact.attributes["certainty"] == "uncertain"
    assert fact.attributes["experiencer"] == "family"
    assert fact.effective_time == {"precision": "year", "start": "2024"}


def test_laboratory_fact_preserves_range_unit_and_partial_date() -> None:
    fields = {
        "value": {"numeric": 7.2, "reference_range": {"low": 4.0, "high": 8.0}},
        "unit": "mmol/L",
        "status": "final",
        "effective_time": {"precision": "month", "start": "2026-03"},
    }

    result = _normalize(_request(_fragment(fields=fields), profile="laboratory"))

    assert result.ok
    assert result.value is not None
    assert result.value.fact.value["reference_range"] == {"high": 8.0, "low": 4.0}
    assert result.value.fact.unit == "mmol/L"
    assert result.value.fact.effective_time["start"] == "2026-03"


@pytest.mark.parametrize(
    ("profile", "status", "fields"),
    [
        (
            "medication",
            "active",
            {
                "assertion": "affirmed",
                "experiencer": "patient",
                "value": {"code": "synthetic-medication", "dose": 5},
            },
        ),
        (
            "procedure",
            "completed",
            {
                "assertion": "affirmed",
                "experiencer": "patient",
                "effective_time": {"precision": "day", "start": "2026-02-03"},
                "value": {"code": "synthetic-procedure"},
            },
        ),
        (
            "observation",
            "final",
            {
                "assertion": "affirmed",
                "experiencer": "patient",
                "effective_time": {
                    "precision": "second",
                    "start": "2026-02-03T10:00:00Z",
                },
                "value": {"code": "synthetic-observation"},
            },
        ),
        (
            "social_determinant",
            "historical",
            {
                "assertion": "affirmed",
                "certainty": "probable",
                "experiencer": "patient",
                "effective_time": {"precision": "year", "start": "2022"},
                "value": {"code": "synthetic-housing-state"},
            },
        ),
    ],
)
def test_common_fact_profiles_preserve_declared_fields(
    profile: str, status: str, fields: dict[str, object]
) -> None:
    result = _normalize(
        _request(_fragment(fields=fields | {"status": status}), profile=profile)
    )

    assert result.ok
    assert result.value is not None
    assert result.value.fact.fact_type == profile
    assert result.value.fact.status == status


def test_missing_required_value_is_unknown_not_invented() -> None:
    fields = _condition_fields()
    fields.pop("value")

    result = _normalize(_request(_fragment(fields=fields)))

    assert result.state is StoreState.UNKNOWN
    assert result.code == "fact_unknown"
    assert result.value is not None
    assert result.value.fact.value is None
    assert result.value.field_states["value"] == "unknown"


def test_explicit_partial_and_unsupported_states_remain_typed() -> None:
    partial = _normalize(
        _request(_fragment(field_states={"effective_time": "partial"}))
    )
    unsupported = _normalize(_request(_fragment(field_states={"value": "unsupported"})))

    assert partial.state is StoreState.PARTIAL
    assert partial.code == "fact_partial"
    assert unsupported.state is StoreState.UNSUPPORTED
    assert unsupported.code == "fact_value_unsupported"


def test_conflicting_component_values_do_not_create_a_fact() -> None:
    first = _fragment()
    second = _fragment(
        fields=_condition_fields() | {"status": "resolved"},
        component="clinical.status",
        kind="status",
    )

    result = _normalize(_request(first, second))

    assert result.state is StoreState.CONFLICT
    assert result.code == "component_field_conflict"
    assert result.value is None


def test_explicit_component_conflict_state_is_not_serialized_as_success() -> None:
    result = _normalize(_request(_fragment(field_states={"status": "conflict"})))

    assert result.state is StoreState.CONFLICT
    assert result.code == "component_state_conflict"


def test_every_fact_requires_existing_evidence() -> None:
    no_evidence = _normalize(_request(_fragment(evidence_ids=())))
    missing = ClinicalFactNormalizer().normalize(
        _request(_fragment()), evidence_by_id={}
    )

    assert no_evidence.state is StoreState.CONFLICT
    assert no_evidence.code == "evidence_required"
    assert missing.state is StoreState.CONFLICT
    assert missing.code == "evidence_not_found"


def test_component_outputs_are_preserved_but_default_metadata_is_value_free() -> None:
    payload = {"candidate": "synthetic", "source_text": CANARY}
    result = _normalize(_request(_fragment(payload=payload)))

    assert result.value is not None
    public = json.dumps(result.value.to_dict(), sort_keys=True)
    protected = json.dumps(result.value.protected_component_outputs(), sort_keys=True)
    assert CANARY not in public
    assert "source_text" not in public
    assert CANARY in protected


def test_mapping_adapter_uses_only_explicit_paths_and_does_not_guess() -> None:
    adapter = MappingFactAdapter(
        component="clinical.mapping",
        component_version="1.0.0",
        output_schema="mapping.output",
        output_schema_version="1.0.0",
        kind="extraction",
        field_paths={
            "value": "result.concept",
            "status": "result.state",
        },
    )
    output = {
        "result": {"concept": {"code": "synthetic"}},
        "unmapped_status": "active",
    }

    result = adapter.adapt(output, evidence_ids=(EVIDENCE_ID,))

    assert result.ok
    assert result.value is not None
    assert result.value.fields == {"value": {"code": "synthetic"}}
    assert result.value.component_output.output == output


def test_mapping_adapter_returns_typed_malformed_outcomes() -> None:
    adapter = MappingFactAdapter(
        component="clinical.mapping",
        component_version="1.0.0",
        output_schema="mapping.output",
        output_schema_version="1.0.0",
        kind="value",
        field_paths={"value": "value"},
    )

    non_mapping = adapter.adapt("bad", evidence_ids=(EVIDENCE_ID,))  # type: ignore[arg-type]
    nonfinite = adapter.adapt({"value": float("nan")}, evidence_ids=(EVIDENCE_ID,))

    assert non_mapping.state is StoreState.UNSUPPORTED
    assert non_mapping.code == "component_not_mapping"
    assert nonfinite.state is StoreState.FAILURE
    assert nonfinite.code == "component_contract_invalid"


@pytest.mark.parametrize(
    "effective_time",
    [
        {"precision": "day", "start": "2026-02-30"},
        {"precision": "month", "start": "2026"},
        {"precision": "second", "start": "2026-01-01T00:00:00"},
        {"precision": "day", "start": "2026-02-03", "end": "2025-01-01"},
    ],
)
def test_invalid_temporal_values_are_typed_unsupported(
    effective_time: dict[str, str],
) -> None:
    fields = _condition_fields() | {"effective_time": effective_time}

    result = _normalize(_request(_fragment(fields=fields)))

    assert result.state is StoreState.UNSUPPORTED
    assert result.code == "effective_time_unsupported"


def test_unsupported_status_and_axes_are_not_guessed() -> None:
    status = _normalize(
        _request(_fragment(fields=_condition_fields() | {"status": "other"}))
    )
    assertion = _normalize(
        _request(_fragment(fields=_condition_fields() | {"assertion": "maybe"}))
    )

    assert status.state is StoreState.UNSUPPORTED
    assert status.code == "fact_status_unsupported"
    assert assertion.state is StoreState.UNSUPPORTED
    assert assertion.code == "fact_axis_unsupported"


def test_cross_sentence_relation_participants_must_be_in_provenance() -> None:
    participant = RelationParticipant(
        role="indication",
        target_id=PARENT_FACT_ID,
        target_type="fact",
    )
    fields = _condition_fields() | {"relation_participants": [participant.to_dict()]}
    accepted = _normalize(
        _request(
            _fragment(fields=fields),
            parent_fact_ids=(PARENT_FACT_ID,),
        )
    )
    rejected = _normalize(_request(_fragment(fields=fields)))

    assert accepted.ok
    assert accepted.value is not None
    assert accepted.value.fact.attributes["relation_participants"][0]["role"] == (
        "indication"
    )
    assert rejected.state is StoreState.CONFLICT
    assert rejected.code == "relation_provenance_invalid"


def test_amendment_preserves_parent_and_creates_new_derivation() -> None:
    original = _normalize(_request(_fragment()))
    amended = _normalize(
        _request(
            _fragment(fields=_condition_fields() | {"status": "resolved"}),
            parent_fact_ids=(PARENT_FACT_ID,),
        )
    )

    assert original.ok and amended.ok
    assert original.value is not None and amended.value is not None
    assert amended.value.fact.parent_fact_ids == (PARENT_FACT_ID,)
    assert original.value.fact.fact_id != amended.value.fact.fact_id
    assert original.value.fact.derivation_hash != amended.value.fact.derivation_hash


def test_multilingual_component_output_keeps_exact_evidence_offsets() -> None:
    payload = {"mention": "合成症状", "start": 4, "end": 8}
    result = _normalize(_request(_fragment(payload=payload)))

    assert result.ok
    assert result.value is not None
    assert result.value.fact.evidence_ids == (EVIDENCE_ID,)
    assert result.value.protected_component_outputs()[0]["output"] == payload


def test_fragment_order_does_not_change_fact_or_derivation() -> None:
    extraction = _fragment(
        fields={
            "status": "active",
            "value": {"code": "synthetic-condition"},
        }
    )
    context = _fragment(
        fields={
            "assertion": "affirmed",
            "certainty": "certain",
            "experiencer": "patient",
        },
        component="clinical.context",
        kind="assertion",
        payload={"assertion": "affirmed"},
    )

    first = _normalize(_request(extraction, context))
    second = _normalize(_request(context, extraction))

    assert first.value is not None and second.value is not None
    assert first.value.fact == second.value.fact
    assert first.value.to_dict() == second.value.to_dict()


def test_changed_component_output_creates_new_fact_identity() -> None:
    first = _normalize(_request(_fragment(payload={"revision": 1})))
    second = _normalize(_request(_fragment(payload={"revision": 2})))

    assert first.value is not None and second.value is not None
    assert first.value.fact.fact_id != second.value.fact.fact_id


def test_public_schema_validates_normalized_fact() -> None:
    result = _normalize(_request(_fragment()))
    assert result.value is not None
    schema = load_clinical_fact_normalization_schema()

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(result.value.to_dict())


def test_profiles_are_explicit_and_complete() -> None:
    assert set(FACT_PROFILE_SPECS) == {
        "condition",
        "laboratory",
        "medication",
        "observation",
        "procedure",
        "social_determinant",
    }
    assert "unit" in FACT_PROFILE_SPECS["laboratory"].required_fields


def test_duplicate_fragments_are_rejected_without_echoing_output() -> None:
    fragment = _fragment(payload={"private": CANARY})

    with pytest.raises(FactNormalizationError) as exc_info:
        _request(fragment, fragment)

    assert CANARY not in str(exc_info.value)


def test_attribute_metadata_cannot_override_normalizer_provenance() -> None:
    fields = _condition_fields() | {"attributes": {"normalizer_version": "attacker"}}

    result = _normalize(_request(_fragment(fields=fields)))

    assert result.state is StoreState.UNSUPPORTED
    assert result.code == "fact_value_unsupported"


@given(confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_finite_confidence_round_trips_without_state_change(confidence: float) -> None:
    result = _normalize(
        _request(_fragment(fields=_condition_fields() | {"confidence": confidence}))
    )

    assert result.ok
    assert result.value is not None
    assert result.value.fact.confidence == confidence


def test_unknown_component_profile_is_typed_unsupported() -> None:
    result = _normalize(_request(_fragment(), profile="unknown_profile"))

    assert result.state is StoreState.UNSUPPORTED
    assert result.code == "fact_profile_unsupported"


def test_two_evidence_locators_are_sorted_and_bound_to_derivation() -> None:
    fragment = _fragment(evidence_ids=(EVIDENCE_TWO_ID, EVIDENCE_ID))

    result = _normalize(_request(fragment), second=True)

    assert result.ok
    assert result.value is not None
    assert result.value.fact.evidence_ids == (EVIDENCE_ID, EVIDENCE_TWO_ID)


def test_component_envelope_rejects_nonfinite_output_without_echo() -> None:
    with pytest.raises(FactNormalizationError) as exc_info:
        _output({"private": CANARY, "score": float("inf")})

    assert CANARY not in str(exc_info.value)


def test_component_output_is_deeply_immutable_after_custody() -> None:
    payload = {"nested": {"items": [1, 2]}}
    envelope = _output(payload)
    payload["nested"]["items"].append(3)  # type: ignore[index,union-attr]

    assert envelope.to_protected_dict()["output"] == {"nested": {"items": [1, 2]}}
    with pytest.raises(TypeError):
        envelope.output["nested"] = {}  # type: ignore[index]


def test_component_provenance_changes_fact_identity_even_for_same_output() -> None:
    first = _normalize(_request(_fragment(component="clinical.first")))
    second = _normalize(_request(_fragment(component="clinical.second")))

    assert first.value is not None and second.value is not None
    assert first.value.fact.fact_id != second.value.fact.fact_id
    assert first.value.fact.derivation_hash != second.value.fact.derivation_hash


def test_relation_target_type_must_match_opaque_id_prefix() -> None:
    with pytest.raises(FactNormalizationError, match="does not match"):
        RelationParticipant(
            role="source",
            target_id=EVIDENCE_ID,
            target_type="fact",
        )


def test_static_golden_profiles_cover_cross_stage_contract() -> None:
    fixture = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    assert fixture["schema_version"] == "1.0.0"
    assert {item["case_id"] for item in fixture["examples"]} == {
        "amended_observation",
        "cross_sentence_relation",
        "laboratory_range_and_unit",
        "multilingual_span",
        "negated_condition",
        "partial_date_procedure",
        "uncertain_family_history",
    }
    for item in fixture["examples"]:
        fragment = _fragment(fields=item["fields"], payload=item["output"])
        request = _request(
            fragment,
            profile=item["profile"],
            parent_fact_ids=tuple(item["parent_fact_ids"]),
        )
        result = _normalize(request)

        assert result.state.value == item["expected_state"]
        assert result.value is not None
        assert result.value.fact.fact_type == item["profile"]
