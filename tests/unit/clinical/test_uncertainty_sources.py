"""Focused regression tests for typed clinical uncertainty sources."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import (
    CLINICAL_DECISION_SUPPORT_DISCLAIMER,
    SourceSpan,
    UncertaintySource,
    UncertaintySourceError,
    UncertaintySources,
    UncertaintySourceType,
    build_guarded_suggestion,
    build_uncertainty_sources,
    coerce_uncertainty_sources,
    compose_uncertainty_sources,
    conflict_uncertainty,
    disclose_uncertainty_sources,
    evidence_uncertainty,
    guarded_suggestion,
    model_uncertainty,
    policy_uncertainty,
    temporal_uncertainty,
)

RAW_REFERENCE = "synthetic-sensitive-reference-marker"
RAW_REASON = "synthetic-sensitive-reason-marker"


def test_all_uncertainty_types_are_disclosed_without_an_aggregate_score() -> None:
    sources = compose_uncertainty_sources(
        conflict_uncertainty("unresolved"),
        temporal_uncertainty("unresolved"),
        policy_uncertainty("review_required"),
        model_uncertainty("ambiguous"),
        evidence_uncertainty("insufficient", references=(RAW_REFERENCE,)),
    )

    payload = sources.to_dict()

    assert [item["source_type"] for item in payload["active_sources"]] == [
        "evidence",
        "model",
        "policy",
        "temporal",
        "conflict",
    ]
    assert payload["active_source_types"] == [
        "evidence",
        "model",
        "policy",
        "temporal",
        "conflict",
    ]
    assert payload["active_source_count"] == 5
    assert "confidence" not in payload
    assert "score" not in payload
    assert RAW_REFERENCE not in repr(sources)
    assert RAW_REFERENCE not in sources.to_json()


def test_composition_is_deterministic_and_deduplicates_sources() -> None:
    evidence = evidence_uncertainty("missing", references=("synthetic-ref",))
    model = model_uncertainty("ambiguous")

    first = UncertaintySources.compose(model, evidence, evidence)
    second = UncertaintySources.compose(evidence, model)

    assert first == second
    assert first.to_json() == second.to_json()
    assert len(first.sources) == 2


def test_inactive_sources_are_not_disclosed_as_active() -> None:
    sources = compose_uncertainty_sources(
        evidence_uncertainty("stale", active=False),
        conflict_uncertainty("unresolved"),
    )

    assert [source.source_type for source in sources.active_sources] == [
        UncertaintySourceType.CONFLICT
    ]
    assert sources.to_dict()["active_source_count"] == 1
    assert sources.to_dict()["active_sources"][0]["source_type"] == "conflict"


def test_raw_references_are_replaced_by_opaque_digests() -> None:
    source = UncertaintySource(
        UncertaintySourceType.EVIDENCE,
        "missing",
        references=(RAW_REFERENCE,),
    )

    assert source.references[0].startswith("sha256:")
    assert RAW_REFERENCE not in repr(source)
    assert RAW_REFERENCE not in json.dumps(source.to_dict())


def test_invalid_reason_code_fails_without_echoing_the_submitted_value() -> None:
    with pytest.raises(UncertaintySourceError) as exc_info:
        UncertaintySource("model", RAW_REASON)

    assert "reason code is unsupported" in str(exc_info.value)
    assert RAW_REASON not in str(exc_info.value)


def test_source_type_and_reason_aliases_are_typed() -> None:
    source = UncertaintySource(
        kind="evidence",
        reason_code="insufficient_evidence",
    )

    assert source.source_type is UncertaintySourceType.EVIDENCE
    assert source.kind is UncertaintySourceType.EVIDENCE
    assert source.reason_code == "insufficient_evidence"


def test_mapping_round_trip_and_builder_aliases() -> None:
    sources = build_uncertainty_sources(
        evidence={"source_type": "evidence", "reason_code": "missing"},
        model=[{"source_type": "model", "reason_code": "ambiguous"}],
        policy=policy_uncertainty("incomplete"),
    )

    restored = coerce_uncertainty_sources(sources.to_dict())

    assert restored.to_dict() == sources.to_dict()
    assert disclose_uncertainty_sources(sources) == sources.to_dict()


def test_guarded_suggestion_discloses_each_active_source() -> None:
    sources = UncertaintySources.compose(
        evidence_uncertainty("insufficient"),
        model_uncertainty("ambiguous"),
        conflict_uncertainty("unresolved"),
    )

    guarded = build_guarded_suggestion(
        "Review synthetic finding",
        [SourceSpan(start=12, end=28, label="synthetic")],
        0.72,
        uncertainty_sources=sources,
    )
    payload = guarded.to_dict()

    assert payload["disclaimer"] == CLINICAL_DECISION_SUPPORT_DISCLAIMER
    assert payload["uncertainty_sources"]["active_source_count"] == 3
    assert {
        item["source_type"] for item in payload["uncertainty_sources"]["active_sources"]
    } == {"evidence", "model", "conflict"}
    assert payload["confidence"] == pytest.approx(0.72)

    restored = guarded.from_dict(payload)
    assert restored.uncertainty_sources == sources


def test_guarded_decorator_accepts_a_fifth_uncertainty_sources_tuple_item() -> None:
    sources = compose_uncertainty_sources(temporal_uncertainty("unresolved"))

    @guarded_suggestion
    def produce() -> tuple[object, object, float, dict[str, str], UncertaintySources]:
        return (
            "Review synthetic finding",
            [SourceSpan(start=1, end=8)],
            0.5,
            {"producer": "synthetic"},
            sources,
        )

    result = produce()

    assert result.uncertainty_sources == sources
    assert result.to_dict()["uncertainty_sources"]["active_source_types"] == [
        "temporal"
    ]


def test_empty_disclosure_is_deterministic_and_review_safe() -> None:
    disclosure = UncertaintySources().to_dict()

    assert disclosure["active_source_count"] == 0
    assert disclosure["active_source_types"] == []
    assert disclosure["active_sources"] == []
    assert disclosure["disclaimer"]
