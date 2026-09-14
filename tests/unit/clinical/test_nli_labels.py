"""Offline regression tests for the stable clinical NLI label contract."""

from __future__ import annotations

import json
import math

import pytest

from openmed.clinical import (
    NLI_LABELS,
    BackendLabelMapping,
    ClinicalNliResult,
    NliLabel,
    NliScoreValidationError,
    UnknownNliLabelError,
    build_nli_result,
    validate_backend_label_mapping,
)


def _backend_mapping() -> BackendLabelMapping:
    return BackendLabelMapping.from_mapping(
        {
            "class_0": "contradiction",
            "class_1": "neutral",
            "class_2": "entailment",
            "class_3": "abstention",
        },
        backend="synthetic-local",
    )


def test_contract_exposes_exactly_four_canonical_states():
    assert NLI_LABELS == (
        "entailment",
        "contradiction",
        "neutral",
        "abstention",
    )
    assert tuple(item.value for item in NliLabel) == NLI_LABELS
    assert NliLabel.ENTAIL == NliLabel.ENTAILMENT
    assert NliLabel.ABSTAIN == NliLabel.ABSTENTION


def test_backend_mapping_normalizes_labels_and_is_immutable():
    mapping = _backend_mapping()

    assert mapping.resolve(" CLASS_2 ") is NliLabel.ENTAILMENT
    assert mapping["class_0"] is NliLabel.CONTRADICTION
    assert mapping.backend_to_canonical["class_3"] is NliLabel.ABSTENTION
    assert mapping.to_dict() == {
        "schema_version": 1,
        "backend": "synthetic-local",
        "labels": {
            "class_0": "contradiction",
            "class_1": "neutral",
            "class_2": "entailment",
            "class_3": "abstention",
        },
    }
    with pytest.raises(TypeError):
        mapping.labels["class_4"] = NliLabel.NEUTRAL  # type: ignore[index]


def test_mapping_can_require_complete_four_state_coverage():
    complete = validate_backend_label_mapping(
        {"a": "entailment", "b": "contradiction", "c": "neutral", "d": "abstention"},
        backend="synthetic",
        require_complete=True,
    )
    assert set(complete.canonical_labels) == set(NLI_LABELS)

    partial = validate_backend_label_mapping(
        {"a": "entailment", "b": "contradiction"}, backend="synthetic"
    )
    with pytest.raises(ValueError, match="all four"):
        partial.require_complete()


def test_unknown_backend_label_fails_closed_without_echoing_input():
    opaque_label = "synthetic-opaque-label"

    with pytest.raises(UnknownNliLabelError) as error:
        _backend_mapping().resolve(opaque_label)

    assert opaque_label not in str(error.value)


def test_result_maps_backend_label_and_keeps_only_bounded_numeric_scores():
    result = build_nli_result(
        "class_3",
        mapping=_backend_mapping(),
        raw_scores={
            "class_0": 0.10,
            "class_1": 0.20,
            "class_2": 0.30,
            "class_3": 0.40,
        },
        metadata={"margin": 0.25, "raw_scores": {"class_3": 0.40}},
    )

    assert result.label is NliLabel.ABSTENTION
    assert result.state is NliLabel.ABSTENTION
    assert result.backend == "synthetic-local"
    assert result.backend_label == "class_3"
    assert result.score == 0.40
    assert result.raw_scores == {
        "abstention": 0.40,
        "contradiction": 0.10,
        "entailment": 0.30,
        "neutral": 0.20,
    }
    assert result.metadata == {
        "margin": 0.25,
        "raw_scores": {"abstention": 0.40},
    }
    with pytest.raises(TypeError):
        result.metadata["raw_scores"]["abstention"] = 0.1  # type: ignore[index]
    assert result.to_dict()["metadata"] == {
        "margin": 0.25,
        "raw_scores": {"abstention": 0.40},
    }


def test_result_serialization_is_deterministic_and_round_trips():
    result = ClinicalNliResult.from_backend(
        "class_2",
        mapping=_backend_mapping(),
        score=0.875,
    )

    payload = result.to_dict()
    assert "source" not in payload
    assert result.to_json() == result.to_json()
    assert json.loads(result.to_json()) == payload
    assert ClinicalNliResult.from_dict(payload) == result


@pytest.mark.parametrize(
    "bad_score",
    [-0.01, 1.01, math.nan, math.inf, -math.inf, True, "0.5"],
)
def test_scores_reject_non_finite_or_out_of_range_values(bad_score):
    with pytest.raises((TypeError, NliScoreValidationError)):
        ClinicalNliResult.from_backend(
            "class_0",
            mapping=_backend_mapping(),
            score=bad_score,
        )


def test_result_rejects_unknown_score_labels_and_conflicting_aliases():
    with pytest.raises(UnknownNliLabelError):
        ClinicalNliResult.from_backend(
            "class_0",
            mapping=_backend_mapping(),
            raw_scores={"unmapped": 0.5},
        )

    with pytest.raises(ValueError, match="conflicts"):
        ClinicalNliResult.from_backend(
            "class_0",
            mapping=_backend_mapping(),
            raw_scores={"class_0": 0.2},
            score=0.3,
        )


def test_result_validation_does_not_retain_sensitive_input_text():
    source_text = "Synthetic source text that must never enter this contract."
    result = ClinicalNliResult.from_backend(
        "class_2",
        mapping=_backend_mapping(),
        score=0.9,
    )

    serialized = json.dumps(result.to_dict())
    assert source_text not in serialized
    assert source_text not in repr(result)
