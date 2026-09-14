"""Offline regression tests for experiencer-aware clinical NLI pairs."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import (
    CAREGIVER_EXPERIENCER,
    FAMILY_EXPERIENCER,
    NLI_EXPERIENCER_PAIR_ADVISORY,
    NLI_EXPERIENCER_VALUES,
    UNKNOWN_EXPERIENCER,
    ExperiencerCompatibility,
    ExperiencerMetadata,
    NliExperiencerPair,
    build_experiencer_nli_pair,
    build_experiencer_nli_pairs,
    compare_experiencers,
)


def _side(
    text: str,
    experiencer: str | None = None,
    *,
    start: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"text": text}
    if experiencer is not None:
        payload["experiencer"] = experiencer
    if start is not None:
        payload["offset"] = [start, start + len(text)]
    return payload


def test_nli_experiencer_vocabulary_has_four_explicit_classes() -> None:
    assert NLI_EXPERIENCER_VALUES == (
        "patient",
        "family",
        CAREGIVER_EXPERIENCER,
        UNKNOWN_EXPERIENCER,
    )


@pytest.mark.parametrize("experiencer", NLI_EXPERIENCER_VALUES[:-1])
def test_matching_known_experiencers_allow_entailment(experiencer: str) -> None:
    pair = build_experiencer_nli_pair(
        _side("synthetic premise", experiencer),
        _side("synthetic hypothesis", experiencer),
        predicted_label="entailment",
    )

    assert isinstance(pair, NliExperiencerPair)
    assert pair.premise_experiencer_class == experiencer
    assert pair.hypothesis_experiencer_class == experiencer
    assert pair.experiencer_compatibility == "compatible"
    assert pair.experiencer_comparison.status is ExperiencerCompatibility.COMPATIBLE
    assert pair.entailment_allowed is True
    assert pair.entailment_blocked is False
    assert pair.review_required is False
    assert pair.label == "entailment"


@pytest.mark.parametrize(
    ("premise_experiencer", "hypothesis_experiencer"),
    [
        ("family", "patient"),
        ("caregiver", "patient"),
        ("family", "caregiver"),
    ],
)
def test_conflicting_known_experiencers_block_entailment(
    premise_experiencer: str,
    hypothesis_experiencer: str,
) -> None:
    pair = build_experiencer_nli_pair(
        _side("synthetic premise", premise_experiencer),
        _side("synthetic hypothesis", hypothesis_experiencer),
        predicted_label="entailment",
    )

    assert pair.experiencer_compatibility == "incompatible"
    assert pair.experiencer_comparison.reason == "experiencer_conflict"
    assert pair.experiencer_comparison.status is ExperiencerCompatibility.INCOMPATIBLE
    assert pair.entailment_allowed is False
    assert pair.entailment_blocked is True
    assert pair.review_required is True
    assert pair.label == "review_required"
    assert pair.predicted_label == "entailment"
    assert pair.experiencer_reason == "experiencer_conflict"


@pytest.mark.parametrize(
    ("premise", "hypothesis"),
    [
        (_side("synthetic premise"), _side("synthetic hypothesis", "patient")),
        (
            _side("synthetic premise", "unknown"),
            _side("synthetic hypothesis", "patient"),
        ),
        (
            _side("synthetic premise", "patient"),
            _side("synthetic hypothesis", "unknown"),
        ),
        (_side("synthetic premise"), _side("synthetic hypothesis")),
    ],
)
def test_missing_or_unknown_experiencers_are_unresolved(
    premise: dict[str, object],
    hypothesis: dict[str, object],
) -> None:
    pair = build_experiencer_nli_pair(
        premise,
        hypothesis,
        predicted_label="entailment",
    )

    assert pair.experiencer_compatibility == "unresolved"
    assert pair.experiencer_comparison.reason == "experiencer_unresolved"
    assert pair.entailment_allowed is False
    assert pair.label == "review_required"
    assert pair.review_required is True


def test_caregiver_and_unknown_are_preserved_in_nested_metadata() -> None:
    pair = build_experiencer_nli_pair(
        {
            "text": "synthetic caregiver statement",
            "context": {"experiencer": "care giver", "resolved": True},
        },
        {
            "text": "synthetic unresolved statement",
            "clinical_assertion": {"experiencer": "indeterminate"},
        },
    )

    assert pair.premise_experiencer_class == CAREGIVER_EXPERIENCER
    assert pair.premise_experiencer_metadata.is_resolved is True
    assert pair.hypothesis_experiencer_class == UNKNOWN_EXPERIENCER
    assert pair.hypothesis_experiencer_metadata.is_resolved is False


def test_existing_clinical_assertion_and_other_subject_fail_closed() -> None:
    pair = build_experiencer_nli_pair(
        "synthetic premise",
        "synthetic hypothesis",
        premise_experiencer={"experiencer": "other", "source": "cue"},
        hypothesis_experiencer={"experiencer": "family"},
    )

    assert pair.premise_experiencer_class == UNKNOWN_EXPERIENCER
    assert pair.hypothesis_experiencer_class == FAMILY_EXPERIENCER
    assert pair.entailment_allowed is False


def test_explicit_and_nested_experiencer_metadata_must_agree() -> None:
    with pytest.raises(ValueError, match="inconsistent") as error:
        build_experiencer_nli_pair(
            {"text": "synthetic premise", "experiencer": "family"},
            {"text": "synthetic hypothesis", "experiencer": "patient"},
            premise_experiencer="patient",
        )

    assert "family" not in str(error.value)
    assert "patient" not in str(error.value)


def test_metadata_validation_keeps_only_safe_provenance() -> None:
    metadata = ExperiencerMetadata.from_value(
        {
            "experiencer": "family",
            "source": "context",
            "cue_offset": [3, 9],
        }
    )

    assert metadata.experiencer_class == FAMILY_EXPERIENCER
    assert metadata.is_resolved is True
    assert metadata.to_dict() == {
        "experiencer": FAMILY_EXPERIENCER,
        "resolved": True,
        "source": "context",
        "cue_offset": [3, 9],
    }


def test_comparison_is_conservative_for_unknown_and_known_classes() -> None:
    comparison = compare_experiencers("unknown", "unknown")

    assert comparison.status is ExperiencerCompatibility.UNRESOLVED
    assert comparison.compatible is False
    assert comparison.resolved is False
    assert comparison.to_dict() == {
        "status": "unresolved",
        "reason": "experiencer_unresolved",
        "compatible": False,
        "resolved": False,
    }


def test_pair_audit_serialization_is_deterministic_and_value_free() -> None:
    premise = "synthetic premise value that stays local"
    hypothesis = "synthetic hypothesis value that stays local"
    pair = build_experiencer_nli_pair(
        _side(premise, "family", start=4),
        _side(hypothesis, "patient", start=48),
        predicted_label="entailment",
    )

    first = pair.to_json()
    payload = json.loads(first)

    assert first == pair.to_json()
    assert payload["schema_version"] == 1
    assert payload["pair_id"] == pair.pair_id
    assert premise not in first
    assert hypothesis not in first
    assert pair.to_audit_dict() == pair.to_dict()
    assert "human review" in payload["advisory"]
    assert "text_hash" in payload["premise"]
    assert "text" not in payload["premise"]


def test_model_input_is_the_explicit_raw_text_boundary() -> None:
    pair = build_experiencer_nli_pair(
        "synthetic premise",
        "synthetic hypothesis",
        premise_experiencer="patient",
        hypothesis_experiencer="patient",
    )

    assert pair.to_text_pair() == ("synthetic premise", "synthetic hypothesis")
    model_input = pair.to_model_input()
    assert model_input["premise"] == "synthetic premise"
    assert model_input["hypothesis"] == "synthetic hypothesis"
    assert model_input["premise_experiencer"]["experiencer"] == "patient"
    assert "synthetic premise" not in repr(pair)
    assert "synthetic hypothesis" not in repr(pair)


def test_batch_construction_preserves_order_and_is_repeatable() -> None:
    inputs = [
        {
            "premise": _side("synthetic family finding", "family"),
            "hypothesis": _side("synthetic family claim", "family"),
            "predicted_label": "entailment",
        },
        (
            _side("synthetic caregiver finding", "caregiver"),
            _side("synthetic patient claim", "patient"),
        ),
    ]

    pairs = build_experiencer_nli_pairs(inputs)

    assert len(pairs) == 2
    assert pairs[0].premise_experiencer_class == FAMILY_EXPERIENCER
    assert pairs[0].label == "entailment"
    assert pairs[1].experiencer_compatibility == "incompatible"
    assert pairs == build_experiencer_nli_pairs(inputs)


def test_aliases_accept_provider_label_and_score() -> None:
    pair = build_experiencer_nli_pair(
        "synthetic premise",
        "synthetic hypothesis",
        premise_experiencer="patient",
        hypothesis_experiencer="patient",
        nli_label="ENTAIL",
        nli_score=0.875,
    )

    assert pair.predicted_label == "entailment"
    assert pair.predicted_score == 0.875
    assert pair.label == "entailment"


def test_invalid_experiencer_does_not_echo_sensitive_value() -> None:
    sensitive_marker = "synthetic-private-value"

    with pytest.raises(ValueError) as error:
        build_experiencer_nli_pair(
            "synthetic premise",
            "synthetic hypothesis",
            premise_experiencer=sensitive_marker,
            hypothesis_experiencer="patient",
        )

    assert sensitive_marker not in str(error.value)


def test_advisory_requires_human_review() -> None:
    assert "qualified human review" in NLI_EXPERIENCER_PAIR_ADVISORY
    assert "autonomous clinical decision" in NLI_EXPERIENCER_PAIR_ADVISORY
