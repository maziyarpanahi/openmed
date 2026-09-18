"""Offline regression tests for assertion-aware clinical NLI pairs."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import (
    AFFIRMED,
    CERTAIN,
    HISTORICAL,
    HYPOTHETICAL,
    NEGATED,
    NLI_ASSERTION_PAIR_ADVISORY,
    RECENT,
    UNCERTAIN,
    ClinicalAssertion,
    InconsistentAssertionMetadataError,
    MissingAssertionMetadataError,
    NliAssertionPair,
    build_assertion_aware_pair,
    build_nli_pair,
    build_nli_pairs,
)

SYNTHETIC_PREMISE = "synthetic source finding"
SYNTHETIC_HYPOTHESIS = "synthetic target claim"


def _assertion(
    *,
    negation: str = AFFIRMED,
    certainty: str = CERTAIN,
    temporality: str = RECENT,
) -> dict[str, str]:
    return {
        "negation": negation,
        "certainty": certainty,
        "temporality": temporality,
    }


def test_pair_carries_negation_uncertainty_and_hypothetical_state() -> None:
    pair = build_nli_pair(
        {
            "text": SYNTHETIC_PREMISE,
            "assertion": {
                "negation": NEGATED,
                "uncertainty": True,
                "hypothetical": True,
            },
            "start": 11,
            "end": 34,
        },
        {
            "text": SYNTHETIC_HYPOTHESIS,
            "clinical_context": _assertion(),
            "offset": [45, 67],
        },
    )

    assert pair.premise_assertion.negation == NEGATED
    assert pair.premise_assertion.uncertainty == UNCERTAIN
    assert pair.premise_assertion.hypothetical is True
    assert pair.hypothesis_assertion.to_dict()["hypothetical"] is False
    assert pair.premise_offset == (11, 34)
    assert pair.hypothesis_offset == (45, 67)
    assert pair.requires_clinician_review is True


def test_pair_accepts_existing_clinical_assertion_objects() -> None:
    pair = build_assertion_aware_pair(
        SYNTHETIC_PREMISE,
        SYNTHETIC_HYPOTHESIS,
        premise_assertion=ClinicalAssertion(
            negation=NEGATED,
            certainty=UNCERTAIN,
            temporality=HYPOTHETICAL,
        ),
        hypothesis_assertion=ClinicalAssertion(
            negation=AFFIRMED,
            certainty=CERTAIN,
            temporality=RECENT,
        ),
    )

    assert isinstance(pair, NliAssertionPair)
    assert pair.premise_assertion.to_dict()["uncertainty"] == UNCERTAIN
    assert pair.premise_assertion.to_dict()["hypothetical"] is True


def test_missing_pair_assertion_metadata_fails_closed() -> None:
    with pytest.raises(MissingAssertionMetadataError):
        build_nli_pair(SYNTHETIC_PREMISE, SYNTHETIC_HYPOTHESIS)

    with pytest.raises(MissingAssertionMetadataError):
        build_nli_pair(
            SYNTHETIC_PREMISE,
            SYNTHETIC_HYPOTHESIS,
            premise_assertion=_assertion(),
        )


def test_missing_assertion_axis_is_rejected() -> None:
    incomplete = _assertion()
    del incomplete["negation"]

    with pytest.raises(MissingAssertionMetadataError, match="premise assertion"):
        build_nli_pair(
            SYNTHETIC_PREMISE,
            SYNTHETIC_HYPOTHESIS,
            premise_assertion=incomplete,
            hypothesis_assertion=_assertion(),
        )


@pytest.mark.parametrize(
    "metadata",
    (
        {
            "negation": AFFIRMED,
            "certainty": CERTAIN,
            "uncertainty": True,
            "temporality": RECENT,
        },
        {
            "negation": AFFIRMED,
            "negated": True,
            "certainty": CERTAIN,
            "temporality": RECENT,
        },
        {
            "negation": AFFIRMED,
            "certainty": CERTAIN,
            "temporality": RECENT,
            "hypothetical": True,
        },
    ),
)
def test_conflicting_redundant_assertion_fields_are_rejected(
    metadata: dict[str, object],
) -> None:
    with pytest.raises(InconsistentAssertionMetadataError):
        build_nli_pair(
            SYNTHETIC_PREMISE,
            SYNTHETIC_HYPOTHESIS,
            premise_assertion=metadata,
            hypothesis_assertion=_assertion(),
        )


def test_invalid_assertion_value_is_not_echoed() -> None:
    sensitive_marker = "synthetic-private-value"

    with pytest.raises(ValueError) as exc_info:
        build_nli_pair(
            SYNTHETIC_PREMISE,
            SYNTHETIC_HYPOTHESIS,
            premise_assertion={
                "negation": sensitive_marker,
                "certainty": CERTAIN,
                "temporality": RECENT,
            },
            hypothesis_assertion=_assertion(),
        )

    assert sensitive_marker not in str(exc_info.value)


def test_pair_serialization_is_deterministic_and_value_free() -> None:
    pair = build_nli_pair(
        {"text": SYNTHETIC_PREMISE, **_assertion(negation=NEGATED)},
        {"text": SYNTHETIC_HYPOTHESIS, **_assertion()},
    )

    first = pair.to_json()
    second = pair.to_json()
    payload = json.loads(first)

    assert first == second
    assert payload["schema_version"] == 1
    assert payload["pair_id"] == pair.pair_id
    assert SYNTHETIC_PREMISE not in first
    assert SYNTHETIC_HYPOTHESIS not in first
    assert pair.to_audit_dict() == pair.to_dict()


def test_repr_is_value_free_but_model_input_is_explicitly_local() -> None:
    pair = build_nli_pair(
        SYNTHETIC_PREMISE,
        SYNTHETIC_HYPOTHESIS,
        premise_assertion=_assertion(),
        hypothesis_assertion=_assertion(),
    )

    assert SYNTHETIC_PREMISE not in repr(pair)
    assert SYNTHETIC_HYPOTHESIS not in repr(pair)
    assert pair.to_model_input() == {
        "premise": SYNTHETIC_PREMISE,
        "hypothesis": SYNTHETIC_HYPOTHESIS,
        "premise_assertion": pair.premise_assertion.to_dict(),
        "hypothesis_assertion": pair.hypothesis_assertion.to_dict(),
    }
    assert pair.to_text_pair() == (SYNTHETIC_PREMISE, SYNTHETIC_HYPOTHESIS)


def test_batch_construction_preserves_order_and_is_offline() -> None:
    pairs = build_nli_pairs(
        (
            {
                "premise": SYNTHETIC_PREMISE,
                "hypothesis": SYNTHETIC_HYPOTHESIS,
                "premise_assertion": _assertion(negation=NEGATED),
                "hypothesis_assertion": _assertion(),
            },
            (
                {
                    "text": "synthetic second premise",
                    **_assertion(),
                },
                {
                    "text": "synthetic second claim",
                    **_assertion(temporality=HYPOTHETICAL),
                },
            ),
        )
    )

    assert len(pairs) == 2
    assert pairs[0].premise == SYNTHETIC_PREMISE
    assert pairs[1].hypothesis_assertion.hypothetical is True
    assert pairs == build_nli_pairs(pairs)


def test_invalid_offsets_fail_without_echoing_text() -> None:
    with pytest.raises(ValueError):
        build_nli_pair(
            {
                "text": SYNTHETIC_PREMISE,
                "offset": [8, 8],
                **_assertion(),
            },
            {"text": SYNTHETIC_HYPOTHESIS, **_assertion()},
        )


def test_pair_disclaimer_requires_human_review() -> None:
    assert "human review" in NLI_ASSERTION_PAIR_ADVISORY
    assert "autonomous clinical decision" in NLI_ASSERTION_PAIR_ADVISORY


@pytest.mark.parametrize("flag", ["hypothetical", "is_hypothetical"])
def test_historical_nonhypothetical_metadata_round_trips(flag: str) -> None:
    pair = build_nli_pair(
        SYNTHETIC_PREMISE,
        SYNTHETIC_HYPOTHESIS,
        premise_assertion={**_assertion(temporality=HISTORICAL), flag: False},
        hypothesis_assertion=_assertion(),
    )
    rebuilt = build_nli_pair(
        SYNTHETIC_PREMISE,
        SYNTHETIC_HYPOTHESIS,
        premise_assertion=pair.premise_assertion.to_dict(),
        hypothesis_assertion=pair.hypothesis_assertion.to_dict(),
    )
    assert rebuilt == pair


def test_historical_hypothetical_conflict_is_rejected() -> None:
    with pytest.raises(InconsistentAssertionMetadataError):
        build_nli_pair(
            SYNTHETIC_PREMISE,
            SYNTHETIC_HYPOTHESIS,
            premise_assertion={
                **_assertion(temporality=HISTORICAL),
                "hypothetical": True,
            },
            hypothesis_assertion=_assertion(),
        )
