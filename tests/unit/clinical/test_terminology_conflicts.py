"""Focused tests for deterministic, privacy-safe terminology reconciliation."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.grounding import Candidate
from openmed.clinical.terminology.conflicts import (
    DISCARD_CATEGORIES,
    ConflictResolution,
    TerminologyCandidate,
    TerminologyConflictResolver,
    resolve_terminology_conflicts,
)


def _candidate(
    code: str,
    *,
    source: str = "curated",
    version: str = "2025.01",
    exactness: str = "exact",
    score: float = 0.8,
) -> TerminologyCandidate:
    return TerminologyCandidate(
        system="SYNTHETIC",
        code=code,
        display=f"Synthetic display {code}",
        source=source,
        version=version,
        exactness=exactness,
        score=score,
        matched_alias="synthetic query surface",
        synonym="synthetic synonym",
    )


def test_policy_applies_source_version_exactness_and_score_in_order() -> None:
    resolver = TerminologyConflictResolver({"curated": 30, "local": 20, "legacy": 10})
    candidates = (
        _candidate("SOURCE", source="curated", version="2023.01", exactness="fuzzy"),
        _candidate("VERSION", source="curated", version="2025.02"),
        _candidate("LOCAL", source="local", version="2026.01"),
    )

    result = resolver.resolve(reversed(candidates))

    assert isinstance(result, ConflictResolution)
    assert result.selected is not None
    assert result.selected.code == "VERSION"
    assert result.selected_provenance is not None
    assert result.selected_provenance.source == "curated"
    assert result.selected_provenance.version == "2025.02"
    assert result.discarded_categories == (
        "lower_source_priority",
        "older_version",
    )
    assert result.discarded["lower_source_priority"][0].candidate.code == "LOCAL"
    assert result.discarded["older_version"][0].candidate.code == "SOURCE"

    exactness_result = resolver.resolve(
        (
            _candidate("FUZZY", exactness="fuzzy"),
            _candidate("EXACT", exactness="exact"),
        )
    )
    assert exactness_result.selected is not None
    assert exactness_result.selected.code == "EXACT"
    assert exactness_result.discarded_categories == ("less_exact",)

    score_result = resolver.resolve(
        (
            _candidate("LOW", score=0.1),
            _candidate("HIGH", score=0.9),
        )
    )
    assert score_result.selected is not None
    assert score_result.selected.code == "HIGH"
    assert score_result.discarded_categories == ("lower_score",)


def test_same_concept_from_multiple_sources_is_retained_as_duplicate() -> None:
    result = resolve_terminology_conflicts(
        (
            _candidate("SYN-1", source="local"),
            _candidate("SYN-1", source="curated"),
            _candidate("SYN-2", source="curated"),
        ),
        {"curated": 2, "local": 1},
    )

    assert result.selected is not None
    assert result.selected.code in {"SYN-1", "SYN-2"}
    assert len(result.discarded["duplicate"]) == 1
    assert result.discarded["duplicate"][0].candidate.code == "SYN-1"
    assert set(result.discarded) == set(DISCARD_CATEGORIES)


def test_version_order_is_numeric_and_independent_of_input_order() -> None:
    candidates = (
        _candidate("V2", version="2025.2"),
        _candidate("V10", version="2025.10"),
    )

    first = resolve_terminology_conflicts(candidates)
    second = resolve_terminology_conflicts(tuple(reversed(candidates)))

    assert first.selected is not None
    assert first.selected.code == "V10"
    assert first.to_dict() == second.to_dict()


def test_stable_version_outranks_its_prerelease() -> None:
    result = resolve_terminology_conflicts(
        (
            _candidate("RELEASE-CANDIDATE", version="v2.2rc1"),
            _candidate("STABLE", version="2.2"),
        )
    )

    assert result.selected is not None
    assert result.selected.code == "STABLE"
    assert result.discarded_categories == ("older_version",)


def test_existing_grounding_candidate_and_mapping_are_supported() -> None:
    grounding_candidate = Candidate(
        system="SYNTHETIC",
        code="GROUND-1",
        display="Synthetic grounding display",
        score=0.7,
        source="curated",
        matched_alias="synthetic query surface",
        match_kind="exact",
        vocab_version="2025.01",
    )
    result = resolve_terminology_conflicts(
        (
            grounding_candidate,
            {
                "system": "SYNTHETIC",
                "code": "GROUND-2",
                "source": "local",
                "version": "2026.01",
                "match_kind": "exact",
                "matched_alias": "synthetic query surface",
                "display": "Synthetic mapping display",
            },
        ),
        {"curated": 2, "local": 1},
    )

    assert result.selected is not None
    assert result.selected.code == "GROUND-1"
    assert result.selected.version == "2025.01"


def test_serialized_result_omits_candidate_surfaces_and_metadata() -> None:
    result = resolve_terminology_conflicts(
        (
            _candidate("SAFE-1"),
            _candidate("SAFE-2", exactness="fuzzy"),
        )
    )

    payload = result.to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert json.loads(serialized) == payload
    assert "synthetic query surface" not in serialized
    assert "synthetic synonym" not in serialized
    assert "Synthetic display" not in serialized
    assert "matched_alias" not in serialized
    assert "display" not in serialized
    assert payload["selected_provenance"]["code"] == "SAFE-1"


def test_empty_input_abstains_without_network_or_text() -> None:
    result = resolve_terminology_conflicts(())

    assert result.abstained is True
    assert result.selected is None
    assert result.selected_provenance is None
    assert result.discarded_categories == ()
    assert set(result.to_dict()["discarded"]) == set(DISCARD_CATEGORIES)


def test_invalid_candidate_errors_do_not_echo_sensitive_values() -> None:
    sensitive_surface = "Synthetic Patient-0001"

    with pytest.raises(ValueError, match="candidate code") as error:
        TerminologyCandidate(code="", matched_alias=sensitive_surface)

    assert sensitive_surface not in str(error.value)
