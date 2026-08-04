"""Tests for exact offline reference-population risk assessment."""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace

import pytest

from openmed.core.audit import stable_hash
from openmed.risk import PopulationRiskAssessment, assess_population_risk


def _rehash_population_payload(payload: dict[str, object]) -> None:
    payload["integrity_digest"] = stable_hash(
        {
            field: value
            for field, value in payload.items()
            if field != "integrity_digest"
        }
    )


def test_row_level_population_metrics_and_policy_verdict() -> None:
    sample = [
        {"age": 30, "region": "north"},
        {"age": 40, "region": "south"},
    ]
    reference_population = [
        {"age": 30, "region": "north"},
        {"age": 30, "region": "north"},
        {"age": 30, "region": "north"},
        {"age": 40, "region": "south"},
        {"age": 40, "region": "south"},
    ]

    assessment = assess_population_risk(
        sample,
        reference_population,
        ("age", "region"),
        target_k_map=2,
        max_delta_presence=0.5,
    )

    assert assessment.analysis_unit_model == "row"
    assert assessment.sample_row_count == 2
    assert assessment.reference_row_count == 5
    assert assessment.sample_unit_count == 2
    assert assessment.reference_unit_count == 5
    assert assessment.sample_profile_count == 2
    assert assessment.reference_profile_count == 2
    assert assessment.matched_sample_unit_count == 2
    assert assessment.unmatched_sample_unit_count == 0
    assert assessment.population_singleton_count == 0
    assert assessment.population_singleton_rate == 0.0
    assert assessment.achieved_k_map == 2
    assert assessment.max_exact_linkage_risk == 0.5
    assert assessment.mean_exact_linkage_risk == pytest.approx(5 / 12)
    assert assessment.max_delta_presence == 0.5
    assert assessment.mean_delta_presence == pytest.approx(5 / 12)
    assert assessment.reference_model_consistent is True
    assert assessment.meets_k_map is True
    assert assessment.meets_delta_presence is True
    assert assessment.meets_policy is True


def test_keyed_profiles_preserve_joint_row_correlation() -> None:
    sample = [
        {"sample_id": "sample-canary", "site": "north", "code": "alpha"},
        {"sample_id": "sample-canary", "site": "south", "code": "beta"},
    ]
    crossed_reference = [
        {"population_id": "population-canary", "site": "north", "code": "beta"},
        {"population_id": "population-canary", "site": "south", "code": "alpha"},
    ]

    assessment = assess_population_risk(
        sample,
        crossed_reference,
        ("site", "code"),
        sample_privacy_unit="sample_id",
        population_privacy_unit="population_id",
        target_k_map=1,
        max_delta_presence=1.0,
    )

    assert assessment.analysis_unit_model == "keyed"
    assert assessment.sample_unit_count == 1
    assert assessment.reference_unit_count == 1
    assert assessment.matched_sample_unit_count == 0
    assert assessment.unmatched_sample_unit_count == 1
    assert assessment.achieved_k_map == 0
    assert assessment.max_exact_linkage_risk == 1.0
    assert assessment.reference_model_consistent is False
    assert assessment.meets_policy is False


def test_keyed_profiles_preserve_repeated_row_multiplicity() -> None:
    sample = [
        {"sample_id": "sample-canary", "site": "north", "code": "alpha"},
        {"sample_id": "sample-canary", "site": "north", "code": "alpha"},
    ]
    reference_population = [
        {"population_id": "population-canary", "site": "north", "code": "alpha"}
    ]

    assessment = assess_population_risk(
        sample,
        reference_population,
        ("site", "code"),
        sample_privacy_unit="sample_id",
        population_privacy_unit="population_id",
        target_k_map=1,
        max_delta_presence=1.0,
    )

    assert assessment.unmatched_sample_unit_count == 1
    assert assessment.achieved_k_map == 0
    assert assessment.reference_model_consistent is False


def test_keyed_population_frequency_counts_analysis_units() -> None:
    sample = [
        {"sample_id": "sample-a", "site": "north", "code": "alpha"},
        {"sample_id": "sample-a", "site": "south", "code": "beta"},
        {"sample_id": "sample-b", "site": "south", "code": "beta"},
        {"sample_id": "sample-b", "site": "north", "code": "alpha"},
    ]
    reference_population = [
        {"population_id": unit, "site": site, "code": code}
        for unit in ("population-a", "population-b", "population-c")
        for site, code in (("south", "beta"), ("north", "alpha"))
    ]

    assessment = assess_population_risk(
        sample,
        reference_population,
        ("site", "code"),
        sample_privacy_unit="sample_id",
        population_privacy_unit="population_id",
        target_k_map=3,
        max_delta_presence=2 / 3,
    )

    assert assessment.sample_row_count == 4
    assert assessment.reference_row_count == 6
    assert assessment.sample_unit_count == 2
    assert assessment.reference_unit_count == 3
    assert assessment.sample_profile_count == 1
    assert assessment.reference_profile_count == 1
    assert assessment.achieved_k_map == 3
    assert assessment.max_exact_linkage_risk == pytest.approx(1 / 3)
    assert assessment.mean_exact_linkage_risk == pytest.approx(1 / 3)
    assert assessment.max_delta_presence == pytest.approx(2 / 3)
    assert assessment.mean_delta_presence == pytest.approx(2 / 3)
    assert assessment.meets_policy is True


def test_sample_frequency_above_reference_fails_closed() -> None:
    sample = [{"age": 30}, {"age": 30}]
    reference_population = [{"age": 30}]

    assessment = assess_population_risk(
        sample,
        reference_population,
        ("age",),
        target_k_map=1,
        max_delta_presence=1.0,
    )

    assert assessment.matched_sample_unit_count == 2
    assert assessment.unmatched_sample_unit_count == 0
    assert assessment.population_singleton_count == 2
    assert assessment.population_singleton_rate == 1.0
    assert assessment.reference_frequency_violation_count == 1
    assert assessment.reference_frequency_violation_unit_count == 2
    assert assessment.achieved_k_map == 1
    assert assessment.max_exact_linkage_risk == 1.0
    assert assessment.max_delta_presence == 2.0
    assert assessment.mean_delta_presence == 2.0
    assert assessment.reference_model_consistent is False
    assert assessment.meets_k_map is False
    assert assessment.meets_delta_presence is False
    assert assessment.meets_policy is False


def test_unmatched_profiles_receive_conservative_risk_and_zero_k_map() -> None:
    assessment = assess_population_risk(
        [{"age": 30}],
        [{"age": 40}],
        ("age",),
        target_k_map=1,
        max_delta_presence=1.0,
    )

    assert assessment.matched_sample_unit_count == 0
    assert assessment.unmatched_sample_unit_count == 1
    assert assessment.reference_frequency_violation_count == 1
    assert assessment.achieved_k_map == 0
    assert assessment.max_exact_linkage_risk == 1.0
    assert assessment.mean_exact_linkage_risk == 1.0
    assert assessment.max_delta_presence == 1.0
    assert assessment.mean_delta_presence == 1.0
    assert assessment.meets_policy is False


def test_typed_quasi_identifier_encodings_cannot_alias() -> None:
    with pytest.raises(ValueError, match="scalar type"):
        assess_population_risk(
            [{"age": 30}],
            [{"age": 30.0}],
            ("age",),
            target_k_map=1,
            max_delta_presence=1.0,
        )

    assessment = assess_population_risk(
        [{"region": "North"}],
        [{"region": "north"}],
        ("region",),
        target_k_map=1,
        max_delta_presence=1.0,
    )
    assert assessment.unmatched_sample_unit_count == 1


def test_serialization_is_aggregate_only_and_integrity_bound() -> None:
    sample = [
        {
            "sample_id": "SAMPLE-ID-CANARY",
            "age": 34,
            "postal_code": "POSTAL-CANARY",
            "sensitive_note": "DIAGNOSIS-CANARY",
            "source_path": "/private/canary/source.csv",
        }
    ]
    reference_population = [
        {
            "population_id": unit,
            "age": 34,
            "postal_code": "POSTAL-CANARY",
            "sensitive_note": "REFERENCE-NOTE-CANARY",
            "source_path": "/reference/canary/source.csv",
        }
        for unit in ("POPULATION-ID-CANARY-A", "POPULATION-ID-CANARY-B")
    ]

    first = assess_population_risk(
        sample,
        reference_population,
        ("age", "postal_code"),
        sample_privacy_unit="sample_id",
        population_privacy_unit="population_id",
        target_k_map=2,
        max_delta_presence=0.5,
    )
    second = assess_population_risk(
        sample,
        reference_population,
        ("postal_code", "age"),
        sample_privacy_unit="sample_id",
        population_privacy_unit="population_id",
        target_k_map=2,
        max_delta_presence=0.5,
    )

    payload = first.to_json(indent=None)
    parsed = json.loads(payload)
    for canary in (
        "SAMPLE-ID-CANARY",
        "POPULATION-ID-CANARY",
        "POSTAL-CANARY",
        "DIAGNOSIS-CANARY",
        "REFERENCE-NOTE-CANARY",
        "/private/canary/source.csv",
        "/reference/canary/source.csv",
        "sample_id",
        "population_id",
        "postal_code",
        "sensitive_note",
        "source_path",
    ):
        assert canary not in payload
    assert "profiles" not in parsed
    assert "rows" not in parsed
    assert first.digest == second.digest
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", first.digest)


def test_digests_are_order_independent_and_bind_data_schema_and_policy() -> None:
    sample = [{"age": 30}, {"age": 40}]
    reference_population = [{"age": 30}, {"age": 40}, {"age": 40}]

    baseline = assess_population_risk(
        sample,
        reference_population,
        ("age",),
        target_k_map=1,
        max_delta_presence=1.0,
    )
    reordered = assess_population_risk(
        list(reversed(sample)),
        list(reversed(reference_population)),
        ("age",),
        target_k_map=1,
        max_delta_presence=1.0,
    )
    changed_policy = assess_population_risk(
        sample,
        reference_population,
        ("age",),
        target_k_map=1,
        max_delta_presence=0.5,
    )
    changed_reference = assess_population_risk(
        sample,
        [*reference_population, {"age": 40}],
        ("age",),
        target_k_map=1,
        max_delta_presence=1.0,
    )

    assert baseline.sample_digest == reordered.sample_digest
    assert baseline.reference_population_digest == reordered.reference_population_digest
    assert baseline.schema_digest == reordered.schema_digest
    assert baseline.policy_digest == reordered.policy_digest
    assert baseline.digest == reordered.digest
    assert baseline.policy_digest != changed_policy.policy_digest
    assert baseline.schema_digest == changed_policy.schema_digest
    assert baseline.digest != changed_policy.digest
    assert (
        baseline.reference_population_digest
        != changed_reference.reference_population_digest
    )
    assert baseline.schema_digest != changed_reference.schema_digest
    assert baseline.digest != changed_reference.digest


def test_strict_saved_artifact_round_trip() -> None:
    assessment = assess_population_risk(
        [{"age": 30}],
        [{"age": 30}, {"age": 30}],
        ("age",),
        target_k_map=2,
        max_delta_presence=0.5,
    )

    from_dict = PopulationRiskAssessment.from_dict(assessment.to_dict())
    from_json = PopulationRiskAssessment.from_json(assessment.to_json(indent=None))

    assert from_dict == assessment
    assert from_json == assessment
    assert from_dict.digest == assessment.digest
    assert assessment.integrity_digest == assessment.digest
    assert assessment.to_dict()["integrity_digest"] == assessment.digest


def test_integer_valued_public_rates_are_canonicalized_for_round_trip() -> None:
    assessment = assess_population_risk(
        [{"age": 30}],
        [{"age": 30}, {"age": 30}],
        ("age",),
        target_k_map=2,
        max_delta_presence=1.0,
    )
    with_integer_rate = replace(assessment, population_singleton_rate=0)

    assert type(with_integer_rate.population_singleton_rate) is float
    assert (
        PopulationRiskAssessment.from_json(with_integer_rate.to_json(indent=None))
        == with_integer_rate
    )


def test_saved_artifact_tampering_requires_canonical_reconstruction() -> None:
    assessment = assess_population_risk(
        [{"age": 30}],
        [{"age": 30}, {"age": 30}],
        ("age",),
        target_k_map=2,
        max_delta_presence=1.0,
    )
    tampered = assessment.to_dict()
    tampered["target_k_map"] = 3

    with pytest.raises(ValueError, match="integrity digest mismatch"):
        PopulationRiskAssessment.from_dict(tampered)

    tampered["integrity_digest"] = stable_hash(
        {
            field: value
            for field, value in tampered.items()
            if field != "integrity_digest"
        }
    )
    with pytest.raises(ValueError, match="canonical assessment"):
        PopulationRiskAssessment.from_dict(tampered)

    reconstructed = assess_population_risk(
        [{"age": 30}],
        [{"age": 30}, {"age": 30}],
        ("age",),
        target_k_map=3,
        max_delta_presence=1.0,
    )
    assert reconstructed.digest != assessment.digest
    assert PopulationRiskAssessment.from_dict(reconstructed.to_dict()) == reconstructed


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        (
            {"achieved_k_map": 4},
            "achieved_k_map cannot exceed reference_unit_count",
        ),
        (
            {
                "matched_sample_unit_count": 1,
                "unmatched_sample_unit_count": 2,
            },
            "singleton units cannot exceed matched",
        ),
        (
            {"reference_frequency_violation_count": 3},
            "profile count cannot exceed sample profiles",
        ),
        (
            {"reference_frequency_violation_unit_count": 1},
            "profile count cannot exceed affected units",
        ),
        (
            {
                "sample_unit_count": 4,
                "matched_sample_unit_count": 3,
            },
            "sample_unit_count cannot exceed sample_row_count",
        ),
        (
            {"reference_unit_count": 4},
            "reference_unit_count cannot exceed reference_row_count",
        ),
        (
            {
                "sample_profile_count": 3,
                "matched_sample_unit_count": 3,
                "unmatched_sample_unit_count": 0,
            },
            "sample profiles exceed the matched and unmatched profile bounds",
        ),
        (
            {
                "reference_frequency_violation_count": 0,
                "reference_frequency_violation_unit_count": 1,
            },
            "profile and unit counts must both be zero",
        ),
        (
            {"sample_profile_count": 1},
            "cannot combine matched and unmatched partitions",
        ),
        (
            {"reference_frequency_violation_count": 1},
            "do not cover unmatched and matched violation partitions",
        ),
        (
            {"mean_exact_linkage_risk": 0.9},
            "mean exact-linkage risk is below the aggregate count bound",
        ),
    ),
)
def test_rehashed_artifact_rejects_impossible_aggregate_relationships(
    changes: dict[str, object],
    message: str,
) -> None:
    assessment = assess_population_risk(
        [{"age": 30}, {"age": 30}, {"age": 50}],
        [{"age": 30}, {"age": 40}, {"age": 40}],
        ("age",),
        target_k_map=1,
        max_delta_presence=1.0,
    )
    tampered: dict[str, object] = assessment.to_dict()
    tampered.update(changes)
    _rehash_population_payload(tampered)

    with pytest.raises(ValueError, match=message):
        PopulationRiskAssessment.from_dict(tampered)


def test_rehashed_artifact_rejects_unsupported_matched_profile_k_map() -> None:
    assessment = assess_population_risk(
        [{"age": 30}, {"age": 40}],
        [{"age": 30}, {"age": 30}, {"age": 40}, {"age": 40}],
        ("age",),
        target_k_map=2,
        max_delta_presence=0.5,
    )
    tampered: dict[str, object] = assessment.to_dict()
    tampered["reference_row_count"] = 3
    tampered["reference_unit_count"] = 3
    _rehash_population_payload(tampered)

    with pytest.raises(ValueError, match="cannot support every matched sample profile"):
        PopulationRiskAssessment.from_dict(tampered)


def test_rehashed_artifact_rejects_consistent_sample_larger_than_reference() -> None:
    assessment = assess_population_risk(
        [{"age": 30}, {"age": 30}],
        [{"age": 30}, {"age": 30}, {"age": 30}],
        ("age",),
        target_k_map=3,
        max_delta_presence=1.0,
    )
    tampered: dict[str, object] = assessment.to_dict()
    tampered["sample_row_count"] = 4
    tampered["sample_unit_count"] = 4
    tampered["matched_sample_unit_count"] = 4
    _rehash_population_payload(tampered)

    with pytest.raises(ValueError, match="exceed the reference units available"):
        PopulationRiskAssessment.from_dict(tampered)


def test_rehashed_fully_unmatched_artifact_requires_exact_delta_presence() -> None:
    assessment = assess_population_risk(
        [{"age": 30}, {"age": 31}],
        [{"age": 40}, {"age": 41}],
        ("age",),
        target_k_map=1,
        max_delta_presence=1.0,
    )
    tampered: dict[str, object] = assessment.to_dict()
    tampered["max_delta_presence"] = 2.0
    tampered["mean_delta_presence"] = 2.0
    _rehash_population_payload(tampered)

    with pytest.raises(
        ValueError,
        match="maximum delta-presence and matched frequency violations",
    ):
        PopulationRiskAssessment.from_dict(tampered)


def test_rehashed_single_profile_artifact_cannot_flip_delta_policy_to_pass() -> None:
    assessment = assess_population_risk(
        [{"age": 30}, {"age": 30}],
        [{"age": 30}, {"age": 30}],
        ("age",),
        target_k_map=2,
        max_delta_presence=0.5,
    )
    assert assessment.meets_policy is False
    tampered: dict[str, object] = assessment.to_dict()
    tampered.update(
        {
            "max_delta_presence": 0.5,
            "mean_delta_presence": 0.5,
            "meets_delta_presence": True,
            "meets_policy": True,
        }
    )
    _rehash_population_payload(tampered)

    with pytest.raises(ValueError, match="delta-presence is below"):
        PopulationRiskAssessment.from_dict(tampered)


def test_rehashed_nonviolating_singletons_cannot_exceed_matched_profiles() -> None:
    rows = [{"region": "a"}] * 4 + [{"region": "b"}]
    assessment = assess_population_risk(
        rows,
        rows,
        ("region",),
        target_k_map=1,
        max_delta_presence=1.0,
    )
    assert assessment.population_singleton_count == 1
    tampered: dict[str, object] = assessment.to_dict()
    tampered.update(
        {
            "population_singleton_count": 4,
            "population_singleton_rate": 0.8,
            "mean_exact_linkage_risk": 0.84,
        }
    )
    _rehash_population_payload(tampered)

    with pytest.raises(ValueError, match="nonviolating matched profile bound"):
        PopulationRiskAssessment.from_dict(tampered)


def test_population_schema_version_requires_an_exact_integer() -> None:
    assessment = assess_population_risk(
        [{"age": 30}],
        [{"age": 30}],
        ("age",),
        target_k_map=1,
        max_delta_presence=1.0,
    )
    tampered: dict[str, object] = assessment.to_dict()
    tampered["schema_version"] = 1.0
    _rehash_population_payload(tampered)

    with pytest.raises(TypeError, match="must be an integer"):
        PopulationRiskAssessment.from_dict(tampered)


def test_saved_artifact_parser_rejects_schema_smuggling() -> None:
    assessment = assess_population_risk(
        [{"age": 30}],
        [{"age": 30}],
        ("age",),
        target_k_map=1,
        max_delta_presence=1.0,
    )
    unknown = {**assessment.to_dict(), "raw_profiles": ["CANARY"]}
    missing = assessment.to_dict()
    del missing["policy_digest"]

    with pytest.raises(ValueError, match="missing or unknown"):
        PopulationRiskAssessment.from_dict(unknown)
    with pytest.raises(ValueError, match="missing or unknown"):
        PopulationRiskAssessment.from_dict(missing)

    duplicate_key_json = '{"schema_version":1,' + assessment.to_json(indent=None)[1:]
    with pytest.raises(ValueError, match="invalid population-risk JSON"):
        PopulationRiskAssessment.from_json(duplicate_key_json)
    with pytest.raises(ValueError, match="must be an object"):
        PopulationRiskAssessment.from_json("[]")


def test_safe_unicode_and_punctuation_column_names_are_supported() -> None:
    assessment = assess_population_risk(
        [{"patient key": "sample-a", "postal code, région": "75001"}],
        [
            {"population key": "population-a", "postal code, région": "75001"},
            {"population key": "population-b", "postal code, région": "75001"},
        ],
        ("postal code, région",),
        sample_privacy_unit="patient key",
        population_privacy_unit="population key",
        target_k_map=2,
        max_delta_presence=1.0,
    )

    assert assessment.achieved_k_map == 2
    assert assessment.meets_policy is True


@pytest.mark.parametrize(
    ("sample", "reference_population", "message"),
    (
        ([], [{"age": 30}], "sample must contain"),
        ([{"age": 30}], [], "reference_population must contain"),
        ([{}], [{"age": 30}], "empty row"),
        ([{"other": 30}], [{"age": 30}], "quasi-identifier"),
        ([{"age": 30}], [{"other": 30}], "quasi-identifier"),
        ([{"age": 30, "extra": ["nested"]}], [{"age": 30}], "unsupported"),
    ),
)
def test_invalid_rows_are_rejected(
    sample: list[dict[str, object]],
    reference_population: list[dict[str, object]],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        assess_population_risk(
            sample,
            reference_population,
            ("age",),
            target_k_map=1,
            max_delta_presence=1.0,
        )


def test_invalid_privacy_unit_models_are_rejected() -> None:
    with pytest.raises(ValueError, match="compatible"):
        assess_population_risk(
            [{"sample_id": "sample-a", "age": 30}],
            [{"age": 30}],
            ("age",),
            sample_privacy_unit="sample_id",
            target_k_map=1,
            max_delta_presence=1.0,
        )
    with pytest.raises(ValueError, match="cannot also"):
        assess_population_risk(
            [{"sample_id": "sample-a"}],
            [{"population_id": "population-a"}],
            ("sample_id",),
            sample_privacy_unit="sample_id",
            population_privacy_unit="population_id",
            target_k_map=1,
            max_delta_presence=1.0,
        )
    with pytest.raises(ValueError, match="non-empty"):
        assess_population_risk(
            [{"sample_id": "", "age": 30}],
            [{"population_id": "population-a", "age": 30}],
            ("age",),
            sample_privacy_unit="sample_id",
            population_privacy_unit="population_id",
            target_k_map=1,
            max_delta_presence=1.0,
        )


@pytest.mark.parametrize(
    ("sample_unit", "population_unit", "message"),
    (
        (" ", "population-a", "sample privacy unit.*non-empty"),
        ("\tpatient-a", "population-a", "sample privacy unit.*surrounding"),
        ("patient-a ", "population-a", "sample privacy unit.*surrounding"),
        (b"", b"population-a", "sample privacy unit.*non-empty"),
        (b" ", b"population-a", "sample privacy unit.*non-empty"),
        (b"patient-a ", b"population-a", "sample privacy unit.*surrounding"),
        ("patient-a", "\t", "reference_population privacy unit.*non-empty"),
        (
            b"patient-a",
            b" population-a",
            "reference_population privacy unit.*surrounding",
        ),
    ),
)
def test_keyed_privacy_units_require_canonical_non_whitespace_values(
    sample_unit: object,
    population_unit: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        assess_population_risk(
            [{"sample_id": sample_unit, "age": 30}],
            [{"population_id": population_unit, "age": 30}],
            ("age",),
            sample_privacy_unit="sample_id",
            population_privacy_unit="population_id",
            target_k_map=1,
            max_delta_presence=1.0,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({}, "target_k_map must be explicitly set"),
        ({"target_k_map": 1}, "max_delta_presence must be explicitly set"),
        ({"max_delta_presence": 1.0}, "target_k_map must be explicitly set"),
    ),
)
def test_population_policy_thresholds_must_be_explicit(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        assess_population_risk(
            [{"age": 30}],
            [{"age": 30}],
            ("age",),
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("target_k_map", (True, 0, -1, 1.5))
def test_invalid_k_map_threshold_is_rejected(target_k_map: object) -> None:
    with pytest.raises(ValueError, match="target_k_map"):
        assess_population_risk(
            [{"age": 30}],
            [{"age": 30}],
            ("age",),
            target_k_map=target_k_map,  # type: ignore[arg-type]
            max_delta_presence=1.0,
        )


@pytest.mark.parametrize(
    "max_delta_presence",
    (True, -0.01, 1.01, math.inf, math.nan, "0.5"),
)
def test_invalid_delta_presence_threshold_is_rejected(
    max_delta_presence: object,
) -> None:
    with pytest.raises(ValueError, match="max_delta_presence"):
        assess_population_risk(
            [{"age": 30}],
            [{"age": 30}],
            ("age",),
            target_k_map=1,
            max_delta_presence=max_delta_presence,  # type: ignore[arg-type]
        )
