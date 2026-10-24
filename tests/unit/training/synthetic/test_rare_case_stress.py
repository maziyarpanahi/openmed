"""Synthetic, offline coverage for rare-case privacy stress generation."""

from __future__ import annotations

import json
import logging
import math
import socket
from datetime import date

import pytest

from openmed.core.labels import (
    DATE,
    DATE_OF_BIRTH,
    DIRECT_IDENTIFIER,
    QUASI_IDENTIFIER,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
)
from openmed.training.synthetic.rare_case_stress import (
    PREVALENCE_TIERS,
    RARE_CASE_DIMENSIONS,
    RARE_CASE_DIRECT_IDENTIFIER_FIELDS,
    RARE_CASE_QUASI_IDENTIFIER_FIELDS,
    SAFE_HARBOR_FLAGS,
    SYNTHETIC_RARE_CASE_SOURCE,
    SYNTHETIC_REFERENCE_POPULATION,
    RareCaseProfile,
    RareCaseStressConfig,
    default_rare_case_profiles,
    generate_rare_case_stress_set,
    rare_case_vocabulary,
)

_CANARY = "canary-7f3a9c"
_TIER_LEVELS = {"common": RISK_LOW, "uncommon": RISK_MEDIUM, "rare": RISK_HIGH}


def _stress_set(count: int = 30, *, seed: int = 11, **config_kwargs):
    config_kwargs.setdefault("background_classes", 8)
    return generate_rare_case_stress_set(
        count, seed=seed, config=RareCaseStressConfig(**config_kwargs)
    )


def _string_values(stress_set) -> set[str]:
    """Generated non-numeric values that must never appear in reports."""

    values = set()
    for record in stress_set.records:
        for attribute in record.attributes:
            value = str(attribute.value)
            if not value.isdigit():
                values.add(value)
    return values


def test_fixed_seed_reproduces_records_and_digests() -> None:
    first = _stress_set(seed=91)
    second = _stress_set(seed=91)
    other = _stress_set(seed=92)

    assert first == second
    assert first.to_dicts() == second.to_dicts()
    assert first.summary() == second.summary()
    assert first.dataset_hash() == second.dataset_hash()
    assert first.dataset_hash() != other.dataset_hash()
    assert first.summary()["dataset_hash"].startswith("sha256:")


def test_generation_needs_no_network(monkeypatch) -> None:
    def fail_network(*_args, **_kwargs):
        raise AssertionError("rare-case stress generation attempted network access")

    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.setattr(socket, "create_connection", fail_network)

    stress_set = _stress_set(45)

    assert len(stress_set.records) == 45 + 8 * 3


def test_generation_emits_no_log_records(caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        _stress_set(20).summary()

    assert caplog.records == []


def test_default_profiles_cover_every_pairwise_rare_combination() -> None:
    profiles = default_rare_case_profiles()
    stress_set = generate_rare_case_stress_set(len(profiles), seed=3)

    assert len(profiles) == 15
    assert len({profile.profile_id for profile in profiles}) == 15
    assert stress_set.summary()["profile_counts"] == {
        profile.profile_id: 1 for profile in profiles
    }
    by_profile = {record.profile_id: record for record in stress_set.records}
    for profile in profiles:
        record = by_profile[profile.profile_id]
        assert record.case_kind == "stress"
        assert record.rare_dimensions == profile.rare
        for attribute in record.stress_attributes:
            assert attribute.tier == profile.tier_for(attribute.dimension)

    assert [len(default_rare_case_profiles(order)) for order in range(1, 7)] == [
        6,
        15,
        20,
        15,
        6,
        1,
    ]


def test_configured_profiles_draw_the_requested_tiers() -> None:
    profile = RareCaseProfile.from_tiers(
        {"occupation": "rare", "age": "uncommon", "geography": "common"}
    )
    assert profile == RareCaseProfile(rare=("occupation",), uncommon=("age",))
    assert profile.profile_id == "rare:occupation;uncommon:age"
    assert RareCaseProfile().profile_id == "common"
    assert RareCaseProfile(rare=("family_relationship", "age", "age")).rare == (
        "age",
        "family_relationship",
    )

    stress_set = _stress_set(12, profiles=(profile,), background_classes=0)
    for record in stress_set.records:
        tiers = {item.dimension: item.tier for item in record.stress_attributes}
        assert tiers == {
            "geography": "common",
            "age": "uncommon",
            "date": "common",
            "diagnosis": "common",
            "occupation": "rare",
            "family_relationship": "common",
        }
        age = record.to_row()["age"]
        assert 85 <= age <= 89


def test_vocabulary_tiers_are_complete_ordered_and_value_free() -> None:
    vocabulary = rare_case_vocabulary()

    assert tuple(vocabulary) == RARE_CASE_DIMENSIONS
    codes = [entry["code"] for entries in vocabulary.values() for entry in entries]
    assert len(codes) == len(set(codes))
    for dimension, entries in vocabulary.items():
        by_tier = {
            tier: [entry["frequency"] for entry in entries if entry["tier"] == tier]
            for tier in PREVALENCE_TIERS
        }
        assert all(by_tier.values()), dimension
        assert min(by_tier["common"]) > max(by_tier["uncommon"])
        assert min(by_tier["uncommon"]) > max(by_tier["rare"])
        assert all(0 < frequency <= 1 for frequency in by_tier["common"])
        for entry in entries:
            assert entry["code"].startswith(f"{dimension}.{entry['tier']}.")

    serialized = json.dumps(vocabulary)
    for value in _string_values(_stress_set(60)):
        assert value not in serialized


def test_risk_contributions_follow_the_synthetic_priors() -> None:
    stress_set = _stress_set(60)

    for record in stress_set.records:
        stress = record.stress_attributes
        assert [item.dimension for item in stress] == list(RARE_CASE_DIMENSIONS)
        assert [item.field for item in stress] == list(
            RARE_CASE_QUASI_IDENTIFIER_FIELDS
        )
        for item in stress:
            assert item.surprisal_bits == pytest.approx(
                -math.log2(item.frequency), abs=1e-4
            )
            assert item.risk_level == _TIER_LEVELS[item.tier]
            assert item.sensitive_attribute is (item.dimension == "diagnosis")
        assert record.combined_surprisal_bits == pytest.approx(
            sum(item.surprisal_bits for item in stress), abs=1e-4
        )
        assert sum(item.contribution_share for item in stress) == pytest.approx(
            1.0, abs=1e-4
        )
        dominant = max(stress, key=lambda item: item.surprisal_bits)
        assert record.dominant_dimension == dominant.dimension
        assert record.expected_population_matches == pytest.approx(
            SYNTHETIC_REFERENCE_POPULATION
            * math.prod(item.frequency for item in stress),
            rel=1e-4,
        )
        assert 0.0 <= record.population_unique_probability <= 1.0

        if record.case_kind == "background":
            assert record.rare_dimensions == ()
            assert record.expected_population_matches > 1.0
        else:
            # Any two rare dimensions push the tuple below one expected match.
            assert len(record.rare_dimensions) == 2
            assert record.expected_population_matches < 1.0
            assert record.population_unique_probability > 0.5

    background = [r for r in stress_set.records if r.case_kind == "background"]
    stress = [r for r in stress_set.records if r.case_kind == "stress"]
    assert min(r.combined_surprisal_bits for r in stress) > max(
        r.combined_surprisal_bits for r in background
    )


def test_identifier_classes_reuse_core_label_policy() -> None:
    stress_set = _stress_set(60)

    seen_dates = set()
    for record in stress_set.records:
        direct = [
            item for item in record.attributes if item.dimension == "direct_identifier"
        ]
        assert [item.field for item in direct] == list(
            RARE_CASE_DIRECT_IDENTIFIER_FIELDS
        )
        for item in direct:
            assert item.identifier_class == DIRECT_IDENTIFIER
            assert item.risk_level == RISK_HIGH
            assert item.tier is None and item.frequency is None
        birth_date = next(
            item for item in record.attributes if item.dimension == "date"
        )
        seen_dates.add(birth_date.tier)
        if birth_date.tier == "rare":
            assert birth_date.label == DATE_OF_BIRTH
            assert birth_date.identifier_class == DIRECT_IDENTIFIER
            assert birth_date in record.direct_identifiers
        else:
            assert birth_date.label == DATE
            assert birth_date.identifier_class == QUASI_IDENTIFIER
            assert birth_date in record.quasi_identifiers
        for item in record.stress_attributes:
            if item.dimension != "date":
                assert item.identifier_class == QUASI_IDENTIFIER
        family = record.stress_attributes[-1]
        assert family.label is None
        assert family.relative_role in {"spouse", "sibling", "child"}
    assert {"common", "rare"} <= seen_dates


def test_family_roles_follow_the_family_history_vocabulary() -> None:
    from openmed.clinical.family_history import _CANONICAL_RELATIVE_ROLES

    stress_set = _stress_set(
        30, profiles=(RareCaseProfile(rare=("family_relationship",)),)
    )
    roles = {
        record.stress_attributes[-1].relative_role for record in stress_set.records
    }

    assert roles <= _CANONICAL_RELATIVE_ROLES


def test_spans_rows_and_training_records_align() -> None:
    stress_set = _stress_set(30)

    for record in stress_set.records:
        assert record.text.startswith("Synthetic rare-case stress note.\n")
        row = record.to_row()
        assert row["record_id"] == record.record_id
        for attribute in record.attributes:
            start, end = attribute.span
            assert record.text[start:end] == str(attribute.value)
            assert row[attribute.field] == attribute.value
        assert isinstance(row["age"], int)

        payload = record.to_dict()
        json.dumps(payload, allow_nan=False)
        assert payload["metadata"] == {
            "synthetic": True,
            "source": SYNTHETIC_RARE_CASE_SOURCE,
            "restricted_data": False,
            "schema_version": "openmed.training.rare_case_stress.v1",
        }
        assert payload["fields"] == {
            key: value for key, value in row.items() if key != "record_id"
        }
        for span in payload["spans"]:
            assert span["label"] is not None
            text = payload["text"][span["start"] : span["end"]]
            assert text == str(payload["fields"][span["field"]])
        assert len(payload["spans"]) == len(record.attributes) - 1


def test_birth_dates_are_valid_and_consistent_with_age() -> None:
    for reference in (date(2025, 6, 30), date(2024, 1, 15)):
        stress_set = _stress_set(
            90,
            profiles=default_rare_case_profiles(1),
            reference_date=reference,
        )
        leap_days = 0
        for record in stress_set.records:
            row = record.to_row()
            parts = [int(part) for part in row["birth_date"].split("-")]
            if len(parts) == 3:
                born = date(*parts)
                leap_days += (born.month, born.day) == (2, 29)
                age = reference.year - born.year
                age -= (reference.month, reference.day) < (born.month, born.day)
                assert age == row["age"]
            else:
                assert reference.year - parts[0] - row["age"] in {0, 1}
        assert leap_days > 0


def test_impossible_leap_day_stressor_fails_closed() -> None:
    config = RareCaseStressConfig(
        profiles=(RareCaseProfile(rare=("date",), uncommon=("age",)),),
        reference_date=date(1988, 6, 30),
    )

    with pytest.raises(ValueError, match="leap-day") as error:
        generate_rare_case_stress_set(64, seed=5, config=config)
    assert "1899" not in str(error.value)


def test_safe_harbor_flags_describe_the_released_values() -> None:
    stress_set = _stress_set(90)

    counts = stress_set.summary()["safe_harbor_flag_counts"]
    assert set(counts) <= set(SAFE_HARBOR_FLAGS)
    for flag in ("age_over_89", "small_geographic_unit", "name"):
        assert counts[flag] > 0
    for record in stress_set.records:
        row = record.to_row()
        flags = {
            item.dimension: set(item.safe_harbor_flags)
            for item in record.stress_attributes
        }
        assert ("age_over_89" in flags["age"]) is (row["age"] > 89)
        assert ("date_indicates_age_over_89" in flags["date"]) is (row["age"] > 89)
        assert ("date_more_specific_than_year" in flags["date"]) is (
            "-" in row["birth_date"]
        )
        geography = record.stress_attributes[0]
        small = "small_geographic_unit" in flags["geography"]
        assert small is (geography.frequency * SYNTHETIC_REFERENCE_POPULATION <= 20_000)


def test_background_classes_are_k_anonymous() -> None:
    stress_set = _stress_set(15, background_classes=6, background_class_size=4)
    summary = stress_set.summary()

    background = [r for r in stress_set.records if r.case_kind == "background"]
    assert len(background) == 24
    assert all(record.equivalence_class_size >= 4 for record in background)
    assert summary["background_record_count"] == 24
    assert summary["stress_record_count"] == 15
    assert summary["stress_singleton_count"] == 15
    assert summary["dataset_singleton_count"] == 15
    assert summary["min_equivalence_class_size"] == 1


def test_truth_labels_match_openmed_privacy_attack_suites() -> None:
    from openmed.eval.attacks import linkage_attack
    from openmed.risk.reid import risk_report

    stress_set = _stress_set(45, background_classes=10)
    released = stress_set.rows(include_direct_identifiers=False)
    truth = {r.record_id for r in stress_set.records if r.is_dataset_singleton}

    assert all("patient_name" not in row for row in released)
    report = risk_report(
        released, quasi_identifier_fields=stress_set.quasi_identifier_fields
    )
    assert {item["record_id"] for item in report["singleton_records"]} == truth
    assert report["k_min"] == min(r.equivalence_class_size for r in stress_set.records)

    # Linking the release back to the identified rows re-identifies exactly the
    # designed singletons; background classes stay ambiguous.
    attack = linkage_attack(
        released,
        stress_set.rows(),
        quasi_identifiers=stress_set.quasi_identifier_fields,
    )
    assert attack.unique_matches == len(truth)
    assert attack.no_matches == 0
    outcomes = {detail["record_id"]: detail["outcome"] for detail in attack.details}
    for record in stress_set.records:
        expected = "unique" if record.is_dataset_singleton else "ambiguous"
        assert outcomes[record.record_id] == expected

    text_report = risk_report(list(stress_set.to_dicts()))
    categories = {item["category"] for item in text_report["quasi_identifiers"]}
    assert {"age", "date", "geography", "rare_condition"} <= categories


def test_direct_identifiers_can_be_disabled() -> None:
    stress_set = _stress_set(20, include_direct_identifiers=False)

    assert stress_set.direct_identifier_fields == ()
    assert stress_set.summary()["direct_identifier_fields"] == []
    for record in stress_set.records:
        assert all(item.dimension != "direct_identifier" for item in record.attributes)
        assert "Patient:" not in record.text and "MRN:" not in record.text
        assert set(record.to_row()) == {"record_id", *RARE_CASE_QUASI_IDENTIFIER_FIELDS}


def test_reports_and_reprs_contain_no_generated_values() -> None:
    stress_set = _stress_set(60)
    values = _string_values(stress_set)
    assert values

    reports = json.dumps(
        {
            "summary": stress_set.summary(),
            "labels": [record.risk_labels() for record in stress_set.records],
            "config": stress_set.config.to_dict(),
        },
        allow_nan=False,
    )
    rendered = reports + repr(stress_set)
    for value in values:
        assert value not in rendered
    assert '"value"' not in reports
    assert '"text"' not in reports
    assert "Synthetic rare-case stress note" not in rendered


@pytest.mark.parametrize(
    "call",
    [
        lambda: generate_rare_case_stress_set(-1),
        lambda: generate_rare_case_stress_set(True),
        lambda: generate_rare_case_stress_set(1, seed=_CANARY),
        lambda: generate_rare_case_stress_set(1, config={"seed": _CANARY}),
        lambda: RareCaseProfile(rare=(_CANARY,)),
        lambda: RareCaseProfile(rare=_CANARY),
        lambda: RareCaseProfile(rare=("age",), uncommon=("age",)),
        lambda: RareCaseProfile.from_tiers({"age": _CANARY}),
        lambda: RareCaseProfile().tier_for(_CANARY),
        lambda: RareCaseStressConfig(profiles=(_CANARY,)),
        lambda: RareCaseStressConfig(background_classes=-1),
        lambda: RareCaseStressConfig(background_class_size=1),
        lambda: RareCaseStressConfig(include_direct_identifiers=_CANARY),
        lambda: RareCaseStressConfig(reference_date=_CANARY),
        lambda: RareCaseStressConfig(reference_date=date(1850, 1, 1)),
        lambda: default_rare_case_profiles(7),
    ],
)
def test_invalid_inputs_fail_without_echoing_values(call) -> None:
    with pytest.raises((TypeError, ValueError)) as error:
        call()

    assert _CANARY not in str(error.value)
