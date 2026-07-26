"""Tests for patient-level release assessment and anonymization."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from openmed.risk import (
    AnonymityPolicy,
    anonymize_release,
    assess_release,
    release_dataset_digest,
    risk_report,
    safe_risk_summary,
    validate_released_output,
)
from openmed.structured import read_table, write_table


def _balanced_rows() -> list[dict[str, object]]:
    return [
        {
            "patient_id": "p-001",
            "patient_name": "Alice Example",
            "age": 31,
            "zip": "10001",
            "visit_date": "2024-01-01",
            "disease": "flu",
        },
        {
            "patient_id": "p-002",
            "patient_name": "Bob Example",
            "age": 32,
            "zip": "10002",
            "visit_date": "2024-01-02",
            "disease": "cold",
        },
        {
            "patient_id": "p-003",
            "patient_name": "Carol Example",
            "age": 41,
            "zip": "20001",
            "visit_date": "2024-01-03",
            "disease": "flu",
        },
        {
            "patient_id": "p-004",
            "patient_name": "Dan Example",
            "age": 42,
            "zip": "20002",
            "visit_date": "2024-01-04",
            "disease": "cold",
        },
    ]


def _policy(**overrides) -> AnonymityPolicy:
    values = {
        "quasi_identifiers": ("age", "zip", "visit_date"),
        "sensitive_attributes": ("disease",),
        "direct_identifiers": ("patient_name",),
        "privacy_unit": "patient_id",
        "target_k": 2,
        "target_l": 2,
        "target_t": 1.0,
    }
    values.update(overrides)
    return AnonymityPolicy(**values)


def test_assessment_is_aggregate_safe_and_deterministic() -> None:
    rows = [
        {"age": 30, "zip": "10001", "disease": "canary-flu"},
        {"age": 30, "zip": "10001", "disease": "canary-cold"},
        {"age": 40, "zip": "20001", "disease": "canary-flu"},
        {"age": 40, "zip": "20001", "disease": "canary-cold"},
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("age", "zip"),
        sensitive_attributes=("disease",),
        target_k=2,
        target_l=2,
        target_t=0.0,
    )

    first = assess_release(rows, policy)
    second = assess_release(rows, policy)

    assert first == second
    assert first.meets_policy is True
    assert first.achieved_k == 2
    assert first.singleton_class_count == 0
    assert first.attributes[0].achieved_l == 2
    assert first.attributes[0].achieved_t == 0.0
    payload = first.to_json()
    assert "canary-flu" not in payload
    assert "canary-cold" not in payload
    assert "10001" not in payload
    assert "equivalence_classes" not in payload
    assert "members" not in payload


def test_repeated_encounters_do_not_inflate_patient_level_k() -> None:
    rows = [
        {
            "patient_id": "patient-a",
            "age": 50,
            "zip": "10001",
            "disease": "a",
        },
        {
            "patient_id": "patient-a",
            "age": 50,
            "zip": "10001",
            "disease": "b",
        },
        {
            "patient_id": "patient-b",
            "age": 50,
            "zip": "10001",
            "disease": "a",
        },
        {
            "patient_id": "patient-b",
            "age": 50,
            "zip": "10001",
            "disease": "b",
        },
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("age", "zip"),
        sensitive_attributes=("disease",),
        privacy_unit="patient_id",
        target_k=2,
    )

    report = assess_release(rows, policy)

    assert report.row_count == 4
    assert report.privacy_unit_count == 2
    assert report.achieved_k == 2
    assert report.meets_policy is True


def test_multivalued_patient_qis_use_joint_profiles_and_disclose_a_warning() -> None:
    rows = [
        {"patient_id": "a", "facility": "north", "disease": "x"},
        {"patient_id": "a", "facility": "south", "disease": "x"},
        {"patient_id": "b", "facility": "north", "disease": "x"},
        {"patient_id": "b", "facility": "south", "disease": "x"},
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("facility",),
        sensitive_attributes=("disease",),
        privacy_unit="patient_id",
        target_k=2,
    )

    report = assess_release(rows, policy)

    assert report.achieved_k == 2
    assert any(
        "joint ordered-multiset fingerprints: facility" in item
        for item in report.warnings
    )
    serialized = report.to_json()
    assert "north" not in serialized
    assert "south" not in serialized


def test_crossed_longitudinal_qi_pairings_remain_distinct() -> None:
    rows = [
        {"patient_id": "a", "age": 30, "facility": "north"},
        {"patient_id": "a", "age": 40, "facility": "south"},
        {"patient_id": "b", "age": 30, "facility": "south"},
        {"patient_id": "b", "age": 40, "facility": "north"},
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("age", "facility"),
        privacy_unit="patient_id",
        target_k=2,
    )

    report = assess_release(rows, policy)

    assert report.privacy_unit_count == 2
    assert report.class_count == 2
    assert report.achieved_k == 1
    assert report.meets_policy is False


def test_longitudinal_qi_profiles_preserve_repeated_row_multiplicity() -> None:
    rows = [
        {"patient_id": "a", "facility": "north"},
        {"patient_id": "a", "facility": "north"},
        {"patient_id": "b", "facility": "north"},
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("facility",),
        privacy_unit="patient_id",
        target_k=2,
    )

    report = assess_release(rows, policy)

    assert report.privacy_unit_count == 2
    assert report.class_count == 2
    assert report.achieved_k == 1
    assert report.meets_policy is False
    assert any(
        "row multiplicity can distinguish privacy units" in item
        for item in report.warnings
    )


def test_multivalued_patient_qis_cannot_collapse_to_their_first_value() -> None:
    rows = [
        {"patient_id": "a", "age": 30},
        {"patient_id": "a", "age": 40},
        {"patient_id": "b", "age": 30},
        {"patient_id": "b", "age": 30},
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("age",),
        privacy_unit="patient_id",
        target_k=2,
    )

    report = assess_release(rows, policy)

    assert report.privacy_unit_count == 2
    assert report.achieved_k == 1
    assert report.meets_policy is False


def test_longitudinal_profiles_recanonicalize_after_qi_coarsening() -> None:
    rows = [
        {"patient_id": "a", "age": 30, "facility": "north"},
        {"patient_id": "a", "age": 40, "facility": "south"},
        {"patient_id": "b", "age": 30, "facility": "south"},
        {"patient_id": "b", "age": 40, "facility": "north"},
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("age", "facility"),
        privacy_unit="patient_id",
        target_k=2,
    )

    result = anonymize_release(
        rows,
        policy,
        hierarchies={
            "age": [
                {"name": "exact", "loss": 0.0},
                {
                    "name": "all-ages",
                    "loss": 0.5,
                    "values": {"30": "all", "40": "all"},
                },
            ],
            "facility": [{"name": "exact", "loss": 0.0}],
        },
    )

    assert result.after.meets_policy is True
    assert result.after.achieved_k == 2
    assert {row["age"] for row in result.records} == {"all"}
    assert {row["facility"] for row in result.records} == {"north", "south"}


def test_full_qi_suppression_preserves_longitudinal_row_multiplicity() -> None:
    rows = [
        {"patient_id": "a", "facility": "north"},
        {"patient_id": "a", "facility": "north"},
        {"patient_id": "b", "facility": "north"},
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("facility",),
        privacy_unit="patient_id",
        target_k=2,
    )

    with pytest.raises(ValueError, match="No generalization satisfies"):
        anonymize_release(rows, policy)


def test_missing_null_and_empty_patient_qis_remain_distinct() -> None:
    rows = [
        {"patient_id": "a"},
        {"patient_id": "b", "age": None},
        {"patient_id": "c", "age": ""},
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("age",),
        privacy_unit="patient_id",
        target_k=2,
    )

    report = assess_release(rows, policy)

    assert report.achieved_k == 1
    assert report.class_count == 3


def test_anonymization_revalidates_and_separates_sensitive_records() -> None:
    rows = [
        {**row, "source_account": f"acct-{index}"}
        for index, row in enumerate(_balanced_rows())
    ]

    result = anonymize_release(
        rows,
        _policy(direct_identifiers=("patient_name", "source_account")),
    )

    assert result.after.meets_policy is True
    assert result.after.achieved_k >= 2
    assert len(result.records) == len(rows)
    assert all("patient_id" not in row for row in result.records)
    assert all("patient_name" not in row for row in result.records)
    assert all("source_account" not in row for row in result.records)
    assert result.generalization.optimum_proven is True
    assert result.generalization.nodes_evaluated > 0
    assert result.generalization.search_complete is True
    assert result.generalization.suppression_subsets_evaluated > 0
    assert (
        result.generalization.suppression_subsets_evaluated
        == result.generalization.suppression_subsets_possible
    )
    assert result.hierarchy_digest.startswith("sha256:")
    assert result.utility.released_rows == len(rows)
    safe = result.to_safe_json()
    for canary in ("p-001", "Alice Example", "10001", "2024-01-01"):
        assert canary not in safe
    assert '"records"' not in safe

    validation = validate_released_output(result.records, result)
    assert validation.passed is True
    assert validation.digest_matches is True
    assert validation.direct_identifier_columns == ()


def test_anonymization_result_repr_excludes_sensitive_records() -> None:
    rows = [
        {
            **row,
            "disease": f"repr-sensitive-canary-{index}",
        }
        for index, row in enumerate(_balanced_rows())
    ]
    result = anonymize_release(
        rows,
        _policy(target_l=1),
    )

    rendered = repr(result)

    assert "records=" not in rendered
    assert "repr-sensitive-canary" not in rendered


def test_hierarchy_digest_binds_supplied_hierarchy_configuration() -> None:
    rows = [
        {"patient_id": "a", "facility": "north"},
        {"patient_id": "b", "facility": "south"},
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("facility",),
        privacy_unit="patient_id",
        target_k=1,
    )

    default = anonymize_release(rows, policy)
    supplied = anonymize_release(
        rows,
        policy,
        hierarchies={
            "facility": [
                {
                    "name": "Alice Canary exact",
                    "loss": 0.0,
                },
                {"name": "suppressed", "loss": 1.0, "default": "*"},
            ]
        },
    )

    assert default.hierarchy_digest != supplied.hierarchy_digest
    assert supplied.to_safe_dict()["hierarchy_digest"] == supplied.hierarchy_digest
    assert "Alice Canary" not in supplied.to_safe_json()


def test_coarsening_has_nonzero_loss_and_utility_evidence() -> None:
    rows = [
        {"patient_id": "a", "facility": "north"},
        {"patient_id": "b", "facility": "south"},
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("facility",),
        privacy_unit="patient_id",
        target_k=2,
    )

    result = anonymize_release(
        rows,
        policy,
        hierarchies={
            "facility": [
                {"name": "exact", "loss": 0.0},
                {"name": "collapsed", "loss": 0.5, "default": "*"},
            ]
        },
    )

    assert {row["facility"] for row in result.records} == {"*"}
    assert result.generalization.levels[0][1] == 1
    assert result.generalization.generalization_loss == pytest.approx(0.5)
    assert result.generalization.information_loss == pytest.approx(0.5)
    assert result.utility.quasi_identifier_cell_change_rate == 1.0


def test_multi_valued_qi_fingerprint_cannot_collide_with_a_literal() -> None:
    import openmed.risk.kanon as kanon_module
    import openmed.risk.release as release_module

    internal_value, is_multi = release_module._projected_value(
        [{"code": "A"}, {"code": "B"}],
        "code",
        quasi_identifier=True,
    )
    assert is_multi is True
    injected_literal = kanon_module._exact_qi_value("code", internal_value)
    rows = [
        {"patient_id": "p1", "code": "A"},
        {"patient_id": "p1", "code": "B"},
        {"patient_id": "p2", "code": injected_literal},
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("code",),
        privacy_unit="patient_id",
        target_k=2,
    )

    assessment = assess_release(rows, policy)

    assert assessment.privacy_unit_count == 2
    assert assessment.class_count == 2
    assert assessment.achieved_k == 1
    assert assessment.meets_policy is False


@pytest.mark.parametrize(
    ("rows", "missing_count"),
    [
        (
            [
                {"patient_id": "a"},
                {"patient_id": "b", "value": "present"},
            ],
            1,
        ),
        (
            [
                {"patient_id": "a", "value": None},
                {"patient_id": "b", "value": None},
            ],
            2,
        ),
        (
            [
                {"patient_id": "a", "value": ""},
                {"patient_id": "b", "value": ""},
            ],
            2,
        ),
    ],
)
def test_exact_level_preserves_missing_state_utility(
    rows: list[dict[str, object]],
    missing_count: int,
) -> None:
    policy = AnonymityPolicy(
        quasi_identifiers=("value",),
        privacy_unit="patient_id",
        target_k=1,
    )

    result = anonymize_release(rows, policy)

    assert result.generalization.levels[0][1] == 0
    assert result.utility.quasi_identifier_cells_changed == 0
    assert result.utility.missing_qi_cells_before == missing_count
    assert result.utility.missing_qi_cells_after == missing_count


def test_output_validation_detects_tampering_and_identifier_survival() -> None:
    result = anonymize_release(_balanced_rows(), _policy())
    tampered = [dict(row) for row in result.records]
    tampered[0]["age"] = 999
    tampered[0]["patient_id"] = "should-not-survive"

    validation = validate_released_output(tampered, result)

    assert validation.passed is False
    assert validation.digest_matches is False
    assert validation.direct_identifier_columns == ("patient_id",)
    assert "should-not-survive" not in json.dumps(validation.to_dict())


def test_manual_output_validation_cannot_pass_without_expected_digest() -> None:
    result = anonymize_release(_balanced_rows(), _policy())
    valid = validate_released_output(result.records, result)
    unbound = type(valid)(
        row_count=valid.row_count,
        expected_row_count=valid.expected_row_count,
        dataset_digest=valid.dataset_digest,
        expected_digest=None,
        schema_digest=valid.schema_digest,
        expected_schema_digest=valid.expected_schema_digest,
        direct_identifier_columns=valid.direct_identifier_columns,
        policy_revalidated_before_identifier_removal=(
            valid.policy_revalidated_before_identifier_removal
        ),
        typed_digest_comparison_available=valid.typed_digest_comparison_available,
        policy_value_encoding_preserved=valid.policy_value_encoding_preserved,
    )

    assert unbound.digest_matches is None
    assert unbound.passed is False


def test_output_validation_uses_lexical_digest_for_delimited_files() -> None:
    rows = [
        {**row, "encounter_count": index + 1}
        for index, row in enumerate(_balanced_rows())
    ]
    result = anonymize_release(
        rows,
        _policy(non_sensitive_attributes=("encounter_count",)),
    )
    reread = [
        {field: "" if value is None else str(value) for field, value in row.items()}
        for row in result.records
    ]

    valid = validate_released_output(
        reread,
        result,
        preserve_scalar_types=False,
    )
    reread[0]["encounter_count"] = "999"
    tampered = validate_released_output(
        reread,
        result,
        preserve_scalar_types=False,
    )

    assert valid.passed is True
    assert valid.digest_matches is True
    assert valid.typed_digest_comparison_available is False
    assert tampered.passed is False
    assert tampered.digest_matches is False


def test_output_validation_rejects_delimited_policy_type_collisions() -> None:
    rows = [
        {"group": "same", "disease": 1},
        {"group": "same", "disease": "1"},
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("group",),
        sensitive_attributes=("disease",),
        target_k=2,
        target_l=2,
    )
    result = anonymize_release(rows, policy)
    reread = [
        {field: "" if value is None else str(value) for field, value in row.items()}
        for row in result.records
    ]

    validation = validate_released_output(
        reread,
        result,
        preserve_scalar_types=False,
    )

    assert validation.digest_matches is True
    assert validation.policy_value_encoding_preserved is False
    assert validation.passed is False


def test_anonymization_result_records_are_immutable() -> None:
    result = anonymize_release(_balanced_rows(), _policy())

    with pytest.raises(TypeError):
        result.records[0]["age"] = 999  # type: ignore[index]

    assert validate_released_output(result.records, result).passed is True


def test_anonymization_result_rejects_cross_run_record_and_evidence_splicing() -> None:
    strong_policy = AnonymityPolicy(
        quasi_identifiers=("group",),
        sensitive_attributes=("disease",),
        target_k=2,
        target_l=2,
    )
    weak_policy = AnonymityPolicy(
        quasi_identifiers=("group",),
        sensitive_attributes=("disease",),
        target_k=2,
        target_l=1,
    )
    strong = anonymize_release(
        [
            {"group": "same", "disease": "flu"},
            {"group": "same", "disease": "cold"},
        ],
        strong_policy,
    )
    weak = anonymize_release(
        [
            {"group": "same", "disease": "flu"},
            {"group": "same", "disease": "flu"},
        ],
        weak_policy,
    )

    with pytest.raises(ValueError, match="after assessment does not match"):
        replace(
            strong,
            records=weak.records,
            released_dataset_digest=weak.released_dataset_digest,
            released_schema_digest=weak.released_schema_digest,
        )


def test_signed_float_zero_cannot_inflate_release_l_diversity() -> None:
    policy = AnonymityPolicy(
        quasi_identifiers=("group",),
        sensitive_attributes=("measurement",),
        target_k=2,
        target_l=2,
    )

    assessment = assess_release(
        [
            {"group": "same", "measurement": -0.0},
            {"group": "same", "measurement": 0.0},
        ],
        policy,
    )

    assert assessment.attributes[0].achieved_l == 1
    assert assessment.meets_policy is False


def test_unicode_canonical_equivalents_cannot_inflate_release_l_diversity() -> None:
    policy = AnonymityPolicy(
        quasi_identifiers=("group",),
        sensitive_attributes=("condition",),
        target_k=2,
        target_l=2,
    )

    assessment = assess_release(
        [
            {"group": "same", "condition": "café"},
            {"group": "same", "condition": "cafe\u0301"},
        ],
        policy,
    )

    assert assessment.attributes[0].achieved_l == 1
    assert assessment.meets_policy is False


def test_privacy_unit_ids_preserve_exact_unicode_representation() -> None:
    assessment = assess_release(
        [
            {"patient_id": "café", "age": 40},
            {"patient_id": "cafe\u0301", "age": 40},
        ],
        AnonymityPolicy(
            quasi_identifiers=("age",),
            privacy_unit="patient_id",
            target_k=1,
        ),
    )

    assert assessment.row_count == 2
    assert assessment.privacy_unit_count == 2
    assert assessment.achieved_k == 2


def test_privacy_unit_unicode_aliases_cannot_create_false_longitudinal_k() -> None:
    rows = [
        {"patient_id": "é", "event": "A"},
        {"patient_id": "e\u0301", "event": "B"},
        {"patient_id": "third", "event": "A"},
        {"patient_id": "third", "event": "B"},
    ]
    assessment = assess_release(
        rows,
        AnonymityPolicy(
            quasi_identifiers=("event",),
            privacy_unit="patient_id",
            target_k=2,
        ),
    )

    assert assessment.privacy_unit_count == 3
    assert assessment.achieved_k == 1
    assert assessment.class_count == 3
    assert assessment.meets_policy is False


def test_exact_match_policy_uses_published_qi_case_and_tracks_real_changes() -> None:
    rows = [
        {"city": "Paris", "measure": 1},
        {"city": "paris", "measure": 2},
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("city",),
        non_sensitive_attributes=("measure",),
        target_k=2,
    )

    assessment = assess_release(rows, policy)
    result = anonymize_release(
        rows,
        policy,
        hierarchies={
            "city": [
                {"name": "exact", "loss": 0.0},
                {
                    "name": "case-normalized",
                    "loss": 0.5,
                    "values": {"Paris": "paris", "paris": "paris"},
                },
                {"name": "suppressed", "loss": 1.0, "default": "*"},
            ]
        },
    )

    assert assessment.achieved_k == 1
    assert assessment.meets_policy is False
    assert result.after.achieved_k == 2
    assert {row["city"] for row in result.records} == {"paris"}
    assert result.utility.quasi_identifier_cells_changed == 1
    assert result.generalization.affected_privacy_units == (("city", 1),)


def test_exact_match_policy_preserves_unicode_and_datetime_representation() -> None:
    first_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
    second_time = datetime(
        2020,
        1,
        1,
        1,
        tzinfo=timezone(timedelta(hours=1)),
    )
    rows = [
        {"city": "café", "event_time": first_time},
        {"city": "cafe\u0301", "event_time": second_time},
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("city", "event_time"),
        target_k=2,
    )

    assessment = assess_release(rows, policy)

    assert first_time == second_time
    assert assessment.achieved_k == 1
    assert assessment.class_count == 2
    assert assessment.meets_policy is False


def test_signed_zero_qis_cannot_create_false_delimited_k(tmp_path) -> None:
    rows = [{"measurement": -0.0}, {"measurement": 0.0}]
    policy = AnonymityPolicy(
        quasi_identifiers=("measurement",),
        target_k=2,
    )

    assessment = assess_release(rows, policy)
    result = anonymize_release(
        rows,
        policy,
        hierarchies={
            "measurement": [
                {"name": "exact", "loss": 0.0},
                {"name": "suppressed", "loss": 1.0, "default": "*"},
            ]
        },
    )
    path = write_table(tmp_path / "release.csv", result.records)
    reread = read_table(path)
    validation = validate_released_output(
        reread,
        result,
        preserve_scalar_types=False,
    )

    assert assessment.achieved_k == 1
    assert assessment.meets_policy is False
    assert result.after.achieved_k == 2
    assert {row["measurement"] for row in result.records} == {"*"}
    assert validation.passed is True


def test_t_closeness_uses_published_sensitive_representation() -> None:
    policy = AnonymityPolicy(
        quasi_identifiers=("group",),
        sensitive_attributes=("condition",),
        target_k=2,
        target_t=0.0,
    )

    assessment = assess_release(
        [
            {"group": "A", "condition": "café"},
            {"group": "A", "condition": "café"},
            {"group": "B", "condition": "cafe\u0301"},
            {"group": "B", "condition": "cafe\u0301"},
        ],
        policy,
    )

    assert assessment.attributes[0].achieved_l == 1
    assert assessment.attributes[0].achieved_t == pytest.approx(0.5)
    assert assessment.meets_policy is False


def test_release_digest_preserves_high_precision_decimal_values() -> None:
    first = Decimal("1.0000000000000000000000000000000000000001")
    second = Decimal("1.0000000000000000000000000000000000000002")

    assert release_dataset_digest([{"measurement": first}]) != (
        release_dataset_digest([{"measurement": second}])
    )


def test_anonymization_result_rejects_high_precision_decimal_tampering() -> None:
    first = Decimal("1.0000000000000000000000000000000000000001")
    second = Decimal("1.0000000000000000000000000000000000000002")
    policy = AnonymityPolicy(
        quasi_identifiers=("group",),
        non_sensitive_attributes=("measurement",),
        target_k=2,
    )
    result = anonymize_release(
        [
            {"group": "same", "measurement": first},
            {"group": "same", "measurement": first},
        ],
        policy,
    )
    tampered = [dict(row) for row in result.records]
    tampered[0]["measurement"] = second

    with pytest.raises(ValueError, match="released_dataset_digest does not match"):
        replace(result, records=tuple(tampered))


@pytest.mark.parametrize(
    ("source_value", "replacement_value"),
    [
        (0.0, -0.0),
        (Decimal("1.0"), Decimal("1.00")),
    ],
)
def test_anonymization_result_digest_binds_exact_qi_representation(
    source_value: object,
    replacement_value: object,
) -> None:
    result = anonymize_release(
        [{"measurement": source_value}, {"measurement": source_value}],
        AnonymityPolicy(
            quasi_identifiers=("measurement",),
            target_k=2,
        ),
    )
    tampered = tuple({"measurement": replacement_value} for _row in result.records)

    assert release_dataset_digest(tampered) != result.released_dataset_digest
    with pytest.raises(ValueError, match="released_dataset_digest does not match"):
        replace(result, records=tampered)


def test_exported_release_digests_reject_non_string_column_names() -> None:
    with pytest.raises(TypeError, match="column names must be strings"):
        release_dataset_digest([{1: "a", "1": "a"}])


def test_every_source_column_requires_an_explicit_role() -> None:
    rows = [{"age": 30, "disease": "x", "review_note": "synthetic"}]
    policy = AnonymityPolicy(
        quasi_identifiers=("age",),
        sensitive_attributes=("disease",),
        target_k=1,
    )

    with pytest.raises(ValueError, match="explicit release role"):
        assess_release(rows, policy)

    reviewed = AnonymityPolicy(
        quasi_identifiers=("age",),
        sensitive_attributes=("disease",),
        excluded_attributes=("review_note",),
        target_k=1,
    )
    result = anonymize_release(rows, reviewed)
    assert "review_note" not in result.records[0]


def test_non_string_column_names_cannot_collide_with_reviewed_roles() -> None:
    rows = [
        {"age": 30, "None": "reviewed", None: "collision-canary"},
        {"age": 30, "None": "reviewed", None: "collision-canary"},
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("age",),
        non_sensitive_attributes=("None",),
        target_k=2,
    )

    with pytest.raises(TypeError, match="column names must be strings"):
        anonymize_release(rows, policy)


def test_duplicate_dataframe_columns_cannot_collapse_before_role_review() -> None:
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    frame = pd.DataFrame(
        [
            [30, "reviewed", "collision-canary"],
            [30, "reviewed", "collision-canary"],
        ],
        columns=["age", "safe", "safe"],
    )
    policy = AnonymityPolicy(
        quasi_identifiers=("age",),
        non_sensitive_attributes=("safe",),
        target_k=2,
    )

    with pytest.raises(ValueError, match="column names must be unique"):
        anonymize_release(frame, policy)


def test_direct_dataframe_release_normalizes_pandas_timestamps() -> None:
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    frame = pd.DataFrame(
        {
            "service_date": [
                pd.Timestamp("2025-03-01T08:00:00Z"),
                pd.Timestamp("2025-03-01T08:00:00Z"),
            ]
        }
    )
    policy = AnonymityPolicy(
        quasi_identifiers=("service_date",),
        target_k=2,
    )

    report = assess_release(frame, policy)
    result = anonymize_release(frame, policy)

    assert report.achieved_k == 2
    assert isinstance(result.records[0]["service_date"], datetime)
    assert validate_released_output(result.records, result).passed is True


def test_direct_dataframe_release_rejects_sub_microsecond_timestamp_collapse() -> None:
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    frame = pd.DataFrame(
        {
            "service_date": [
                pd.Timestamp("2025-03-01T08:00:00.000000001Z"),
                pd.Timestamp("2025-03-01T08:00:00.000000002Z"),
            ]
        }
    )
    policy = AnonymityPolicy(
        quasi_identifiers=("service_date",),
        target_k=2,
    )

    with pytest.raises(ValueError, match="sub-microsecond precision"):
        assess_release(frame, policy)


def test_release_digests_support_type_preserving_clinical_scalars() -> None:
    rows = [
        {
            "group": "same",
            "service_date": date(2024, 1, 1),
            "amount": Decimal("1.20"),
            "blob": b"a",
        },
        {
            "group": "same",
            "service_date": date(2024, 1, 2),
            "amount": Decimal("2.30"),
            "blob": b"b",
        },
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("group",),
        non_sensitive_attributes=("amount", "blob", "service_date"),
        target_k=2,
    )

    result = anonymize_release(rows, policy)
    validation = validate_released_output(result.records, result)

    assert validation.passed is True
    assert validation.schema_matches is True
    assert isinstance(result.records[0]["service_date"], date)
    assert isinstance(result.records[0]["amount"], Decimal)
    assert isinstance(result.records[0]["blob"], bytes)


def test_policy_rejects_quasi_identifier_sensitive_overlap() -> None:
    with pytest.raises(ValueError, match="cannot overlap"):
        AnonymityPolicy(
            quasi_identifiers=("age",),
            sensitive_attributes=("age",),
            target_k=1,
        )


def test_multivalued_sensitive_attributes_fail_closed_for_l_diversity() -> None:
    rows = [
        {"patient_id": "a", "age": 30, "disease": "flu"},
        {"patient_id": "a", "age": 30, "disease": "cold"},
        {"patient_id": "b", "age": 30, "disease": "flu"},
        {"patient_id": "b", "age": 30, "disease": "diabetes"},
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("age",),
        sensitive_attributes=("disease",),
        privacy_unit="patient_id",
        target_k=2,
        target_l=2,
    )

    with pytest.raises(ValueError, match="Multi-valued sensitive"):
        assess_release(rows, policy)


def test_anonymization_suppresses_complete_privacy_units() -> None:
    rows = [
        {"patient_id": "a", "age": 30, "zip": "10001", "disease": "x"},
        {"patient_id": "a", "age": 30, "zip": "10001", "disease": "y"},
        {"patient_id": "b", "age": 30, "zip": "10001", "disease": "x"},
        {"patient_id": "b", "age": 30, "zip": "10001", "disease": "y"},
        {"patient_id": "outlier", "age": 99, "zip": "99999", "disease": "x"},
        {"patient_id": "outlier", "age": 99, "zip": "99999", "disease": "y"},
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("age", "zip"),
        sensitive_attributes=("disease",),
        privacy_unit="patient_id",
        target_k=2,
        suppression_limit=1,
    )

    result = anonymize_release(rows, policy)

    assert result.generalization.suppressed_privacy_units == 1
    assert result.generalization.suppressed_rows == 2
    assert result.utility.released_rows == 4
    assert len(result.records) == 4
    assert result.after.privacy_unit_count == 2


def test_anonymization_fails_closed_when_search_budget_is_too_small() -> None:
    with pytest.raises(ValueError, match="search budget"):
        anonymize_release(
            _balanced_rows(),
            _policy(max_lattice_nodes=10),
        )


def test_privacy_unit_errors_do_not_echo_identifier_values() -> None:
    rows = [
        {"patient_id": "secret-person", "age": 30, "zip": "10001"},
        {"age": 30, "zip": "10001"},
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("age", "zip"),
        privacy_unit="patient_id",
        target_k=2,
    )

    with pytest.raises(ValueError) as caught:
        assess_release(rows, policy)

    assert "secret-person" not in str(caught.value)
    assert "row offset 1" in str(caught.value)


def test_privacy_unit_surrounding_whitespace_fails_closed_without_echo() -> None:
    policy = AnonymityPolicy(
        quasi_identifiers=("age",),
        privacy_unit="patient_id",
        target_k=2,
    )

    with pytest.raises(ValueError, match="surrounding whitespace") as caught:
        assess_release(
            [
                {"patient_id": "person-a", "age": 30},
                {"patient_id": " person-a ", "age": 30},
            ],
            policy,
        )

    assert "person-a" not in str(caught.value)


def test_safe_risk_summary_drops_raw_values_keys_and_record_ids() -> None:
    detailed = risk_report(
        [
            {
                "record_id": "patient-canary",
                "age": 94,
                "city": "Riverton-canary",
            }
        ]
    )

    safe = safe_risk_summary(detailed)
    serialized = json.dumps(safe, sort_keys=True)

    assert safe["singleton_record_count"] == 1
    assert safe["quasi_identifier_count"] >= 1
    assert "patient-canary" not in serialized
    assert "Riverton-canary" not in serialized
    assert "normalized_value" not in serialized
    assert "quasi_identifier_key" not in serialized


def test_safe_risk_summary_allow_lists_categories_without_echoing_unknowns() -> None:
    safe = safe_risk_summary(
        {
            "record_count": 1,
            "quasi_identifiers": [{"category": "sensitive-category-canary"}],
        }
    )
    serialized = json.dumps(safe, sort_keys=True)

    assert safe["quasi_identifier_categories"] == {"unknown": 1}
    assert "sensitive-category-canary" not in serialized


def test_missing_sensitive_values_cannot_inflate_l_diversity() -> None:
    policy = AnonymityPolicy(
        quasi_identifiers=("age",),
        sensitive_attributes=("disease",),
        target_k=2,
        target_l=2,
    )

    with pytest.raises(ValueError, match="cannot count toward l-diversity"):
        assess_release(
            [
                {"age": 40, "disease": "flu"},
                {"age": 40, "disease": None},
            ],
            policy,
        )


@pytest.mark.parametrize("ambiguous_value", [" flu ", b""])
def test_ambiguous_sensitive_values_cannot_inflate_l_diversity(
    ambiguous_value: object,
) -> None:
    policy = AnonymityPolicy(
        quasi_identifiers=("age",),
        sensitive_attributes=("disease",),
        target_k=2,
        target_l=2,
    )

    with pytest.raises(ValueError, match="cannot count toward l-diversity"):
        assess_release(
            [
                {"age": 40, "disease": "flu"},
                {"age": 40, "disease": ambiguous_value},
            ],
            policy,
        )


def test_nonfinite_qi_and_privacy_unit_values_fail_closed() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        assess_release(
            [{"age": float("nan")}],
            AnonymityPolicy(quasi_identifiers=("age",), target_k=1),
        )
    with pytest.raises(ValueError, match="privacy_unit must be finite"):
        assess_release(
            [{"patient_id": float("inf"), "age": 40}],
            AnonymityPolicy(
                quasi_identifiers=("age",),
                privacy_unit="patient_id",
                target_k=1,
            ),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_k": 0}, "target_k"),
        ({"target_l": 2, "sensitive_attributes": ()}, "sensitive"),
        ({"target_t": 0.2, "sensitive_attributes": ()}, "sensitive"),
        ({"l_metric": "recursive"}, "l_metric"),
        ({"max_lattice_nodes": 0}, "max_lattice_nodes"),
        ({"max_suppression_subsets": 0}, "max_suppression_subsets"),
        ({"quasi_identifiers": (" Alice Canary",)}, "column names"),
        ({"privacy_unit": "patient\nidentifier"}, "column names"),
    ],
)
def test_policy_rejects_ambiguous_or_unsafe_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "quasi_identifiers": ("age",),
        "target_k": 2,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        AnonymityPolicy(**values)
