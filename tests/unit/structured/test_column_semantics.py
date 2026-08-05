"""Tests for review-first structured column semantic classification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.core.anonymizer.providers import validate_mrn as public_validate_mrn
from openmed.core.anonymizer.providers.clinical_ids import validate_mrn
from openmed.core.labels import (
    CANONICAL_LABELS,
    canonical_label_for_column_semantic,
)
from openmed.interop import get_adapter
from openmed.structured import (
    ACTION_GENERALIZE,
    ACTION_KEEP,
    ACTION_MANUAL_REVIEW,
    ACTION_ROUTE_TO_DEIDENTIFY,
    ACTION_SUPPRESS,
    ROLE_DIRECT_ID,
    ROLE_SAFE,
    classify_columns,
    write_auto_policy,
)

FIXTURE_DIRECTORY = Path(__file__).parents[2] / "fixtures" / "structured"
FIXTURE_PATH = FIXTURE_DIRECTORY / "column_semantics_40.csv"
LABELS_PATH = FIXTURE_DIRECTORY / "column_semantics_40.labels.json"


def _expected_semantics() -> dict[str, str]:
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))


def test_labeled_40_column_fixture_meets_accuracy_and_direct_id_safety() -> None:
    expected = _expected_semantics()

    policy = classify_columns(FIXTURE_PATH)
    decisions = policy["columns"]
    accuracy = sum(
        decisions[column]["semantic_type"] == semantic_type
        for column, semantic_type in expected.items()
    ) / len(expected)

    assert len(expected) == 40
    assert set(decisions) == set(expected)
    assert accuracy >= 0.85
    assert policy["summary"]["abstained_column_count"] == 0
    assert all(
        decision["canonical_label"] in CANONICAL_LABELS
        for decision in decisions.values()
    )
    assert all(
        decision["column_role"] != ROLE_SAFE
        for decision in decisions.values()
        if decision["semantic_type"]
        in {
            "email_address",
            "medical_record_number",
            "nhs_number",
            "person_name",
            "phone_number",
            "social_security_number",
            "street_address",
        }
    )


def test_ambiguous_headers_are_resolved_by_shared_identifier_validators() -> None:
    policy = classify_columns(FIXTURE_PATH)

    decision = policy["columns"]["col1"]
    assert decision["semantic_type"] == "nhs_number"
    assert decision["canonical_label"] == "ID_NUM"
    assert decision["column_role"] == ROLE_DIRECT_ID
    assert decision["recommended_action"] == ACTION_SUPPRESS
    assert decision["abstained"] is False
    assert "validated-nhs-checksum-ratio=1.000000" in decision["evidence"]
    assert policy["columns"]["value"]["semantic_type"] == "medical_record_number"
    assert policy["columns"]["value"]["column_role"] == ROLE_DIRECT_ID
    assert policy["columns"]["code"]["semantic_type"] == "clinical_code"
    assert policy["columns"]["code"]["recommended_action"] == ACTION_GENERALIZE


def test_low_confidence_column_abstains_to_manual_review(tmp_path: Path) -> None:
    source = tmp_path / "ambiguous.csv"
    source.write_text("mystery\nalpha-foo\nbeta-bar\n", encoding="utf-8")

    policy = classify_columns(source)
    decision = policy["columns"]["mystery"]

    assert decision["semantic_type"] == "unknown"
    assert decision["inferred_semantic_type"] == "unknown"
    assert decision["recommended_action"] == ACTION_MANUAL_REVIEW
    assert decision["abstained"] is True
    assert policy["summary"]["abstained_columns"] == ["mystery"]


def test_ambiguous_decimal_value_is_classified_as_lab_value(tmp_path: Path) -> None:
    source = tmp_path / "measurement.csv"
    source.write_text("value\n4.2\n5.7\n6.1\n", encoding="utf-8")

    decision = classify_columns(source)["columns"]["value"]

    assert decision["semantic_type"] == "lab_value"
    assert decision["recommended_action"] == ACTION_KEEP


def test_configured_threshold_turns_a_prediction_into_abstention() -> None:
    decision = classify_columns(FIXTURE_PATH, confidence_threshold=0.99)["columns"][
        "status"
    ]

    assert decision["semantic_type"] == "unknown"
    assert decision["inferred_semantic_type"] == "categorical"
    assert decision["confidence"] < 0.99
    assert decision["recommended_action"] == ACTION_MANUAL_REVIEW


def test_auto_policy_is_editable_persistable_and_unapplied(tmp_path: Path) -> None:
    policy = classify_columns(FIXTURE_PATH)
    policy["columns"]["status"]["recommended_action"] = ACTION_KEEP
    output = tmp_path / "review-policy.json"

    write_auto_policy(policy, output)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    serialized = json.dumps(persisted, sort_keys=True)

    assert persisted["review_status"] == "pending"
    assert persisted["review_required"] is True
    assert persisted["applied"] is False
    assert persisted["columns"]["status"]["recommended_action"] == ACTION_KEEP
    assert "column_semantics_40.csv" not in serialized
    assert "MRN-7000001" not in serialized


def test_pandas_requires_review_then_routes_each_action_explicitly() -> None:
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    get_adapter("pandas")
    frame = pd.read_csv(FIXTURE_PATH, keep_default_na=False).head(2)
    policy = frame.openmed.classify_columns()
    generalized: list[str] = []

    def deidentifier(text: str, **_: object) -> str:
        return "[DEIDENTIFIED]" if text else text

    def generalizer(value: object, semantic_type: str) -> str:
        generalized.append(semantic_type)
        return f"generalized:{semantic_type}"

    with pytest.raises(ValueError, match="reviewed=True"):
        frame.openmed.apply_auto_policy(policy)

    transformed = frame.openmed.apply_auto_policy(
        policy,
        reviewed=True,
        deidentifier=deidentifier,
        generalizer=generalizer,
    )

    assert "patient_name" not in transformed
    assert "mrn" not in transformed
    assert transformed["clinical_note"].tolist() == [
        "[DEIDENTIFIED]",
        "[DEIDENTIFIED]",
    ]
    assert transformed["admission_date"].tolist() == [
        "generalized:date",
        "generalized:date",
    ]
    assert transformed["code"].tolist() == [
        "generalized:clinical_code",
        "generalized:clinical_code",
    ]
    assert transformed["lab_value"].tolist() == frame["lab_value"].tolist()
    assert "date" in generalized
    assert "clinical_code" in generalized
    assert frame["patient_name"].tolist()[0] == "Synthetic Patient A"
    assert policy["applied"] is False


def test_unresolved_abstention_cannot_be_applied() -> None:
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    get_adapter("pandas")
    frame = pd.DataFrame({"mystery": ["alpha-foo", "beta-bar"]})
    policy = frame.openmed.classify_columns()

    with pytest.raises(ValueError, match="unresolved manual-review columns"):
        frame.openmed.apply_auto_policy(policy, reviewed=True)


def test_default_apply_path_uses_structured_generalization_primitives() -> None:
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    get_adapter("pandas")
    frame = pd.DataFrame(
        {
            "admission_date": ["2026-04-05"],
            "age": [52],
            "zip_code": ["02142"],
            "code": ["M54.5"],
        }
    )
    policy = frame.openmed.classify_columns()

    transformed = frame.openmed.apply_auto_policy(policy, reviewed=True)

    assert transformed.to_dict("records") == [
        {
            "admission_date": "2026",
            "age": "50-54",
            "zip_code": "021",
            "code": "*",
        }
    ]


def test_shared_mrn_validator_and_semantic_label_mapping() -> None:
    assert validate_mrn("MRN-1234567") is True
    assert public_validate_mrn("MRN-1234567") is True
    assert validate_mrn("1234567") is False
    assert canonical_label_for_column_semantic("medical_record_number") == "ID_NUM"
    with pytest.raises(KeyError, match="unknown column semantic type"):
        canonical_label_for_column_semantic("not-a-semantic-type")


def test_policy_routes_free_text_codes_and_dates() -> None:
    policy = classify_columns(FIXTURE_PATH)

    assert policy["columns"]["clinical_note"]["recommended_action"] == (
        ACTION_ROUTE_TO_DEIDENTIFY
    )
    assert policy["columns"]["diagnosis_code"]["recommended_action"] == (
        ACTION_GENERALIZE
    )
    assert policy["columns"]["admission_date"]["recommended_action"] == (
        ACTION_GENERALIZE
    )
