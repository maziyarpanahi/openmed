"""Regression tests for semantic direct-identifier field-name matching."""

import pytest

from openmed.risk import AnonymityPolicy, assess_release


@pytest.mark.parametrize(
    "field",
    (
        "validity_score",
        "fluid_balance",
        "grid_cell",
        "candidate_score",
        "account_status",
        "contact_preference",
        "patient_count",
        "member_count",
        "record_count",
        "visit_count",
        "claim_count",
        "number_of_visits",
        "policy_status",
        "diagnosis_name",
        "patient_diagnosis_name",
        "condition_name",
        "member_condition_name",
        "procedure_name",
        "medication_name",
        "facility_name",
    ),
)
def test_incidental_id_substrings_are_not_direct_identifiers(field: str) -> None:
    rows = [
        {"age": 40, field: "low"},
        {"age": 40, field: "high"},
    ]

    assessment = assess_release(
        rows,
        AnonymityPolicy(
            quasi_identifiers=("age",),
            non_sensitive_attributes=(field,),
            target_k=2,
        ),
    )

    assert assessment.meets_policy is True


@pytest.mark.parametrize(
    "field",
    (
        "patient_id",
        "patientId",
        "medical_record_number",
        "diagnosis_id",
        "email_address",
        "SSNNumber",
        "given_name",
        "api_key",
        "visit_date_email",
        "mother_maiden_name",
        "emergency_contact_name",
        "emergency_contact",
        "next_of_kin",
        "bank_account",
        "user_name",
        "login_name",
        "screen_name",
        "account_name",
        "middle_name",
        "surname",
        "forename",
        "legal_name",
        "birth_name",
        "nickname",
        "alias",
        "patient_number",
        "subject_number",
        "member_number",
        "record_number",
        "chart_number",
        "claim_number",
        "NHS_number",
        "ABHA_number",
        "medicare_number",
        "medicaid_number",
        "health_plan_number",
        "health_insurance_number",
        "insurance_number",
        "policy_number",
        "subscriber_number",
        "medical_record_no",
        "patient_num",
        "hospital_number",
        "admission_number",
        "encounter_number",
        "visit_number",
        "patient_key",
        "subject_key",
        "member_key",
        "record_key",
        "user_key",
        "account_key",
        "account",
        "uuid",
        "guid",
        "npi",
        "device_serial",
        "serial_number",
        "license_plate",
        "driver_license",
        "web_url",
        "ip",
        "ip_address",
        "client_ip",
        "source_ip",
        "mobile_number",
        "mobile",
        "cell_number",
        "imei",
        "device_imei",
        "imsi",
        "photo",
        "profile_photo",
        "patient_photo",
        "full_face_photo",
    ),
)
def test_true_identifier_names_still_fail_without_a_removal_role(field: str) -> None:
    rows = [
        {"age": 40, field: "canary-a"},
        {"age": 40, field: "canary-b"},
    ]

    with pytest.raises(ValueError, match="direct identifiers"):
        assess_release(
            rows,
            AnonymityPolicy(
                quasi_identifiers=("age",),
                non_sensitive_attributes=(field,),
                target_k=2,
            ),
        )
