"""Identifier aliases must not silently fall through to a kept OTHER label."""

import pytest

from openmed.core.clinical_label_map import clinical_label, clinical_label_map
from openmed.core.clinical_policy import resolve_clinical_policy


@pytest.mark.parametrize(
    "label",
    [
        "biometric_identifier",
        "certificate_license_number",
        "customer_id",
        "device_identifier",
        "employee_id",
        "health_plan_beneficiary_number",
        "http_cookie",
        "tax_id",
        "unique_id",
        "bank_routing_number",
        "fax_number",
        "ipv4",
        "ipv6",
        "swift_bic",
        "vehicle_identifier",
        "coordinate",
    ],
)
@pytest.mark.parametrize("prefix", ["", "B-", "I-"])
def test_direct_identifiers_have_explicit_redacting_projection(label, prefix):
    canonical = clinical_label(prefix + label)
    assert canonical != "OTHER"
    assert resolve_clinical_policy().profile.action_for(canonical) == "mask"


def test_clinical_context_mapping_is_explicit_and_unknown_labels_are_rejected():
    assert clinical_label("blood_type") == "OTHER"
    assert clinical_label("race_ethnicity") == "ETHNICITY"
    assert clinical_label("company_name") == "ORGANIZATION"
    assert clinical_label("date_time") == "DATE"
    with pytest.raises(ValueError, match="reviewed"):
        clinical_label("future_unmapped_identifier")
    with pytest.raises(ValueError, match="reviewed"):
        clinical_label_map(["O", "B-future_unmapped_identifier"])


def test_ontology_requires_outside_label():
    with pytest.raises(ValueError, match="outside"):
        clinical_label_map(["B-EMAIL", "I-EMAIL"])
