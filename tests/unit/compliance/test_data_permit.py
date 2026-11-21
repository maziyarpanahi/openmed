"""Offline permit enforcement using synthetic export schemas."""

import pytest

from openmed.compliance.data_permit import DataPermit, OutputForm, check_export_manifest

PERMIT = DataPermit(("research",), ("age_band", "count"), (OutputForm.AGGREGATE,))


def test_matching_final_export_schema_allowed():
    decision = check_export_manifest(
        PERMIT, purpose="research", fields=("count",), output_form="aggregate"
    )
    assert decision.allowed
    decision.require_allowed()


def test_report_records_only_unpermitted_field_codes_and_fixed_reasons():
    result = check_export_manifest(
        PERMIT,
        purpose="other_study",
        fields=("count", "postal_code"),
        output_form="pseudonymised",
    )
    assert result.disallowed_fields == ("postal_code",)
    assert result.violations == (
        "purpose_not_permitted",
        "output_form_not_permitted",
        "fields_not_permitted",
    )
    assert "other_study" not in repr(result)
    with pytest.raises(PermissionError, match="^export_outside_data_permit$"):
        result.require_allowed()


@pytest.mark.parametrize("form", ["aggregate", "anonymised"])
def test_pseudonymised_output_refused_for_each_other_form(form):
    permit = DataPermit(("research",), ("count",), (form,))
    assert not check_export_manifest(
        permit, purpose="research", fields=("count",), output_form="pseudonymised"
    ).allowed


@pytest.mark.parametrize(
    "fields", [(), {"count": "private example"}, ("private example",)]
)
def test_malformed_schema_refused_without_values(fields):
    with pytest.raises(ValueError) as exc:
        check_export_manifest(
            PERMIT, purpose="research", fields=fields, output_form="aggregate"
        )
    assert "private example" not in str(exc.value)
