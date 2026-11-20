"""Offline wire and runtime contract checks for intended-use declarations."""

import json
from importlib.resources import files

import jsonschema
import pytest

from openmed.core.intended_use import IntendedUse, IntendedUseError, RegulatoryTier


def declaration():
    return dict(
        schema_version="1.0.0",
        regulatory_tier="restricted",
        purpose="measurement",
        intended_users=["clinician"],
        excluded_uses=["diagnosis", "autonomous_action"],
        input_modalities=["waveform"],
        languages=["zxx"],
        human_oversight="clinician_review",
        autonomous_clinical_action=False,
        summary="Displays a measurement for review.",
    )


def schema():
    return json.loads(
        files("openmed.core.schemas.json")
        .joinpath("intended_use.schema.json")
        .read_text()
    )


def test_round_trip():
    original = declaration()
    jsonschema.validate(original, schema())
    item = IntendedUse.from_dict(original)
    wire = item.to_dict()
    jsonschema.validate(wire, schema())
    assert IntendedUse.from_dict(wire).digest() == item.digest()
    assert (
        IntendedUse.from_dict({**original, "purpose": "research"}).digest()
        != item.digest()
    )
    assert {
        "notice_required",
        "clinician_mode_required",
    } <= item.regulatory_tier.obligations
    assert all(isinstance(tier.obligations, frozenset) for tier in RegulatoryTier)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("regulatory_tier", "mystery", "unknown_regulatory_tier"),
        ("purpose", "unknown", "unknown_purpose"),
        ("autonomous_clinical_action", True, "autonomous_clinical_action_forbidden"),
        ("summary", "This diagnoses conditions.", "prohibited_summary_claim"),
        ("summary", "FDA-cleared.", "prohibited_summary_claim"),
        ("summary", "CE-marked.", "prohibited_summary_claim"),
    ],
)
def test_rejections(field, value, reason):
    payload = {**declaration(), field: value}
    with pytest.raises(IntendedUseError) as failure:
        IntendedUse.from_dict(payload)
    assert str(failure.value) == failure.value.code == reason
    assert not jsonschema.Draft202012Validator(schema()).is_valid(payload)
