"""Shared native wire fixture is produced by the actual Python composer."""

import json
from pathlib import Path

from openmed.clinical.brief import build_clinical_brief
from tests.unit.clinical.test_brief import fixture_context


def test_native_packet_matches_fixed_generator_composition():
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/clinical/brief_parity/verified.json"
        ).read_text()
    )
    value, context = fixture_context()
    brief = build_clinical_brief(value, model="extractive", context=context)
    assert fixture["source"] == value.deidentified_text
    assert fixture["generator_output"] == brief.summary
    assert fixture["evaluation_json"] == json.dumps(
        brief.to_response(), sort_keys=True, separators=(",", ":")
    )
