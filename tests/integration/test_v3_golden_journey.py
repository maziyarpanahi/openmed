"""Single-gate coverage for the v3 five-source synthetic Journey."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema.validators import validator_for

from openmed.eval.golden_journey import (
    load_golden_journey_scenario,
    load_golden_journey_schema,
    run_golden_journey,
    semantic_diff,
)

pytestmark = pytest.mark.integration

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "journey" / "v3"
SCENARIO = FIXTURE_ROOT / "scenario.json"
GOLDEN = FIXTURE_ROOT / "golden.json"


def test_five_source_golden_journey_covers_every_v3_public_surface(
    tmp_path: Path,
) -> None:
    scenario = load_golden_journey_scenario(SCENARIO)
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    actual = run_golden_journey(scenario, work_dir=tmp_path)
    schema = load_golden_journey_schema()
    validator = validator_for(schema)
    validator.check_schema(schema)
    assert not tuple(validator(schema).iter_errors(actual))
    assert semantic_diff(expected, actual) == []

    assert actual["synthetic"] is True
    assert actual["provenance"]["classification"] == "synthetic"
    assert [item["format"] for item in actual["sources"]] == [
        "text",
        "fhir_r4",
        "hl7v2",
        "csv",
        "dicom_sr",
    ]
    assert len(actual["pipeline"]["runs"]) == 5
    assert actual["pipeline"]["replay"]["replayed"] is True
    assert all(
        [item["stage"] for item in run["stage_states"]]
        == actual["pipeline"]["stage_order"]
        for run in actual["pipeline"]["runs"]
    )

    facts = actual["facts"]
    assert (
        sum(
            item["value"] == {"code": "condition-alpha", "system": "synthetic"}
            for item in facts
        )
        == 3
    )
    assert any(item["attributes"]["assertion"] == "negated" for item in facts)
    assert any(item["attributes"]["certainty"] == "possible" for item in facts)
    assert any(item["effective_time"].get("precision") == "month" for item in facts)
    assert any(
        item["parent_fact_ids"] and item["status"] == "corrected" for item in facts
    )
    assert actual["unit_conversion"] == {
        "canonical_magnitude": 0.9,
        "canonical_unit": "g/L",
        "original_fact_id": actual["unit_conversion"]["original_fact_id"],
        "original_unit": "mg/dL",
        "status": "ok",
    }
    assert actual["conflicts"][0]["status"] == "open"
    assert actual["resolutions"][0]["action"] == "select"
    assert actual["identity"]["state"] == "ambiguous"
    assert actual["identity"]["review_required"] is True

    assert len(actual["journey"]["events"]) == 6
    assert actual["omop"]["state"] == "partial"
    assert actual["omop"]["reason_code"] == "projection_information_loss"
    assert actual["omop"]["violations"] == []
    assert actual["cohort"]["membership_counts"]["met"] == 1
    assert actual["dataset"]["export_profile"] == "metadata_only"
    assert actual["dataset"]["snapshot"]["record_count"] == 1
    assert actual["state_matrix"] == {
        "ambiguous_identity": "ambiguous",
        "conflict": "conflict",
        "denied_consent": "denied",
        "empty_query": "empty",
        "failure": "failure",
        "partial_projection": "partial",
        "unknown": "unknown",
        "unsupported": "unsupported",
    }

    rendered = json.dumps(actual, sort_keys=True)
    assert all(str(item["payload"]) not in rendered for item in scenario["sources"])
    assert "document_text" not in rendered
