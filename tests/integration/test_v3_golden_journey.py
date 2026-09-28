"""Single-gate coverage for the v3 five-source synthetic Journey."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jsonschema.validators import validator_for

from openmed.eval.golden_journey import (
    load_golden_journey_scenario,
    load_golden_journey_schema,
    run_golden_journey,
    semantic_diff,
)
from openmed.integrations.sql.journey_views import (
    query_journey_view,
    render_journey_view_schema,
    validate_journey_analytics_sql,
)
from openmed.mcp.server import build_mcp_tool_handlers
from openmed.mcp.tool_registry import validate_registered_tool_output
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.journey_resources import (
    JourneyResourceCatalog,
    JourneyResourceKind,
    JourneyResourceRecord,
    JourneyResourceState,
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


def test_five_source_journey_has_rest_mcp_and_sql_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read one executed synthetic journey through three public surfaces."""

    scenario = load_golden_journey_scenario(SCENARIO)
    report = run_golden_journey(scenario, work_dir=tmp_path)
    journey = report["journey"]
    snapshot = journey["snapshot"]
    events = journey["events"]
    assert {source_id for event in events for source_id in event["source_ids"]} == {
        source["source_id"] for source in report["sources"]
    }

    # Every record below comes from this run, including the facts already
    # projected by the golden runner and its six evidence-linked events.
    records = [
        JourneyResourceRecord.from_dict(item)
        for item in report["api_results"]["success"]["resources"]
    ]
    records.append(
        JourneyResourceRecord(
            resource_type=JourneyResourceKind.JOURNEY,
            resource_id="journey_" + snapshot["snapshot_id"].split("_", 1)[1],
            namespace="default",
            data={
                "subject_id": snapshot["subject_id"],
                "snapshot_id": snapshot["snapshot_id"],
                "event_count": len(events),
            },
        )
    )
    records.extend(
        JourneyResourceRecord(
            resource_type=JourneyResourceKind.JOURNEY_EVENT,
            resource_id=event["event_id"],
            namespace="default",
            data={
                "subject_id": snapshot["subject_id"],
                "event_type": event["event_type"],
                "effective_at": event["fact"]["effective_time"]["start"],
                "fact_id": event["fact"]["fact_id"],
            },
        )
        for event in events
    )
    catalog = JourneyResourceCatalog(records)

    class FakeLoader:
        def __init__(self, config: object) -> None:
            self.config = config

    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    app = create_app()
    app.state.journey_resources = catalog
    with TestClient(app, base_url="http://127.0.0.1") as client:
        rest_journey = client.get(
            "/v1/journey/resources", params={"resource_type": "journey"}
        )
        rest_events = client.get(
            "/v1/journey/resources", params={"resource_type": "journey_event"}
        )
    assert rest_journey.status_code == rest_events.status_code == 200
    assert rest_journey.json()["state"] == rest_events.json()["state"] == "success"

    handlers = build_mcp_tool_handlers(None, journey_catalog_provider=lambda: catalog)
    mcp_journey = handlers["openmed_read_journey"]()
    assert validate_registered_tool_output("openmed_read_journey", mcp_journey) == (
        mcp_journey
    )
    sql_events = query_journey_view(
        catalog, "journey_events", purpose="care_review", limit=20
    )
    assert sql_events.state is JourneyResourceState.SUCCESS

    rest_journey_page = rest_journey.json()
    rest_event_page = rest_events.json()
    assert mcp_journey["resources"] == rest_journey_page["resources"]
    assert mcp_journey["resources"][0]["data"] == {
        "subject_id": snapshot["subject_id"],
        "snapshot_id": snapshot["snapshot_id"],
        "event_count": len(events),
    }
    assert (
        {row["resource_id"] for row in sql_events.rows}
        == {item["resource_id"] for item in rest_event_page["resources"]}
        == {event["event_id"] for event in events}
    )
    assert {json.loads(row["data_json"])["fact_id"] for row in sql_events.rows} == {
        event["fact"]["fact_id"] for event in events
    }

    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE journey_resource_records ("
            "resource_id TEXT PRIMARY KEY, resource_type TEXT, schema_version TEXT, "
            "compatibility_policy TEXT, state TEXT, version INTEGER, "
            "revision INTEGER, namespace TEXT, data_json TEXT, extensions_json TEXT)"
        )
        connection.executemany(
            "INSERT INTO journey_resource_records VALUES "
            "(:resource_id, :resource_type, :schema_version, "
            ":compatibility_policy, :state, :version, :revision, "
            ":namespace, :data_json, :extensions_json)",
            [dict(row) for row in sql_events.rows],
        )
        connection.executescript(render_journey_view_schema())
        sql_rows = connection.execute(
            validate_journey_analytics_sql(
                "SELECT resource_id, data_json FROM journey_events LIMIT 20"
            )
        ).fetchall()
    assert {resource_id for resource_id, _ in sql_rows} == {
        event["event_id"] for event in events
    }
    assert {json.loads(data_json)["fact_id"] for _, data_json in sql_rows} == {
        event["fact"]["fact_id"] for event in events
    }
    assert (
        rest_journey_page["page_info"]["snapshot_digest"]
        == rest_event_page["page_info"]["snapshot_digest"]
        == mcp_journey["snapshot"]["digest"]
        == sql_events.snapshot_digest
        == catalog.snapshot_digest
    )
    rendered = json.dumps(
        [
            rest_journey_page,
            rest_event_page,
            mcp_journey,
            sql_events.to_dict(),
            sql_rows,
        ]
    )
    assert all(str(source["payload"]) not in rendered for source in scenario["sources"])
