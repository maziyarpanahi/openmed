"""Cross-surface tests for versioned Journey resources."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from openmed.integrations.sql.journey_views import (
    JOURNEY_SQL_VIEWS,
    JourneySQLCredential,
    query_journey_view,
    render_journey_view_schema,
    validate_journey_analytics_sql,
)
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.client import OpenMedClient
from openmed.service.journey_resources import (
    JOURNEY_RESOURCE_SCHEMA_VERSION,
    RESOURCE_FIELDS,
    JourneyAccessPolicy,
    JourneyResourceCatalog,
    JourneyResourceKind,
    JourneyResourceQuery,
    JourneyResourceRecord,
    JourneyResourceState,
    migrate_resource_record,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "fixtures" / "service" / "journey_resources.json"
SQL_SNAPSHOT = ROOT / "docs" / "api" / "journey-views.sql"
OPENAPI_SNAPSHOT = ROOT / "docs" / "api" / "openapi.json"
GRAPHQL_SNAPSHOT = ROOT / "docs" / "api" / "graphql-schema.graphql"
PAGE_SCHEMA = (
    ROOT / "openmed" / "core" / "schemas" / "json" / "journey_resource_page.schema.json"
)
LOOPBACK_BASE_URL = "http://127.0.0.1"


class FakeLoader:
    """Minimal offline loader for service construction."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, Any]:
        return {}


def _catalog() -> JourneyResourceCatalog:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return JourneyResourceCatalog(
        JourneyResourceRecord.from_dict(item) for item in payload
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    app = create_app()
    app.state.journey_resources = _catalog()
    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as test_client:
        yield test_client


def test_python_contract_paginates_and_binds_cursor_to_query() -> None:
    catalog = _catalog()
    query = JourneyResourceQuery(
        resource_type=JourneyResourceKind.FACT,
        first=1,
        fields=("subject_id", "concept"),
    )

    first = catalog.list_resources(query)
    assert first.state is JourneyResourceState.SUCCESS
    assert first.page_info.has_next_page is True
    assert first.page_info.page_size == 1
    assert set(first.resources[0]["data"]) == {"subject_id", "concept"}

    second = catalog.list_resources(
        JourneyResourceQuery(
            resource_type=query.resource_type,
            first=1,
            fields=query.fields,
            after=first.page_info.end_cursor,
        )
    )
    assert second.state is JourneyResourceState.SUCCESS
    assert second.page_info.has_next_page is False
    assert second.resources[0]["resource_id"] != first.resources[0]["resource_id"]

    mismatched = catalog.list_resources(
        JourneyResourceQuery(
            resource_type=query.resource_type,
            first=1,
            fields=("assertion",),
            after=first.page_info.end_cursor,
        )
    )
    assert mismatched.state is JourneyResourceState.FAILURE
    assert mismatched.code == "cursor_query_mismatch"
    assert mismatched.resources == ()

    changed_catalog = JourneyResourceCatalog(
        [
            *(
                JourneyResourceRecord.from_dict(item)
                for item in json.loads(FIXTURE.read_text(encoding="utf-8"))
            ),
            JourneyResourceRecord(
                resource_type=JourneyResourceKind.FACT,
                resource_id="fact_newsnapshot0000",
                namespace="default",
                data={"concept": "synthetic.condition.gamma"},
            ),
        ]
    )
    stale = changed_catalog.list_resources(
        JourneyResourceQuery(
            resource_type=query.resource_type,
            first=1,
            fields=query.fields,
            after=first.page_info.end_cursor,
        )
    )
    assert stale.state is JourneyResourceState.FAILURE
    assert stale.code == "cursor_snapshot_changed"

    schema = json.loads(PAGE_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(first.to_dict())


def test_policy_denial_and_non_success_states_are_not_success() -> None:
    catalog = _catalog()
    denied = catalog.list_resources(
        JourneyResourceQuery(
            resource_type=JourneyResourceKind.FACT,
            namespace="restricted",
        )
    )
    assert denied.state is JourneyResourceState.DENIED
    assert denied.code == "namespace_denied"
    assert denied.resources == ()

    policy = JourneyAccessPolicy(
        fields_by_resource={
            **RESOURCE_FIELDS,
            JourneyResourceKind.FACT: frozenset({"subject_id"}),
        }
    )
    field_denied = catalog.list_resources(
        JourneyResourceQuery(
            resource_type=JourneyResourceKind.FACT,
            fields=("concept",),
        ),
        policy=policy,
    )
    assert field_denied.state is JourneyResourceState.DENIED
    assert field_denied.code == "field_denied"

    unknown_catalog = JourneyResourceCatalog(
        [
            JourneyResourceRecord(
                resource_type=JourneyResourceKind.FACT,
                resource_id="fact_unknown00000000",
                namespace="default",
                data={},
                state=JourneyResourceState.UNKNOWN,
            )
        ]
    )
    unknown = unknown_catalog.list_resources(
        JourneyResourceQuery(resource_type=JourneyResourceKind.FACT)
    )
    assert unknown.state is JourneyResourceState.UNKNOWN
    assert unknown.code == "resource_unknown"


@pytest.mark.parametrize(
    "state",
    [JourneyResourceState.FAILURE, JourneyResourceState.UNSUPPORTED],
)
def test_terminal_resource_states_return_value_free_pages(
    state: JourneyResourceState,
) -> None:
    catalog = JourneyResourceCatalog(
        [
            JourneyResourceRecord(
                resource_type=JourneyResourceKind.FACT,
                resource_id=f"fact_{state.value + '0' * 16}"[:21],
                namespace="default",
                data={},
                state=state,
            )
        ]
    )

    page = catalog.list_resources(
        JourneyResourceQuery(resource_type=JourneyResourceKind.FACT)
    )

    assert page.state is state
    assert page.code == f"resource_{state.value}"
    assert page.resources == ()


@pytest.mark.parametrize(
    ("query_overrides", "policy_overrides", "code"),
    [
        ({"role": "guest"}, {}, "role_denied"),
        ({"consent_state": "unknown"}, {}, "consent_unknown"),
        ({"consent_state": "withdrawn"}, {}, "consent_withdrawn"),
        ({"export_policy": "full_record"}, {}, "export_policy_denied"),
        (
            {},
            {
                "required_attributes_by_resource": {
                    JourneyResourceKind.FACT: frozenset({"approved_device"})
                }
            },
            "attribute_denied",
        ),
    ],
)
def test_access_context_fails_closed_with_reconstructable_decision(
    query_overrides: dict[str, Any],
    policy_overrides: dict[str, Any],
    code: str,
) -> None:
    query = JourneyResourceQuery(
        resource_type=JourneyResourceKind.FACT,
        **query_overrides,
    )

    page = _catalog().list_resources(
        query, policy=JourneyAccessPolicy(**policy_overrides)
    )

    assert page.state is JourneyResourceState.DENIED
    assert page.code == code
    assert page.resources == ()
    policy_payload = page.policy.to_dict()
    assert policy_payload.pop("request_digest") == query.access_request_digest
    assert policy_payload.pop("decision_id").startswith("decision_")
    assert policy_payload == {
        "allowed_fields": [],
        "attributes": list(query.attributes),
        "code": code,
        "consent_state": query.consent_state,
        "export_policy": query.export_policy,
        "namespace": query.namespace,
        "policy_version": JOURNEY_RESOURCE_SCHEMA_VERSION,
        "purpose": query.purpose,
        "role": query.role,
        "state": "denied",
    }


def test_cursor_cannot_be_reused_under_a_different_access_context() -> None:
    catalog = _catalog()
    original = JourneyResourceQuery(
        resource_type=JourneyResourceKind.FACT,
        first=1,
    )
    cursor = catalog.list_resources(original).page_info.end_cursor
    assert cursor is not None

    confused_deputy = catalog.list_resources(
        JourneyResourceQuery(
            resource_type=JourneyResourceKind.FACT,
            role="researcher",
            first=1,
            after=cursor,
        )
    )

    assert confused_deputy.state is JourneyResourceState.FAILURE
    assert confused_deputy.code == "cursor_query_mismatch"
    assert confused_deputy.resources == ()


@pytest.mark.parametrize("first", [0, 101])
def test_page_size_abuse_limits_fail_before_execution(first: int) -> None:
    with pytest.raises(ValueError, match="first must be between"):
        JourneyResourceQuery(resource_type=JourneyResourceKind.FACT, first=first)


def test_records_reject_sensitive_or_unallowlisted_data() -> None:
    with pytest.raises(ValueError, match="sensitive field"):
        JourneyResourceRecord(
            resource_type=JourneyResourceKind.FACT,
            resource_id="fact_sensitive000000",
            namespace="default",
            data={"raw_text": "synthetic value"},
        )

    with pytest.raises(ValueError, match="non-success"):
        JourneyResourceRecord(
            resource_type=JourneyResourceKind.FACT,
            resource_id="fact_failure00000000",
            namespace="default",
            state=JourneyResourceState.FAILURE,
            data={"concept": "synthetic.condition"},
        )


def test_same_major_migration_preserves_unknown_extensions() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))[0]
    payload["schema_version"] = "1.2.0"
    payload["producer_hint"] = {"format": "future"}

    migrated = migrate_resource_record(payload)

    assert migrated["schema_version"] == JOURNEY_RESOURCE_SCHEMA_VERSION
    assert migrated["data"] == payload["data"]
    assert migrated["extensions"]["producer_hint"] == {"format": "future"}

    payload["schema_version"] = "2.0.0"
    with pytest.raises(ValueError, match="unsupported Journey resource schema major"):
        migrate_resource_record(payload)


def test_rest_graphql_python_and_sql_return_equivalent_facts(
    client: TestClient,
) -> None:
    catalog = _catalog()
    query = JourneyResourceQuery(
        resource_type=JourneyResourceKind.FACT,
        first=2,
        fields=("subject_id", "concept", "assertion"),
    )
    expected = catalog.list_resources(query).to_dict()

    rest = client.get(
        "/v1/journey/resources",
        params={
            "resource_type": "fact",
            "namespace": "default",
            "purpose": "care_review",
            "first": 2,
            "fields": "subject_id,concept,assertion",
        },
    )
    assert rest.status_code == 200
    assert rest.json() == expected

    graphql = client.post(
        "/graphql",
        json={
            "query": """
                {
                  journeyResources(
                    resourceType: FACT
                    first: 2
                    fields: ["subject_id", "concept", "assertion"]
                  ) {
                    state code schemaVersion compatibilityPolicy
                    resources {
                      resourceType resourceId namespace data state version
                      revision schemaVersion compatibilityPolicy extensions
                    }
                    pageInfo {
                      hasNextPage endCursor pageSize snapshotDigest
                    }
                    policy {
                      state namespace purpose allowedFields code policyVersion
                    }
                  }
                }
            """
        },
    )
    assert graphql.status_code == 200
    graph_page = graphql.json()["data"]["journeyResources"]
    assert graph_page["state"] == expected["state"]
    assert graph_page["schemaVersion"] == expected["schema_version"]
    assert (
        graph_page["pageInfo"]["snapshotDigest"]
        == expected["page_info"]["snapshot_digest"]
    )
    assert [item["data"] for item in graph_page["resources"]] == [
        item["data"] for item in expected["resources"]
    ]

    sql = query_journey_view(
        catalog,
        "journey_facts",
        purpose="care_review",
        limit=2,
        fields=query.fields,
    )
    assert sql.state is JourneyResourceState.SUCCESS
    assert [json.loads(row["data_json"]) for row in sql.rows] == [
        item["data"] for item in expected["resources"]
    ]
    assert sql.snapshot_digest == expected["page_info"]["snapshot_digest"]


def test_rest_graphql_and_sql_enforce_the_same_access_context(
    client: TestClient,
) -> None:
    rest = client.get(
        "/v1/journey/resources",
        params={
            "resource_type": "fact",
            "role": "guest",
            "attributes": "approved_device",
            "consent_state": "active",
            "export_policy": "metadata_only",
        },
    )
    assert rest.status_code == 200
    assert rest.json()["state"] == "denied"
    assert rest.json()["code"] == "role_denied"
    assert rest.json()["resources"] == []

    graphql = client.post(
        "/graphql",
        json={
            "query": """
                {
                  journeyResources(
                    resourceType: FACT
                    role: "clinician"
                    attributes: ["approved_device"]
                    consentState: "withdrawn"
                    exportPolicy: "metadata_only"
                  ) {
                    state code resources { resourceId }
                    policy {
                      state role attributes consentState exportPolicy code
                    }
                  }
                }
            """
        },
    )
    assert graphql.status_code == 200
    graph_page = graphql.json()["data"]["journeyResources"]
    assert graph_page["state"] == "denied"
    assert graph_page["code"] == "consent_withdrawn"
    assert graph_page["resources"] == []
    assert graph_page["policy"]["consentState"] == "withdrawn"

    sql = query_journey_view(
        _catalog(),
        "journey_facts",
        role="researcher",
        consent_state="active",
        export_policy="full_record",
    )
    assert sql.state is JourneyResourceState.DENIED
    assert sql.code == "export_policy_denied"
    assert sql.rows == ()


def test_rest_limits_and_access_denial_remain_typed(client: TestClient) -> None:
    too_large = client.get(
        "/v1/journey/resources",
        params={"resource_type": "fact", "first": 101},
    )
    assert too_large.status_code == 422
    assert "synthetic.condition.alpha" not in too_large.text

    denied = client.get(
        "/v1/journey/resources",
        params={"resource_type": "fact", "namespace": "restricted"},
    )
    assert denied.status_code == 200
    assert denied.json()["state"] == "denied"
    assert denied.json()["resources"] == []


def test_python_client_encodes_bounded_journey_query() -> None:
    observed: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["path"] = request.url.path
        observed["query"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={
                "state": "empty",
                "code": "no_resources",
                "resources": [],
                "page_info": {},
                "policy": {},
                "schema_version": "1.0.0",
                "compatibility_policy": "same_major",
            },
        )

    with OpenMedClient(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    ) as api:
        response = api.journey_resources(
            "fact",
            first=2,
            fields=("subject_id", "concept"),
        )

    assert response["state"] == "empty"
    assert observed == {
        "path": "/v1/journey/resources",
        "query": {
            "first": "2",
            "namespace": "default",
            "purpose": "care_review",
            "role": "clinician",
            "consent_state": "active",
            "export_policy": "metadata_only",
            "resource_type": "fact",
            "fields": "subject_id,concept",
        },
    }


def test_sql_views_are_generated_and_analytics_credentials_cannot_write() -> None:
    assert SQL_SNAPSHOT.read_text(encoding="utf-8") == render_journey_view_schema()
    assert len(JOURNEY_SQL_VIEWS) == 8
    credential = JourneySQLCredential()
    assert credential.read_only is True
    with pytest.raises(ValueError, match="read-only"):
        JourneySQLCredential(read_only=False)

    assert (
        validate_journey_analytics_sql("SELECT resource_id FROM journey_facts LIMIT 10")
        == "SELECT resource_id FROM journey_facts LIMIT 10"
    )
    for statement in (
        "SELECT resource_id FROM journey_facts",
        "SELECT * FROM journey_facts LIMIT 10",
        "DELETE FROM journey_facts LIMIT 10",
        "SELECT resource_id FROM unrestricted_table LIMIT 10",
    ):
        with pytest.raises(ValueError):
            validate_journey_analytics_sql(statement)

    unsupported = query_journey_view(_catalog(), "journey_unknown")
    assert unsupported.state is JourneyResourceState.UNSUPPORTED
    denied = query_journey_view(
        _catalog(),
        "journey_facts",
        credential=JourneySQLCredential(allowed_views=frozenset()),
    )
    assert denied.state is JourneyResourceState.DENIED
    invalid = query_journey_view(
        _catalog(),
        "journey_facts",
        namespace="unsafe\nvalue",
        role="guest",
        attributes=("external_device",),
        consent_state="unknown",
        export_policy="full_record",
        limit=101,
    )
    assert invalid.state is JourneyResourceState.FAILURE
    assert invalid.code == "sql_limit_invalid"
    assert invalid.policy["namespace"] == "invalid"
    assert invalid.policy["role"] == "guest"
    assert invalid.policy["attributes"] == ["external_device"]
    assert invalid.policy["consent_state"] == "unknown"
    assert invalid.policy["export_policy"] == "full_record"
    assert str(invalid.policy["decision_id"]).startswith("decision_")
    assert str(invalid.policy["request_digest"]).startswith("sha256:")


def test_sql_snapshot_creates_only_views_over_caller_owned_table() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE journey_resource_records (
          resource_id TEXT, resource_type TEXT, schema_version TEXT,
          compatibility_policy TEXT, state TEXT, version INTEGER,
          revision INTEGER, namespace TEXT, data_json TEXT,
          extensions_json TEXT
        )
        """
    )
    connection.executescript(render_journey_view_schema())
    views = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'view'"
        )
    }
    assert views == {view.name for view in JOURNEY_SQL_VIEWS}
    connection.close()


def test_openapi_graphql_and_sql_snapshots_cover_the_same_resource_contract() -> None:
    expected_types = {kind.value for kind in JourneyResourceKind}
    openapi = json.loads(OPENAPI_SNAPSHOT.read_text(encoding="utf-8"))
    assert (
        set(
            openapi["components"]["schemas"]["JourneyResourceResponse"]["properties"][
                "resource_type"
            ]["enum"]
        )
        == expected_types
    )
    operation = openapi["paths"]["/v1/journey/resources"]["get"]
    parameters = {item["name"]: item for item in operation["parameters"]}
    assert parameters["resource_type"]["required"] is True
    assert parameters["first"]["schema"]["maximum"] == 100

    graphql = GRAPHQL_SNAPSHOT.read_text(encoding="utf-8")
    assert "journeyResources(" in graphql
    assert "type Mutation" not in graphql
    for kind in JourneyResourceKind:
        assert f"  {kind.name}" in graphql

    sql_resource_types = {view.resource_type.value for view in JOURNEY_SQL_VIEWS}
    assert sql_resource_types <= expected_types
