"""Generated read-only SQL views over the Journey resource contract."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from openmed.clinical.journey_contracts import canonical_digest
from openmed.guard.query_safety import validate_bounded_read_only_sql
from openmed.service.journey_resources import (
    JOURNEY_RESOURCE_SCHEMA_VERSION,
    JourneyAccessPolicy,
    JourneyResourceCatalog,
    JourneyResourceKind,
    JourneyResourceQuery,
    JourneyResourceState,
    parse_access_attributes,
)

JOURNEY_SQL_SCHEMA_VERSION: Final = JOURNEY_RESOURCE_SCHEMA_VERSION
_VIEW_RE = re.compile(r"^journey_[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True, slots=True)
class JourneySQLView:
    """One generated SQL view and its canonical Journey resource family."""

    name: str
    resource_type: JourneyResourceKind

    def __post_init__(self) -> None:
        if _VIEW_RE.fullmatch(self.name) is None:
            raise ValueError("Journey SQL view name must be controlled")


JOURNEY_SQL_VIEWS: Final[tuple[JourneySQLView, ...]] = (
    JourneySQLView("journey_artifacts", JourneyResourceKind.ARTIFACT),
    JourneySQLView("journey_facts", JourneyResourceKind.FACT),
    JourneySQLView("journey_evidence", JourneyResourceKind.EVIDENCE),
    JourneySQLView("journey_current_facts", JourneyResourceKind.CURRENT_FACT),
    JourneySQLView("journey_events", JourneyResourceKind.JOURNEY_EVENT),
    JourneySQLView("journey_mappings", JourneyResourceKind.MAPPING),
    JourneySQLView("journey_cohort_runs", JourneyResourceKind.COHORT_RUN),
    JourneySQLView("journey_dataset_manifests", JourneyResourceKind.DATASET_MANIFEST),
)
_VIEW_BY_NAME: Final = MappingProxyType({view.name: view for view in JOURNEY_SQL_VIEWS})


@dataclass(frozen=True, slots=True)
class JourneySQLCredential:
    """A least-privilege analytics identity that can never authorize writes."""

    credential_id: str = "analytics_local"
    read_only: bool = True
    allowed_views: frozenset[str] = frozenset(_VIEW_BY_NAME)

    def __post_init__(self) -> None:
        if re.fullmatch(r"^[a-z][a-z0-9_.:/-]{0,127}$", self.credential_id) is None:
            raise ValueError("Journey SQL credential ID must be controlled")
        if not self.read_only:
            raise ValueError("Journey SQL analytics credentials must be read-only")
        if not self.allowed_views.issubset(_VIEW_BY_NAME):
            raise ValueError("Journey SQL credential contains an unsupported view")


@dataclass(frozen=True, slots=True)
class JourneySQLQueryResult:
    """Transport-neutral SQL projection with explicit non-success state."""

    state: JourneyResourceState
    code: str | None
    rows: tuple[Mapping[str, Any], ...]
    schema_version: str
    compatibility_policy: str
    snapshot_digest: str
    policy: Mapping[str, Any]

    def __post_init__(self) -> None:
        state = JourneyResourceState(self.state)
        if state is JourneyResourceState.SUCCESS and self.code is not None:
            raise ValueError("successful Journey SQL results cannot carry a code")
        if state is not JourneyResourceState.SUCCESS and self.code is None:
            raise ValueError("non-success Journey SQL results require a code")
        if (
            state
            in {
                JourneyResourceState.DENIED,
                JourneyResourceState.EMPTY,
                JourneyResourceState.FAILURE,
                JourneyResourceState.UNSUPPORTED,
            }
            and self.rows
        ):
            raise ValueError("this Journey SQL state cannot carry rows")
        object.__setattr__(self, "state", state)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "compatibility_policy": self.compatibility_policy,
            "policy": dict(self.policy),
            "rows": [dict(row) for row in self.rows],
            "schema_version": self.schema_version,
            "snapshot_digest": self.snapshot_digest,
            "state": self.state.value,
        }


def validate_journey_analytics_sql(
    sql: str,
    *,
    credential: JourneySQLCredential | None = None,
    max_rows: int = 100,
) -> str:
    """Reject writes, unbounded reads, and access outside allowlisted views."""

    active = credential or JourneySQLCredential()
    return validate_bounded_read_only_sql(
        sql,
        allowed_views=tuple(sorted(active.allowed_views)),
        max_rows=max_rows,
    )


def query_journey_view(
    catalog: JourneyResourceCatalog,
    view_name: str,
    *,
    namespace: str = "default",
    purpose: str = "analytics",
    role: str = "researcher",
    attributes: Sequence[str] = (),
    consent_state: str = "active",
    export_policy: str = "metadata_only",
    limit: int = 20,
    offset: int = 0,
    fields: Sequence[str] = (),
    policy: JourneyAccessPolicy | None = None,
    credential: JourneySQLCredential | None = None,
) -> JourneySQLQueryResult:
    """Execute one SQL-view-shaped query through the shared Python catalog."""

    active_credential = credential or JourneySQLCredential()

    def terminal(state: JourneyResourceState, code: str) -> JourneySQLQueryResult:
        return _terminal_sql_result(
            catalog,
            state,
            code,
            namespace,
            purpose,
            role,
            attributes,
            consent_state,
            export_policy,
        )

    view = _VIEW_BY_NAME.get(view_name)
    if view is None:
        return terminal(JourneyResourceState.UNSUPPORTED, "sql_view_unsupported")
    if view_name not in active_credential.allowed_views:
        return terminal(JourneyResourceState.DENIED, "sql_view_denied")
    if type(offset) is not int or offset < 0 or offset > 1_000_000:
        return terminal(JourneyResourceState.FAILURE, "sql_offset_invalid")
    if type(limit) is not int or not 1 <= limit <= 100:
        return terminal(JourneyResourceState.FAILURE, "sql_limit_invalid")
    try:
        query = JourneyResourceQuery(
            resource_type=view.resource_type,
            namespace=namespace,
            purpose=purpose,
            role=role,
            attributes=parse_access_attributes(attributes),
            consent_state=consent_state,
            export_policy=export_policy,
            first=limit,
            fields=tuple(fields),
        )
    except ValueError:
        return terminal(JourneyResourceState.FAILURE, "sql_query_invalid")
    if offset:
        query = JourneyResourceQuery(
            resource_type=query.resource_type,
            namespace=query.namespace,
            purpose=query.purpose,
            role=query.role,
            attributes=query.attributes,
            consent_state=query.consent_state,
            export_policy=query.export_policy,
            first=query.first,
            fields=query.fields,
            after=catalog.cursor_for_offset(query, offset),
        )
    page = catalog.list_resources(query, policy=policy)
    rows = tuple(_sql_row(item) for item in page.resources)
    return JourneySQLQueryResult(
        state=page.state,
        code=page.code,
        rows=rows,
        schema_version=page.schema_version,
        compatibility_policy=page.compatibility_policy,
        snapshot_digest=page.page_info.snapshot_digest,
        policy=page.policy.to_dict(),
    )


def render_journey_view_schema() -> str:
    """Generate deterministic ANSI-compatible read-only view definitions."""

    statements = [
        "-- OpenMed Journey read-only views",
        f"-- schema_version: {JOURNEY_SQL_SCHEMA_VERSION}",
        "-- compatibility_policy: same_major",
        "-- Base table is caller-owned; grant analytics roles SELECT on views only.",
    ]
    for view in JOURNEY_SQL_VIEWS:
        statements.extend(
            (
                "",
                f"CREATE VIEW {view.name} AS",
                "SELECT resource_id, resource_type, schema_version,",
                "       compatibility_policy, state, version, revision,",
                "       namespace, data_json, extensions_json",
                "FROM journey_resource_records",
                f"WHERE resource_type = '{view.resource_type.value}';",
            )
        )
    return "\n".join(statements) + "\n"


def _sql_row(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "compatibility_policy": item["compatibility_policy"],
            "data_json": json.dumps(
                item["data"], ensure_ascii=True, separators=(",", ":"), sort_keys=True
            ),
            "extensions_json": json.dumps(
                item["extensions"],
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
            "namespace": item["namespace"],
            "resource_id": item["resource_id"],
            "resource_type": item["resource_type"],
            "revision": item["revision"],
            "schema_version": item["schema_version"],
            "state": item["state"],
            "version": item["version"],
        }
    )


def _terminal_sql_result(
    catalog: JourneyResourceCatalog,
    state: JourneyResourceState,
    code: str,
    namespace: str,
    purpose: str,
    role: str = "researcher",
    attributes: Sequence[str] = (),
    consent_state: str = "active",
    export_policy: str = "metadata_only",
) -> JourneySQLQueryResult:
    safe_namespace = _safe_controlled(namespace)
    safe_purpose = _safe_controlled(purpose)
    safe_role = _safe_controlled(role)
    safe_export_policy = _safe_controlled(export_policy)
    safe_consent_state = (
        consent_state
        if consent_state in {"active", "unknown", "withdrawn"}
        else "unknown"
    )
    safe_attributes = list(parse_access_attributes(attributes))
    request_digest = canonical_digest(
        {
            "attributes": safe_attributes,
            "consent_state": safe_consent_state,
            "export_policy": safe_export_policy,
            "namespace": safe_namespace,
            "purpose": safe_purpose,
            "role": safe_role,
        }
    )
    decision_digest = canonical_digest(
        {
            "code": code,
            "request_digest": request_digest,
            "state": state.value,
        }
    )
    return JourneySQLQueryResult(
        state=state,
        code=code,
        rows=(),
        schema_version=JOURNEY_SQL_SCHEMA_VERSION,
        compatibility_policy="same_major",
        snapshot_digest=catalog.snapshot_digest,
        policy={
            "allowed_fields": [],
            "attributes": safe_attributes,
            "code": code if state is JourneyResourceState.DENIED else None,
            "consent_state": safe_consent_state,
            "decision_id": (f"decision_{decision_digest.removeprefix('sha256:')[:32]}"),
            "export_policy": safe_export_policy,
            "namespace": safe_namespace,
            "policy_version": JOURNEY_SQL_SCHEMA_VERSION,
            "purpose": safe_purpose,
            "request_digest": request_digest,
            "role": safe_role,
            "state": (
                JourneyResourceState.DENIED.value
                if state is JourneyResourceState.DENIED
                else JourneyResourceState.SUCCESS.value
            ),
        },
    )


def _safe_controlled(value: Any) -> str:
    if isinstance(value, str) and re.fullmatch(r"^[a-z][a-z0-9_.:/-]{0,127}$", value):
        return value
    return "invalid"


__all__ = [
    "JOURNEY_SQL_SCHEMA_VERSION",
    "JOURNEY_SQL_VIEWS",
    "JourneySQLCredential",
    "JourneySQLQueryResult",
    "JourneySQLView",
    "query_journey_view",
    "render_journey_view_schema",
    "validate_journey_analytics_sql",
]
