"""Tests for the REST, GraphQL and gRPC protocol map and its drift check.

The committed map must match every transport. Each negative control mutates
one discovered surface or one map entry and asserts the exact value-free
finding the drift check reports.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from openmed.service import protocol_map
from openmed.service.api_compatibility import discover_service_error_categories
from openmed.service.protocol_map import (
    DEFAULT_PROTOCOL_MAP_PATH,
    ObservedOutcome,
    ProtocolDriftError,
    ProtocolMapError,
    TransportSurface,
    check_protocol_map,
    discover_transport_surface,
    load_protocol_map,
    parse_protocol_map,
)

SENTINEL = "SYNTH-SENTINEL-PROTOCOL-0042"
ANALYZE = "/openmed.service.v1.OpenMedService/Analyze"


@pytest.fixture(scope="module")
def document() -> dict[str, Any]:
    return json.loads(DEFAULT_PROTOCOL_MAP_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def surface() -> TransportSurface:
    return discover_transport_surface()


def issues(report: Any) -> list[dict[str, str]]:
    return [issue.to_dict() for issue in report.issues]


def changed(document: dict[str, Any], edit: Callable[[dict[str, Any]], None]) -> Any:
    updated = copy.deepcopy(document)
    edit(updated)
    return parse_protocol_map(updated)


def outcome_entry(document: dict[str, Any], outcome_id: str) -> dict[str, Any]:
    return next(item for item in document["outcomes"] if item["id"] == outcome_id)


def test_committed_protocol_map_matches_every_transport(
    surface: TransportSurface,
) -> None:
    report = check_protocol_map(surface=surface)

    assert report.is_clean, issues(report)
    assert report.contract_version == load_protocol_map().contract_version
    assert report.counts["operations"] == 4
    assert report.assert_clean() is report


def test_discovered_surface_covers_each_transport(surface: TransportSurface) -> None:
    assert {"authentication_required", "mtls_certificate_required"} <= (
        surface.rest_codes
    )
    assert surface.rest_stream_event_codes == {
        "bad_request",
        "internal_error",
        "timeout",
    }
    assert surface.graphql_codes == {
        "OPENMED_FORBIDDEN",
        "OPENMED_RESOLVER_ERROR",
        "model_not_served",
    }
    assert surface.grpc_failures["insufficient_scope"] == (
        "PERMISSION_DENIED",
        "permission_denied",
    )
    assert (
        "request.text"
        in surface.grpc_request_paths[
            "/openmed.service.v1.OpenMedService/StreamDeidentify"
        ]
    )
    assert "spans[].entityType" in surface.graphql_response_paths["deidentify"]


def test_rest_discovery_includes_mtls_and_request_size_codes() -> None:
    codes = set(discover_service_error_categories())

    assert {
        "mtls_certificate_invalid",
        "mtls_certificate_required",
        "request_size_exceeded",
        "request_size_invalid",
    } <= codes


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda s: {"rest_codes": s.rest_codes | {"brand_new_failure"}},
            {
                "change": "unmapped_code",
                "transport": "rest",
                "code": "brand_new_failure",
            },
        ),
        (
            lambda s: {"rest_codes": s.rest_codes - {"privacy_gateway_blocked"}},
            {
                "change": "stale_code",
                "transport": "rest",
                "code": "privacy_gateway_blocked",
            },
        ),
        (
            lambda s: {
                "rest_stream_event_codes": s.rest_stream_event_codes | {"stream_lost"}
            },
            {
                "change": "unmapped_stream_event_code",
                "transport": "rest",
                "code": "stream_lost",
            },
        ),
        (
            lambda s: {
                "graphql_codes": s.graphql_codes | {"OPENMED_GRAPHQL_LIMIT_EXCEEDED"}
            },
            {
                "change": "unmapped_code",
                "transport": "graphql",
                "code": "OPENMED_GRAPHQL_LIMIT_EXCEEDED",
            },
        ),
        (
            lambda s: {"graphql_codes": s.graphql_codes - {"model_not_served"}},
            {
                "change": "stale_code",
                "transport": "graphql",
                "code": "model_not_served",
            },
        ),
        (
            lambda s: {
                "grpc_failures": {
                    **s.grpc_failures,
                    "egress_denied": ("PERMISSION_DENIED", "permission_denied"),
                }
            },
            {"change": "unmapped_code", "transport": "grpc", "code": "egress_denied"},
        ),
        (
            lambda s: {
                "grpc_failures": {
                    **s.grpc_failures,
                    "rate_limited": ("UNAVAILABLE", "resource_exhausted"),
                }
            },
            {"change": "status_mismatch", "transport": "grpc", "code": "rate_limited"},
        ),
        (
            lambda s: {
                "grpc_failures": {
                    **s.grpc_failures,
                    "rate_limited": ("RESOURCE_EXHAUSTED", "unavailable"),
                }
            },
            {
                "change": "category_mismatch",
                "transport": "grpc",
                "code": "rate_limited",
            },
        ),
        (
            lambda s: {
                "grpc_request_paths": {
                    **s.grpc_request_paths,
                    ANALYZE: s.grpc_request_paths[ANALYZE] | {"deadline_ms"},
                }
            },
            {
                "change": "unmapped_field",
                "transport": "grpc",
                "operation": "analyze",
                "field": "deadline_ms",
            },
        ),
        (
            lambda s: {
                "graphql_input_fields": {
                    **s.graphql_input_fields,
                    "analyze": s.graphql_input_fields["analyze"] - {"keepAlive"},
                }
            },
            {
                "change": "stale_field",
                "transport": "graphql",
                "operation": "analyze",
                "field": "keepAlive",
            },
        ),
        (
            lambda s: {
                "graphql_response_paths": {
                    **s.graphql_response_paths,
                    "deidentify": s.graphql_response_paths["deidentify"]
                    - {"spans[].replacement"},
                }
            },
            {
                "change": "stale_field",
                "transport": "graphql",
                "operation": "deidentify",
                "field": "spans[].replacement",
            },
        ),
        (
            lambda s: {
                "graphql_field_scopes": {**s.graphql_field_scopes, "summarize": ()}
            },
            {
                "change": "unmapped_operation",
                "transport": "graphql",
                "operation": "summarize",
            },
        ),
        (
            lambda s: {
                "grpc_request_paths": {
                    **s.grpc_request_paths,
                    "/openmed.service.v1.OpenMedService/Summarize": frozenset(),
                }
            },
            {
                "change": "unmapped_operation",
                "transport": "grpc",
                "operation": "/openmed.service.v1.OpenMedService/Summarize",
            },
        ),
        (
            lambda s: {
                "rest_route_scopes": {
                    **s.rest_route_scopes,
                    ("POST", "/analyze"): ("analyze:admin",),
                }
            },
            {"change": "scope_mismatch", "transport": "rest", "operation": "analyze"},
        ),
        (
            lambda s: {
                "grpc_routes": {**s.grpc_routes, ANALYZE: ("POST", "/pii/extract")}
            },
            {"change": "route_mismatch", "transport": "grpc", "operation": "analyze"},
        ),
        (
            lambda s: {
                "graphql_arguments": {
                    **s.graphql_arguments,
                    "analyze": {"input": "AnalyzeInputV2"},
                }
            },
            {
                "change": "argument_mismatch",
                "transport": "graphql",
                "operation": "analyze",
            },
        ),
    ],
)
def test_transport_drift_is_reported_without_values(
    surface: TransportSurface,
    mutation: Callable[[TransportSurface], dict[str, Any]],
    expected: dict[str, str],
) -> None:
    drifted = dataclasses.replace(surface, **mutation(surface))

    report = check_protocol_map(surface=drifted)

    assert issues(report) == [expected]
    with pytest.raises(ProtocolDriftError) as excinfo:
        report.assert_clean()
    assert excinfo.value.report is report


def test_unsafe_emitted_codes_are_reported_without_the_value(
    surface: TransportSurface,
) -> None:
    drifted = dataclasses.replace(
        surface, rest_codes=surface.rest_codes | {f"leak {SENTINEL}"}
    )

    report = check_protocol_map(surface=drifted)

    assert issues(report) == [{"change": "invalid_code", "transport": "rest"}]
    assert SENTINEL not in report.to_json()


def test_category_differences_must_be_declared(
    document: dict[str, Any], surface: TransportSurface
) -> None:
    def drop_rest_validation_divergence(updated: dict[str, Any]) -> None:
        updated["known_divergences"] = [
            item
            for item in updated["known_divergences"]
            if item["id"] != "rest_limits_reported_as_validation"
        ]

    report = check_protocol_map(
        changed(document, drop_rest_validation_divergence), surface=surface
    )

    assert issues(report) == [
        {
            "change": "undeclared_divergence",
            "transport": "rest",
            "outcome": "stream_chunk_too_large",
        },
        {
            "change": "undeclared_divergence",
            "transport": "rest",
            "outcome": "text_too_long",
        },
    ]


def test_divergences_that_no_longer_diverge_are_stale(
    document: dict[str, Any], surface: TransportSurface
) -> None:
    def declare_matching_outcome(updated: dict[str, Any]) -> None:
        updated["known_divergences"][0]["outcomes"].append("rate_limited")

    report = check_protocol_map(
        changed(document, declare_matching_outcome), surface=surface
    )

    assert issues(report) == [
        {"change": "stale_divergence", "transport": "rest", "outcome": "rate_limited"}
    ]


def test_missing_transport_variants_need_a_reason(
    document: dict[str, Any], surface: TransportSurface
) -> None:
    def drop_grpc_variants(updated: dict[str, Any]) -> None:
        outcome_entry(updated, "rate_limited")["grpc"] = []

    report = check_protocol_map(changed(document, drop_grpc_variants), surface=surface)

    assert issues(report) == [
        {"change": "unmapped_outcome", "transport": "grpc", "outcome": "rate_limited"},
        {"change": "unused_code", "transport": "grpc", "code": "rate_limited"},
    ]


def test_codes_cannot_be_both_mapped_and_rest_only(
    document: dict[str, Any], surface: TransportSurface
) -> None:
    def duplicate(updated: dict[str, Any]) -> None:
        updated["rest_only_codes"]["grounding"].append("forbidden")

    report = check_protocol_map(changed(document, duplicate), surface=surface)

    assert issues(report) == [
        {"change": "ambiguous_code", "transport": "rest", "code": "forbidden"}
    ]


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.update(schema_version=2),
        lambda d: d.update(contract_version=SENTINEL),
        lambda d: d.update(categories=["unauthenticated"]),
        lambda d: d["codes"]["rest"].update({SENTINEL: {"status": 400}}),
        lambda d: d["codes"]["rest"]["forbidden"].update(status=SENTINEL),
        lambda d: d["outcomes"][0].update(category=SENTINEL),
        lambda d: d["outcomes"][0]["rest"].append({"code": "not_in_the_table"}),
        lambda d: d["outcomes"][0]["grpc"].append({"status": "CANCELLED"}),
        lambda d: d["operations"][0]["request_fields"].append({"field": SENTINEL}),
        lambda d: d["operations"][0]["response_fields"][0].pop("normalize"),
        lambda d: d["operations"][0].update(graphql={"supported": False}),
        lambda d: d["known_divergences"][0].update(kind=SENTINEL),
        lambda d: d.update(outcomes=[]),
    ],
)
def test_invalid_maps_fail_closed_without_echoing_values(
    document: dict[str, Any], edit: Callable[[dict[str, Any]], None]
) -> None:
    updated = copy.deepcopy(document)
    edit(updated)

    with pytest.raises(ProtocolMapError) as excinfo:
        parse_protocol_map(updated)

    assert SENTINEL not in str(excinfo.value)


def test_outcomes_classify_observations(document: dict[str, Any]) -> None:
    parsed = parse_protocol_map(document)
    scope = parsed.outcome("insufficient_scope")
    deadline = parsed.outcome("deadline_exceeded")

    assert scope.expects(ObservedOutcome("rest", 403, "forbidden"))
    assert scope.expects(
        ObservedOutcome("graphql", 200, "OPENMED_FORBIDDEN", envelope="graphql")
    )
    assert scope.expects(
        ObservedOutcome("grpc", "PERMISSION_DENIED", "insufficient_scope")
    )
    assert not scope.expects(ObservedOutcome("rest", 401, "forbidden"))
    assert deadline.expects(
        ObservedOutcome("rest", 200, "timeout", delivery="stream_event")
    )
    assert deadline.expects(ObservedOutcome("grpc", "DEADLINE_EXCEEDED"))
    record = parsed.classify(
        ObservedOutcome("graphql", 401, "authentication_required", envelope="rest")
    )
    assert record == {
        "transport": "graphql",
        "status": 401,
        "code": "authentication_required",
        "envelope": "rest",
        "category": "unauthenticated",
        "outcomes": ["authentication_required", "credentials_malformed"],
    }
    assert parsed.classify(ObservedOutcome("rest", 418, SENTINEL))["category"] is None
    assert parsed.operation("extract").unsupported.keys() == {"graphql"}
    with pytest.raises(KeyError):
        parsed.outcome(SENTINEL)


def test_cli_reports_status_and_value_free_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert protocol_map.main([]) == 0
    assert "in sync" in capsys.readouterr().out

    assert protocol_map.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["issues"] == []
    assert payload["contract"] == "openmed.service.protocol-map"

    broken = tmp_path / "protocol-map.json"
    broken.write_text(json.dumps({"schema_version": 1, "note": SENTINEL}))
    assert protocol_map.main(["--map", str(broken), "--json"]) == 2
    output = capsys.readouterr().out
    assert json.loads(output) == {"schema_version": 1, "error": "invalid_protocol_map"}
    assert SENTINEL not in output
