"""Focused offline tests for the MCP authorization conformance matrix."""

from __future__ import annotations

import copy
import json
import socket
from pathlib import Path
from typing import Any

import pytest

from openmed.mcp.authorization_conformance import (
    AUTHORIZATION_CONFORMANCE_SCHEMA_VERSION,
    DEFAULT_FIXTURE_PATH,
    ConformanceCoverageError,
    ConformanceFixtureError,
    ConformanceManifest,
    MockAuthorizationTransport,
    load_conformance_manifest,
    render_conformance_matrix,
    run_conformance,
    validate_case_coverage,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MATRIX_PATH = REPO_ROOT / "docs" / "security" / "mcp-authorization-matrix.md"

FOCUSED_CASE_IDS = frozenset(
    {
        "positive-read-tool",
        "positive-approved-state-change",
        "negative-wrong-audience",
        "negative-missing-resource-indicator",
        "negative-token-passthrough",
        "negative-invalid-pkce",
        "negative-redirect-mismatch",
        "negative-insufficient-tool-scope",
        "negative-oversized-payload",
        "negative-unapproved-state-change",
    }
)


def test_manifest_is_versioned_synthetic_and_covers_required_boundaries() -> None:
    manifest = load_conformance_manifest()

    assert DEFAULT_FIXTURE_PATH.is_file()
    assert manifest.schema_version == AUTHORIZATION_CONFORMANCE_SCHEMA_VERSION
    assert manifest.synthetic is True
    assert len(manifest.cases) == 10
    assert {case.kind for case in manifest.cases} == {"positive", "negative"}
    assert {case.expected_failure for case in manifest.cases if case.is_negative} == {
        "wrong_audience",
        "missing_resource_indicator",
        "token_passthrough",
        "invalid_pkce",
        "redirect_mismatch",
        "insufficient_tool_scope",
        "oversized_payload",
        "unapproved_state_change",
    }


def test_declared_cases_are_exactly_covered_by_focused_tests() -> None:
    manifest = load_conformance_manifest()

    validate_case_coverage(manifest, FOCUSED_CASE_IDS)
    with pytest.raises(ConformanceCoverageError) as error:
        validate_case_coverage(manifest, FOCUSED_CASE_IDS - {"negative-invalid-pkce"})

    assert error.value.missing == ("negative-invalid-pkce",)


def test_every_negative_fixture_fails_at_its_declared_boundary() -> None:
    manifest = load_conformance_manifest()
    report = run_conformance(manifest, covered_case_ids=FOCUSED_CASE_IDS)
    results = {result.case_id: result for result in report.results}

    assert report.ok
    for case in manifest.cases:
        result = results[case.case_id]
        if case.is_negative:
            assert result.failure_category == case.expected_failure
            assert result.failure_boundary == case.failure_boundary
        else:
            assert result.failure_category is None
            assert result.failure_boundary is None


def test_runner_is_deterministic_and_reports_only_safe_result_fields() -> None:
    manifest = load_conformance_manifest()

    first = run_conformance(manifest, covered_case_ids=FOCUSED_CASE_IDS)
    second = run_conformance(manifest, covered_case_ids=FOCUSED_CASE_IDS)

    assert first.to_json() == second.to_json()
    assert set(first.to_dict()) == {"results"}
    assert all(
        set(result.to_dict()) == {"case_id", "failure_category"}
        for result in first.results
    )
    report_text = first.to_json()
    assert "Bearer " not in report_text
    assert "Authorization:" not in report_text
    assert "mcp.synthetic.invalid" not in report_text
    assert "clinical payload" not in report_text


def test_positive_and_negative_cases_complete_without_network_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = MockAuthorizationTransport()

    def fail_network(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("authorization conformance attempted network egress")

    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_network)
    monkeypatch.setattr(socket, "create_connection", fail_network)

    report = run_conformance(
        load_conformance_manifest(),
        transport=transport,
        covered_case_ids=FOCUSED_CASE_IDS,
    )

    assert report.ok
    assert "tool_call" in transport.operations
    assert "token_exchange" in transport.operations


def test_matrix_is_generated_from_manifest_and_makes_no_certification_claim() -> None:
    manifest = load_conformance_manifest()
    generated = render_conformance_matrix(manifest)

    assert MATRIX_PATH.read_text(encoding="utf-8") == generated
    assert "not a certification claim" in generated
    for category in (
        "wrong_audience",
        "missing_resource_indicator",
        "token_passthrough",
        "invalid_pkce",
        "redirect_mismatch",
        "insufficient_tool_scope",
        "oversized_payload",
        "unapproved_state_change",
    ):
        assert category in generated


def test_manifest_and_errors_do_not_echo_sensitive_fixture_values() -> None:
    raw = json.loads(DEFAULT_FIXTURE_PATH.read_text(encoding="utf-8"))
    mutated = copy.deepcopy(raw)
    secret = "Bearer synthetic-token-should-never-surface"
    mutated["cases"][0]["tool_call"]["payload_bytes"] = secret

    with pytest.raises(ConformanceFixtureError) as error:
        ConformanceManifest.from_mapping(mutated)

    assert secret not in str(error.value)
