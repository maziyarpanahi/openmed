"""Focused tests for the offline service API compatibility gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openmed.service.api_compatibility import (
    ERROR_CATEGORIES_EXTENSION,
    STABLE_ERROR_CATEGORIES,
    APICompatibilityError,
    APIContractError,
    build_live_contract,
    check_api_compatibility,
    compare_contracts,
    discover_service_error_categories,
    load_contract,
)


def _contract(
    *,
    include_health: bool = True,
    required_fields: list[str] | None = None,
    error_categories: list[str] | None = None,
) -> dict[str, Any]:
    paths: dict[str, Any] = {
        "/analyze": {
            "post": {
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/AnalyzeRequest"}
                        }
                    }
                }
            }
        }
    }
    if include_health:
        paths["/health"] = {"get": {"responses": {"200": {}}}}
    fields = required_fields if required_fields is not None else ["text"]
    return {
        "openapi": "3.1.0",
        "info": {"title": "Synthetic service", "version": "0.0.0"},
        "paths": paths,
        "components": {
            "schemas": {
                "AnalyzeRequest": {
                    "type": "object",
                    "properties": {
                        field: {"type": "string"}
                        for field in {"text", "synthetic_required", "optional_mode"}
                    },
                    "required": fields,
                    "example": "SYNTHETIC_INPUT_SENTINEL",
                }
            }
        },
        ERROR_CATEGORIES_EXTENSION: (
            error_categories
            if error_categories is not None
            else list(STABLE_ERROR_CATEGORIES)
        ),
    }


def test_matching_contract_is_deterministic_and_compatible() -> None:
    contract = _contract()

    first = compare_contracts(contract, contract)
    second = compare_contracts(contract, contract)

    assert first.is_compatible
    assert first.breaking == ()
    assert first.to_json() == second.to_json()
    assert first.to_dict()["summary"]["breaking"] == 0


def test_removed_route_required_field_and_error_category_are_breaking() -> None:
    before = _contract(error_categories=["validation_error", "timeout"])
    after = _contract(
        include_health=False,
        required_fields=["text", "synthetic_required"],
        error_categories=["validation_error"],
    )

    report = compare_contracts(before, after)
    changes = {issue.change for issue in report.breaking}

    assert changes == {
        "error_category_removed",
        "operation_removed",
        "required_field_added",
    }
    assert all(
        "SYNTHETIC_INPUT_SENTINEL" not in json.dumps(issue.to_dict())
        for issue in report.breaking
    )
    with pytest.raises(APICompatibilityError) as raised:
        report.assert_compatible()
    assert "SYNTHETIC_INPUT_SENTINEL" not in str(raised.value)
    assert "GET /health" in str(raised.value)


def test_additive_route_optional_field_and_error_category_are_allowed() -> None:
    before = _contract(error_categories=["validation_error"])
    after = _contract(error_categories=["validation_error", "new_category"])
    after["paths"]["/models"] = {"get": {"responses": {"200": {}}}}
    after["components"]["schemas"]["AnalyzeRequest"]["properties"]["optional_mode"] = {
        "type": "string"
    }

    report = compare_contracts(before, after)

    assert report.is_compatible
    assert {issue.change for issue in report.added} == {
        "error_category_added",
        "operation_added",
    }


def test_nested_required_field_uses_a_schema_path() -> None:
    before = _contract()
    before["components"]["schemas"]["AnalyzeRequest"]["properties"]["options"] = {
        "type": "object",
        "properties": {"mode": {"type": "string"}},
    }
    after = json.loads(json.dumps(before))
    after["components"]["schemas"]["AnalyzeRequest"]["properties"]["options"][
        "required"
    ] = ["mode"]

    report = compare_contracts(before, after)

    assert report.breaking[0].change == "required_field_added"
    assert report.breaking[0].schema_path == (
        "#/components/schemas/AnalyzeRequest/properties/options/properties/mode"
    )


def test_check_gate_reads_baseline_and_uses_an_offline_app(tmp_path: Path) -> None:
    baseline_path = tmp_path / "openapi.json"
    baseline = _contract()
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    class SyntheticApp:
        def openapi(self) -> dict[str, Any]:
            return baseline

    report = check_api_compatibility(baseline_path, app=SyntheticApp())

    assert report.is_compatible
    assert (
        load_contract(baseline_path).operation_keys
        == build_live_contract(SyntheticApp()).operation_keys
    )


def test_invalid_error_categories_are_rejected_without_echoing_values() -> None:
    contract = _contract(error_categories=["SYNTHETIC_INPUT_SENTINEL"])

    with pytest.raises(APIContractError) as raised:
        compare_contracts(contract, contract)

    assert "SYNTHETIC_INPUT_SENTINEL" not in str(raised.value)


def test_checked_in_contract_matches_live_service_surface() -> None:
    report = check_api_compatibility()

    assert report.is_compatible


def test_live_error_categories_are_discovered_from_local_service_source() -> None:
    assert discover_service_error_categories() == STABLE_ERROR_CATEGORIES
