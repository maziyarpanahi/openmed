"""Tests for the portable SDK configuration schema, vectors and projection."""

from __future__ import annotations

import json
import math
from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from openmed.core import portable_policy
from openmed.core.config import (
    LOW_RESOURCE_PII_MODEL,
    LOW_RESOURCE_PII_REVISION,
    OpenMedConfig,
    config_schema_path,
)
from openmed.core.offline import OFFLINE_ENV_VAR
from openmed.core.sdk_config import (
    ISSUE_CODES,
    ONNX_VARIANTS,
    OPENMED_CONFIG_FIELDS,
    POLICY_PROFILES,
    SDK_CONFIG_DEFAULTS,
    SDK_CONFIG_FIELDS,
    SDK_CONFIG_SCHEMA_ID,
    SdkConfigError,
    SdkConfigIssue,
    load_sdk_config_schema,
    resolve_sdk_config,
    validate_sdk_config,
)

ROOT = Path(__file__).resolve().parents[3]
VECTORS = ROOT / "tests" / "fixtures" / "parity" / "sdk_config_v1.json"
DOC = ROOT / "docs" / "configuration" / "sdk-config.md"


def _vectors() -> dict:
    return json.loads(VECTORS.read_text(encoding="utf-8"))


def _validator() -> Draft202012Validator:
    return Draft202012Validator(load_sdk_config_schema())


def _strings(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        found = set(value)
        for item in value.values():
            found |= _strings(item)
        return found
    if isinstance(value, list):
        return {text for item in value for text in _strings(item)}
    return set()


@pytest.fixture(autouse=True)
def _no_offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OFFLINE_ENV_VAR, raising=False)
    monkeypatch.delenv("OPENMED_PROFILE", raising=False)


def test_schema_is_valid_draft_2020_12_with_the_portable_keys() -> None:
    schema = load_sdk_config_schema()
    Draft202012Validator.check_schema(schema)

    assert schema["$id"] == SDK_CONFIG_SCHEMA_ID
    assert schema["additionalProperties"] is False
    assert tuple(sorted(schema["properties"])) == SDK_CONFIG_FIELDS
    assert {
        name: prop["default"] for name, prop in schema["properties"].items()
    } == dict(SDK_CONFIG_DEFAULTS)
    assert SDK_CONFIG_DEFAULTS["local_only"] is True


def test_schema_enums_match_python_and_the_portable_policy_subset() -> None:
    properties = load_sdk_config_schema()["properties"]

    assert tuple(properties["onnx_variant"]["enum"]) == ONNX_VARIANTS
    assert tuple(properties["policy_profile"]["enum"]) == POLICY_PROFILES
    assert POLICY_PROFILES == portable_policy.PORTABLE_PROFILES
    assert SDK_CONFIG_DEFAULTS["policy_profile"] == portable_policy.DEFAULT_PROFILE


def test_types_and_bounds_align_with_config_schema() -> None:
    config_schema = json.loads(config_schema_path().read_text(encoding="utf-8"))
    config_properties = config_schema["properties"]
    sdk_properties = load_sdk_config_schema()["properties"]

    assert config_properties["local_only"]["type"] == "boolean"
    assert sdk_properties["local_only"]["type"] == "boolean"
    assert (
        config_properties["onnx_variant"]["enum"]
        == sdk_properties["onnx_variant"]["enum"]
    )
    for sdk_name, config_name in (
        ("model_id", "pii_model"),
        ("model_revision", "pii_model_revision"),
    ):
        assert config_properties[config_name]["type"] == ["string", "null"]
        branch_types = {branch["type"] for branch in sdk_properties[sdk_name]["oneOf"]}
        assert branch_types == {"string", "null"}
    assert OpenMedConfig.__dataclass_fields__["onnx_variant"].default == "auto"
    assert SDK_CONFIG_DEFAULTS["onnx_variant"] == "auto"


def test_each_field_maps_to_openmed_config_or_is_documented_per_call() -> None:
    config_fields = {item.name for item in dataclass_fields(OpenMedConfig)}
    config_schema = json.loads(config_schema_path().read_text(encoding="utf-8"))
    document = DOC.read_text(encoding="utf-8")

    assert tuple(OPENMED_CONFIG_FIELDS) == SDK_CONFIG_FIELDS
    for sdk_name, config_name in OPENMED_CONFIG_FIELDS.items():
        assert f"`{sdk_name}`" in document
        if config_name is None:
            assert sdk_name not in config_fields
            continue
        assert config_name in config_fields
        assert config_name in config_schema["properties"]
        assert f"`OpenMedConfig.{config_name}`" in document


def test_fixture_header_is_synthetic_versioned_and_matches_python() -> None:
    vectors = _vectors()

    assert vectors["synthetic"] is True
    assert vectors["version"] == 1
    assert vectors["sdk_config_schema"] == SDK_CONFIG_SCHEMA_ID
    assert tuple(vectors["fields"]) == SDK_CONFIG_FIELDS
    assert vectors["defaults"] == dict(SDK_CONFIG_DEFAULTS)
    assert tuple(vectors["issue_codes"]) == ISSUE_CODES
    ids = [case["id"] for case in vectors["valid"] + vectors["invalid"]]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case", _vectors()["valid"], ids=lambda case: case["id"])
def test_valid_vectors_resolve_in_python_and_the_schema(case: dict) -> None:
    assert validate_sdk_config(case["config"]) == ()
    assert list(_validator().iter_errors(case["config"])) == []
    assert resolve_sdk_config(case["config"]) == case["resolved"]
    assert list(resolve_sdk_config(case["config"])) == list(SDK_CONFIG_FIELDS)


@pytest.mark.parametrize("case", _vectors()["invalid"], ids=lambda case: case["id"])
def test_invalid_vectors_fail_closed_with_the_expected_codes(case: dict) -> None:
    issues = validate_sdk_config(case["config"])

    assert [issue.to_dict() for issue in issues] == case["errors"]
    assert next(_validator().iter_errors(case["config"]), None) is not None
    with pytest.raises(SdkConfigError) as raised:
        resolve_sdk_config(case["config"])
    assert raised.value.issues == issues


@pytest.mark.parametrize("case", _vectors()["invalid"], ids=lambda case: case["id"])
def test_error_expectations_are_value_free(case: dict) -> None:
    for error in case["errors"]:
        assert set(error) == {"code", "field"}
        assert error["code"] in ISSUE_CODES
        assert error["field"] is None or error["field"] in SDK_CONFIG_FIELDS
    controlled = set(ISSUE_CODES) | set(SDK_CONFIG_FIELDS)
    assert _strings(case["errors"]) <= controlled | {"code", "field"}
    message = str(SdkConfigError(validate_sdk_config(case["config"])))
    prefix = "invalid SDK configuration: "
    assert message.startswith(prefix)
    tokens = {
        token
        for item in message.removeprefix(prefix).split(", ")
        for token in item.split(":")
    }
    assert tokens <= controlled


def test_unknown_keys_fail_closed_without_naming_them() -> None:
    issues = validate_sdk_config({"synthetic_secret_key": 1})

    assert issues == (SdkConfigIssue("unknown_field", None),)
    with pytest.raises(SdkConfigError) as raised:
        resolve_sdk_config({"synthetic_secret_key": 1})
    assert "synthetic_secret_key" not in str(raised.value)


def test_non_finite_threshold_is_out_of_range() -> None:
    for value in (math.nan, math.inf, -math.inf):
        assert validate_sdk_config({"confidence_threshold": value}) == (
            SdkConfigIssue("out_of_range", "confidence_threshold"),
        )


def test_to_sdk_config_projects_the_default_config() -> None:
    projected = OpenMedConfig().to_sdk_config()

    assert projected == {
        "confidence_threshold": 0.5,
        "local_only": False,
        "model_id": None,
        "model_revision": None,
        "onnx_variant": "auto",
        "policy_profile": "hipaa_safe_harbor",
    }
    assert list(_validator().iter_errors(projected)) == []


def test_to_sdk_config_projects_pinned_profile_settings() -> None:
    config = OpenMedConfig.from_profile("low_resource", local_only=True)

    projected = config.to_sdk_config(
        confidence_threshold=0.7, policy_profile="strict_no_leak"
    )

    assert projected == {
        "confidence_threshold": 0.7,
        "local_only": True,
        "model_id": LOW_RESOURCE_PII_MODEL,
        "model_revision": LOW_RESOURCE_PII_REVISION,
        "onnx_variant": "int8",
        "policy_profile": "strict_no_leak",
    }
    assert resolve_sdk_config(projected) == projected


@pytest.mark.parametrize(
    ("config", "kwargs", "issue"),
    [
        (
            {"pii_model": "/synthetic/private/models/pii"},
            {},
            SdkConfigIssue("invalid_format", "model_id"),
        ),
        (
            {"pii_model_revision": "main"},
            {},
            SdkConfigIssue("missing_dependency", "model_revision"),
        ),
        (
            {},
            {"confidence_threshold": 1.2},
            SdkConfigIssue("out_of_range", "confidence_threshold"),
        ),
        (
            {},
            {"policy_profile": "china_pipl"},
            SdkConfigIssue("not_allowed_value", "policy_profile"),
        ),
    ],
)
def test_to_sdk_config_fails_closed_outside_the_subset(
    config: dict, kwargs: dict, issue: SdkConfigIssue
) -> None:
    with pytest.raises(SdkConfigError) as raised:
        OpenMedConfig(**config).to_sdk_config(**kwargs)

    assert raised.value.issues == (issue,)
    assert "private" not in str(raised.value)
    assert "china_pipl" not in str(raised.value)
