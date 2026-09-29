from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from importlib import resources
from typing import Any, get_args

import pytest
from jsonschema import Draft202012Validator

from openmed.risk.budget import (
    CURRENT_EPSILON_POLICY_SCHEMA_VERSION,
    EPSILON_POLICY_CONFIG_RESOURCE,
    CompositionRule,
    EpsilonPolicy,
    load_epsilon_policies,
)
from openmed.risk.privacy_budget_schema import (
    EPSILON_POLICY_JSON_SCHEMA_ID,
    MAX_FINITE_POLICY_NUMBER,
    PRIVACY_BUDGET_POLICY_JSON_SCHEMA_ID,
    export_epsilon_policy_schema,
    export_epsilon_policy_schema_json,
    export_privacy_budget_policy_schema,
    export_privacy_budget_policy_schema_json,
)

_SCOPE = "clinical_release_default"
_CONFIG_SHA256 = "6034c30fdd69c5f80db7e360f6cbf4a5617917e70eba52d91d878292b01b4349"
_RECORD_SHA256 = "247f1f0464b7e0fe8880796d0fd40be88223bc828045a412f80cb863a97cd24e"


def _policy(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "max_epsilon": 3.0,
        "max_delta": 1e-06,
        "composition": "advanced",
        "delta_prime": 1e-09,
    }
    payload.update(overrides)
    return payload


def _config(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": CURRENT_EPSILON_POLICY_SCHEMA_VERSION,
        "policies": {_SCOPE: _policy() if policy is None else policy},
    }


def _committed_config() -> dict[str, Any]:
    resource = (
        resources.files("openmed.risk")
        .joinpath("data")
        .joinpath(EPSILON_POLICY_CONFIG_RESOURCE)
    )
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _config_validator() -> Draft202012Validator:
    schema = export_privacy_budget_policy_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _record_validator() -> Draft202012Validator:
    schema = export_epsilon_policy_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _set(payload: dict[str, Any], path: tuple[Any, ...], value: Any) -> dict[str, Any]:
    cursor: Any = payload
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value
    return payload


def _refs(value: Any) -> list[str]:
    if isinstance(value, dict):
        refs = [value["$ref"]] if "$ref" in value else []
        for nested in value.values():
            refs.extend(_refs(nested))
        return refs
    if isinstance(value, list):
        return [ref for nested in value for ref in _refs(nested)]
    return []


@pytest.mark.parametrize(
    ("schema", "expected_id"),
    [
        (export_privacy_budget_policy_schema(), PRIVACY_BUDGET_POLICY_JSON_SCHEMA_ID),
        (export_epsilon_policy_schema(), EPSILON_POLICY_JSON_SCHEMA_ID),
    ],
)
def test_schemas_are_draft_2020_12_and_resolve_entirely_locally(
    schema: dict[str, Any],
    expected_id: str,
) -> None:
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == expected_id
    assert _refs(schema)
    assert all(ref.startswith("#/$defs/") for ref in _refs(schema))
    assert set(_refs(schema)) <= {f"#/$defs/{name}" for name in schema["$defs"]}
    Draft202012Validator.check_schema(schema)


def test_exported_schema_ids_are_unique() -> None:
    assert PRIVACY_BUDGET_POLICY_JSON_SCHEMA_ID != EPSILON_POLICY_JSON_SCHEMA_ID


def test_committed_policy_config_satisfies_the_exported_schema() -> None:
    committed = _committed_config()

    _config_validator().validate(committed)
    assert set(committed["policies"]) == set(load_epsilon_policies())


def test_committed_policy_records_satisfy_the_record_schema() -> None:
    validator = _record_validator()

    for scope, policy in load_epsilon_policies().items():
        record = policy.to_dict()
        assert record["scope"] == scope
        validator.validate(record)


def test_record_schema_tracks_the_runtime_policy_fields() -> None:
    schema = export_epsilon_policy_schema()

    assert schema["required"] == [field.name for field in fields(EpsilonPolicy)]
    assert set(schema["properties"]) == {field.name for field in fields(EpsilonPolicy)}


def test_schema_tracks_the_closed_composition_enum() -> None:
    schema = export_privacy_budget_policy_schema()
    expected = list(get_args(CompositionRule))

    assert schema["$defs"]["compositionRule"]["enum"] == expected
    assert expected == ["basic", "advanced"]
    for rule in ("rdp", "gdp", "advanced-composition", ""):
        payload = _config(_policy(composition=rule))
        assert not _config_validator().is_valid(payload)
        with pytest.raises(ValueError):
            EpsilonPolicy.from_mapping(_SCOPE, _policy(composition=rule))


@pytest.mark.parametrize("field", ["max_epsilon", "max_delta", "delta_prime"])
@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_schema_and_loader_reject_infinite_policy_numbers(
    field: str,
    value: float,
) -> None:
    policy = _policy(**{field: value})
    record = EpsilonPolicy(
        scope=_SCOPE,
        max_epsilon=1.0,
        max_delta=1e-06,
    ).to_dict()

    assert not _config_validator().is_valid(_config(policy))
    assert not _record_validator().is_valid(record | {field: value})
    with pytest.raises(ValueError):
        EpsilonPolicy.from_mapping(_SCOPE, policy)


@pytest.mark.parametrize("field", ["max_epsilon", "max_delta", "delta_prime"])
def test_loader_rejects_nan_which_has_no_json_representation(field: str) -> None:
    policy = _policy(**{field: float("nan")})

    with pytest.raises(ValueError):
        EpsilonPolicy.from_mapping(_SCOPE, policy)
    with pytest.raises(ValueError):
        json.dumps(_config(policy), allow_nan=False)
    assert json.dumps(_config(_policy()), allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_epsilon", 0.0),
        ("max_epsilon", -1.5),
        ("max_delta", 0.0),
        ("max_delta", 1.0),
        ("max_delta", 1.5),
        ("delta_prime", -1e-09),
    ],
)
def test_schema_and_loader_agree_on_bound_violations(
    field: str,
    value: float,
) -> None:
    policy = _policy(**{field: value})

    assert not _config_validator().is_valid(_config(policy))
    with pytest.raises(ValueError):
        EpsilonPolicy.from_mapping(_SCOPE, policy)


def test_schema_binds_finite_numeric_bounds() -> None:
    definitions = export_privacy_budget_policy_schema()["$defs"]

    assert definitions["epsilonBound"]["maximum"] == MAX_FINITE_POLICY_NUMBER
    assert definitions["epsilonBound"]["exclusiveMinimum"] == 0
    assert definitions["deltaBound"]["exclusiveMinimum"] == 0
    assert definitions["deltaBound"]["exclusiveMaximum"] == 1
    assert definitions["deltaPrime"]["minimum"] == 0
    assert definitions["deltaPrime"]["maximum"] == MAX_FINITE_POLICY_NUMBER


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), 2),
        (("policies",), {}),
        (("policies",), []),
        (("policies", _SCOPE), {}),
        (("policies", _SCOPE, "max_epsilon"), "3.0"),
        (("policies", _SCOPE, "max_epsilon"), True),
        (("policies", _SCOPE, "composition"), "rdp"),
        (("policies", _SCOPE, "unbounded"), True),
        (("extra",), True),
    ],
)
def test_schema_rejects_permissive_and_malformed_policies(
    path: tuple[Any, ...],
    value: Any,
) -> None:
    assert not _config_validator().is_valid(_set(_config(), path, value))


@pytest.mark.parametrize("field", ["schema_version", "policies"])
def test_schema_rejects_incomplete_policy_documents(field: str) -> None:
    payload = _config()
    payload.pop(field)

    assert not _config_validator().is_valid(payload)


def test_schema_requires_advanced_composition_slack() -> None:
    without_slack = _config(_policy()).copy()
    without_slack["policies"][_SCOPE].pop("delta_prime")
    zero_slack = _config(_policy(delta_prime=0.0))
    basic = _config(_policy(composition="basic", delta_prime=0.0))

    assert not _config_validator().is_valid(without_slack)
    assert not _config_validator().is_valid(zero_slack)
    assert _config_validator().is_valid(basic)
    with pytest.raises(ValueError):
        EpsilonPolicy.from_mapping(_SCOPE, _policy(delta_prime=0.0))


def test_schema_defaults_a_missing_composition_to_basic() -> None:
    policy = {"max_epsilon": 8.0, "max_delta": 1e-05}
    payload = _config(policy)

    assert _config_validator().is_valid(payload)
    loaded = EpsilonPolicy.from_mapping(_SCOPE, policy)
    assert loaded.composition == "basic"
    assert loaded.delta_prime == 0.0
    _record_validator().validate(loaded.to_dict())


def test_schema_rejects_unknown_record_fields() -> None:
    record = EpsilonPolicy.from_mapping(_SCOPE, _policy()).to_dict()

    assert _record_validator().is_valid(record)
    record["accountant"] = "rdp"
    assert not _record_validator().is_valid(record)


def test_every_schema_object_is_closed() -> None:
    config_schema = export_privacy_budget_policy_schema()
    record_schema = export_epsilon_policy_schema()

    assert config_schema["additionalProperties"] is False
    assert config_schema["$defs"]["policy"]["additionalProperties"] is False
    assert record_schema["additionalProperties"] is False
    assert config_schema["$defs"]["policyRecord"]["additionalProperties"] is False


def test_schema_exports_are_byte_stable() -> None:
    first_config = export_privacy_budget_policy_schema_json()
    second_config = export_privacy_budget_policy_schema_json()
    first_record = export_epsilon_policy_schema_json()
    second_record = export_epsilon_policy_schema_json()

    assert first_config == second_config
    assert first_record == second_record
    assert json.loads(first_config) == export_privacy_budget_policy_schema()
    assert json.loads(first_record) == export_epsilon_policy_schema()
    assert hashlib.sha256(first_config.encode("utf-8")).hexdigest() == _CONFIG_SHA256
    assert hashlib.sha256(first_record.encode("utf-8")).hexdigest() == _RECORD_SHA256
