"""Synthetic server-validation, privacy and pre-review refusal contracts."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from openmed.core.offline import HF_OFFLINE_ENV_VARS, network_blocked_if_offline
from openmed.interop.fhir import (
    MAX_SERVER_VALIDATION_ISSUES,
    MAX_SERVER_VALIDATION_RESOURCE_BYTES,
    FHIRServerValidationIssue,
    FHIRServerValidationReason,
    FHIRServerValidationResult,
    FHIRServerValidationStatus,
    FHIRValidationResponse,
    preflight_server_validation,
)
from openmed.interop.fhir_capability_preflight import (
    FHIRWritePlan,
    preflight_write_plan,
)

VALIDATE = "http://hl7.org/fhir/OperationDefinition/Resource-validate"
SOURCE_SENTINEL = "SYNTHETIC-SOURCE-DO-NOT-EMIT-0042"


def _statement() -> dict[str, Any]:
    return {
        "resourceType": "CapabilityStatement",
        "fhirVersion": "4.0.1",
        "rest": [
            {
                "mode": "server",
                "resource": [
                    {
                        "type": "Patient",
                        "interaction": [{"code": "create"}, {"code": "update"}],
                        "operation": [{"name": "validate", "definition": VALIDATE}],
                    }
                ],
            }
        ],
    }


def _outcome(severity: str = "information") -> dict[str, Any]:
    return {
        "resourceType": "OperationOutcome",
        "issue": [
            {
                "severity": severity,
                "code": "informational" if severity == "information" else "required",
                "diagnostics": SOURCE_SENTINEL,
                "details": {"text": SOURCE_SENTINEL},
                "expression": ["Patient.name[42].family"],
            }
        ],
        "text": {"div": SOURCE_SENTINEL},
    }


class _Handle:
    def __repr__(self) -> str:
        raise AssertionError("credential handle must not be rendered")


class _Transport:
    def __init__(self, outcome: Any = None, *, status_code: int = 200) -> None:
        self.outcome = _outcome() if outcome is None else outcome
        self.status_code = status_code
        self.calls: list[dict[str, Any]] = []

    def validate(self, resource_type: str, **kwargs: Any) -> FHIRValidationResponse:
        self.calls.append({"resource_type": resource_type, **kwargs})
        return FHIRValidationResponse(self.status_code, self.outcome)


def _run(default_transport: Any, **overrides: Any) -> FHIRServerValidationResult:
    arguments: dict[str, Any] = {
        "mode": "create",
        "enabled": True,
        "transport": default_transport,
        "credential_handle": _Handle(),
    }
    arguments.update(overrides)
    statement = arguments.pop("statement", _statement())
    resource = arguments.pop(
        "resource",
        {
            "resourceType": "Patient",
            "name": [{"family": SOURCE_SENTINEL}],
        },
    )
    return preflight_server_validation(statement, resource, **arguments)


def test_default_disabled_does_not_inspect_inputs_or_transport() -> None:
    class Untouched(Mapping[str, Any]):
        def __getitem__(self, key: str) -> Any:
            pytest.fail("disabled preflight must not inspect protected inputs")

        def __iter__(self) -> Any:
            pytest.fail("disabled preflight must not inspect protected inputs")

        def __len__(self) -> int:
            pytest.fail("disabled preflight must not inspect protected inputs")

    class UntouchedTransport:
        @property
        def validate(self) -> Any:
            pytest.fail("disabled preflight must not resolve transport")

    result = preflight_server_validation(
        Untouched(), Untouched(), mode="update", transport=UntouchedTransport()
    )
    assert result.status is FHIRServerValidationStatus.DISABLED
    assert not result.is_valid
    assert result.request_digest is None


@pytest.mark.parametrize("mode", ["create", "update"])
def test_target_bound_transport_receives_intended_mode_and_only_update_instance(
    mode: str,
) -> None:
    transport = _Transport()
    resource = {"resourceType": "Patient", "name": [{"family": SOURCE_SENTINEL}]}
    handle = _Handle()
    arguments = {"resource_id": "synthetic-instance"} if mode == "update" else {}
    result = _run(
        transport, mode=mode, resource=resource, credential_handle=handle, **arguments
    )
    assert result.is_valid
    assert len(transport.calls) == 1
    request = transport.calls[0]
    assert request["resource_type"] == "Patient"
    assert request["instance_id"] == arguments.get("resource_id")
    assert request["credential_handle"] is handle
    assert request["timeout_seconds"] == 10.0
    assert request["parameters"]["resourceType"] == "Parameters"
    assert request["parameters"]["parameter"][0] == {"name": "mode", "valueCode": mode}
    sent = request["parameters"]["parameter"][1]["resource"]
    assert sent == resource and sent is not resource
    assert sent["name"] is not resource["name"]
    assert len(result.request_digest or "") == 64


@pytest.mark.parametrize(
    "severity,valid",
    [("fatal", False), ("error", False), ("warning", True), ("information", True)],
)
def test_http_200_is_classified_by_issues_not_transport_success(
    severity: str, valid: bool
) -> None:
    result = _run(_Transport(_outcome(severity)))
    assert result.is_valid is valid
    assert result.reason_code is (
        FHIRServerValidationReason.VALIDATED
        if valid
        else FHIRServerValidationReason.ERROR_ISSUES
    )
    assert result.issues[0].severity == severity
    assert result.issues[0].element_paths == ("Patient.name[].family",)


def test_warning_policy_can_stop_preview() -> None:
    result = _run(_Transport(_outcome("warning")), block_warnings=True)
    assert not result.is_valid
    assert result.reason_code is FHIRServerValidationReason.WARNING_ISSUES


def test_nominated_profile_is_in_protected_parameters_and_request_binding_only() -> (
    None
):
    profile = "http://hl7.org/fhir/StructureDefinition/Patient"
    transport = _Transport()
    result = _run(transport, profile_uri=profile)
    assert result.is_valid
    assert transport.calls[0]["parameters"]["parameter"][2] == {
        "name": "profile",
        "valueUri": profile,
    }
    assert profile not in json.dumps(result.to_dict())
    assert result.request_digest != _run(_Transport()).request_digest


@pytest.mark.parametrize(
    "profile",
    [
        "",
        "file:/synthetic/schema",
        "https://synthetic.invalid/profile?secret=x",
        "https://user:secret@synthetic.invalid/profile",
        "urn:synthetic:profile\n",
        42,
    ],
)
def test_invalid_profile_is_not_sent_or_reflected(profile: Any) -> None:
    transport = _Transport()
    result = _run(transport, profile_uri=profile)
    assert result.reason_code is FHIRServerValidationReason.CONFIGURATION_INVALID
    assert transport.calls == []


def test_response_repr_excludes_protected_outcome() -> None:
    response = FHIRValidationResponse(200, _outcome())
    assert SOURCE_SENTINEL not in repr(response)


@pytest.mark.parametrize(
    "status", [201, 204, 301, 400, 401, 403, 404, 405, 429, 500, 501, 503]
)
def test_non_200_never_proves_validation_even_with_informational_outcome(
    status: int,
) -> None:
    transport = _Transport(status_code=status)
    result = _run(transport)
    assert not result.is_valid
    assert result.reason_code is FHIRServerValidationReason.VALIDATION_UNAVAILABLE
    assert result.issues == ()
    assert len(transport.calls) == 1


def test_transport_exception_is_value_free_and_not_retried(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Failing:
        calls = 0

        def validate(self, *args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            raise RuntimeError(SOURCE_SENTINEL)

    transport = Failing()
    result = _run(transport)
    assert result.reason_code is FHIRServerValidationReason.TRANSPORT_FAILED
    assert transport.calls == 1
    assert SOURCE_SENTINEL not in repr(result)
    assert SOURCE_SENTINEL not in json.dumps(result.to_dict())
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    "shape",
    [
        "absent",
        "wrong_definition",
        "client_only",
        "other_resource",
        "conflicting",
        "cross_rest",
    ],
)
def test_unsupported_cached_operation_stops_before_transport_resolution(
    shape: str,
) -> None:
    statement = _statement()
    target = statement["rest"][0]["resource"][0]
    if shape == "absent":
        target.pop("operation")
    elif shape == "wrong_definition":
        target["operation"][0]["definition"] = "urn:synthetic:other-operation"
    elif shape == "client_only":
        statement["rest"][0]["mode"] = "client"
    elif shape == "other_resource":
        target["type"] = "Observation"
    elif shape == "conflicting":
        target["operation"].append(
            {"name": "validate", "definition": "urn:synthetic:conflict"}
        )
    else:
        target.pop("operation")
        statement["rest"].append(
            {
                "mode": "server",
                "operation": [{"name": "validate", "definition": VALIDATE}],
            }
        )

    class Unresolved:
        @property
        def validate(self) -> Any:
            pytest.fail("unsupported metadata must stop before transport resolution")

    result = _run(Unresolved(), statement=statement)
    assert result.status is FHIRServerValidationStatus.UNSUPPORTED
    assert result.request_digest is None


def test_rest_level_operation_and_known_canonical_version_are_supported() -> None:
    statement = _statement()
    statement["rest"][0]["resource"][0].pop("operation")
    statement["rest"][0]["operation"] = [
        {"name": "validate", "definition": VALIDATE + "|4.0.1"}
    ]
    assert _run(_Transport(), statement=statement).is_valid


@pytest.mark.parametrize(
    "operation",
    [
        None,
        {},
        [None],
        [{}],
        [{"name": "validate"}],
        [{"name": SOURCE_SENTINEL, "definition": None}],
    ],
)
def test_malformed_operation_metadata_never_sends_resource(operation: Any) -> None:
    statement = _statement()
    statement["rest"][0]["resource"][0]["operation"] = operation
    transport = _Transport()
    result = _run(transport, statement=statement)
    assert result.reason_code is FHIRServerValidationReason.CAPABILITY_MALFORMED
    assert transport.calls == []
    assert SOURCE_SENTINEL not in json.dumps(result.to_dict())


def test_operation_bound_is_not_truncated_to_a_supported_entry() -> None:
    statement = _statement()
    statement["rest"][0]["resource"][0]["operation"] *= 257
    transport = _Transport()
    assert (
        _run(transport, statement=statement).reason_code
        is FHIRServerValidationReason.CAPABILITY_MALFORMED
    )
    assert transport.calls == []


def test_r5_metadata_is_explicitly_unsupported_without_contact() -> None:
    statement = _statement()
    statement["fhirVersion"] = "5.0.0"
    transport = _Transport()
    assert (
        _run(transport, statement=statement).reason_code
        is FHIRServerValidationReason.VERSION_UNSUPPORTED
    )
    assert transport.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": 1},
        {"mode": "delete"},
        {"mode": SOURCE_SENTINEL},
        {"block_warnings": "yes"},
        {"timeout_seconds": True},
        {"timeout_seconds": 0},
        {"timeout_seconds": 61},
        {"timeout_seconds": float("inf")},
        {"timeout_seconds": float("nan")},
        {"credential_handle": None},
        {"credential_handle": SOURCE_SENTINEL},
        {"transport": None},
    ],
)
def test_invalid_configuration_is_controlled_and_makes_no_request(
    kwargs: dict[str, Any],
) -> None:
    transport = _Transport()
    result = _run(transport, **kwargs)
    assert result.reason_code is FHIRServerValidationReason.CONFIGURATION_INVALID
    assert transport.calls == []
    assert SOURCE_SENTINEL not in repr(result)


def test_unbounded_timeout_and_literal_credential_collections_are_refused() -> None:
    for arguments in (
        {"timeout_seconds": 10**1000},
        {"credential_handle": {"secret": SOURCE_SENTINEL}},
        {"credential_handle": [SOURCE_SENTINEL]},
    ):
        transport = _Transport()
        result = _run(transport, **arguments)
        assert result.reason_code is FHIRServerValidationReason.CONFIGURATION_INVALID
        assert transport.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "update"},
        {"mode": "update", "resource_id": "Patient/0042"},
        {
            "mode": "update",
            "resource_id": "synthetic-target",
            "resource": {"resourceType": "Patient", "id": "synthetic-other"},
        },
        {"resource_id": "synthetic-instance"},
        {"resource": {"resourceType": "Patient", "x": float("nan")}},
        {"resource": {"resourceType": "Patient", "x": object()}},
        {"resource": {"resourceType": "Patient", 42: SOURCE_SENTINEL}},
        {"resource": {"resourceType": SOURCE_SENTINEL}},
        {
            "resource": {
                "resourceType": "Patient",
                "x": "x" * (MAX_SERVER_VALIDATION_RESOURCE_BYTES + 1),
            }
        },
        {
            "resource": {
                "resourceType": "Patient",
                "x": "\u4e2d" * (MAX_SERVER_VALIDATION_RESOURCE_BYTES // 2),
            }
        },
    ],
)
def test_invalid_protected_resource_or_instance_is_not_sent(
    kwargs: dict[str, Any],
) -> None:
    transport = _Transport()
    result = _run(transport, **kwargs)
    assert result.reason_code is FHIRServerValidationReason.RESOURCE_INVALID
    assert transport.calls == []
    assert SOURCE_SENTINEL not in repr(result)


def test_cyclic_deep_and_excessive_node_inputs_are_refused_without_request() -> None:
    cyclic: dict[str, Any] = {"resourceType": "Patient"}
    cyclic["self"] = cyclic
    deep: dict[str, Any] = {"resourceType": "Patient"}
    current = deep
    for _ in range(33):
        current["child"] = {}
        current = current["child"]
    for resource in (cyclic, deep, {"resourceType": "Patient", "x": [None] * 65_537}):
        transport = _Transport()
        assert (
            _run(transport, resource=resource).reason_code
            is FHIRServerValidationReason.RESOURCE_INVALID
        )
        assert transport.calls == []


@pytest.mark.parametrize(
    "outcome",
    [
        {},
        {"resourceType": "Patient"},
        {"resourceType": "OperationOutcome", "issue": []},
        {"resourceType": "OperationOutcome", "issue": [{}]},
        {
            "resourceType": "OperationOutcome",
            "issue": [{"severity": SOURCE_SENTINEL, "code": "required"}],
        },
        {
            "resourceType": "OperationOutcome",
            "issue": [{"severity": "warning", "code": SOURCE_SENTINEL}],
        },
        {
            "resourceType": "OperationOutcome",
            "issue": [
                {
                    "severity": "warning",
                    "code": "required",
                    "expression": SOURCE_SENTINEL,
                }
            ],
        },
    ],
)
def test_malformed_server_outcomes_cannot_pass_or_echo_source(outcome: Any) -> None:
    result = _run(_Transport(outcome))
    assert result.reason_code is FHIRServerValidationReason.OUTCOME_MALFORMED
    assert SOURCE_SENTINEL not in repr(result)
    assert SOURCE_SENTINEL not in json.dumps(result.to_dict())


def test_issue_and_expression_bounds_refuse_instead_of_hiding_late_errors() -> None:
    for key, values in (
        ("issues", [_outcome()["issue"][0]] * (MAX_SERVER_VALIDATION_ISSUES + 1)),
        ("expressions", ["Patient.name"] * 17),
    ):
        outcome = _outcome()
        if key == "issues":
            outcome["issue"] = values
        else:
            outcome["issue"][0]["expression"] = values
        assert (
            _run(_Transport(outcome)).reason_code
            is FHIRServerValidationReason.OUTCOME_MALFORMED
        )


def test_all_protected_response_fields_are_discarded_with_conservative_paths() -> None:
    outcome = _outcome("warning")
    outcome["issue"][0]["expression"] = [
        "Patient.name[42].family",
        "Patient.name[].given",
        "Patient.identifier[0042].value",
        "Patient.name.where(family = '" + SOURCE_SENTINEL + "')",
        "Patient.name = 0042",
        "Patient/0042",
        "Patient." + SOURCE_SENTINEL,
        "Patient.name\u200b.family",
        "Patient.name\n.family",
        "Patient.\u540d\u5b57",
        "Observation.code",
        "Patient.foo.identifier",
        {"secret": SOURCE_SENTINEL},
    ]
    outcome["issue"][0]["location"] = [SOURCE_SENTINEL]
    outcome["issue"][0]["details"]["coding"] = [{"code": SOURCE_SENTINEL}]
    result = _run(_Transport(outcome))
    assert result.is_valid
    assert result.issues[0].element_paths == (
        "Patient.identifier[].value",
        "Patient.name[].family",
        "Patient.name[].given",
    )
    assert result.discarded_path_count == 10
    rendered = json.dumps(result.to_dict()) + repr(result) + json.dumps(asdict(result))
    assert SOURCE_SENTINEL not in rendered
    assert "Patient/0042" not in rendered
    assert "[0042]" not in rendered
    assert "diagnostics" not in rendered
    assert "location" not in rendered
    assert "where(" not in rendered
    assert "\u540d\u5b57" not in rendered


def test_identical_inputs_are_deterministic_and_changed_request_has_new_digest() -> (
    None
):
    before = _statement()
    outcome = _outcome()
    original = copy.deepcopy(outcome)
    first = _run(_Transport(outcome))
    second = _run(_Transport(outcome))
    assert first.to_dict() == second.to_dict()
    assert outcome == original
    assert before == _statement()
    changed = _run(_Transport(), resource={"resourceType": "Patient", "active": False})
    update = _run(_Transport(), mode="update", resource_id="synthetic-instance")
    assert (
        len({first.request_digest, changed.request_digest, update.request_digest}) == 3
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"severity": SOURCE_SENTINEL, "code": "required"},
        {"severity": "error", "code": SOURCE_SENTINEL},
        {
            "severity": "error",
            "code": "required",
            "element_paths": ("Patient.name.where(value = '0042')",),
        },
    ],
)
def test_public_issue_constructor_refuses_value_bearing_metadata(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError) as error:
        FHIRServerValidationIssue(**kwargs)
    assert SOURCE_SENTINEL not in str(error.value)
    assert "0042" not in str(error.value)


@pytest.mark.integration
@pytest.mark.parametrize(
    "severity,preview_count", [("error", 0), ("warning", 1), ("information", 1)]
)
def test_offline_application_checks_capability_and_validation_before_preview(
    monkeypatch: pytest.MonkeyPatch, severity: str, preview_count: int
) -> None:
    for name in HF_OFFLINE_ENV_VARS:
        monkeypatch.setenv(name, "1")
    previews: list[str] = []
    transport = _Transport(_outcome(severity))
    with network_blocked_if_offline(local_only=True):
        capability = preflight_write_plan(
            _statement(), FHIRWritePlan("create", "Patient")
        )
        if capability.is_compatible:
            validation = _run(transport)
            if validation.is_valid:
                previews.append("synthetic-preview")
    assert len(previews) == preview_count
    assert len(transport.calls) == 1


def test_documented_fake_transport_example_runs_offline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from openmed.core.models import ModelLoader

    def refuse_model(*args: Any, **kwargs: Any) -> None:
        pytest.fail("server-validation example must not load a model")

    monkeypatch.setattr(ModelLoader, "load_model", refuse_model)
    for name in HF_OFFLINE_ENV_VARS:
        monkeypatch.setenv(name, "1")
    guide = Path(__file__).resolve().parents[4] / "docs/interop/fhir-write-preflight.md"
    blocks = re.findall(
        r"(?ms)^```python\n(.*?)^```", guide.read_text(encoding="utf-8")
    )
    runnable = [block for block in blocks if block.startswith("# Runnable:")]
    assert runnable
    with network_blocked_if_offline(local_only=True):
        for block in runnable:
            exec(compile(block, "server_validation_example.py", "exec"), {})
    captured = capsys.readouterr()
    assert captured.out == "{'status': 'blocked', 'reason_code': 'error_issues'}\n"
    assert captured.err == ""
