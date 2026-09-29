"""Offline tests for anonymous federated capability/requirement comparison."""

from __future__ import annotations

import hashlib
import json
import socket

import pytest

from openmed.training import federated_compatibility as compat_module
from openmed.training.federated_compatibility import (
    DEFAULT_MAX_DECLARED_CAPABILITIES,
    FEDERATED_COMPATIBILITY_FIELDS,
    FEDERATED_COMPATIBILITY_REASON_CODES,
    FEDERATED_COMPATIBILITY_SCHEMA_VERSION,
    MAX_DECLARED_CAPABILITIES,
    MAX_PROTOCOL_VERSION,
    MIN_PROTOCOL_VERSION,
    FederatedClientCapabilityEnvelope,
    FederatedCompatibilityError,
    FederatedCompatibilityFinding,
    FederatedCompatibilityReasonCode,
    FederatedCompatibilityReport,
    FederatedCompatibilityVerdict,
    FederatedResourceClass,
    FederatedRoundRequirement,
    FederatedSecureAggregationMode,
    FederatedTrainingBackend,
    check_federated_compatibility,
)
from openmed.training.federated_metrics import FederatedPrivacyMechanism
from tests.fixtures.private_learning_forbidden import FORBIDDEN_FIELD_CASES

SENTINEL = "synthetic_private_sentinel_2982"
_GOLDEN_REPORT_SHA256 = (
    "25afd6aad5224cf0fd5cf5fc0cf552872cde2fd637bd2c6311b3d41f0b984f8d"
)

_SUPPORTED_REASONS = {
    "protocol_version": FederatedCompatibilityReasonCode.PROTOCOL_VERSION_SUPPORTED,
    "training_backend": FederatedCompatibilityReasonCode.TRAINING_BACKEND_SUPPORTED,
    "model_format": FederatedCompatibilityReasonCode.MODEL_FORMAT_SUPPORTED,
    "adapter_format": FederatedCompatibilityReasonCode.ADAPTER_FORMAT_SUPPORTED,
    "quantization_format": FederatedCompatibilityReasonCode.QUANTIZATION_SUPPORTED,
    "resource_class": FederatedCompatibilityReasonCode.RESOURCE_CLASS_SUFFICIENT,
    "deterministic_kernels": (
        FederatedCompatibilityReasonCode.DETERMINISTIC_KERNELS_SUPPORTED
    ),
    "privacy_mechanism": FederatedCompatibilityReasonCode.PRIVACY_MECHANISM_SUPPORTED,
    "secure_aggregation": FederatedCompatibilityReasonCode.SECURE_AGGREGATION_SUPPORTED,
}


def _requirement(**overrides: object) -> FederatedRoundRequirement:
    values: dict[str, object] = {
        "protocol_version": 2,
        "training_backend": FederatedTrainingBackend.TORCH,
        "model_format": "safetensors",
        "adapter_format": "dense",
        "quantization_format": None,
        "resource_class": FederatedResourceClass.MEDIUM,
        "deterministic_kernels": False,
        "privacy_mechanism": FederatedPrivacyMechanism.GAUSSIAN,
        "secure_aggregation": FederatedSecureAggregationMode.PAIRWISE_MASKING,
    }
    values.update(overrides)
    return FederatedRoundRequirement(**values)  # type: ignore[arg-type]


def _capability(**overrides: object) -> FederatedClientCapabilityEnvelope:
    values: dict[str, object] = {
        "training_backends": (FederatedTrainingBackend.TORCH,),
        "model_formats": ("safetensors",),
        "adapter_formats": ("dense",),
        "privacy_mechanisms": (FederatedPrivacyMechanism.GAUSSIAN,),
        "secure_aggregation_modes": (FederatedSecureAggregationMode.PAIRWISE_MASKING,),
        "quantization_formats": (),
        "minimum_protocol_version": 1,
        "maximum_protocol_version": 3,
        "resource_class": FederatedResourceClass.MEDIUM,
        "deterministic_kernels": False,
    }
    values.update(overrides)
    return FederatedClientCapabilityEnvelope(**values)  # type: ignore[arg-type]


def _required_reason(
    report: FederatedCompatibilityReport, field: str
) -> FederatedCompatibilityReasonCode | None:
    for finding in report.findings:
        if finding.field == field and finding.required:
            return finding.reason
    return None


def _optional_fields(report: FederatedCompatibilityReport) -> list[str]:
    return [
        finding.field
        for finding in report.findings
        if finding.reason
        is (FederatedCompatibilityReasonCode.OPTIONAL_CAPABILITY_DIFFERENCE)
    ]


def _cannot_reach_network(*args: object, **kwargs: object) -> None:
    raise AssertionError("network access attempted")


# --- constants --------------------------------------------------------------


def test_schema_version_constant() -> None:
    assert (
        FEDERATED_COMPATIBILITY_SCHEMA_VERSION
        == "openmed.training.federated_compatibility.v1"
    )


def test_reason_codes_are_a_sorted_closed_set() -> None:
    assert FEDERATED_COMPATIBILITY_REASON_CODES == tuple(
        sorted(FederatedCompatibilityReasonCode, key=lambda code: code.value)
    )
    assert len(FEDERATED_COMPATIBILITY_REASON_CODES) == 22
    assert len(set(FEDERATED_COMPATIBILITY_REASON_CODES)) == 22
    for code in FEDERATED_COMPATIBILITY_REASON_CODES:
        assert isinstance(code, str)
        assert code.value == code.value.lower()


def test_declared_capability_caps_and_protocol_bounds() -> None:
    assert DEFAULT_MAX_DECLARED_CAPABILITIES < MAX_DECLARED_CAPABILITIES
    assert MAX_DECLARED_CAPABILITIES == 128
    assert MIN_PROTOCOL_VERSION == 1
    assert MAX_PROTOCOL_VERSION == 10_000


def test_fields_are_ordered_and_closed() -> None:
    assert FEDERATED_COMPATIBILITY_FIELDS == (
        "protocol_version",
        "training_backend",
        "model_format",
        "adapter_format",
        "quantization_format",
        "resource_class",
        "deterministic_kernels",
        "privacy_mechanism",
        "secure_aggregation",
    )


def test_verdicts_and_backends_are_string_enums() -> None:
    assert [entry.value for entry in FederatedCompatibilityVerdict] == [
        "compatible",
        "review_required",
        "incompatible",
    ]
    assert [entry.value for entry in FederatedTrainingBackend] == [
        "torch",
        "mlx",
        "onnx",
        "coreml",
    ]
    assert [entry.value for entry in FederatedResourceClass] == [
        "small",
        "medium",
        "large",
        "xlarge",
    ]
    assert [entry.value for entry in FederatedSecureAggregationMode] == [
        "disabled",
        "pairwise_masking",
        "shamir",
    ]
    assert [entry.value for entry in FederatedPrivacyMechanism] == [
        "threshold_only",
        "laplace",
        "gaussian",
    ]


# --- comparison behaviour ---------------------------------------------------


def test_baseline_is_compatible() -> None:
    report = check_federated_compatibility(_requirement(), _capability())
    assert report.verdict is FederatedCompatibilityVerdict.COMPATIBLE
    assert report.ok is True
    assert report.incompatible_fields == ()
    assert report.review_fields == ()
    assert [finding.field for finding in report.findings] == [
        field
        for field in FEDERATED_COMPATIBILITY_FIELDS
        if field != "quantization_format"
    ]
    for finding in report.findings:
        assert finding.reason is _SUPPORTED_REASONS[finding.field]
        assert finding.required is True


@pytest.mark.parametrize(
    ("required_version", "minimum", "maximum", "reason"),
    [
        (2, 2, 2, "protocol_version_supported"),
        (1, 1, 3, "protocol_version_supported"),
        (3, 1, 3, "protocol_version_supported"),
        (1, 2, 4, "protocol_version_below_minimum"),
        (5, 2, 4, "protocol_version_above_maximum"),
        (1, None, None, "protocol_version_unknown"),
    ],
)
def test_protocol_version_table(
    required_version: int, minimum: int | None, maximum: int | None, reason: str
) -> None:
    report = check_federated_compatibility(
        _requirement(protocol_version=required_version),
        _capability(minimum_protocol_version=minimum, maximum_protocol_version=maximum),
    )
    assert _required_reason(report, "protocol_version") is (
        FederatedCompatibilityReasonCode(reason)
    )
    expected_verdict = (
        FederatedCompatibilityVerdict.COMPATIBLE
        if reason == "protocol_version_supported"
        else FederatedCompatibilityVerdict.INCOMPATIBLE
    )
    assert report.verdict is expected_verdict


def test_declared_protocol_range_flag() -> None:
    assert _capability().declares_protocol_range() is True
    undeclared = _capability(
        minimum_protocol_version=None, maximum_protocol_version=None
    )
    assert undeclared.declares_protocol_range() is False


@pytest.mark.parametrize(
    ("field", "override", "reason"),
    [
        (
            "training_backend",
            {"training_backends": (FederatedTrainingBackend.MLX,)},
            "training_backend_unsupported",
        ),
        (
            "model_format",
            {"model_formats": ("onnx",)},
            "model_format_unsupported",
        ),
        (
            "adapter_format",
            {"adapter_formats": ("lora",)},
            "adapter_format_unsupported",
        ),
    ],
)
def test_backend_and_format_mismatch_table(
    field: str, override: dict[str, object], reason: str
) -> None:
    report = check_federated_compatibility(_requirement(), _capability(**override))
    assert report.verdict is FederatedCompatibilityVerdict.INCOMPATIBLE
    assert report.incompatible_fields == (field,)
    assert _required_reason(report, field) is FederatedCompatibilityReasonCode(reason)


def test_unsupported_backend_and_adapter_mismatch() -> None:
    backend = check_federated_compatibility(
        _requirement(training_backend=FederatedTrainingBackend.COREML),
        _capability(training_backends=(FederatedTrainingBackend.ONNX,)),
    )
    assert backend.verdict is FederatedCompatibilityVerdict.INCOMPATIBLE
    assert _required_reason(backend, "training_backend") is (
        FederatedCompatibilityReasonCode.TRAINING_BACKEND_UNSUPPORTED
    )
    adapter = check_federated_compatibility(
        _requirement(adapter_format="qlora"),
        _capability(adapter_formats=("lora",)),
    )
    assert adapter.verdict is FederatedCompatibilityVerdict.INCOMPATIBLE
    assert _required_reason(adapter, "adapter_format") is (
        FederatedCompatibilityReasonCode.ADAPTER_FORMAT_UNSUPPORTED
    )


def test_missing_mandatory_declarations_fail_closed() -> None:
    report = check_federated_compatibility(
        _requirement(),
        _capability(
            training_backends=(),
            model_formats=(),
            adapter_formats=(),
            privacy_mechanisms=(),
            secure_aggregation_modes=(),
            resource_class=None,
            deterministic_kernels=None,
        ),
    )
    assert report.verdict is FederatedCompatibilityVerdict.INCOMPATIBLE
    assert report.incompatible_fields == (
        "training_backend",
        "model_format",
        "adapter_format",
        "resource_class",
        "deterministic_kernels",
        "privacy_mechanism",
        "secure_aggregation",
    )
    assert report.review_fields == ()
    for finding in report.findings:
        assert finding.required is True
        if finding.field == "protocol_version":
            assert (
                finding.reason
                is FederatedCompatibilityReasonCode.PROTOCOL_VERSION_SUPPORTED
            )
        else:
            assert finding.reason is FederatedCompatibilityReasonCode.CAPABILITY_UNKNOWN


def test_optional_capability_differences_require_review() -> None:
    report = check_federated_compatibility(
        _requirement(),
        _capability(
            training_backends=(
                FederatedTrainingBackend.TORCH,
                FederatedTrainingBackend.MLX,
            ),
            model_formats=("safetensors", "mlx"),
            adapter_formats=("dense", "lora"),
            quantization_formats=("int8",),
            resource_class=FederatedResourceClass.XLARGE,
            deterministic_kernels=True,
            privacy_mechanisms=(
                FederatedPrivacyMechanism.GAUSSIAN,
                FederatedPrivacyMechanism.LAPLACE,
            ),
            secure_aggregation_modes=(
                FederatedSecureAggregationMode.PAIRWISE_MASKING,
                FederatedSecureAggregationMode.SHAMIR,
            ),
        ),
    )
    assert report.verdict is FederatedCompatibilityVerdict.REVIEW_REQUIRED
    assert report.ok is False
    assert report.incompatible_fields == ()
    assert report.review_fields == (
        "training_backend",
        "model_format",
        "adapter_format",
        "quantization_format",
        "resource_class",
        "deterministic_kernels",
        "privacy_mechanism",
        "secure_aggregation",
    )
    assert _optional_fields(report) == list(report.review_fields)
    for finding in report.findings:
        if finding.field in report.review_fields and not finding.required:
            assert finding.reason in {
                FederatedCompatibilityReasonCode.OPTIONAL_CAPABILITY_DIFFERENCE,
                FederatedCompatibilityReasonCode.RESOURCE_CLASS_SUFFICIENT,
            }


def test_optional_difference_keeps_the_supported_reason() -> None:
    report = check_federated_compatibility(
        _requirement(), _capability(resource_class=FederatedResourceClass.LARGE)
    )
    fields = [(finding.field, finding.reason) for finding in report.findings]
    assert (
        "resource_class",
        FederatedCompatibilityReasonCode.RESOURCE_CLASS_SUFFICIENT,
    ) in fields
    assert (
        "resource_class",
        FederatedCompatibilityReasonCode.OPTIONAL_CAPABILITY_DIFFERENCE,
    ) in fields
    assert report.review_fields == ("resource_class",)


@pytest.mark.parametrize(
    ("required_quantization", "declared", "reason", "review"),
    [
        (None, (), "quantization_supported", False),
        (None, ("int8",), None, True),
        ("int8", ("int8",), "quantization_supported", False),
        ("int8", ("int8", "int4"), "quantization_supported", True),
        ("int8", ("int4",), "quantization_unsupported", False),
        ("int8", (), "capability_unknown", False),
    ],
)
def test_quantization_table(
    required_quantization: str | None,
    declared: tuple[str, ...],
    reason: str | None,
    review: bool,
) -> None:
    report = check_federated_compatibility(
        _requirement(quantization_format=required_quantization),
        _capability(quantization_formats=declared),
    )
    if required_quantization is None:
        assert _required_reason(report, "quantization_format") is None
        assert _optional_fields(report) == (["quantization_format"] if declared else [])
        expected = (
            FederatedCompatibilityVerdict.REVIEW_REQUIRED
            if declared
            else FederatedCompatibilityVerdict.COMPATIBLE
        )
        assert report.verdict is expected
        return
    assert _required_reason(report, "quantization_format") is (
        FederatedCompatibilityReasonCode(reason)
    )
    assert _optional_fields(report) == (["quantization_format"] if review else [])
    if review:
        assert report.verdict is FederatedCompatibilityVerdict.REVIEW_REQUIRED
    elif reason == "quantization_supported":
        assert report.verdict is FederatedCompatibilityVerdict.COMPATIBLE
    else:
        assert report.verdict is FederatedCompatibilityVerdict.INCOMPATIBLE


def test_resource_class_insufficient_is_incompatible() -> None:
    report = check_federated_compatibility(
        _requirement(resource_class=FederatedResourceClass.LARGE),
        _capability(resource_class=FederatedResourceClass.SMALL),
    )
    assert report.verdict is FederatedCompatibilityVerdict.INCOMPATIBLE
    assert report.incompatible_fields == ("resource_class",)
    assert _required_reason(report, "resource_class") is (
        FederatedCompatibilityReasonCode.RESOURCE_CLASS_INSUFFICIENT
    )


def test_deterministic_kernels_table() -> None:
    required = check_federated_compatibility(
        _requirement(deterministic_kernels=True),
        _capability(deterministic_kernels=True),
    )
    assert required.verdict is FederatedCompatibilityVerdict.COMPATIBLE
    missing = check_federated_compatibility(
        _requirement(deterministic_kernels=True),
        _capability(deterministic_kernels=False),
    )
    assert missing.verdict is FederatedCompatibilityVerdict.INCOMPATIBLE
    assert _required_reason(missing, "deterministic_kernels") is (
        FederatedCompatibilityReasonCode.DETERMINISTIC_KERNELS_MISSING
    )
    extra = check_federated_compatibility(
        _requirement(), _capability(deterministic_kernels=True)
    )
    assert extra.verdict is FederatedCompatibilityVerdict.REVIEW_REQUIRED
    assert _optional_fields(extra) == ["deterministic_kernels"]


def test_privacy_mechanism_and_secure_aggregation_table() -> None:
    privacy = check_federated_compatibility(
        _requirement(privacy_mechanism=FederatedPrivacyMechanism.LAPLACE),
        _capability(),
    )
    assert privacy.verdict is FederatedCompatibilityVerdict.INCOMPATIBLE
    assert _required_reason(privacy, "privacy_mechanism") is (
        FederatedCompatibilityReasonCode.PRIVACY_MECHANISM_UNSUPPORTED
    )
    aggregation = check_federated_compatibility(
        _requirement(secure_aggregation=FederatedSecureAggregationMode.SHAMIR),
        _capability(),
    )
    assert aggregation.verdict is FederatedCompatibilityVerdict.INCOMPATIBLE
    assert _required_reason(aggregation, "secure_aggregation") is (
        FederatedCompatibilityReasonCode.SECURE_AGGREGATION_UNSUPPORTED
    )


def test_mixed_report_is_ordered_and_fail_closed() -> None:
    report = check_federated_compatibility(
        _requirement(protocol_version=9, adapter_format="qlora"),
        _capability(
            adapter_formats=("lora",),
            resource_class=FederatedResourceClass.XLARGE,
        ),
    )
    assert report.verdict is FederatedCompatibilityVerdict.INCOMPATIBLE
    assert report.incompatible_fields == ("protocol_version", "adapter_format")
    assert report.review_fields == ("resource_class",)
    assert [finding.field for finding in report.findings] == [
        "protocol_version",
        "training_backend",
        "model_format",
        "adapter_format",
        "resource_class",
        "resource_class",
        "deterministic_kernels",
        "privacy_mechanism",
        "secure_aggregation",
    ]


def test_requirement_fields_drop_optional_quantization() -> None:
    assert _requirement().fields() == (
        "protocol_version",
        "training_backend",
        "model_format",
        "adapter_format",
        "resource_class",
        "deterministic_kernels",
        "privacy_mechanism",
        "secure_aggregation",
    )
    assert _requirement(quantization_format="int8").fields() == tuple(
        FEDERATED_COMPATIBILITY_FIELDS
    )


def test_finding_required_flag_matches_the_reason_code() -> None:
    report = check_federated_compatibility(
        _requirement(quantization_format="int8"),
        _capability(quantization_formats=("int8",), deterministic_kernels=True),
    )
    for finding in report.findings:
        optional = (
            finding.reason
            is FederatedCompatibilityReasonCode.OPTIONAL_CAPABILITY_DIFFERENCE
        )
        assert finding.required is not optional


# --- rendering --------------------------------------------------------------


def test_report_json_is_byte_stable() -> None:
    first = check_federated_compatibility(_requirement(), _capability())
    second = check_federated_compatibility(_requirement(), _capability())
    assert first.to_json() == second.to_json()
    assert first.to_json().endswith("}\n")
    assert json.loads(first.to_json())["schema_version"] == (
        FEDERATED_COMPATIBILITY_SCHEMA_VERSION
    )


def test_golden_report_digest() -> None:
    report = check_federated_compatibility(
        _requirement(quantization_format="int8"),
        _capability(quantization_formats=("int8",)),
    )
    digest = hashlib.sha256(report.to_json().encode("utf-8")).hexdigest()
    assert digest == _GOLDEN_REPORT_SHA256


def test_report_key_order() -> None:
    report = check_federated_compatibility(_requirement(), _capability())
    assert list(report.to_dict()) == [
        "verdict",
        "findings",
        "incompatible_fields",
        "review_fields",
        "schema_version",
    ]
    assert list(report.findings[0].to_dict()) == ["field", "reason", "required"]
    assert "ok" not in report.to_dict()


def test_requirement_key_order() -> None:
    assert list(_requirement().to_dict()) == [
        "protocol_version",
        "training_backend",
        "model_format",
        "adapter_format",
        "quantization_format",
        "resource_class",
        "deterministic_kernels",
        "privacy_mechanism",
        "secure_aggregation",
        "schema_version",
    ]


def test_capability_key_order() -> None:
    assert list(_capability().to_dict()) == [
        "training_backends",
        "model_formats",
        "adapter_formats",
        "privacy_mechanisms",
        "secure_aggregation_modes",
        "quantization_formats",
        "minimum_protocol_version",
        "maximum_protocol_version",
        "resource_class",
        "deterministic_kernels",
        "schema_version",
    ]


def test_capability_declarations_are_canonicalized() -> None:
    envelope = FederatedClientCapabilityEnvelope(
        training_backends=(
            FederatedTrainingBackend.MLX,
            FederatedTrainingBackend.TORCH,
            FederatedTrainingBackend.MLX,
        ),
        model_formats=("onnx", "safetensors", "onnx"),
        adapter_formats=("lora", "dense"),
        privacy_mechanisms=(
            FederatedPrivacyMechanism.LAPLACE,
            FederatedPrivacyMechanism.GAUSSIAN,
        ),
        secure_aggregation_modes=(
            FederatedSecureAggregationMode.SHAMIR,
            FederatedSecureAggregationMode.DISABLED,
        ),
        quantization_formats=("int4", "int8"),
        minimum_protocol_version=2,
        maximum_protocol_version=2,
        resource_class=FederatedResourceClass.LARGE,
        deterministic_kernels=True,
    )
    assert envelope.training_backends == (
        FederatedTrainingBackend.MLX,
        FederatedTrainingBackend.TORCH,
    )
    assert envelope.model_formats == ("onnx", "safetensors")
    assert envelope.adapter_formats == ("dense", "lora")
    assert envelope.privacy_mechanisms == (
        FederatedPrivacyMechanism.GAUSSIAN,
        FederatedPrivacyMechanism.LAPLACE,
    )
    assert envelope.secure_aggregation_modes == (
        FederatedSecureAggregationMode.DISABLED,
        FederatedSecureAggregationMode.SHAMIR,
    )
    assert envelope.to_dict() == envelope.to_dict()


# --- privacy ----------------------------------------------------------------


def test_report_never_echoes_declared_values() -> None:
    report = check_federated_compatibility(
        _requirement(), _capability(model_formats=(SENTINEL,))
    )
    text = report.to_json()
    assert SENTINEL not in text
    assert SENTINEL not in json.dumps(report.to_dict())


@pytest.mark.parametrize(
    "case",
    FORBIDDEN_FIELD_CASES,
    ids=[case.reason_code for case in FORBIDDEN_FIELD_CASES],
)
def test_forbidden_field_is_rejected_without_echo(case) -> None:
    requirement_payload = _requirement().to_dict()
    requirement_payload[case.field] = case.value
    with pytest.raises(FederatedCompatibilityError) as requirement_error:
        FederatedRoundRequirement.from_dict(requirement_payload)
    assert case.field not in str(requirement_error.value)
    assert str(case.value) not in str(requirement_error.value)

    capability_payload = _capability().to_dict()
    capability_payload[case.field] = case.value
    with pytest.raises(FederatedCompatibilityError) as capability_error:
        FederatedClientCapabilityEnvelope.from_dict(capability_payload)
    assert case.field not in str(capability_error.value)
    assert str(case.value) not in str(capability_error.value)
    assert str(requirement_error.value).islower()


def test_capability_from_json_rejects_payloads_without_echo() -> None:
    payload = json.loads(_capability().to_json())
    payload["site_id"] = SENTINEL
    with pytest.raises(FederatedCompatibilityError) as error:
        FederatedClientCapabilityEnvelope.from_json(json.dumps(payload))
    assert SENTINEL not in str(error.value)
    assert "site_id" not in str(error.value)


def test_compatibility_check_is_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "socket", _cannot_reach_network)
    monkeypatch.setattr(socket, "create_connection", _cannot_reach_network)
    report = check_federated_compatibility(_requirement(), _capability())
    assert report.ok is True
    assert report.to_json().endswith("}\n")


# --- round trips ------------------------------------------------------------


def test_requirement_round_trip() -> None:
    requirement = _requirement(quantization_format="int8")
    assert FederatedRoundRequirement.from_dict(requirement.to_dict()) == requirement
    assert FederatedRoundRequirement.from_json(requirement.to_json()) == requirement


def test_capability_round_trip() -> None:
    envelope = _capability(
        minimum_protocol_version=None,
        maximum_protocol_version=None,
        resource_class=None,
        deterministic_kernels=None,
    )
    assert FederatedClientCapabilityEnvelope.from_dict(envelope.to_dict()) == envelope
    assert FederatedClientCapabilityEnvelope.from_json(envelope.to_json()) == envelope


def test_requirement_from_dict_coerces_enum_names() -> None:
    payload = _requirement().to_dict()
    payload["training_backend"] = "coreml"
    payload["resource_class"] = "large"
    payload["privacy_mechanism"] = "laplace"
    payload["secure_aggregation"] = "disabled"
    loaded = FederatedRoundRequirement.from_dict(payload)
    assert loaded.training_backend is FederatedTrainingBackend.COREML
    assert loaded.resource_class is FederatedResourceClass.LARGE
    assert loaded.privacy_mechanism is FederatedPrivacyMechanism.LAPLACE
    assert loaded.secure_aggregation is FederatedSecureAggregationMode.DISABLED


def test_capability_from_dict_coerces_enum_names() -> None:
    payload = _capability().to_dict()
    payload["training_backends"] = ["coreml", "torch"]
    payload["privacy_mechanisms"] = ["gaussian"]
    payload["secure_aggregation_modes"] = ["disabled"]
    payload["resource_class"] = "large"
    loaded = FederatedClientCapabilityEnvelope.from_dict(payload)
    assert loaded.training_backends == (
        FederatedTrainingBackend.COREML,
        FederatedTrainingBackend.TORCH,
    )
    assert loaded.privacy_mechanisms == (FederatedPrivacyMechanism.GAUSSIAN,)
    assert loaded.secure_aggregation_modes == (FederatedSecureAggregationMode.DISABLED,)
    assert loaded.resource_class is FederatedResourceClass.LARGE


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "requirement",
        3,
        {"protocol_version": 1},
        {**_requirement().to_dict(), "site_id": SENTINEL},
        {**_requirement().to_dict(), "protocol_version": True},
        {**_requirement().to_dict(), "protocol_version": 0},
        {**_requirement().to_dict(), "protocol_version": MAX_PROTOCOL_VERSION + 1},
        {**_requirement().to_dict(), "model_format": "Safetensors"},
        {**_requirement().to_dict(), "adapter_format": "dense lora"},
        {**_requirement().to_dict(), "quantization_format": "INT8"},
        {**_requirement().to_dict(), "deterministic_kernels": "yes"},
        {**_requirement().to_dict(), "schema_version": "v0"},
    ],
)
def test_requirement_rejects_invalid_payloads(payload: object) -> None:
    with pytest.raises(FederatedCompatibilityError):
        FederatedRoundRequirement.from_dict(payload)


@pytest.mark.parametrize(
    "bad",
    ["{}", "nope", "[]", "{,}"],
)
def test_requirement_from_json_rejects_invalid_text(bad: str) -> None:
    with pytest.raises(FederatedCompatibilityError):
        FederatedRoundRequirement.from_json(bad)
    with pytest.raises(FederatedCompatibilityError):
        FederatedRoundRequirement.from_json({"protocol_version": 1})


@pytest.mark.parametrize(
    "overrides",
    [
        {"training_backends": [FederatedTrainingBackend.TORCH]},
        {"model_formats": ["safetensors"]},
        {"adapter_formats": ["dense"]},
        {"privacy_mechanisms": ["gaussian"]},
        {"secure_aggregation_modes": ["shamir"]},
        {"quantization_formats": ["int8"]},
        {"model_formats": (1,)},
        {"training_backends": ("torch",)},
        {"deterministic_kernels": "yes"},
        {"resource_class": "medium"},
        {"minimum_protocol_version": "1"},
        {"minimum_protocol_version": 1, "maximum_protocol_version": None},
        {"minimum_protocol_version": 4, "maximum_protocol_version": 2},
        {"minimum_protocol_version": 0},
        {"maximum_protocol_version": MAX_PROTOCOL_VERSION + 1},
        {
            "model_formats": tuple(
                f"fmt{index}" for index in range(MAX_DECLARED_CAPABILITIES + 1)
            )
        },
    ],
)
def test_capability_rejects_invalid_declarations(overrides: dict[str, object]) -> None:
    with pytest.raises(FederatedCompatibilityError):
        _capability(**overrides)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "capability",
        7,
        {"training_backends": []},
        {"model_formats": []},
        {**_capability().to_dict(), "site_id": SENTINEL},
        {**_capability().to_dict(), "training_backends": ["nccl"]},
        {**_capability().to_dict(), "minimum_protocol_version": True},
    ],
)
def test_capability_rejects_invalid_payloads(payload: object) -> None:
    with pytest.raises(FederatedCompatibilityError):
        FederatedClientCapabilityEnvelope.from_dict(payload)


def test_capability_from_json_rejects_invalid_text() -> None:
    with pytest.raises(FederatedCompatibilityError):
        FederatedClientCapabilityEnvelope.from_json("{")
    with pytest.raises(FederatedCompatibilityError):
        FederatedClientCapabilityEnvelope.from_json([1, 2])


@pytest.mark.parametrize("field", [None, "site_id", "protocol_version_extra", 3])
def test_finding_rejects_unknown_fields(field: object) -> None:
    with pytest.raises(FederatedCompatibilityError):
        FederatedCompatibilityFinding(
            field=field,  # type: ignore[arg-type]
            reason=FederatedCompatibilityReasonCode.CAPABILITY_UNKNOWN,
        )


@pytest.mark.parametrize("reason", [None, "capability_unknown", "unknown", 3])
def test_finding_rejects_unknown_reasons(reason: object) -> None:
    with pytest.raises(FederatedCompatibilityError):
        FederatedCompatibilityFinding(
            field="protocol_version",
            reason=reason,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("reason", "required"),
    [
        (
            FederatedCompatibilityReasonCode.OPTIONAL_CAPABILITY_DIFFERENCE,
            True,
        ),
        (FederatedCompatibilityReasonCode.PROTOCOL_VERSION_SUPPORTED, False),
        (FederatedCompatibilityReasonCode.CAPABILITY_UNKNOWN, False),
        (FederatedCompatibilityReasonCode.PROTOCOL_VERSION_SUPPORTED, 1),
        (FederatedCompatibilityReasonCode.PROTOCOL_VERSION_SUPPORTED, None),
    ],
)
def test_finding_rejects_required_flag_mismatch(
    reason: FederatedCompatibilityReasonCode, required: object
) -> None:
    with pytest.raises(FederatedCompatibilityError):
        FederatedCompatibilityFinding(
            field="protocol_version",
            reason=reason,
            required=required,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("verdict", "findings", "incompatible_fields", "review_fields"),
    [
        (
            "compatible",
            (),
            (),
            (),
        ),
        (
            FederatedCompatibilityVerdict.COMPATIBLE,
            (object(),),
            (),
            (),
        ),
        (
            FederatedCompatibilityVerdict.INCOMPATIBLE,
            (
                FederatedCompatibilityFinding(
                    field="adapter_format",
                    reason=FederatedCompatibilityReasonCode.ADAPTER_FORMAT_UNSUPPORTED,
                ),
                FederatedCompatibilityFinding(
                    field="protocol_version",
                    reason=FederatedCompatibilityReasonCode.PROTOCOL_VERSION_UNKNOWN,
                ),
            ),
            ("protocol_version", "adapter_format"),
            (),
        ),
        (
            FederatedCompatibilityVerdict.INCOMPATIBLE,
            (
                FederatedCompatibilityFinding(
                    field="protocol_version",
                    reason=FederatedCompatibilityReasonCode.PROTOCOL_VERSION_UNKNOWN,
                ),
            ),
            (),
            (),
        ),
        (
            FederatedCompatibilityVerdict.COMPATIBLE,
            (
                FederatedCompatibilityFinding(
                    field="protocol_version",
                    reason=FederatedCompatibilityReasonCode.PROTOCOL_VERSION_UNKNOWN,
                ),
            ),
            ("protocol_version",),
            (),
        ),
        (
            FederatedCompatibilityVerdict.REVIEW_REQUIRED,
            (),
            (),
            ("protocol_version",),
        ),
    ],
)
def test_report_rejects_inconsistent_records(
    verdict: object,
    findings: object,
    incompatible_fields: object,
    review_fields: object,
) -> None:
    with pytest.raises(FederatedCompatibilityError):
        FederatedCompatibilityReport(
            verdict=verdict,  # type: ignore[arg-type]
            findings=findings,  # type: ignore[arg-type]
            incompatible_fields=incompatible_fields,  # type: ignore[arg-type]
            review_fields=review_fields,  # type: ignore[arg-type]
        )


def test_report_rejects_unsupported_schema_version() -> None:
    with pytest.raises(FederatedCompatibilityError):
        FederatedCompatibilityReport(
            verdict=FederatedCompatibilityVerdict.COMPATIBLE,
            findings=(),
            incompatible_fields=(),
            review_fields=(),
            schema_version="openmed.training.federated_compatibility.v0",
        )


def test_check_rejects_non_records() -> None:
    with pytest.raises(FederatedCompatibilityError):
        check_federated_compatibility(object(), _capability())  # type: ignore[arg-type]
    with pytest.raises(FederatedCompatibilityError):
        check_federated_compatibility(_requirement(), object())  # type: ignore[arg-type]
    with pytest.raises(FederatedCompatibilityError):
        check_federated_compatibility(None, None)  # type: ignore[arg-type]


def test_baseline_error_messages_stay_lowercase() -> None:
    with pytest.raises(FederatedCompatibilityError) as error:
        _requirement(model_format="Safetensors")
    message = str(error.value)
    assert message == message.lower()
    assert "Safetensors" not in message


# --- package wiring ---------------------------------------------------------


def test_module_all_is_complete() -> None:
    assert frozenset(compat_module.__all__) == frozenset(
        {
            "DEFAULT_MAX_DECLARED_CAPABILITIES",
            "FEDERATED_COMPATIBILITY_FIELDS",
            "FEDERATED_COMPATIBILITY_REASON_CODES",
            "FEDERATED_COMPATIBILITY_SCHEMA_VERSION",
            "MAX_PROTOCOL_VERSION",
            "MIN_PROTOCOL_VERSION",
            "FederatedClientCapabilityEnvelope",
            "FederatedCompatibilityError",
            "FederatedCompatibilityFinding",
            "FederatedCompatibilityReasonCode",
            "FederatedCompatibilityReport",
            "FederatedCompatibilityVerdict",
            "FederatedResourceClass",
            "FederatedRoundRequirement",
            "FederatedSecureAggregationMode",
            "FederatedTrainingBackend",
            "check_federated_compatibility",
        }
    )
    for name in compat_module.__all__:
        assert hasattr(compat_module, name)


def test_lazy_training_exports_resolve() -> None:
    import openmed.training as training

    for name in compat_module.__all__:
        assert name in training.__all__
        assert getattr(training, name) is getattr(compat_module, name)
