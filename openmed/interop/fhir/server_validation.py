"""Opt-in FHIR R4 server validation with bounded, value-free outcomes.

Only an explicitly injected, target-bound transport can contact a server. This
module supplies no HTTP client, discovers no endpoint and resolves no secret.
The proposed resource remains protected local input; only controlled issue
codes, structural paths, counts and a request digest cross the result boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Protocol
from urllib.parse import urlsplit

from ..fhir_capability_preflight import (
    CapabilityStatementError,
    FHIRPreflightReason,
    parse_capability_statement,
)
from .versions import SUPPORTED_RESOURCE_TYPES

__all__ = [
    "MAX_SERVER_VALIDATION_ISSUES",
    "MAX_SERVER_VALIDATION_RESOURCE_BYTES",
    "FHIRServerValidationIssue",
    "FHIRServerValidationReason",
    "FHIRServerValidationResult",
    "FHIRServerValidationStatus",
    "FHIRValidationResponse",
    "FHIRValidationTransport",
    "preflight_server_validation",
]

MAX_SERVER_VALIDATION_ISSUES: Final = 128
MAX_SERVER_VALIDATION_RESOURCE_BYTES: Final = 1_048_576
_MAX_OPERATIONS: Final = 256
_MAX_PATHS_PER_ISSUE: Final = 16
_VALIDATE_DEFINITION: Final = (
    "http://hl7.org/fhir/OperationDefinition/Resource-validate"
)
_VALIDATE_DEFINITIONS: Final = frozenset(
    {
        _VALIDATE_DEFINITION,
        *(_VALIDATE_DEFINITION + "|" + v for v in ("4.0", "4.0.0", "4.0.1")),
    }
)
_RESOURCE_TYPE: Final = re.compile(r"[A-Z][A-Za-z0-9]{0,63}\Z")
_INSTANCE_ID: Final = re.compile(r"[A-Za-z0-9.-]{1,64}\Z")
_DIGEST: Final = re.compile(r"[0-9a-f]{64}\Z")
_PATH_SEGMENT: Final = re.compile(r"([A-Za-z][A-Za-z0-9]*)(\[(?:[0-9]{1,10})?\])?\Z")
_SEVERITIES: Final = frozenset({"fatal", "error", "warning", "information"})
_ISSUE_CODES: Final = frozenset(
    "invalid structure required value invariant security login unknown expired "
    "forbidden suppressed processing not-supported duplicate multiple-matches "
    "not-found deleted too-long code-invalid extension too-costly business-rule "
    "conflict transient lock-error no-store exception timeout incomplete throttled "
    "informational".split()
)
# A conservative vocabulary of structural element names, not a FHIR schema.
# Unknown/custom names and all function/filter/comparison expressions are dropped.
_ELEMENT_NAMES: Final = frozenset(
    "resourceType id meta versionId lastUpdated profile security tag implicitRules "
    "language text status div contained extension modifierExtension url valueString "
    "valueCode valueBoolean valueInteger valueDecimal valueDate valueDateTime "
    "valueUri valueReference valueQuantity valueCodeableConcept identifier use type "
    "system value period start end assigner active name family given prefix suffix "
    "telecom gender birthDate deceasedBoolean deceasedDateTime address line city "
    "district state postalCode country maritalStatus multipleBirthBoolean "
    "multipleBirthInteger photo contact relationship communication preferred "
    "generalPractitioner managingOrganization link other reference display coding "
    "code version userSelected subject encounter effectiveDateTime effectivePeriod "
    "effectiveTiming effectiveInstant issued performer dataAbsentReason "
    "interpretation bodySite method specimen device referenceRange low high appliesTo "
    "age hasMember derivedFrom component category note authorReference authorString "
    "time clinicalStatus verificationStatus severity onsetDateTime onsetAge "
    "onsetPeriod onsetRange onsetString abatementDateTime abatementAge abatementPeriod "
    "abatementRange abatementString recordedDate recorder asserter stage summary "
    "assessment evidence detail intent priority doNotPerform medicationCodeableConcept "
    "medicationReference authoredOn requester reasonCode reasonReference dosageInstruction "
    "dosage route doseAndRate doseQuantity rateQuantity timing repeat frequency "
    "duration durationUnit periodUnit boundsPeriod count event when offset quantity "
    "unit comparator entry fullUrl resource request response ifMatch ifNoneExist "
    "location description title content attachment data contentType size hash "
    "creation context author custodian relatesTo target actor agent entity role "
    "who what source site outcome action recorded purposeOfEvent".split()
)


class FHIRServerValidationStatus(str, Enum):
    """Controlled terminal classifications; only passed confirms validation."""

    DISABLED = "disabled"
    UNSUPPORTED = "unsupported"
    PASSED = "passed"
    BLOCKED = "blocked"


class FHIRServerValidationReason(str, Enum):
    """Fixed reasons that never echo server or application content."""

    DISABLED = "disabled"
    CONFIGURATION_INVALID = "configuration_invalid"
    CAPABILITY_MALFORMED = "capability_malformed"
    VERSION_UNSUPPORTED = "version_unsupported"
    RESOURCE_UNSUPPORTED = "resource_unsupported"
    OPERATION_UNSUPPORTED = "operation_unsupported"
    RESOURCE_INVALID = "resource_invalid"
    TRANSPORT_FAILED = "transport_failed"
    VALIDATION_UNAVAILABLE = "validation_unavailable"
    OUTCOME_MALFORMED = "outcome_malformed"
    ERROR_ISSUES = "error_issues"
    WARNING_ISSUES = "warning_issues"
    VALIDATED = "validated"


def _element_path(value: Any) -> str | None:
    if type(value) is not str or not value or len(value) > 256:
        return None
    parts = value.split(".")
    if len(parts) > 16 or parts[0] not in SUPPORTED_RESOURCE_TYPES:
        return None
    normalized = [parts[0]]
    for part in parts[1:]:
        match = _PATH_SEGMENT.fullmatch(part)
        if match is None or match[1] not in _ELEMENT_NAMES:
            return None
        normalized.append(match[1] + ("[]" if match[2] is not None else ""))
    return ".".join(normalized)


@dataclass(frozen=True, slots=True)
class FHIRServerValidationIssue:
    """A controlled severity/code and conservative structural element paths.

    Args:
        severity: One of the four FHIR R4 issue severities.
        code: A FHIR R4 issue-type code.
        element_paths: Normalized structural paths without values or indices.
    """

    severity: str
    code: str
    element_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.severity) is not str or self.severity not in _SEVERITIES:
            raise ValueError("validation issue severity is invalid")
        if type(self.code) is not str or self.code not in _ISSUE_CODES:
            raise ValueError("validation issue code is invalid")
        paths = tuple(self.element_paths)
        if len(paths) > _MAX_PATHS_PER_ISSUE or any(
            _element_path(path) != path for path in paths
        ):
            raise ValueError("validation element paths are invalid")
        object.__setattr__(self, "element_paths", tuple(sorted(set(paths))))

    def to_dict(self) -> dict[str, Any]:
        """Return controlled metadata without the original OperationOutcome."""
        return {
            "severity": self.severity,
            "code": self.code,
            "element_paths": list(self.element_paths),
        }


@dataclass(frozen=True, slots=True)
class FHIRServerValidationResult:
    """Value-free preflight evidence, never write authority or clinical assurance.

    Args:
        status: Controlled validation classification.
        reason_code: Controlled terminal reason.
        issues: Bounded sanitized issues; no diagnostics or resource content.
        discarded_path_count: Number of unrepresentable server expressions.
        request_digest: SHA-256 binding resource, intended mode and instance ID.
            It does not bind a server identity, policy, permission or approval.
    """

    status: FHIRServerValidationStatus
    reason_code: FHIRServerValidationReason
    issues: tuple[FHIRServerValidationIssue, ...] = ()
    discarded_path_count: int = 0
    request_digest: str | None = None

    def __post_init__(self) -> None:
        try:
            status = FHIRServerValidationStatus(self.status)
            reason = FHIRServerValidationReason(self.reason_code)
        except (TypeError, ValueError):
            raise ValueError("validation result classification is invalid") from None
        issues = tuple(self.issues)
        if len(issues) > MAX_SERVER_VALIDATION_ISSUES or any(
            not isinstance(issue, FHIRServerValidationIssue) for issue in issues
        ):
            raise ValueError("validation result issues are invalid")
        if type(self.discarded_path_count) is not int or not (
            0
            <= self.discarded_path_count
            <= MAX_SERVER_VALIDATION_ISSUES * _MAX_PATHS_PER_ISSUE
        ):
            raise ValueError("discarded path count is invalid")
        if self.request_digest is not None and (
            type(self.request_digest) is not str
            or _DIGEST.fullmatch(self.request_digest) is None
        ):
            raise ValueError("validation request digest is invalid")
        if status is FHIRServerValidationStatus.PASSED and (
            reason is not FHIRServerValidationReason.VALIDATED
            or not issues
            or self.request_digest is None
            or any(issue.severity in {"fatal", "error"} for issue in issues)
        ):
            raise ValueError("passed validation result is inconsistent")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason_code", reason)
        object.__setattr__(self, "issues", issues)

    @property
    def is_valid(self) -> bool:
        """Return true only when the enabled server preflight passed."""
        return self.status is FHIRServerValidationStatus.PASSED

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible controlled outcomes, counts and a digest."""
        return {
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "issues": [issue.to_dict() for issue in self.issues],
            "issue_count": len(self.issues),
            "discarded_path_count": self.discarded_path_count,
            "request_digest": self.request_digest,
        }


@dataclass(frozen=True, slots=True)
class FHIRValidationResponse:
    """Protected transport response; retain only the sanitized result as evidence.

    Args:
        status_code: HTTP status from the target-bound validation transport.
        outcome: Parsed response resource, potentially containing protected data.
            It is excluded from repr and must not be logged or serialized as evidence.
    """

    status_code: int
    outcome: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.status_code) is not int or not 100 <= self.status_code <= 599:
            raise ValueError("validation HTTP status is invalid")
        if not isinstance(self.outcome, Mapping):
            raise ValueError("validation response resource must be a mapping")


class FHIRValidationTransport(Protocol):
    """Application-owned transport bound to the intended target FHIR server.

    Implementations issue one POST to the resource type's ``$validate`` endpoint
    or the exact instance endpoint for update. They resolve the opaque handle
    in credential custody, enforce timeout/body limits, and disable redirects,
    logging of protected inputs/responses and automatic retries. This protocol
    has no bundled HTTP implementation and cannot interrupt a nonconforming one.
    """

    def validate(
        self,
        resource_type: str,
        *,
        instance_id: str | None,
        parameters: Mapping[str, Any],
        credential_handle: object,
        timeout_seconds: float,
    ) -> FHIRValidationResponse:
        """Send protected Parameters to the already-bound validation target.

        Args:
            resource_type: Valid resource type from the proposed resource.
            instance_id: Exact update target ID, or None for create validation.
            parameters: FHIR Parameters with intended mode and proposed resource.
            credential_handle: Opaque custody handle, never an exposed secret.
            timeout_seconds: Bounded transport deadline supplied by the caller.

        Returns:
            The protected HTTP status and parsed OperationOutcome response.
        """
        ...


def _stop(
    reason: FHIRServerValidationReason,
    *,
    unsupported: bool = False,
    request_digest: str | None = None,
) -> FHIRServerValidationResult:
    return FHIRServerValidationResult(
        FHIRServerValidationStatus.UNSUPPORTED
        if unsupported
        else FHIRServerValidationStatus.BLOCKED,
        reason,
        request_digest=request_digest,
    )


def _declares_validation(statement: Mapping[str, Any], resource_type: str) -> bool:
    count = 0
    supported = False
    conflicting = False
    for rest in statement["rest"]:
        if rest["mode"] != "server":
            continue
        resources = rest.get("resource", [])
        target_present = any(
            resource["type"] == resource_type for resource in resources
        )
        operation_groups = [(target_present, rest.get("operation", []))]
        operation_groups.extend(
            (resource["type"] == resource_type, resource.get("operation", []))
            for resource in resources
        )
        for relevant, operations in operation_groups:
            if type(operations) is not list:
                raise ValueError("capability operations are malformed")
            count += len(operations)
            if count > _MAX_OPERATIONS:
                raise ValueError("capability operation limit exceeded")
            for operation in operations:
                if not isinstance(operation, Mapping):
                    raise ValueError("capability operation is malformed")
                name, definition = operation.get("name"), operation.get("definition")
                if type(name) is not str or not name or len(name) > 64:
                    raise ValueError("capability operation name is malformed")
                if (
                    type(definition) is not str
                    or not definition
                    or len(definition) > 512
                ):
                    raise ValueError("capability operation definition is malformed")
                if relevant and name == "validate":
                    supported |= definition in _VALIDATE_DEFINITIONS
                    conflicting |= definition not in _VALIDATE_DEFINITIONS
    return supported and not conflicting


def _parse_issues(
    outcome: Any, resource_type: str
) -> tuple[tuple[FHIRServerValidationIssue, ...], int]:
    if (
        not isinstance(outcome, Mapping)
        or outcome.get("resourceType") != "OperationOutcome"
    ):
        raise ValueError("validation outcome is malformed")
    raw = outcome.get("issue")
    if type(raw) is not list or not 1 <= len(raw) <= MAX_SERVER_VALIDATION_ISSUES:
        raise ValueError("validation issues are malformed")
    issues: list[FHIRServerValidationIssue] = []
    discarded = 0
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("validation issue is malformed")
        severity, code = item.get("severity"), item.get("code")
        if type(severity) is not str or severity not in _SEVERITIES:
            raise ValueError("validation severity is malformed")
        if type(code) is not str or code not in _ISSUE_CODES:
            raise ValueError("validation code is malformed")
        expressions = item.get("expression", [])
        if type(expressions) is not list or len(expressions) > _MAX_PATHS_PER_ISSUE:
            raise ValueError("validation expressions are malformed")
        paths: set[str] = set()
        for expression in expressions:
            path = _element_path(expression)
            if path is None or path.split(".", 1)[0] != resource_type:
                discarded += 1
            else:
                paths.add(path)
        issues.append(FHIRServerValidationIssue(severity, code, tuple(sorted(paths))))
    issues.sort(key=lambda issue: (issue.severity, issue.code, issue.element_paths))
    return tuple(issues), discarded


def _resource_json(resource: Mapping[str, Any]) -> bytes:
    if len(resource) > 65_536:
        raise ValueError("resource structure exceeds the validation bound")
    copied = dict(resource)
    pending: list[tuple[Any, int]] = [(copied, 0)]
    visited = 0
    text_size = 0
    while pending:
        value, depth = pending.pop()
        visited += 1
        if depth > 32 or visited > 65_536:
            raise ValueError("resource structure exceeds the validation bound")
        if type(value) is dict:
            if len(value) > 65_536:
                raise ValueError("resource structure exceeds the validation bound")
            children = list(value.values())
            if any(type(key) is not str or len(key) > 256 for key in value):
                raise ValueError("resource JSON keys are invalid")
            text_size += sum(len(key) for key in value)
        elif type(value) is list:
            children = value
        elif type(value) is str:
            text_size += len(value)
            children = []
        elif type(value) is int:
            if value.bit_length() > 4096:
                raise ValueError("resource JSON number exceeds the validation bound")
            children = []
        elif type(value) is float:
            if not math.isfinite(value):
                raise ValueError("resource JSON number is invalid")
            children = []
        elif value is None or type(value) is bool:
            children = []
        else:
            raise ValueError("resource must contain only JSON values")
        if text_size > MAX_SERVER_VALIDATION_RESOURCE_BYTES:
            raise ValueError("resource text exceeds the validation bound")
        if visited + len(pending) + len(children) > 65_536:
            raise ValueError("resource structure exceeds the validation bound")
        pending.extend((child, depth + 1) for child in children)
    encoded = json.dumps(
        copied,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_SERVER_VALIDATION_RESOURCE_BYTES:
        raise ValueError("resource bytes exceed the validation bound")
    return encoded


def _profile_uri_valid(value: Any) -> bool:
    if value is None:
        return True
    if (
        type(value) is not str
        or not 1 <= len(value) <= 2048
        or not value.isascii()
        or re.search(r"[\x00-\x20\x7f]", value)
    ):
        return False
    try:
        parsed = urlsplit(value)
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            return False
        if parsed.scheme == "urn":
            return bool(parsed.path) and not parsed.netloc
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and bool(parsed.path)
            and (parsed.port is None or 0 < parsed.port <= 65535)
        )
    except ValueError:
        return False


def preflight_server_validation(
    statement: Mapping[str, Any],
    resource: Mapping[str, Any],
    *,
    mode: str,
    enabled: bool = False,
    transport: FHIRValidationTransport | None = None,
    credential_handle: object | None = None,
    resource_id: str | None = None,
    profile_uri: str | None = None,
    block_warnings: bool = False,
    timeout_seconds: float = 10.0,
) -> FHIRServerValidationResult:
    """Optionally validate one proposed create/update before application review.

    Args:
        statement: Already-cached FHIR R4 capability metadata for the same server.
        resource: Protected JSON resource, materialized only after authority and
            minimum-data checks. It is not retained in the returned result.
        mode: Intended ``create`` or ``update`` validation mode.
        enabled: Explicit opt-in. False returns before touching inputs or transport.
        transport: Trusted transport already bound to the intended write server.
        credential_handle: Opaque application custody handle; scalar secrets are rejected.
        resource_id: Exact instance ID required for update and absent for create.
            A resource's supplied ID must match this update target.
        profile_uri: Optional, application-approved canonical profile URI sent
            as a Parameters value. It never selects a transport destination.
        block_warnings: Whether warnings also prevent a passed result.
        timeout_seconds: Positive finite transport deadline of at most 60 seconds.

    Returns:
        Immutable sanitized evidence. HTTP 200 with error/fatal issues blocks;
        warnings alone pass by default. Unsupported, disabled, failed or malformed
        results do not establish validation. Passing never authorizes a write or
        guarantees acceptance after concurrent server changes. No retry occurs.
    """
    reason = FHIRServerValidationReason
    if type(enabled) is not bool:
        return _stop(reason.CONFIGURATION_INVALID)
    if not enabled:
        return FHIRServerValidationResult(
            FHIRServerValidationStatus.DISABLED, reason.DISABLED
        )
    if (
        type(mode) is not str
        or mode not in {"create", "update"}
        or type(block_warnings) is not bool
        or not _profile_uri_valid(profile_uri)
        or type(timeout_seconds) not in {int, float}
        or not 0 < timeout_seconds <= 60
        or not math.isfinite(timeout_seconds)
        or credential_handle is None
        or isinstance(
            credential_handle,
            (
                str,
                bytes,
                bytearray,
                memoryview,
                bool,
                int,
                float,
                list,
                tuple,
                dict,
                set,
                frozenset,
            ),
        )
    ):
        return _stop(reason.CONFIGURATION_INVALID)
    try:
        capabilities = parse_capability_statement(statement)
    except CapabilityStatementError as exc:
        if exc.reason_code is FHIRPreflightReason.FHIR_VERSION_NOT_SUPPORTED:
            return _stop(reason.VERSION_UNSUPPORTED, unsupported=True)
        return _stop(reason.CAPABILITY_MALFORMED)
    except Exception:
        return _stop(reason.CAPABILITY_MALFORMED)
    try:
        resource_type = resource.get("resourceType")
        if (
            type(resource_type) is not str
            or _RESOURCE_TYPE.fullmatch(resource_type) is None
        ):
            return _stop(reason.RESOURCE_INVALID)
    except Exception:
        return _stop(reason.RESOURCE_INVALID)
    try:
        if capabilities.for_resource(resource_type) is None:
            return _stop(reason.RESOURCE_UNSUPPORTED, unsupported=True)
        if not _declares_validation(statement, resource_type):
            return _stop(reason.OPERATION_UNSUPPORTED, unsupported=True)
    except Exception:
        return _stop(reason.CAPABILITY_MALFORMED)
    try:
        send = getattr(transport, "validate", None)
    except Exception:
        return _stop(reason.CONFIGURATION_INVALID)
    if not callable(send):
        return _stop(reason.CONFIGURATION_INVALID)
    try:
        if mode == "update":
            if (
                type(resource_id) is not str
                or _INSTANCE_ID.fullmatch(resource_id) is None
            ):
                return _stop(reason.RESOURCE_INVALID)
            if resource.get("id", resource_id) != resource_id:
                return _stop(reason.RESOURCE_INVALID)
        elif resource_id is not None:
            return _stop(reason.RESOURCE_INVALID)
        encoded = _resource_json(resource)
        parameters: dict[str, Any] = {
            "resourceType": "Parameters",
            "parameter": [
                {"name": "mode", "valueCode": mode},
                {"name": "resource", "resource": json.loads(encoded)},
            ],
        }
        if profile_uri is not None:
            parameters["parameter"].append({"name": "profile", "valueUri": profile_uri})
        bound = json.dumps(
            {"parameters": parameters, "instance_id": resource_id},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        digest = hashlib.sha256(bound).hexdigest()
    except Exception:
        return _stop(reason.RESOURCE_INVALID)
    try:
        response = send(
            resource_type,
            instance_id=resource_id,
            parameters=parameters,
            credential_handle=credential_handle,
            timeout_seconds=float(timeout_seconds),
        )
    except Exception:
        return _stop(reason.TRANSPORT_FAILED, request_digest=digest)
    if (
        not isinstance(response, FHIRValidationResponse)
        or type(response.status_code) is not int
    ):
        return _stop(reason.OUTCOME_MALFORMED, request_digest=digest)
    if response.status_code != 200:
        return _stop(reason.VALIDATION_UNAVAILABLE, request_digest=digest)
    try:
        issues, discarded = _parse_issues(response.outcome, resource_type)
    except Exception:
        return _stop(reason.OUTCOME_MALFORMED, request_digest=digest)
    if any(issue.severity in {"fatal", "error"} for issue in issues):
        status, terminal = FHIRServerValidationStatus.BLOCKED, reason.ERROR_ISSUES
    elif block_warnings and any(issue.severity == "warning" for issue in issues):
        status, terminal = FHIRServerValidationStatus.BLOCKED, reason.WARNING_ISSUES
    else:
        status, terminal = FHIRServerValidationStatus.PASSED, reason.VALIDATED
    return FHIRServerValidationResult(status, terminal, issues, discarded, digest)
