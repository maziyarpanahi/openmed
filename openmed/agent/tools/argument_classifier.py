"""Classify and govern sensitive values in structured tool arguments.

Classification is declarative: trusted local code supplies typed paths and
developer-authored data-class identifiers. The classifier never sends values
to a detector or network service. Reports retain only schema paths, data
classes, and keyed value hashes.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Generic, TypeVar, cast

from openmed.agent.identifiers import GovernanceIdError, PolicyId
from openmed.agent.permissions.grants import (
    CapabilityGrantManifest,
    CapabilityGrantRequest,
    CapabilityGrantVerifier,
)

ARGUMENT_CLASSIFICATION_REPORT_SCHEMA_VERSION: Final = (
    "openmed.agent.argument_classification.v1"
)
ARGUMENT_VALUE_HASH_ALGORITHM: Final = "hmac-sha256"

_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_NAMESPACE = rf"{_LABEL}(?:\.{_LABEL})+"
_LOCAL_NAME = r"[a-z][a-z0-9-]{0,63}"
_NUMBER = r"(?:0|[1-9][0-9]*)"
_VERSION = rf"{_NUMBER}\.{_NUMBER}\.{_NUMBER}"
_DATA_CLASS_RE = re.compile(rf"data:{_NAMESPACE}/{_LOCAL_NAME}(?:@{_VERSION})?")
_FIELD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,127}")
_HASH_RE = re.compile(r"hmac-sha256:[0-9a-f]{64}")
_REPORT_BUCKETS = ("allowed", "redacted", "blocked")

JsonScalar = str | int | float | bool | None
JsonValue = (
    JsonScalar | dict[str, "JsonValue"] | list["JsonValue"] | tuple["JsonValue", ...]
)
PathSegment = str | int
_T = TypeVar("_T")


class ArgumentAction(str, Enum):
    """A policy decision for one classified data class."""

    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


class ArgumentClassifierError(ValueError):
    """Base class for value-free argument-classification failures."""

    def __init__(self, code: str, field_name: str) -> None:
        self.code = code
        self.field_name = field_name
        super().__init__(f"{field_name}: {code}")


class ArgumentClassifierValidationError(ArgumentClassifierError):
    """Raised when rules, policies, keys, or arguments are malformed."""


class ArgumentClassificationCoverageError(ArgumentClassifierError):
    """Raised when a leaf has no unique classification rule."""


class ArgumentDispatchBlockedError(ArgumentClassifierError):
    """Raised before dispatch when at least one classified value is blocked."""

    def __init__(self, report: "ArgumentClassificationReport") -> None:
        self.report = report
        super().__init__("blocked_by_policy", "arguments")


@dataclass(frozen=True, slots=True, repr=False)
class ArgumentPathRule:
    """Assign one data class to a scalar path.

    A ``"*"`` segment matches any list or tuple index. Other strings are
    developer-authored mapping fields and integers match an exact sequence
    index.
    """

    path: tuple[PathSegment, ...]
    data_class: str

    def __post_init__(self) -> None:
        if type(self.path) is not tuple or not self.path:
            raise ArgumentClassifierValidationError("invalid_path", "rules")
        for segment in self.path:
            if type(segment) is int:
                if segment < 0:
                    raise ArgumentClassifierValidationError("invalid_path", "rules")
            elif type(segment) is str:
                if segment != "*" and _FIELD_RE.fullmatch(segment) is None:
                    raise ArgumentClassifierValidationError("invalid_path", "rules")
            else:
                raise ArgumentClassifierValidationError("invalid_path", "rules")
        _validate_data_class(self.data_class, "rules")

    def __repr__(self) -> str:
        """Return a representation that omits policy metadata."""

        return "ArgumentPathRule(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class ArgumentDataClassDecision:
    """Bind a data class to allow, redact, or block behavior.

    ``replacement`` is used only for redaction. It must be a JSON scalar so a
    policy can choose a schema-compatible placeholder without copying the
    sensitive source value.
    """

    data_class: str
    action: ArgumentAction
    replacement: JsonScalar = None

    def __post_init__(self) -> None:
        _validate_data_class(self.data_class, "decisions")
        if type(self.action) is not ArgumentAction:
            raise ArgumentClassifierValidationError("invalid_action", "decisions")
        _validate_scalar(self.replacement, "decisions")
        if self.action is not ArgumentAction.REDACT and self.replacement is not None:
            raise ArgumentClassifierValidationError(
                "replacement_requires_redact", "decisions"
            )

    def __repr__(self) -> str:
        """Return a representation that omits replacement values."""

        return "ArgumentDataClassDecision(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class ArgumentClassificationPolicy:
    """A complete data-class policy bound to a signed policy profile."""

    policy_profile: str
    decisions: tuple[ArgumentDataClassDecision, ...]

    def __post_init__(self) -> None:
        try:
            PolicyId.parse(self.policy_profile)
        except GovernanceIdError as exc:
            raise ArgumentClassifierValidationError(
                "invalid_policy_profile", "policy"
            ) from exc
        if type(self.decisions) is not tuple or not self.decisions:
            raise ArgumentClassifierValidationError("invalid_decisions", "policy")
        if not all(type(item) is ArgumentDataClassDecision for item in self.decisions):
            raise ArgumentClassifierValidationError("invalid_decisions", "policy")
        ordered = tuple(sorted(self.decisions, key=lambda item: item.data_class))
        if len({item.data_class for item in ordered}) != len(ordered):
            raise ArgumentClassifierValidationError("duplicate_data_class", "policy")
        object.__setattr__(self, "decisions", ordered)

    def __repr__(self) -> str:
        """Return a representation that omits the policy contents."""

        return "ArgumentClassificationPolicy(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class ArgumentFinding:
    """A PHI-free classification record for one scalar value."""

    path: str
    data_class: str
    value_hash: str

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path.startswith("/"):
            raise ArgumentClassifierValidationError("invalid_finding", "report")
        _validate_data_class(self.data_class, "report")
        if (
            type(self.value_hash) is not str
            or _HASH_RE.fullmatch(self.value_hash) is None
        ):
            raise ArgumentClassifierValidationError("invalid_finding", "report")

    def to_dict(self) -> dict[str, str]:
        """Return the path, data class, and keyed hash only."""

        return {
            "path": self.path,
            "data_class": self.data_class,
            "value_hash": self.value_hash,
        }

    def __repr__(self) -> str:
        """Return a value-free representation."""

        return "ArgumentFinding(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class ArgumentClassificationReport:
    """Deterministic findings grouped by the applied decision."""

    allowed: tuple[ArgumentFinding, ...]
    redacted: tuple[ArgumentFinding, ...]
    blocked: tuple[ArgumentFinding, ...]
    schema_version: str = ARGUMENT_CLASSIFICATION_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ARGUMENT_CLASSIFICATION_REPORT_SCHEMA_VERSION:
            raise ArgumentClassifierValidationError("invalid_schema_version", "report")
        for bucket in _REPORT_BUCKETS:
            findings = getattr(self, bucket)
            if type(findings) is not tuple or not all(
                type(item) is ArgumentFinding for item in findings
            ):
                raise ArgumentClassifierValidationError("invalid_findings", "report")
            ordered = tuple(sorted(findings, key=lambda item: item.path))
            if len({item.path for item in ordered}) != len(ordered):
                raise ArgumentClassifierValidationError("duplicate_path", "report")
            object.__setattr__(self, bucket, ordered)
        all_paths = [
            item.path for bucket in _REPORT_BUCKETS for item in getattr(self, bucket)
        ]
        if len(set(all_paths)) != len(all_paths):
            raise ArgumentClassifierValidationError("duplicate_path", "report")

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic evidence without argument values."""

        return {
            "schema_version": self.schema_version,
            "allowed": [item.to_dict() for item in self.allowed],
            "redacted": [item.to_dict() for item in self.redacted],
            "blocked": [item.to_dict() for item in self.blocked],
        }

    def to_json(self) -> str:
        """Serialize deterministic evidence as canonical JSON."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def __repr__(self) -> str:
        """Return a value-free representation."""

        return "ArgumentClassificationReport(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class ClassifiedArguments:
    """Schema-preserving arguments and their PHI-free report."""

    arguments: dict[str, JsonValue]
    report: ArgumentClassificationReport

    def __repr__(self) -> str:
        """Return a representation that never displays arguments."""

        return "ClassifiedArguments(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class ClassifiedDispatchResult(Generic[_T]):
    """A tool result paired with its PHI-free pre-dispatch report."""

    value: _T
    report: ArgumentClassificationReport

    def __repr__(self) -> str:
        """Return a representation that never displays the tool result."""

        return "ClassifiedDispatchResult(<redacted>)"


class ArgumentClassifier:
    """Walk JSON-shaped arguments under a complete local classification policy."""

    def __init__(
        self,
        rules: Sequence[ArgumentPathRule],
        policy: ArgumentClassificationPolicy,
        *,
        hash_key: bytes,
    ) -> None:
        if isinstance(rules, (str, bytes, bytearray)):
            raise ArgumentClassifierValidationError("invalid_rules", "rules")
        try:
            rule_tuple = tuple(rules)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ArgumentClassifierValidationError("invalid_rules", "rules") from None
        if not rule_tuple or not all(
            type(item) is ArgumentPathRule for item in rule_tuple
        ):
            raise ArgumentClassifierValidationError("invalid_rules", "rules")
        if type(policy) is not ArgumentClassificationPolicy:
            raise ArgumentClassifierValidationError("invalid_policy", "policy")
        if type(hash_key) is not bytes or len(hash_key) < 32:
            raise ArgumentClassifierValidationError("invalid_hash_key", "hash_key")
        decision_classes = {item.data_class for item in policy.decisions}
        if any(rule.data_class not in decision_classes for rule in rule_tuple):
            raise ArgumentClassifierValidationError("missing_decision", "policy")
        if len({rule.path for rule in rule_tuple}) != len(rule_tuple):
            raise ArgumentClassifierValidationError("duplicate_rule", "rules")
        self.rules = rule_tuple
        self.policy = policy
        self._hash_key = hash_key

    def classify(self, arguments: Mapping[str, Any]) -> ClassifiedArguments:
        """Classify every scalar leaf and return a schema-preserving copy."""

        if type(arguments) is not dict:
            raise ArgumentClassifierValidationError("invalid_arguments", "arguments")
        findings: dict[ArgumentAction, list[ArgumentFinding]] = {
            action: [] for action in ArgumentAction
        }
        transformed = cast(dict[str, JsonValue], self._walk(arguments, (), findings))
        report = ArgumentClassificationReport(
            allowed=tuple(findings[ArgumentAction.ALLOW]),
            redacted=tuple(findings[ArgumentAction.REDACT]),
            blocked=tuple(findings[ArgumentAction.BLOCK]),
        )
        return ClassifiedArguments(arguments=transformed, report=report)

    def _walk(
        self,
        value: Any,
        path: tuple[PathSegment, ...],
        findings: dict[ArgumentAction, list[ArgumentFinding]],
    ) -> JsonValue:
        if type(value) is dict:
            transformed: dict[str, JsonValue] = {}
            for key, child in value.items():
                if type(key) is not str or _FIELD_RE.fullmatch(key) is None:
                    raise ArgumentClassifierValidationError(
                        "invalid_argument_field", "arguments"
                    )
                transformed[key] = self._walk(child, (*path, key), findings)
            return transformed
        if type(value) in (list, tuple):
            children = [
                self._walk(child, (*path, index), findings)
                for index, child in enumerate(value)
            ]
            return tuple(children) if type(value) is tuple else children

        _validate_scalar(value, "arguments")
        matches = [rule for rule in self.rules if _path_matches(rule.path, path)]
        if len(matches) != 1:
            code = "unclassified_argument" if not matches else "ambiguous_rule"
            raise ArgumentClassificationCoverageError(code, "rules")
        rule = matches[0]
        decision = next(
            item for item in self.policy.decisions if item.data_class == rule.data_class
        )
        finding = ArgumentFinding(
            path=_render_path(path),
            data_class=rule.data_class,
            value_hash=_hash_scalar(
                value,
                path=path,
                data_class=rule.data_class,
                key=self._hash_key,
            ),
        )
        findings[decision.action].append(finding)
        if decision.action is ArgumentAction.ALLOW:
            return cast(JsonScalar, value)
        if decision.action is ArgumentAction.REDACT:
            return decision.replacement
        return None

    def __repr__(self) -> str:
        """Return a representation that omits rules, policy, and key material."""

        return "ArgumentClassifier(<redacted>)"


def dispatch_with_argument_classification(
    manifest: CapabilityGrantManifest | Mapping[str, Any] | str | bytes | None,
    request: CapabilityGrantRequest,
    verifier: CapabilityGrantVerifier,
    classifier: ArgumentClassifier,
    arguments: Mapping[str, Any],
    dispatch: Callable[[dict[str, JsonValue]], _T],
    *,
    now: int | None = None,
) -> ClassifiedDispatchResult[_T]:
    """Verify the policy grant, classify arguments, then invoke the tool.

    The classifier policy must exactly match the signed request policy profile.
    Blocked findings raise before ``dispatch`` is called. Redacted findings are
    replaced in the copied argument tree; the caller's input is never mutated.
    """

    if type(request) is not CapabilityGrantRequest:
        raise ArgumentClassifierValidationError("invalid_request", "request")
    if type(verifier) is not CapabilityGrantVerifier:
        raise ArgumentClassifierValidationError("invalid_verifier", "grant")
    if type(classifier) is not ArgumentClassifier:
        raise ArgumentClassifierValidationError("invalid_classifier", "classifier")
    if not callable(dispatch):
        raise ArgumentClassifierValidationError("invalid_dispatch", "dispatch")
    if request.policy_profile != classifier.policy.policy_profile:
        raise ArgumentClassifierValidationError("policy_profile_mismatch", "policy")

    verifier.verify(manifest, request, now=now)
    classified = classifier.classify(arguments)
    if classified.report.blocked:
        raise ArgumentDispatchBlockedError(classified.report)
    value = dispatch(classified.arguments)
    return ClassifiedDispatchResult(value=value, report=classified.report)


def _validate_data_class(value: object, field_name: str) -> str:
    if type(value) is not str or _DATA_CLASS_RE.fullmatch(value) is None:
        raise ArgumentClassifierValidationError("invalid_data_class", field_name)
    return value


def _validate_scalar(value: object, field_name: str) -> JsonScalar:
    if value is None or type(value) in (str, int, bool):
        return cast(JsonScalar, value)
    if type(value) is float and math.isfinite(value):
        return value
    raise ArgumentClassifierValidationError("invalid_argument_value", field_name)


def _path_matches(
    pattern: tuple[PathSegment, ...], path: tuple[PathSegment, ...]
) -> bool:
    if len(pattern) != len(path):
        return False
    return all(
        expected == actual or (expected == "*" and type(actual) is int)
        for expected, actual in zip(pattern, path)
    )


def _render_path(path: tuple[PathSegment, ...]) -> str:
    return "/" + "/".join(str(segment) for segment in path)


def _hash_scalar(
    value: JsonScalar,
    *,
    path: tuple[PathSegment, ...],
    data_class: str,
    key: bytes,
) -> str:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        context = f"{_render_path(path)}\0{data_class}\0".encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ArgumentClassifierValidationError(
            "invalid_argument_value", "arguments"
        ) from None
    digest = hmac.new(key, context + serialized, hashlib.sha256).hexdigest()
    return f"{ARGUMENT_VALUE_HASH_ALGORITHM}:{digest}"


__all__ = [
    "ARGUMENT_CLASSIFICATION_REPORT_SCHEMA_VERSION",
    "ARGUMENT_VALUE_HASH_ALGORITHM",
    "ArgumentAction",
    "ArgumentClassificationCoverageError",
    "ArgumentClassificationPolicy",
    "ArgumentClassificationReport",
    "ArgumentClassifier",
    "ArgumentClassifierError",
    "ArgumentClassifierValidationError",
    "ArgumentDataClassDecision",
    "ArgumentDispatchBlockedError",
    "ArgumentFinding",
    "ArgumentPathRule",
    "ClassifiedArguments",
    "ClassifiedDispatchResult",
    "JsonScalar",
    "JsonValue",
    "dispatch_with_argument_classification",
]
