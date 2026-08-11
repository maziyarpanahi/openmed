"""Offline validation for conservative FHIR ``Observation`` extensions.

FHIR permits arbitrary extensions, but an exporter should not silently turn
uncertain evidence into an extension that downstream systems interpret as a
fact.  This module therefore validates only an explicit, local allowlist.  It
does not load StructureDefinitions, resolve canonical URLs, or contact a FHIR
server.

The primary entry point, :func:`check_observation_extensions`, returns
deterministic, JSON-serializable findings.  :func:`validate_observation_extensions`
adapts those findings to the shared FHIR ``OperationOutcome`` shape.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypedDict
from urllib.parse import urlparse

from .codeable_concept import MEDICAL_DEVICE_ASSIST_EXTENSION_URL
from .grounded import COREFERENCE_EVIDENCE_EXTENSION_URL
from .operation_outcome import to_operation_outcome

__all__ = [
    "DEFAULT_OBSERVATION_EXTENSION_RULES",
    "FHIR_R4",
    "FHIR_R5",
    "OBSERVATION_EXTENSION_BASE_URL",
    "OBSERVATION_UNKNOWN_STATE_CODES",
    "OBSERVATION_UNKNOWN_STATE_EXTENSION_URL",
    "ObservationExtensionFinding",
    "ObservationExtensionSpec",
    "check_observation_extensions",
    "validate_observation_extensions",
]

FHIR_R4 = "R4"
FHIR_R5 = "R5"
_FHIR_VERSIONS = frozenset({FHIR_R4, FHIR_R5})

OBSERVATION_EXTENSION_BASE_URL = "https://openmed.ai/fhir/StructureDefinition/"
OBSERVATION_UNKNOWN_STATE_EXTENSION_URL = (
    f"{OBSERVATION_EXTENSION_BASE_URL}observation-unknown-state"
)
OBSERVATION_UNKNOWN_STATE_CODES = (
    "unknown",
    "not-asked",
    "asked-unknown",
    "temporarily-unknown",
)

_FHIR_VALUE_FIELDS = frozenset(
    {
        "valueBase64Binary",
        "valueBoolean",
        "valueCanonical",
        "valueCode",
        "valueDate",
        "valueDateTime",
        "valueDecimal",
        "valueId",
        "valueInstant",
        "valueInteger",
        "valueMarkdown",
        "valueOid",
        "valuePositiveInt",
        "valueQuantity",
        "valueReference",
        "valueString",
        "valueTime",
        "valueUnsignedInt",
        "valueUri",
        "valueUrl",
        "valueUuid",
        "valueAddress",
        "valueAge",
        "valueAnnotation",
        "valueAttachment",
        "valueCodeableConcept",
        "valueContactPoint",
        "valueCount",
        "valueDistance",
        "valueDuration",
        "valueHumanName",
        "valueIdentifier",
        "valueMoney",
        "valuePeriod",
        "valueRange",
        "valueRatio",
        "valueRatioRange",
        "valueSignature",
        "valueTiming",
    }
)
_ALLOWED_EXTENSION_KEYS = frozenset({"id", "url", "extension"})
_INFERRED_KEYS = frozenset(
    {"confidence", "inference", "inferred", "model", "probability"}
)
_INFERRED_MARKERS = frozenset({"derived", "estimated", "inferred", "predicted"})
_CANONICAL_URI_SCHEMES = frozenset({"http", "https", "urn"})
_SAFE_EXPRESSION = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\[\]]*$")
_CODE_VALUE = re.compile(r"^\S(?:.*\S)?$")
_DATE_VALUE = re.compile(r"^\d{4}(?:-\d{2}(?:-\d{2})?)?(?:T.*)?$")


class ObservationExtensionFinding(TypedDict):
    """Value-free finding shape returned by the Observation checker."""

    finding_code: str
    severity: Literal["error", "warning"]
    code: str
    diagnostics: str
    expression: list[str]


@dataclass(frozen=True)
class ObservationExtensionSpec:
    """Offline contract for one allowlisted extension URL.

    ``value_types`` contains FHIR ``value[x]`` field names such as
    ``"valueCode"``.  A spec with no value types may instead declare nested
    extension specs through ``nested``.  ``max_occurs=None`` means unbounded.

    ``allowed_values`` is deliberately a shape-level allowlist for explicit
    code-like states.  The validator compares a value but never places it in a
    finding or exception.
    """

    value_types: tuple[str, ...] = ()
    min_occurs: int = 0
    max_occurs: int | None = 1
    allowed_values: tuple[str, ...] = ()
    nested: Mapping[str, "ObservationExtensionSpec"] = field(default_factory=dict)
    fhir_versions: frozenset[str] = field(
        default_factory=lambda: frozenset(_FHIR_VERSIONS)
    )
    explicit_only: bool = True

    def __post_init__(self) -> None:
        value_types = _normalise_string_tuple(self.value_types, "value_types")
        allowed_values = _normalise_string_tuple(self.allowed_values, "allowed_values")
        versions = _normalise_versions(self.fhir_versions)
        nested = self.nested
        if not isinstance(nested, Mapping):
            raise TypeError("nested must be a mapping")
        normalised_nested: dict[str, ObservationExtensionSpec] = {}
        for url, raw_spec in nested.items():
            if not isinstance(url, str) or not url.strip():
                raise ValueError("nested extension keys must be non-empty strings")
            normalised_nested[url.strip()] = (
                raw_spec
                if isinstance(raw_spec, ObservationExtensionSpec)
                else _coerce_spec(raw_spec)
            )

        if any(
            value_type not in _FHIR_VALUE_FIELDS and value_type != "value[x]"
            for value_type in value_types
        ):
            raise ValueError("value_types contains an unsupported FHIR value field")
        if not isinstance(self.min_occurs, int) or isinstance(self.min_occurs, bool):
            raise TypeError("min_occurs must be an integer")
        if self.min_occurs < 0:
            raise ValueError("min_occurs must not be negative")
        if self.max_occurs is not None:
            if not isinstance(self.max_occurs, int) or isinstance(
                self.max_occurs, bool
            ):
                raise TypeError("max_occurs must be an integer or None")
            if self.max_occurs < self.min_occurs:
                raise ValueError("max_occurs must not be less than min_occurs")
        if not isinstance(self.explicit_only, bool):
            raise TypeError("explicit_only must be a boolean")

        object.__setattr__(self, "value_types", value_types)
        object.__setattr__(self, "allowed_values", allowed_values)
        object.__setattr__(self, "fhir_versions", versions)
        object.__setattr__(self, "nested", MappingProxyType(normalised_nested))


def _spec(
    *,
    value_types: Sequence[str] = (),
    min_occurs: int = 0,
    max_occurs: int | None = 1,
    allowed_values: Sequence[str] = (),
    nested: Mapping[str, ObservationExtensionSpec] | None = None,
) -> ObservationExtensionSpec:
    """Create a compact built-in extension spec."""

    return ObservationExtensionSpec(
        value_types=tuple(value_types),
        min_occurs=min_occurs,
        max_occurs=max_occurs,
        allowed_values=tuple(allowed_values),
        nested=nested or {},
    )


def _default_extension_rules() -> Mapping[str, ObservationExtensionSpec]:
    """Build the small built-in allowlist without reading external metadata."""

    mention_shape = {
        "start": _spec(value_types=("valueUnsignedInt",), min_occurs=1),
        "end": _spec(value_types=("valueUnsignedInt",), min_occurs=1),
        "textHash": _spec(value_types=("valueString",), min_occurs=1),
    }
    coreference_shape = {
        "clusterId": _spec(value_types=("valueString",), min_occurs=1),
        "representative": _spec(
            min_occurs=1,
            nested=mention_shape,
        ),
        "supportingMention": _spec(
            max_occurs=None,
            nested=mention_shape,
        ),
    }
    assist_shape = {
        "assist_only": _spec(value_types=("valueBoolean",), min_occurs=1),
        "autonomous_decision": _spec(value_types=("valueBoolean",), min_occurs=1),
        "evidence_start": _spec(value_types=("valueUnsignedInt",), min_occurs=1),
        "evidence_end": _spec(value_types=("valueUnsignedInt",), min_occurs=1),
        "disclaimer": _spec(value_types=("valueString",), min_occurs=1),
    }
    return MappingProxyType(
        {
            OBSERVATION_UNKNOWN_STATE_EXTENSION_URL: _spec(
                value_types=("valueCode",),
                allowed_values=OBSERVATION_UNKNOWN_STATE_CODES,
            ),
            COREFERENCE_EVIDENCE_EXTENSION_URL: _spec(nested=coreference_shape),
            MEDICAL_DEVICE_ASSIST_EXTENSION_URL: _spec(nested=assist_shape),
        }
    )


def check_observation_extensions(
    observation: Any,
    *,
    fhir_version: str = FHIR_R4,
    version: str | None = None,
    mode: str | None = None,
    fhir_release: str | None = None,
    allowed_extensions: Mapping[str, Any] | Sequence[str] | None = None,
    extension_rules: Mapping[str, Any] | Sequence[str] | None = None,
    extension_allowlist: Mapping[str, Any] | Sequence[str] | None = None,
    allowlist: Mapping[str, Any] | Sequence[str] | None = None,
    rules: Mapping[str, Any] | Sequence[str] | None = None,
    expression: str = "Observation",
) -> list[ObservationExtensionFinding]:
    """Return deterministic findings for allowlisted ``Observation`` extensions.

    Args:
        observation: Candidate FHIR Observation mapping. It is never mutated.
        fhir_version: ``"R4"`` or ``"R5"``; dotted release aliases are also
            accepted (for example ``"4.0.1"`` and ``"5.0.0"``).
        version, mode, fhir_release: Compatibility aliases for callers that
            name the FHIR mode differently. Supplying conflicting aliases is
            rejected before any resource content is inspected.
        allowed_extensions: Optional URL-to-spec allowlist. The other
            allowlist parameters are compatibility aliases; provide at most
            one of them. A sequence of URLs uses a generic single-value shape.
        expression: Safe FHIRPath-style root used in returned expressions.

    Returns:
        A list of FHIR-shaped, value-free finding dictionaries. An empty list
        means that every supplied extension matched the offline allowlist.

    Raises:
        TypeError or ValueError: If the FHIR mode or caller-supplied allowlist
            is malformed. Malformed resource content becomes a finding.
    """

    selected_version = _resolve_version(
        fhir_version,
        version=version,
        mode=mode,
        fhir_release=fhir_release,
    )
    selected_rules = _resolve_rules(
        allowed_extensions,
        extension_rules=extension_rules,
        extension_allowlist=extension_allowlist,
        allowlist=allowlist,
        rules=rules,
    )
    root = _safe_expression(expression)
    findings: list[ObservationExtensionFinding] = []

    if not isinstance(observation, Mapping):
        findings.append(
            _finding(
                "invalid-observation",
                "error",
                "invalid",
                "Observation must be an object.",
                root,
            )
        )
        return findings

    if observation.get("resourceType") != "Observation":
        findings.append(
            _finding(
                "invalid-resource-type",
                "error",
                "invalid",
                "resourceType must be Observation.",
                f"{root}.resourceType",
            )
        )

    raw_extensions = observation.get("extension")
    if raw_extensions is None:
        return findings
    if not isinstance(raw_extensions, list):
        findings.append(
            _finding(
                "extension-not-array",
                "error",
                "structure",
                "Observation.extension must be an array.",
                f"{root}.extension",
            )
        )
        return findings

    counts: dict[str, int] = {}
    first_indexes: dict[str, int] = {}
    for index, raw_extension in enumerate(raw_extensions):
        extension_path = f"{root}.extension[{index}]"
        if not isinstance(raw_extension, Mapping):
            findings.append(
                _finding(
                    "invalid-extension",
                    "error",
                    "structure",
                    "Extension entries must be objects.",
                    extension_path,
                )
            )
            continue

        url = raw_extension.get("url")
        if not _is_canonical_url(url):
            findings.append(
                _finding(
                    "invalid-extension-url",
                    "error",
                    "value",
                    "Extension URL must be a non-empty absolute URI.",
                    f"{extension_path}.url",
                )
            )
            continue
        counts[url] = counts.get(url, 0) + 1
        first_indexes.setdefault(url, index)

        spec = selected_rules.get(url)
        if spec is None:
            findings.append(
                _finding(
                    "unsupported-extension-url",
                    "error",
                    "not-supported",
                    "Extension URL is not in the offline Observation allowlist.",
                    f"{extension_path}.url",
                )
            )
            continue
        _validate_extension(
            raw_extension,
            spec,
            selected_version,
            extension_path,
            findings,
        )

    for url in sorted(counts, key=lambda item: first_indexes[item]):
        spec = selected_rules.get(url)
        if spec is None:
            continue
        count = counts[url]
        if not _within_cardinality(count, spec):
            findings.append(
                _finding(
                    "extension-cardinality",
                    "error",
                    "structure",
                    "Extension cardinality is outside the offline allowlist.",
                    f"{root}.extension",
                )
            )

    for url, spec in selected_rules.items():
        if spec.min_occurs and counts.get(url, 0) < spec.min_occurs:
            findings.append(
                _finding(
                    "extension-cardinality",
                    "error",
                    "required",
                    "A required allowlisted extension is missing.",
                    f"{root}.extension",
                )
            )

    return findings


def validate_observation_extensions(observation: Any, **kwargs: Any) -> dict[str, Any]:
    """Return an R4 ``OperationOutcome`` for Observation extension findings."""

    return to_operation_outcome(check_observation_extensions(observation, **kwargs))


def _validate_extension(
    extension: Mapping[str, Any],
    spec: ObservationExtensionSpec,
    fhir_version: str,
    path: str,
    findings: list[ObservationExtensionFinding],
) -> None:
    """Validate one known extension and recursively validate its children."""

    if fhir_version not in spec.fhir_versions:
        findings.append(
            _finding(
                "unsupported-fhir-version",
                "error",
                "not-supported",
                "Extension is not supported in the requested FHIR mode.",
                f"{path}.url",
            )
        )

    value_fields = sorted(
        key for key in extension if _looks_like_value_field(key) and key != "value"
    )
    unsupported_fields = []
    inferred_content = False
    for key in extension:
        if not isinstance(key, str):
            unsupported_fields.append(key)
            continue
        canonical_key = key.casefold()
        if canonical_key in _INFERRED_KEYS:
            inferred_content = True
        if key not in _ALLOWED_EXTENSION_KEYS and key not in value_fields:
            unsupported_fields.append(key)
    if inferred_content:
        findings.append(
            _finding(
                "inferred-extension-content",
                "error",
                "business-rule",
                "Inferred extension content is not accepted; provide an explicit state.",
                path,
            )
        )
    if unsupported_fields:
        findings.append(
            _finding(
                "unsupported-extension-fields",
                "error",
                "not-supported",
                "Extension contains unsupported fields.",
                path,
            )
        )

    raw_nested = extension.get("extension")
    has_nested = "extension" in extension
    nested_items: list[Mapping[str, Any]] = []
    if has_nested:
        if not isinstance(raw_nested, list):
            findings.append(
                _finding(
                    "nested-extension-not-array",
                    "error",
                    "structure",
                    "Nested extension content must be an array.",
                    f"{path}.extension",
                )
            )
        else:
            nested_items = [item for item in raw_nested if isinstance(item, Mapping)]
            if len(nested_items) != len(raw_nested):
                findings.append(
                    _finding(
                        "invalid-nested-extension",
                        "error",
                        "structure",
                        "Nested extension entries must be objects.",
                        f"{path}.extension",
                    )
                )

    if len(value_fields) > 1:
        findings.append(
            _finding(
                "multiple-extension-values",
                "error",
                "structure",
                "An extension must contain only one value[x] field.",
                path,
            )
        )
    if value_fields and has_nested:
        findings.append(
            _finding(
                "mixed-extension-content",
                "error",
                "invariant",
                "An extension cannot contain both value[x] and nested extensions.",
                path,
            )
        )

    if value_fields:
        value_field = value_fields[0]
        if value_field not in spec.value_types and "value[x]" not in spec.value_types:
            findings.append(
                _finding(
                    "unsupported-extension-value-type",
                    "error",
                    "value",
                    "Extension value type is not permitted by the offline allowlist.",
                    f"{path}.{value_field}",
                )
            )
        else:
            value = extension.get(value_field)
            if not _valid_value_shape(value_field, value):
                findings.append(
                    _finding(
                        "invalid-extension-value-shape",
                        "error",
                        "value",
                        "Extension value has an unsupported shape.",
                        f"{path}.{value_field}",
                    )
                )
            is_inferred_marker = (
                isinstance(value, str) and value.casefold() in _INFERRED_MARKERS
            )
            if is_inferred_marker or (
                spec.allowed_values and value not in spec.allowed_values
            ):
                finding_code = (
                    "inferred-extension-content"
                    if is_inferred_marker
                    else "invalid-explicit-state"
                )
                diagnostics = (
                    "Inferred extension content is not accepted; provide an explicit state."
                    if finding_code == "inferred-extension-content"
                    else "Extension value is not an explicit allowlisted state."
                )
                findings.append(
                    _finding(
                        finding_code,
                        "error",
                        "value",
                        diagnostics,
                        f"{path}.{value_field}",
                    )
                )
    elif not has_nested or (isinstance(raw_nested, list) and not nested_items):
        findings.append(
            _finding(
                "missing-extension-content",
                "error",
                "required",
                "Extension must contain an explicit value[x] or nested content.",
                path,
            )
        )

    if not has_nested:
        return
    if not nested_items:
        if spec.nested:
            findings.append(
                _finding(
                    "missing-extension-content",
                    "error",
                    "required",
                    "Extension must contain an explicit value[x] or nested content.",
                    f"{path}.extension",
                )
            )
        return
    if not spec.nested:
        findings.append(
            _finding(
                "unsupported-nested-content",
                "error",
                "not-supported",
                "Nested extension content is not permitted by the offline allowlist.",
                f"{path}.extension",
            )
        )
        return

    counts: dict[str, int] = {}
    first_indexes: dict[str, int] = {}
    for index, child in enumerate(nested_items):
        child_path = f"{path}.extension[{index}]"
        child_url = child.get("url")
        if not _is_nested_url(child_url):
            findings.append(
                _finding(
                    "invalid-nested-extension-url",
                    "error",
                    "value",
                    "Nested extension URL must be a non-empty code.",
                    f"{child_path}.url",
                )
            )
            continue
        counts[child_url] = counts.get(child_url, 0) + 1
        first_indexes.setdefault(child_url, index)
        child_spec = _find_nested_spec(spec.nested, child_url)
        if child_spec is None:
            findings.append(
                _finding(
                    "unsupported-nested-extension-url",
                    "error",
                    "not-supported",
                    "Nested extension URL is not in the offline allowlist.",
                    f"{child_path}.url",
                )
            )
            continue
        _validate_extension(child, child_spec, fhir_version, child_path, findings)

    for child_url in sorted(counts, key=lambda item: first_indexes[item]):
        child_spec = _find_nested_spec(spec.nested, child_url)
        if child_spec is not None and not _within_cardinality(
            counts[child_url], child_spec
        ):
            findings.append(
                _finding(
                    "nested-extension-cardinality",
                    "error",
                    "structure",
                    "Nested extension cardinality is outside the offline allowlist.",
                    f"{path}.extension",
                )
            )
    for child_url, child_spec in spec.nested.items():
        count = _nested_count(counts, child_url)
        if child_spec.min_occurs and count < child_spec.min_occurs:
            findings.append(
                _finding(
                    "nested-extension-cardinality",
                    "error",
                    "required",
                    "A required nested extension is missing.",
                    f"{path}.extension",
                )
            )


def _resolve_rules(
    allowed_extensions: Mapping[str, Any] | Sequence[str] | None,
    *,
    extension_rules: Mapping[str, Any] | Sequence[str] | None,
    extension_allowlist: Mapping[str, Any] | Sequence[str] | None,
    allowlist: Mapping[str, Any] | Sequence[str] | None,
    rules: Mapping[str, Any] | Sequence[str] | None,
) -> Mapping[str, ObservationExtensionSpec]:
    supplied = [
        candidate
        for candidate in (
            allowed_extensions,
            extension_rules,
            extension_allowlist,
            allowlist,
            rules,
        )
        if candidate is not None
    ]
    if len(supplied) > 1:
        raise ValueError("provide only one Observation extension allowlist")
    if not supplied:
        return DEFAULT_OBSERVATION_EXTENSION_RULES

    raw_rules = supplied[0]
    if isinstance(raw_rules, Mapping):
        result: dict[str, ObservationExtensionSpec] = {}
        for url, raw_spec in raw_rules.items():
            if not isinstance(url, str) or not _is_canonical_url(url):
                raise ValueError("allowlist keys must be absolute extension URLs")
            result[url] = _coerce_spec(raw_spec)
        return MappingProxyType(result)
    if isinstance(raw_rules, (str, bytes)) or not isinstance(raw_rules, Sequence):
        raise TypeError("Observation extension allowlist must be a mapping or sequence")
    result = {}
    for url in raw_rules:
        if not isinstance(url, str) or not _is_canonical_url(url):
            raise ValueError("allowlist entries must be absolute extension URLs")
        result[url] = ObservationExtensionSpec(
            value_types=tuple(sorted(_FHIR_VALUE_FIELDS))
        )
    return MappingProxyType(result)


def _coerce_spec(raw_spec: Any) -> ObservationExtensionSpec:
    if isinstance(raw_spec, ObservationExtensionSpec):
        return raw_spec
    if raw_spec is None:
        return ObservationExtensionSpec(value_types=tuple(sorted(_FHIR_VALUE_FIELDS)))
    if not isinstance(raw_spec, Mapping):
        raise TypeError("each Observation extension rule must be a spec or mapping")

    value_types = raw_spec.get("value_types", raw_spec.get("value_type", ()))
    allowed_values = raw_spec.get(
        "allowed_values",
        raw_spec.get("allowed_codes", raw_spec.get("codes", ())),
    )
    nested = raw_spec.get(
        "nested",
        raw_spec.get("children", raw_spec.get("nested_extensions", {})),
    )
    min_occurs = raw_spec.get(
        "min_occurs",
        raw_spec.get("min", raw_spec.get("min_cardinality", 0)),
    )
    max_occurs = raw_spec.get(
        "max_occurs",
        raw_spec.get("max", raw_spec.get("max_cardinality", 1)),
    )
    if max_occurs == "*":
        max_occurs = None
    fhir_versions = raw_spec.get(
        "fhir_versions",
        raw_spec.get("versions", _FHIR_VERSIONS),
    )
    explicit_only = raw_spec.get(
        "explicit_only",
        raw_spec.get("require_explicit", not raw_spec.get("allow_inferred", False)),
    )
    return ObservationExtensionSpec(
        value_types=value_types,
        min_occurs=min_occurs,
        max_occurs=max_occurs,
        allowed_values=allowed_values,
        nested=nested,
        fhir_versions=fhir_versions,
        explicit_only=explicit_only,
    )


def _resolve_version(
    fhir_version: str,
    *,
    version: str | None,
    mode: str | None,
    fhir_release: str | None,
) -> str:
    aliases = [candidate for candidate in (version, mode, fhir_release) if candidate]
    normalised_aliases = {_normalise_version(candidate) for candidate in aliases}
    if len(normalised_aliases) > 1:
        raise ValueError("FHIR version aliases must agree")
    if normalised_aliases:
        alias = next(iter(normalised_aliases))
        if _normalise_version(fhir_version) not in {FHIR_R4, alias}:
            raise ValueError("FHIR version aliases must agree")
        return alias
    return _normalise_version(fhir_version)


def _normalise_version(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("FHIR version must be a string")
    canonical = value.strip().casefold()
    if canonical in {"r4", "4", "4.0", "4.0.1"}:
        return FHIR_R4
    if canonical in {"r5", "5", "5.0", "5.0.0"}:
        return FHIR_R5
    raise ValueError("FHIR version must be R4 or R5")


def _normalise_versions(values: Any) -> frozenset[str]:
    if isinstance(values, str):
        values = (values,)
    if isinstance(values, (set, frozenset)):
        values = tuple(sorted(values))
    if isinstance(values, (bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError("fhir_versions must be a sequence")
    versions = frozenset(_normalise_version(value) for value in values)
    if not versions:
        raise ValueError("fhir_versions must not be empty")
    return versions


def _normalise_string_tuple(values: Any, label: str) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    if isinstance(values, (set, frozenset)):
        values = tuple(sorted(values))
    if isinstance(values, (bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(value.strip() for value in values if isinstance(value, str))
    if len(result) != len(values) or any(not value for value in result):
        raise ValueError(f"{label} must contain non-empty strings")
    return result


def _is_canonical_url(value: Any) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if any(character.isspace() for character in value):
        return False
    parsed = urlparse(value)
    if parsed.scheme not in _CANONICAL_URI_SCHEMES:
        return False
    if parsed.scheme in {"http", "https"}:
        return bool(parsed.netloc)
    return bool(parsed.path)


def _is_nested_url(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and not any(character.isspace() for character in value)
    )


def _find_nested_spec(
    specs: Mapping[str, ObservationExtensionSpec],
    child_url: str,
) -> ObservationExtensionSpec | None:
    direct = specs.get(child_url)
    if direct is not None:
        return direct
    local_name = child_url.rsplit("/", 1)[-1]
    return specs.get(local_name)


def _nested_count(counts: Mapping[str, int], key: str) -> int:
    """Count a nested rule keyed by either its local or canonical URL."""

    names = {key, key.rsplit("/", 1)[-1]}
    return sum(
        count
        for actual_url, count in counts.items()
        if actual_url in names or actual_url.rsplit("/", 1)[-1] in names
    )


def _within_cardinality(
    count: int,
    spec: ObservationExtensionSpec,
) -> bool:
    return count >= spec.min_occurs and (
        spec.max_occurs is None or count <= spec.max_occurs
    )


def _looks_like_value_field(key: Any) -> bool:
    return isinstance(key, str) and key.startswith("value")


def _valid_value_shape(field: str, value: Any) -> bool:
    if field == "valueBoolean":
        return isinstance(value, bool)
    if field in {"valueInteger", "valuePositiveInt", "valueUnsignedInt"}:
        if not isinstance(value, int) or isinstance(value, bool):
            return False
        if field == "valuePositiveInt":
            return value > 0
        if field == "valueUnsignedInt":
            return value >= 0
        return True
    if field == "valueDecimal":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    if field == "valueCode":
        return isinstance(value, str) and bool(_CODE_VALUE.fullmatch(value))
    if field in {
        "valueId",
        "valueMarkdown",
        "valueOid",
        "valueString",
        "valueTime",
        "valueUuid",
    }:
        return isinstance(value, str) and bool(value) and value == value.strip()
    if field in {"valueDate", "valueDateTime", "valueInstant"}:
        return isinstance(value, str) and bool(_DATE_VALUE.fullmatch(value))
    if field in {"valueUri", "valueUrl", "valueCanonical"}:
        return _is_canonical_url(value)
    if field in {
        "valueBase64Binary",
        "valueContactPoint",
    }:
        return isinstance(value, str) and bool(value)
    if field in {"valueQuantity", "valueReference", "valueCodeableConcept"}:
        return isinstance(value, Mapping)
    return isinstance(value, Mapping)


def _safe_expression(value: Any) -> str:
    if isinstance(value, str):
        candidate = value.strip()
        if _SAFE_EXPRESSION.fullmatch(candidate):
            return candidate
    return "Observation"


def _finding(
    finding_code: str,
    severity: Literal["error", "warning"],
    code: str,
    diagnostics: str,
    expression: str,
) -> ObservationExtensionFinding:
    return {
        "finding_code": finding_code,
        "severity": severity,
        "code": code,
        "diagnostics": diagnostics,
        "expression": [expression] if expression else [],
    }


DEFAULT_OBSERVATION_EXTENSION_RULES = _default_extension_rules()
