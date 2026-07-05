"""Versioned tool schema registry for OpenMed MCP and adapter surfaces."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from inspect import Parameter, Signature
from typing import Any, Optional

from openmed.core.pii_i18n import DEFAULT_PII_MODELS, SUPPORTED_LANGUAGES

JsonSchema = dict[str, Any]
JsonObject = dict[str, Any]

_SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_STABILITY_VALUES = frozenset({"experimental", "stable", "deprecated"})
_CLINICAL_STAGE_VALUES = (
    "detect",
    "context",
    "sections",
    "relations",
    "ground",
    "export",
    "risk",
)


class ToolSchemaValidationError(ValueError):
    """Raised when a tool payload does not match its declared JSON schema."""


class ToolCompatibilityError(ValueError):
    """Raised when a tool schema change violates registry compatibility rules."""


@dataclass(frozen=True)
class ToolParameter:
    """One callable parameter and its JSON-schema declaration."""

    name: str
    schema: Mapping[str, Any]
    annotation: Any = Any
    default: Any = Parameter.empty
    description: str = ""

    @property
    def required(self) -> bool:
        """Return whether this parameter has no callable default."""

        return self.default is Parameter.empty

    def schema_property(self) -> JsonSchema:
        """Return this parameter's JSON-schema property."""

        payload = deepcopy(dict(self.schema))
        if self.description:
            payload.setdefault("description", self.description)
        if self.default is not Parameter.empty:
            payload.setdefault("default", self.default)
        return payload

    def signature_parameter(self) -> Parameter:
        """Return the inspect signature parameter for a rendered tool."""

        return Parameter(
            self.name,
            kind=Parameter.POSITIONAL_OR_KEYWORD,
            default=self.default,
            annotation=self.annotation,
        )


@dataclass(frozen=True)
class ToolSpec:
    """Canonical schema contract for one OpenMed agent-facing tool."""

    name: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    version: str = "1.0.0"
    stability: str = "stable"
    parameters: Sequence[ToolParameter] = ()

    def __post_init__(self) -> None:
        """Normalize mutable inputs and validate core metadata."""

        if not _SEMVER_RE.match(self.version):
            raise ValueError(f"tool spec {self.name!r} has invalid semver")
        if self.stability not in _STABILITY_VALUES:
            known = ", ".join(sorted(_STABILITY_VALUES))
            raise ValueError(
                f"tool spec {self.name!r} stability must be one of {known}"
            )
        object.__setattr__(self, "input_schema", deepcopy(dict(self.input_schema)))
        object.__setattr__(self, "output_schema", deepcopy(dict(self.output_schema)))
        object.__setattr__(self, "parameters", tuple(self.parameters))

    @property
    def signature(self) -> Signature:
        """Return the generated Python signature for framework renderers."""

        return Signature(
            [parameter.signature_parameter() for parameter in self.parameters],
            return_annotation=dict[str, Any],
        )

    def document(self) -> JsonObject:
        """Return a machine-readable schema document for this tool."""

        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "stability": self.stability,
            "input_schema": deepcopy(dict(self.input_schema)),
            "output_schema": deepcopy(dict(self.output_schema)),
        }


class ToolRegistry:
    """In-memory registry of named, versioned tool specifications."""

    def __init__(self, specs: Iterable[ToolSpec] = ()) -> None:
        self._specs: dict[tuple[str, str], ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        """Register one tool spec version."""

        key = (spec.name, spec.version)
        if key in self._specs:
            raise ValueError(f"duplicate tool spec {spec.name!r} {spec.version!r}")
        self._specs[key] = spec

    def get(self, name: str, version: str | None = None) -> ToolSpec:
        """Return one spec by name and optional version."""

        if version is not None:
            try:
                return self._specs[(name, version)]
            except KeyError as exc:
                raise KeyError(f"unknown tool spec {name!r} {version!r}") from exc

        candidates = [spec for spec in self._specs.values() if spec.name == name]
        if not candidates:
            raise KeyError(f"unknown tool spec {name!r}")
        return max(candidates, key=lambda spec: _semver_key(spec.version))

    def latest_specs(self) -> tuple[ToolSpec, ...]:
        """Return the latest version of each registered tool, sorted by name."""

        names = sorted({name for name, _version in self._specs})
        return tuple(self.get(name) for name in names)

    def all_specs(self) -> tuple[ToolSpec, ...]:
        """Return all registered specs sorted by name and semantic version."""

        return tuple(
            sorted(
                self._specs.values(),
                key=lambda spec: (spec.name, _semver_key(spec.version)),
            )
        )

    def document(self) -> JsonObject:
        """Return the generated machine-readable registry document."""

        return {
            "schema_version": "1.0.0",
            "tools": [spec.document() for spec in self.all_specs()],
        }


def input_schema(parameters: Sequence[ToolParameter]) -> JsonSchema:
    """Build an object input schema from canonical parameters."""

    required = [parameter.name for parameter in parameters if parameter.required]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            parameter.name: parameter.schema_property() for parameter in parameters
        },
        "required": required,
    }


def render_mcp_tool(
    spec: ToolSpec,
    handler: Callable[..., JsonObject],
) -> Callable[..., JsonObject]:
    """Render a FastMCP-compatible callable from a tool spec and handler."""

    def _tool(*args: Any, **kwargs: Any) -> JsonObject:
        bound = spec.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return invoke_tool(spec, handler, **bound.arguments)

    _tool.__name__ = f"_{spec.name}_tool"
    _tool.__doc__ = spec.description
    _tool.__signature__ = spec.signature  # type: ignore[attr-defined]
    _tool.__annotations__ = {
        parameter.name: parameter.annotation for parameter in spec.parameters
    }
    _tool.__annotations__["return"] = dict[str, Any]
    return _tool


def invoke_tool(
    spec: ToolSpec,
    handler: Callable[..., JsonObject],
    **kwargs: Any,
) -> JsonObject:
    """Invoke a tool handler and validate its structured output."""

    result = handler(**kwargs)
    return validate_tool_output(spec, result)


def validate_registered_tool_output(name: str, payload: Any) -> JsonObject:
    """Validate an output payload against the latest registered spec for *name*."""

    return validate_tool_output(TOOL_REGISTRY.get(name), payload)


def validate_tool_output(spec: ToolSpec, payload: Any) -> JsonObject:
    """Validate a tool output payload against its declared schema."""

    errors: list[str] = []
    _validate_schema(payload, spec.output_schema, "$", errors)
    if errors:
        preview = "; ".join(errors[:4])
        raise ToolSchemaValidationError(
            f"{spec.name} output failed schema {spec.version}: {preview}"
        )
    if not isinstance(payload, Mapping):
        raise ToolSchemaValidationError(f"{spec.name} output must be an object")
    return dict(payload)


def render_tool_registry_document(
    registry: ToolRegistry | None = None,
) -> JsonObject:
    """Return the generated registry resource payload."""

    return (registry or TOOL_REGISTRY).document()


def render_adapter_tool_definitions(
    adapter: str,
    specs: Iterable[ToolSpec] | None = None,
) -> tuple[JsonObject, ...]:
    """Render dependency-light adapter tool definitions from registry specs."""

    normalized = str(adapter).strip().lower().replace("-", "_")
    return tuple(
        _adapter_definition(normalized, spec)
        for spec in (
            tuple(specs) if specs is not None else TOOL_REGISTRY.latest_specs()
        )
    )


def render_langchain_tool_definitions(
    specs: Iterable[ToolSpec] | None = None,
) -> tuple[JsonObject, ...]:
    """Render LangChain-compatible JSON tool definitions."""

    return render_adapter_tool_definitions("langchain", specs=specs)


def render_presidio_tool_definitions(
    specs: Iterable[ToolSpec] | None = None,
) -> tuple[JsonObject, ...]:
    """Render Presidio-adapter JSON tool definitions."""

    return render_adapter_tool_definitions("presidio", specs=specs)


def check_tool_registry_compatibility(
    previous: Iterable[ToolSpec],
    current: Iterable[ToolSpec],
) -> None:
    """Raise if *current* breaks tool schemas without a semantic version bump."""

    previous_by_name = _latest_by_name(previous)
    current_by_name = _latest_by_name(current)
    errors: list[str] = []

    for name, old_spec in previous_by_name.items():
        new_spec = current_by_name.get(name)
        if new_spec is None:
            errors.append(f"{name}: tool was removed")
            continue

        old_key = _semver_key(old_spec.version)
        new_key = _semver_key(new_spec.version)
        if new_key < old_key:
            errors.append(
                f"{name}: version regressed from {old_spec.version} "
                f"to {new_spec.version}"
            )
            continue

        breaking = [
            *_schema_breaks(old_spec.input_schema, new_spec.input_schema, "input"),
            *_schema_breaks(old_spec.output_schema, new_spec.output_schema, "output"),
        ]
        schema_changed = (
            old_spec.input_schema != new_spec.input_schema
            or old_spec.output_schema != new_spec.output_schema
        )
        if breaking and new_key == old_key:
            errors.extend(
                f"{name}: breaking schema change without version bump: {issue}"
                for issue in breaking
            )
        elif breaking and new_key[0] == old_key[0]:
            errors.extend(
                f"{name}: breaking schema change requires major version bump: {issue}"
                for issue in breaking
            )
        elif schema_changed and new_key == old_key:
            errors.append(f"{name}: schema changed without version bump")

    if errors:
        raise ToolCompatibilityError("; ".join(errors))


def _adapter_definition(adapter: str, spec: ToolSpec) -> JsonObject:
    return {
        "adapter": adapter,
        "name": spec.name,
        "description": spec.description,
        "version": spec.version,
        "stability": spec.stability,
        "input_schema": deepcopy(dict(spec.input_schema)),
        "output_schema": deepcopy(dict(spec.output_schema)),
    }


def _latest_by_name(specs: Iterable[ToolSpec]) -> dict[str, ToolSpec]:
    latest: dict[str, ToolSpec] = {}
    for spec in specs:
        existing = latest.get(spec.name)
        if existing is None or _semver_key(spec.version) > _semver_key(
            existing.version
        ):
            latest[spec.name] = spec
    return latest


def _semver_key(version: str) -> tuple[int, int, int]:
    match = _SEMVER_RE.match(version)
    if match is None:
        raise ValueError(f"invalid semver {version!r}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _schema_breaks(
    old_schema: Mapping[str, Any],
    new_schema: Mapping[str, Any],
    direction: str,
) -> list[str]:
    errors: list[str] = []
    old_properties = dict(old_schema.get("properties") or {})
    new_properties = dict(new_schema.get("properties") or {})
    old_required = set(_as_str_sequence(old_schema.get("required") or ()))
    new_required = set(_as_str_sequence(new_schema.get("required") or ()))

    for name in sorted(old_required):
        if name not in new_properties:
            errors.append(f"{direction}.{name} required property removed")

    if direction == "input":
        for name in sorted(new_required - old_required):
            errors.append(f"input.{name} new required property added")

    for name, old_property in sorted(old_properties.items()):
        if name not in new_properties:
            errors.append(f"{direction}.{name} property removed")
            continue
        if _schema_type(old_property) != _schema_type(new_properties[name]):
            errors.append(f"{direction}.{name} type changed")
        old_enum = tuple(old_property.get("enum") or ())
        new_enum = tuple(new_properties[name].get("enum") or ())
        if old_enum and new_enum and not set(old_enum) <= set(new_enum):
            errors.append(f"{direction}.{name} enum narrowed")

    if (
        old_schema.get("additionalProperties", True) is True
        and new_schema.get("additionalProperties", True) is False
    ):
        errors.append(f"{direction}.additionalProperties narrowed")

    return errors


def _schema_type(schema: Mapping[str, Any]) -> Any:
    if "type" in schema:
        value = schema["type"]
        if isinstance(value, list):
            return tuple(sorted(str(item) for item in value))
        return value
    if "anyOf" in schema:
        return tuple(_schema_type(item) for item in schema["anyOf"])
    return None


def _validate_schema(
    value: Any,
    schema: Mapping[str, Any],
    path: str,
    errors: list[str],
) -> None:
    if "anyOf" in schema:
        nested_errors: list[list[str]] = []
        for option in schema["anyOf"]:
            option_errors: list[str] = []
            _validate_schema(value, option, path, option_errors)
            if not option_errors:
                return
            nested_errors.append(option_errors)
        first = nested_errors[0][0] if nested_errors and nested_errors[0] else ""
        errors.append(f"{path}: did not match any allowed schema ({first})")
        return

    expected = schema.get("type")
    if expected is not None and not _matches_json_type(value, expected):
        errors.append(
            f"{path}: expected {_type_label(expected)}, got {type(value).__name__}"
        )
        return

    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(repr(item) for item in schema["enum"])
        errors.append(f"{path}: expected one of {allowed}")
        return

    if isinstance(value, Mapping):
        properties = dict(schema.get("properties") or {})
        for required in _as_str_sequence(schema.get("required") or ()):
            if required not in value:
                errors.append(f"{path}.{required}: missing required property")
        for key, item in value.items():
            key_path = f"{path}.{key}"
            if key in properties:
                _validate_schema(item, properties[key], key_path, errors)
                continue
            additional = schema.get("additionalProperties", True)
            if additional is False:
                errors.append(f"{key_path}: unexpected property")
            elif isinstance(additional, Mapping):
                _validate_schema(item, additional, key_path, errors)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_schema(item, item_schema, f"{path}[{index}]", errors)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            errors.append(f"{path}: value below minimum {minimum}")
        if maximum is not None and value > maximum:
            errors.append(f"{path}: value above maximum {maximum}")


def _matches_json_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, Sequence) and not isinstance(expected, str):
        return any(_matches_json_type(value, item) for item in expected)
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        )
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _type_label(expected: Any) -> str:
    if isinstance(expected, Sequence) and not isinstance(expected, str):
        return " or ".join(str(item) for item in expected)
    return str(expected)


def _as_str_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return ()


def _schema(json_type: Any, **kwargs: Any) -> JsonSchema:
    payload: JsonSchema = {"type": json_type}
    payload.update(kwargs)
    return payload


def _nullable(json_type: str, **kwargs: Any) -> JsonSchema:
    return _schema([json_type, "null"], **kwargs)


def _object(
    *,
    properties: Mapping[str, Any] | None = None,
    required: Sequence[str] = (),
    additional: bool | Mapping[str, Any] = True,
) -> JsonSchema:
    return {
        "type": "object",
        "properties": dict(properties or {}),
        "required": list(required),
        "additionalProperties": additional,
    }


def _array(items: Mapping[str, Any]) -> JsonSchema:
    return {"type": "array", "items": dict(items)}


def _parameter(
    name: str,
    schema: Mapping[str, Any],
    annotation: Any,
    default: Any = Parameter.empty,
    description: str = "",
) -> ToolParameter:
    return ToolParameter(
        name=name,
        schema=schema,
        annotation=annotation,
        default=default,
        description=description,
    )


_TEXT_PARAMETER = _parameter(
    "text",
    _schema("string"),
    str,
    description="Clinical text to process.",
)
_MODEL_NAME_PARAMETER = _parameter(
    "model_name",
    _schema("string"),
    str,
    "disease_detection_superclinical",
    "OpenMed model registry key, local model path, or model identifier.",
)
_PII_MODEL_NAME_PARAMETER = _parameter(
    "model_name",
    _schema("string"),
    str,
    DEFAULT_PII_MODELS["en"],
    "PII model registry key, local model path, or model identifier.",
)
_CONFIDENCE_PARAMETER = _parameter(
    "confidence_threshold",
    _nullable("number", minimum=0.0, maximum=1.0),
    Optional[float],
    0.0,
    "Minimum confidence score for returned entities.",
)
_PII_CONFIDENCE_PARAMETER = _parameter(
    "confidence_threshold",
    _schema("number", minimum=0.0, maximum=1.0),
    float,
    0.5,
    "Minimum confidence score for returned PII entities.",
)
_DEID_CONFIDENCE_PARAMETER = _parameter(
    "confidence_threshold",
    _schema("number", minimum=0.0, maximum=1.0),
    float,
    0.7,
    "Minimum confidence score for de-identification.",
)
_KEEP_ALIVE_PARAMETER = _parameter(
    "keep_alive",
    _nullable("string"),
    Optional[str],
    None,
    "Optional model keep-alive duration for the service runtime.",
)
_LANG_PARAMETER = _parameter(
    "lang",
    _schema("string", enum=sorted(SUPPORTED_LANGUAGES)),
    str,
    "en",
    "PII language code.",
)
_NORMALIZE_ACCENTS_PARAMETER = _parameter(
    "normalize_accents",
    _nullable("boolean"),
    Optional[bool],
    None,
    "Override language-specific accent normalization.",
)

_ENTITY_OUTPUT = _object(
    properties={
        "text": _schema("string"),
        "label": _schema("string"),
        "confidence": _schema("number"),
        "start": _nullable("integer"),
        "end": _nullable("integer"),
        "metadata": _object(),
    },
    required=("text", "label", "confidence", "start", "end", "metadata"),
)
_PREDICTION_OUTPUT = _object(
    properties={
        "text": _schema("string"),
        "entities": _array(_ENTITY_OUTPUT),
        "model_name": _schema("string"),
        "timestamp": _schema("string"),
        "processing_time": _nullable("number"),
        "metadata": _nullable("object"),
    },
    required=(
        "text",
        "entities",
        "model_name",
        "timestamp",
        "processing_time",
        "metadata",
    ),
)
_PII_ENTITY_OUTPUT = _object(
    properties={
        **_ENTITY_OUTPUT["properties"],
        "entity_type": _schema("string"),
        "redacted_text": _nullable("string"),
        "canonical_label": _nullable("string"),
        "sources": _array(_schema("string")),
        "evidence": _object(),
        "threshold": _nullable("number"),
        "action": _nullable("string"),
        "surrogate": _nullable("string"),
        "reversible_id": _schema("string"),
    },
    required=(
        "text",
        "label",
        "entity_type",
        "start",
        "end",
        "confidence",
        "redacted_text",
        "canonical_label",
        "sources",
        "evidence",
        "threshold",
        "action",
        "surrogate",
        "metadata",
    ),
)
_DEIDENTIFY_OUTPUT = _object(
    properties={
        "original_text": _schema("string"),
        "deidentified_text": _schema("string"),
        "pii_entities": _array(_PII_ENTITY_OUTPUT),
        "method": _schema("string"),
        "timestamp": _schema("string"),
        "num_entities_redacted": _schema("integer"),
        "metadata": _object(),
        "audit_report": _nullable("object"),
        "mapping": _object(additional=_schema("string")),
    },
    required=(
        "original_text",
        "deidentified_text",
        "pii_entities",
        "method",
        "timestamp",
        "num_entities_redacted",
        "metadata",
        "audit_report",
    ),
)
_MODEL_INFO_OUTPUT = _object(
    properties={
        "key": _schema("string"),
        "model_id": _schema("string"),
        "display_name": _schema("string"),
        "category": _schema("string"),
        "specialization": _schema("string"),
        "description": _schema("string"),
        "entity_types": _array(_schema("string")),
        "size_category": _schema("string"),
        "size_mb": _nullable("integer"),
    },
    required=(
        "key",
        "model_id",
        "display_name",
        "category",
        "specialization",
        "description",
        "entity_types",
        "size_category",
        "size_mb",
    ),
)
_LIST_MODELS_OUTPUT = _object(
    properties={
        "count": _schema("integer"),
        "returned": _schema("integer"),
        "models": _array(_MODEL_INFO_OUTPUT),
    },
    required=("count", "returned", "models"),
)
_PII_LANGUAGE_OUTPUT = _object(
    properties={
        "code": _schema("string"),
        "name": _schema("string"),
        "default_pii_model": _schema("string"),
        "model_count": _schema("integer"),
    },
    required=("code", "name", "default_pii_model", "model_count"),
)
_LIST_PII_LANGUAGES_OUTPUT = _object(
    properties={
        "count": _schema("integer"),
        "languages": _array(_PII_LANGUAGE_OUTPUT),
    },
    required=("count", "languages"),
)
_GENERIC_OBJECT_OUTPUT = _object()
_WORKFLOW_TRACE_STEP_OUTPUT = _object(
    properties={
        "step_id": _schema("string"),
        "tool": _schema("string"),
        "status": _schema(
            "string",
            enum=["completed", "failed", "resumed", "skipped"],
        ),
        "duration_ms": _schema("number"),
        "retry_count": _schema("integer"),
        "attempt_count": _schema("integer"),
        "input_handles": _array(_schema("string")),
        "output_handle": _nullable("string"),
        "error_type": _schema("string"),
        "resumed": _schema("boolean"),
    },
    required=("step_id", "tool", "status", "duration_ms", "retry_count"),
)
_WORKFLOW_RESULT_OUTPUT = _object(
    properties={
        "schema_version": _schema("string"),
        "session_id": _schema("string"),
        "workflow_id": _schema("string"),
        "status": _schema("string", enum=["completed", "failed"]),
        "handles": _object(),
        "final_handle": _nullable("string"),
        "final_output": {},
        "outputs": _object(),
        "trace": _array(_WORKFLOW_TRACE_STEP_OUTPUT),
    },
    required=(
        "schema_version",
        "session_id",
        "workflow_id",
        "status",
        "handles",
        "final_handle",
        "final_output",
        "outputs",
        "trace",
    ),
)
_OPENMED_SPAN_OUTPUT = _object(
    properties={
        "schema_version": _schema("integer"),
        "doc_id": _schema("string"),
        "start": _schema("integer", minimum=0),
        "end": _schema("integer", minimum=0),
        "text_hash": _schema("string"),
        "entity_type": _schema("string"),
        "canonical_label": _schema("string"),
        "policy_label": _schema("string"),
        "regulatory_tags": _array(_schema("string")),
        "score": _nullable("number", minimum=0.0, maximum=1.0),
        "detector": _nullable("string"),
        "evidence": _object(),
        "action": _schema("string"),
        "replacement": _nullable("string"),
        "reversible_id": _nullable("string"),
        "section": _nullable("string"),
        "metadata": _object(),
    },
    required=(
        "schema_version",
        "doc_id",
        "start",
        "end",
        "text_hash",
        "entity_type",
        "canonical_label",
        "policy_label",
        "regulatory_tags",
        "score",
        "detector",
        "evidence",
        "action",
        "replacement",
        "reversible_id",
        "section",
        "metadata",
    ),
    additional=False,
)
_STRUCTURED_ERROR_OUTPUT = _object(
    properties={
        "code": _schema("string"),
        "message": _schema("string"),
        "stage": _nullable("string"),
        "details": _object(),
    },
    required=("code", "message", "stage", "details"),
)
_ERROR_OR_NULL = {"anyOf": [_STRUCTURED_ERROR_OUTPUT, _schema("null")]}
_CLINICAL_TRACE_OUTPUT = _object(
    properties={
        "stage": _schema("string", enum=_CLINICAL_STAGE_VALUES),
        "status": _schema("string"),
        "duration_ms": _nullable("number"),
    },
    required=("stage", "status"),
)
_GROUND_OUTPUT = _object(
    properties={
        "schema_version": _schema("string", enum=["openmed.ground.v1"]),
        "status": _schema("string", enum=["completed", "failed", "unimplemented"]),
        "spans": _array(_OPENMED_SPAN_OUTPUT),
        "grounded_concepts": _array(_object()),
        "error": _ERROR_OR_NULL,
    },
    required=("schema_version", "status", "spans", "grounded_concepts", "error"),
)
_EXPORT_FHIR_OUTPUT = _object(
    properties={
        "schema_version": _schema("string", enum=["openmed.export_fhir.v1"]),
        "status": _schema("string", enum=["completed", "failed", "unimplemented"]),
        "bundle": _object(),
        "resource_count": _schema("integer", minimum=0),
        "error": _ERROR_OR_NULL,
    },
    required=("schema_version", "status", "bundle", "resource_count", "error"),
)
_RISK_SCORE_OUTPUT = _object(
    properties={
        "schema_version": _schema("string", enum=["openmed.risk_score.v1"]),
        "status": _schema("string", enum=["completed", "failed", "unimplemented"]),
        "risk_report": _object(),
        "error": _ERROR_OR_NULL,
    },
    required=("schema_version", "status", "risk_report", "error"),
)
_CLINICAL_PIPELINE_OUTPUT = _object(
    properties={
        "schema_version": _schema("string", enum=["openmed.clinical_pipeline.v1"]),
        "status": _schema(
            "string", enum=["planned", "completed", "rejected", "failed"]
        ),
        "stages": _array(_schema("string", enum=_CLINICAL_STAGE_VALUES)),
        "artifacts": _object(),
        "final_output": {},
        "error": _ERROR_OR_NULL,
        "trace": _array(_CLINICAL_TRACE_OUTPUT),
    },
    required=(
        "schema_version",
        "status",
        "stages",
        "artifacts",
        "final_output",
        "error",
        "trace",
    ),
)
_SPAN_ARRAY_OR_NULL = {"anyOf": [_array(_OPENMED_SPAN_OUTPUT), _schema("null")]}
_STRING_ARRAY_OR_NULL = {"anyOf": [_array(_schema("string")), _schema("null")]}
_OBJECT_ARRAY_OR_NULL = {"anyOf": [_array(_object()), _schema("null")]}
_CLINICAL_STAGES_PARAMETER = _parameter(
    "stages",
    _array(_schema("string", enum=_CLINICAL_STAGE_VALUES)),
    list[str],
    description=(
        "Declared clinical workflow stages in canonical order: "
        "detect, context, sections, relations, ground, export, risk."
    ),
)
_CLINICAL_SPANS_PARAMETER = _parameter(
    "spans",
    _array(_OPENMED_SPAN_OUTPUT),
    list[dict[str, Any]],
    description="Canonical OpenMedSpan artifacts to process.",
)


def _tool_spec(
    *,
    name: str,
    description: str,
    parameters: Sequence[ToolParameter],
    output_schema: Mapping[str, Any],
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        input_schema=input_schema(parameters),
        output_schema=output_schema,
        parameters=parameters,
    )


TOOL_SPECS: tuple[ToolSpec, ...] = (
    _tool_spec(
        name="openmed_analyze_text",
        description="Run OpenMed named-entity recognition on clinical text.",
        parameters=(
            _TEXT_PARAMETER,
            _MODEL_NAME_PARAMETER,
            _CONFIDENCE_PARAMETER,
            _parameter("group_entities", _schema("boolean"), bool, False),
            _parameter(
                "aggregation_strategy",
                _nullable("string", enum=["average", "first", "max", "simple", None]),
                Optional[str],
                "simple",
            ),
            _parameter("sentence_detection", _schema("boolean"), bool, True),
            _parameter("sentence_language", _schema("string"), str, "en"),
            _parameter("sentence_clean", _schema("boolean"), bool, False),
            _parameter("use_fast_tokenizer", _schema("boolean"), bool, True),
            _KEEP_ALIVE_PARAMETER,
        ),
        output_schema=_PREDICTION_OUTPUT,
    ),
    _tool_spec(
        name="openmed_extract_pii",
        description="Extract PII/PHI entities from clinical text.",
        parameters=(
            _TEXT_PARAMETER,
            _PII_MODEL_NAME_PARAMETER,
            _PII_CONFIDENCE_PARAMETER,
            _parameter("use_smart_merging", _schema("boolean"), bool, True),
            _LANG_PARAMETER,
            _NORMALIZE_ACCENTS_PARAMETER,
            _KEEP_ALIVE_PARAMETER,
        ),
        output_schema=_PREDICTION_OUTPUT,
    ),
    _tool_spec(
        name="openmed_deidentify",
        description="De-identify text by masking, removing, replacing, hashing, or shifting.",
        parameters=(
            _TEXT_PARAMETER,
            _parameter(
                "method",
                _schema(
                    "string",
                    enum=["hash", "mask", "remove", "replace", "shift_dates"],
                ),
                str,
                "mask",
            ),
            _PII_MODEL_NAME_PARAMETER,
            _DEID_CONFIDENCE_PARAMETER,
            _parameter("keep_year", _schema("boolean"), bool, False),
            _parameter("shift_dates", _nullable("boolean"), Optional[bool], None),
            _parameter("date_shift_days", _nullable("integer"), Optional[int], None),
            _parameter("keep_mapping", _schema("boolean"), bool, False),
            _parameter("use_smart_merging", _schema("boolean"), bool, True),
            _LANG_PARAMETER,
            _NORMALIZE_ACCENTS_PARAMETER,
            _KEEP_ALIVE_PARAMETER,
        ),
        output_schema=_DEIDENTIFY_OUTPUT,
    ),
    _tool_spec(
        name="openmed_list_models",
        description="List OpenMed model registry entries.",
        parameters=(
            _parameter("category", _nullable("string"), Optional[str], None),
            _parameter("pii_language", _nullable("string"), Optional[str], None),
            _parameter("limit", _schema("integer", minimum=0), int, 50),
        ),
        output_schema=_LIST_MODELS_OUTPUT,
    ),
    _tool_spec(
        name="openmed_list_pii_languages",
        description="List supported PII languages and default models.",
        parameters=(),
        output_schema=_LIST_PII_LANGUAGES_OUTPUT,
    ),
    _tool_spec(
        name="openmed_loaded_models",
        description="Return currently loaded model resources.",
        parameters=(),
        output_schema=_GENERIC_OBJECT_OUTPUT,
    ),
    _tool_spec(
        name="openmed_unload_model",
        description="Unload one inactive model, or all inactive models.",
        parameters=(
            _parameter("model_name", _nullable("string"), Optional[str], None),
            _parameter("all_models", _schema("boolean"), bool, False),
        ),
        output_schema=_GENERIC_OBJECT_OUTPUT,
    ),
    _tool_spec(
        name="openmed_run_workflow",
        description=(
            "Run a stateful multi-step OpenMed workflow with server-side "
            "intermediate handles and PHI-safe egress."
        ),
        parameters=(
            _parameter("pipeline", _object(), dict[str, Any]),
            _parameter("session_id", _nullable("string"), Optional[str], None),
            _parameter("workflow_id", _nullable("string"), Optional[str], None),
        ),
        output_schema=_WORKFLOW_RESULT_OUTPUT,
    ),
    _tool_spec(
        name="openmed_ground",
        description=(
            "Contract for grounding canonical OpenMedSpan artifacts to clinical "
            "coding systems."
        ),
        parameters=(
            _CLINICAL_SPANS_PARAMETER,
            _parameter(
                "vocabularies",
                _STRING_ARRAY_OR_NULL,
                Optional[list[str]],
                None,
                "Optional grounding vocabularies to constrain.",
            ),
            _parameter(
                "max_candidates",
                _schema("integer", minimum=1),
                int,
                5,
                "Maximum grounding candidates per span.",
            ),
            _parameter(
                "allow_external_llm",
                _schema("boolean"),
                bool,
                False,
                "Reserve external-LLM routing for privacy-gateway mediated flows.",
            ),
        ),
        output_schema=_GROUND_OUTPUT,
    ),
    _tool_spec(
        name="openmed_export_fhir",
        description=(
            "Contract for exporting canonical OpenMedSpan artifacts to a FHIR Bundle."
        ),
        parameters=(
            _CLINICAL_SPANS_PARAMETER,
            _parameter(
                "resources",
                _OBJECT_ARRAY_OR_NULL,
                Optional[list[dict[str, Any]]],
                None,
                "Optional prebuilt standalone FHIR resources.",
            ),
            _parameter("doc_id", _schema("string"), str, "workflow"),
            _parameter(
                "bundle_type",
                _schema("string", enum=["collection", "transaction", "batch"]),
                str,
                "collection",
            ),
        ),
        output_schema=_EXPORT_FHIR_OUTPUT,
    ),
    _tool_spec(
        name="openmed_risk_score",
        description=(
            "Contract for residual re-identification risk scoring over "
            "de-identified clinical artifacts."
        ),
        parameters=(
            _CLINICAL_SPANS_PARAMETER,
            _parameter(
                "deidentified_text",
                _nullable("string"),
                Optional[str],
                None,
                "Optional de-identified text for residual-risk scoring.",
            ),
            _parameter(
                "records",
                _OBJECT_ARRAY_OR_NULL,
                Optional[list[dict[str, Any]]],
                None,
                "Optional tabular records for residual-risk scoring.",
            ),
            _parameter(
                "quasi_identifiers",
                _STRING_ARRAY_OR_NULL,
                Optional[list[str]],
                None,
                "Optional quasi-identifier field names.",
            ),
        ),
        output_schema=_RISK_SCORE_OUTPUT,
    ),
    _tool_spec(
        name="openmed_clinical_pipeline",
        description=(
            "Plan and validate a composable clinical workflow over "
            "de-identification, extraction, grounding, FHIR export, and risk "
            "stages."
        ),
        parameters=(
            _CLINICAL_STAGES_PARAMETER,
            _parameter(
                "text",
                _nullable("string"),
                Optional[str],
                None,
                "Optional raw clinical text for future pipeline execution.",
            ),
            _parameter(
                "spans",
                _SPAN_ARRAY_OR_NULL,
                Optional[list[dict[str, Any]]],
                None,
                "Optional canonical OpenMedSpan artifacts.",
            ),
            _parameter(
                "options",
                _object(),
                dict[str, Any],
                {},
                "Execution options reserved for later pipeline handlers.",
            ),
            _parameter(
                "allow_external_llm",
                _schema("boolean"),
                bool,
                False,
                "Reserve external-LLM routing for privacy-gateway mediated flows.",
            ),
            _parameter("session_id", _nullable("string"), Optional[str], None),
            _parameter("workflow_id", _nullable("string"), Optional[str], None),
        ),
        output_schema=_CLINICAL_PIPELINE_OUTPUT,
    ),
)

TOOL_REGISTRY = ToolRegistry(TOOL_SPECS)

__all__ = [
    "TOOL_REGISTRY",
    "TOOL_SPECS",
    "ToolCompatibilityError",
    "ToolParameter",
    "ToolRegistry",
    "ToolSchemaValidationError",
    "ToolSpec",
    "check_tool_registry_compatibility",
    "input_schema",
    "invoke_tool",
    "render_adapter_tool_definitions",
    "render_langchain_tool_definitions",
    "render_mcp_tool",
    "render_presidio_tool_definitions",
    "render_tool_registry_document",
    "validate_registered_tool_output",
    "validate_tool_output",
]
