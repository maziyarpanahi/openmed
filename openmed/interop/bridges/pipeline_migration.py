"""Data-only scanner for declarative pipeline migration plans.

The scanner accepts JSON or an already-parsed mapping.  It never imports a
module, resolves a callable, launches a process, or evaluates configuration.
Executable-looking fields are classified for manual review and omitted from
the generated OpenMed-native stub.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from typing import Any, Final

from openmed.clinical.journey_contracts import (
    canonical_digest,
    canonical_json,
    derived_opaque_id,
)
from openmed.interop.ingest.pipeline_contracts import PIPELINE_STAGES

PIPELINE_MIGRATION_SCHEMA_VERSION: Final = "1.0.0"
PIPELINE_MIGRATION_COMPATIBILITY: Final = "same_major"
MAX_PIPELINE_DESCRIPTION_BYTES: Final = 1_048_576
MAX_PIPELINE_STAGES: Final = 256
MAX_PIPELINE_JSON_DEPTH: Final = 24
MAX_PIPELINE_JSON_NODES: Final = 8_192
MAX_PIPELINE_STRING_CHARS: Final = 4_096

_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{8,128}$")
_EXECUTABLE_KEYS = frozenset(
    {
        "callable",
        "class",
        "cmd",
        "command",
        "entrypoint",
        "exec",
        "executable",
        "import",
        "module",
        "plugin",
        "python",
        "script",
        "shell",
    }
)
_ALIASES: Final[Mapping[str, str]] = {
    "assertion_detection": "assertion",
    "deidentify": "privacy_policy",
    "entity_extraction": "extraction",
    "ingest": "source_adaptation",
    "linker": "grounding",
    "negation": "assertion",
    "ner": "extraction",
    "normalize": "fact_normalization",
    "persist": "durable_writes",
    "quality": "validation",
    "redact": "privacy_policy",
    "relation": "relation_extraction",
    "router": "document_routing",
    "sink": "durable_writes",
    "temporal": "temporality",
    "terminology": "grounding",
}
_CONFIG_FIELDS: Final[Mapping[str, frozenset[str]]] = {
    "source_adaptation": frozenset({"format", "namespace"}),
    "privacy_policy": frozenset({"policy"}),
    "document_routing": frozenset({"media_type", "route"}),
    "extraction": frozenset({"model_id", "threshold"}),
    "assertion": frozenset({"model_id", "threshold"}),
    "temporality": frozenset({"model_id", "threshold"}),
    "relation_extraction": frozenset({"model_id", "threshold"}),
    "grounding": frozenset({"snapshot_id", "threshold"}),
    "fact_normalization": frozenset({"profile"}),
    "validation": frozenset({"profile", "strict"}),
    "durable_writes": frozenset({"backend", "namespace"}),
}


class PipelineMigrationError(ValueError):
    """Raised when a declarative pipeline description is malformed."""


class PipelineMigrationState(str, Enum):
    """Typed migration outcomes."""

    SUCCESS = "success"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    UNSUPPORTED = "unsupported"
    DENIED = "denied"
    FAILURE = "failure"


class PipelineStageDisposition(str, Enum):
    """How one declared stage maps to an OpenMed-native stage."""

    SUPPORTED = "supported"
    MAPPED = "mapped"
    UNSUPPORTED = "unsupported"
    MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True, slots=True)
class PipelineStageMigration:
    """One deterministic stage classification."""

    source_index: int
    source_type: str
    target_stage: str | None
    disposition: PipelineStageDisposition
    reason_code: str
    omitted_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.source_index) is not int or self.source_index < 0:
            raise PipelineMigrationError("source_index must be non-negative")
        _controlled(self.source_type, "source_type")
        if self.target_stage is not None and self.target_stage not in PIPELINE_STAGES:
            raise PipelineMigrationError("target stage is unsupported")
        disposition = PipelineStageDisposition(self.disposition)
        _controlled(self.reason_code, "reason_code")
        omitted = tuple(sorted(set(self.omitted_fields)))
        if any(_CONTROLLED_RE.fullmatch(item) is None for item in omitted):
            raise PipelineMigrationError("omitted fields must be controlled")
        if (
            disposition
            in {
                PipelineStageDisposition.UNSUPPORTED,
                PipelineStageDisposition.MANUAL_REVIEW,
            }
            and self.target_stage is not None
        ):
            raise PipelineMigrationError("blocked stages cannot have a target")
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "omitted_fields", omitted)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "disposition": self.disposition.value,
            "omitted_fields": list(self.omitted_fields),
            "reason_code": self.reason_code,
            "source_index": self.source_index,
            "source_type": self.source_type,
            "target_stage": self.target_stage,
        }


@dataclass(frozen=True, slots=True)
class PipelineMigrationReport:
    """Versioned migration report and executable-code-free native stub."""

    report_id: str
    state: PipelineMigrationState
    source_digest: str
    stages: tuple[PipelineStageMigration, ...]
    native_config: Mapping[str, Any]
    schema_version: str = PIPELINE_MIGRATION_SCHEMA_VERSION
    compatibility_policy: str = PIPELINE_MIGRATION_COMPATIBILITY

    def __post_init__(self) -> None:
        _opaque(self.report_id, "report_id")
        state = PipelineMigrationState(self.state)
        _digest(self.source_digest, "source_digest")
        if not isinstance(self.stages, tuple):
            raise PipelineMigrationError("stages must be a tuple")
        if len(self.stages) > MAX_PIPELINE_STAGES:
            raise PipelineMigrationError("pipeline stage limit exceeded")
        if tuple(item.source_index for item in self.stages) != tuple(
            range(len(self.stages))
        ):
            raise PipelineMigrationError("pipeline stage indexes are invalid")
        native_config = _validate_native_config(self.native_config)
        expected_state = _report_state(self.stages)
        if state is not expected_state:
            raise PipelineMigrationError("migration report state is inconsistent")
        if self.schema_version != PIPELINE_MIGRATION_SCHEMA_VERSION:
            raise PipelineMigrationError("migration schema version is unsupported")
        if self.compatibility_policy != PIPELINE_MIGRATION_COMPATIBILITY:
            raise PipelineMigrationError("migration compatibility is unsupported")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "native_config", native_config)

    @property
    def report_digest(self) -> str:
        """Return a stable digest for the report."""

        return canonical_digest(self.to_dict())

    @property
    def can_auto_migrate(self) -> bool:
        """Return whether the report is a complete automatic migration."""

        return self.state is PipelineMigrationState.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "native_config": _plain(self.native_config),
            "report_id": self.report_id,
            "schema_version": self.schema_version,
            "source_digest": self.source_digest,
            "stages": [item.to_dict() for item in self.stages],
            "state": self.state.value,
        }

    def to_json(self) -> str:
        """Return canonical JSON."""

        return canonical_json(self.to_dict())


def scan_pipeline_json(payload: str | bytes) -> PipelineMigrationReport:
    """Scan bounded JSON without importing or executing referenced code."""

    if isinstance(payload, bytes):
        if len(payload) > MAX_PIPELINE_DESCRIPTION_BYTES:
            raise PipelineMigrationError("pipeline description byte limit exceeded")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            raise PipelineMigrationError("pipeline description must be UTF-8") from None
    elif isinstance(payload, str):
        if len(payload.encode("utf-8")) > MAX_PIPELINE_DESCRIPTION_BYTES:
            raise PipelineMigrationError("pipeline description byte limit exceeded")
        text = payload
    else:
        raise PipelineMigrationError("pipeline description must be text or bytes")
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        raise PipelineMigrationError("pipeline description is invalid JSON") from None
    return scan_pipeline_mapping(parsed)


def scan_pipeline_mapping(payload: Mapping[str, Any]) -> PipelineMigrationReport:
    """Classify stages from an already-parsed declarative mapping."""

    _validate_json_shape(payload)
    if not isinstance(payload, Mapping) or set(payload) - {"name", "stages", "version"}:
        raise PipelineMigrationError("pipeline description fields are invalid")
    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, list):
        raise PipelineMigrationError("pipeline stages must be an array")
    if len(raw_stages) > MAX_PIPELINE_STAGES:
        raise PipelineMigrationError("pipeline stage limit exceeded")
    source_digest = canonical_digest(_plain(payload))
    migrations: list[PipelineStageMigration] = []
    stub_stages: list[dict[str, Any]] = []
    for index, raw_stage in enumerate(raw_stages):
        migration, stub = _classify_stage(index, raw_stage)
        migrations.append(migration)
        if stub is not None:
            stub_stages.append(stub)
    stages = tuple(migrations)
    native_config = {
        "compatibility_policy": PIPELINE_MIGRATION_COMPATIBILITY,
        "schema_version": PIPELINE_MIGRATION_SCHEMA_VERSION,
        "stages": stub_stages,
    }
    report_id = derived_opaque_id(
        "migration_report",
        source_digest,
        [item.to_dict() for item in stages],
        native_config,
    )
    return PipelineMigrationReport(
        report_id=report_id,
        state=_report_state(stages),
        source_digest=source_digest,
        stages=stages,
        native_config=native_config,
    )


def load_pipeline_migration_schema() -> dict[str, Any]:
    """Load the bundled migration-report JSON Schema."""

    resource = resources.files("openmed.core.schemas.json").joinpath(
        "pipeline_migration_report.schema.json"
    )
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _classify_stage(
    index: int, raw_stage: Any
) -> tuple[PipelineStageMigration, dict[str, Any] | None]:
    if not isinstance(raw_stage, Mapping):
        return (
            PipelineStageMigration(
                source_index=index,
                source_type="unknown",
                target_stage=None,
                disposition=PipelineStageDisposition.UNSUPPORTED,
                reason_code="stage_not_object",
            ),
            None,
        )
    raw_type = raw_stage.get("type")
    source_type = (
        raw_type
        if isinstance(raw_type, str) and _CONTROLLED_RE.fullmatch(raw_type)
        else "unknown"
    )
    keys = {str(key).lower() for key in _walk_keys(raw_stage) if isinstance(key, str)}
    executable_fields = tuple(sorted(keys & _EXECUTABLE_KEYS))
    if executable_fields:
        return (
            PipelineStageMigration(
                source_index=index,
                source_type=source_type,
                target_stage=None,
                disposition=PipelineStageDisposition.MANUAL_REVIEW,
                reason_code="executable_field_blocked",
                omitted_fields=executable_fields,
            ),
            None,
        )
    if any(
        not isinstance(key, str) or _CONTROLLED_RE.fullmatch(key) is None
        for key in raw_stage
    ):
        return (
            PipelineStageMigration(
                source_index=index,
                source_type=source_type,
                target_stage=None,
                disposition=PipelineStageDisposition.MANUAL_REVIEW,
                reason_code="invalid_stage_field",
                omitted_fields=("invalid_field_name",),
            ),
            None,
        )
    extra_stage_fields = tuple(
        sorted(
            key
            for key in raw_stage
            if isinstance(key, str)
            and _CONTROLLED_RE.fullmatch(key)
            and key not in {"config", "type"}
        )
    )
    if source_type in PIPELINE_STAGES:
        target = source_type
        disposition = PipelineStageDisposition.SUPPORTED
        reason = "native_stage"
    elif source_type in _ALIASES:
        target = _ALIASES[source_type]
        disposition = PipelineStageDisposition.MAPPED
        reason = "alias_mapped"
    else:
        return (
            PipelineStageMigration(
                source_index=index,
                source_type=source_type,
                target_stage=None,
                disposition=PipelineStageDisposition.UNSUPPORTED,
                reason_code="stage_type_unsupported",
            ),
            None,
        )
    config = raw_stage.get("config", {})
    if not isinstance(config, Mapping):
        return (
            PipelineStageMigration(
                source_index=index,
                source_type=source_type,
                target_stage=None,
                disposition=PipelineStageDisposition.MANUAL_REVIEW,
                reason_code="config_not_object",
            ),
            None,
        )
    if any(
        not isinstance(key, str) or _CONTROLLED_RE.fullmatch(key) is None
        for key in config
    ):
        return (
            PipelineStageMigration(
                source_index=index,
                source_type=source_type,
                target_stage=None,
                disposition=PipelineStageDisposition.MANUAL_REVIEW,
                reason_code="invalid_config_field",
                omitted_fields=("invalid_field_name",),
            ),
            None,
        )
    allowed = _CONFIG_FIELDS[target]
    omitted = tuple(
        sorted(
            set(extra_stage_fields)
            | {
                key
                for key in config
                if isinstance(key, str)
                and _CONTROLLED_RE.fullmatch(key)
                and key not in allowed
            }
        )
    )
    sanitized = {
        key: _safe_config_value(value, key)
        for key, value in sorted(config.items())
        if isinstance(key, str) and key in allowed
    }
    if omitted and disposition is PipelineStageDisposition.SUPPORTED:
        disposition = PipelineStageDisposition.MAPPED
        reason = "unsupported_config_omitted"
    migration = PipelineStageMigration(
        source_index=index,
        source_type=source_type,
        target_stage=target,
        disposition=disposition,
        reason_code=reason,
        omitted_fields=omitted,
    )
    return migration, {"config": sanitized, "stage": target}


def _safe_config_value(value: Any, field_name: str) -> Any:
    if field_name == "threshold":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PipelineMigrationError("threshold must be numeric")
        number = float(value)
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise PipelineMigrationError("threshold must be between zero and one")
        return number
    if field_name == "strict":
        if type(value) is not bool:
            raise PipelineMigrationError("strict must be boolean")
        return value
    return _controlled(value, field_name)


def _report_state(stages: tuple[PipelineStageMigration, ...]) -> PipelineMigrationState:
    if not stages:
        return PipelineMigrationState.UNKNOWN
    dispositions = {item.disposition for item in stages}
    if dispositions & {
        PipelineStageDisposition.UNSUPPORTED,
        PipelineStageDisposition.MANUAL_REVIEW,
    }:
        return PipelineMigrationState.UNSUPPORTED
    if PipelineStageDisposition.MAPPED in dispositions:
        return PipelineMigrationState.PARTIAL
    return PipelineMigrationState.SUCCESS


def _validate_native_config(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "compatibility_policy",
        "schema_version",
        "stages",
    }:
        raise PipelineMigrationError("native config fields are invalid")
    if value["schema_version"] != PIPELINE_MIGRATION_SCHEMA_VERSION:
        raise PipelineMigrationError("native config schema is unsupported")
    if value["compatibility_policy"] != PIPELINE_MIGRATION_COMPATIBILITY:
        raise PipelineMigrationError("native config compatibility is unsupported")
    stages = value["stages"]
    if not isinstance(stages, list):
        raise PipelineMigrationError("native config stages must be an array")
    for item in stages:
        if not isinstance(item, Mapping) or set(item) != {"config", "stage"}:
            raise PipelineMigrationError("native config stage fields are invalid")
        if item["stage"] not in PIPELINE_STAGES or not isinstance(
            item["config"], Mapping
        ):
            raise PipelineMigrationError("native config stage is invalid")
    return json.loads(canonical_json(_plain(value)))


def _validate_json_shape(value: Any) -> None:
    nodes = 0

    def walk(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_PIPELINE_JSON_NODES:
            raise PipelineMigrationError("pipeline JSON node limit exceeded")
        if depth > MAX_PIPELINE_JSON_DEPTH:
            raise PipelineMigrationError("pipeline JSON depth limit exceeded")
        if isinstance(item, str):
            if len(item) > MAX_PIPELINE_STRING_CHARS:
                raise PipelineMigrationError("pipeline string limit exceeded")
            return
        if item is None or isinstance(item, (bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise PipelineMigrationError("pipeline numbers must be finite")
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise PipelineMigrationError("pipeline keys must be strings")
                walk(key, depth + 1)
                walk(child, depth + 1)
            return
        if isinstance(item, list):
            for child in item:
                walk(child, depth + 1)
            return
        raise PipelineMigrationError("pipeline contains a non-JSON value")

    walk(value, 0)


def _walk_keys(value: Any) -> tuple[str, ...]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str):
                keys.append(key)
            keys.extend(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk_keys(child))
    return tuple(keys)


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite number: {value}")


def _controlled(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise PipelineMigrationError(f"{field_name} must be a controlled identifier")
    return value


def _digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise PipelineMigrationError(f"{field_name} must be a SHA-256 digest")
    return value


def _opaque(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise PipelineMigrationError(f"{field_name} must be an opaque identifier")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "MAX_PIPELINE_DESCRIPTION_BYTES",
    "MAX_PIPELINE_JSON_DEPTH",
    "MAX_PIPELINE_JSON_NODES",
    "MAX_PIPELINE_STAGES",
    "MAX_PIPELINE_STRING_CHARS",
    "PIPELINE_MIGRATION_COMPATIBILITY",
    "PIPELINE_MIGRATION_SCHEMA_VERSION",
    "PipelineMigrationError",
    "PipelineMigrationReport",
    "PipelineMigrationState",
    "PipelineStageDisposition",
    "PipelineStageMigration",
    "load_pipeline_migration_schema",
    "scan_pipeline_json",
    "scan_pipeline_mapping",
]
