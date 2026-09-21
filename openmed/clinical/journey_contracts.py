"""Versioned contracts for durable longitudinal clinical data.

The records in this module describe artifacts and derivations; they do not
store source bytes themselves.  They are intentionally local-only, immutable,
and deterministic.  Unknown additive fields from a supported schema-major
version survive a deserialize/serialize round trip so newer producers can be
handled without silently discarding metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from types import MappingProxyType
from typing import Any, ClassVar, Final

JOURNEY_CONTRACT_SCHEMA_VERSION: Final = "1.0.0"
JOURNEY_CONTRACT_SCHEMA_MAJOR: Final = 1
JOURNEY_SCHEMA_PACKAGE: Final = "openmed.core.schemas.json"
JOURNEY_SCHEMA_NAMES: Final = (
    "clinical_artifact",
    "evidence_locator",
    "clinical_fact",
    "conflict_set",
    "resolution_event",
    "dataset_snapshot",
)

EVIDENCE_LOCATION_TYPES: Final = (
    "text_span",
    "json_pointer",
    "message_field",
    "page_box",
    "dicom_element",
    "table_cell",
)
CONFLICT_STATUSES: Final = ("open", "resolved", "superseded")
RESOLUTION_ACTIONS: Final = (
    "select",
    "reject",
    "merge",
    "defer",
    "reopen",
    "supersede",
)
RESOLUTION_ACTOR_TYPES: Final = ("human", "policy")

_DIGEST_RE: Final = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_ID_PREFIX_RE: Final = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_OPAQUE_ID_RE: Final = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_CONTROLLED_RE: Final = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_SCHEMA_VERSION_RE: Final = re.compile(
    r"^(0|[1-9][0-9]*)(?:\.(0|[1-9][0-9]*))?(?:\.(0|[1-9][0-9]*))?$"
)
_MEDIA_TYPE_RE: Final = re.compile(
    r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$"
)
_DICOM_UID_RE: Final = re.compile(r"^[0-9]+(?:\.[0-9]+)*$")
_DICOM_TAG_RE: Final = re.compile(r"^[0-9A-Fa-f]{4},?[0-9A-Fa-f]{4}$")
_MESSAGE_FIELD_RE: Final = re.compile(
    r"^[A-Z0-9][A-Z0-9_-]*(?:\.[1-9][0-9]*)+(?:\[[1-9][0-9]*\])?$"
)
_JSON_POINTER_BAD_ESCAPE_RE: Final = re.compile(r"~(?![01])")
_SAFE_PATH_COMPONENT_RE: Final = re.compile(r"^[A-Za-z0-9_.@+=,-]{1,255}$")
_MISSING: Final = object()

__all__ = [
    "CONFLICT_STATUSES",
    "EVIDENCE_LOCATION_TYPES",
    "JOURNEY_CONTRACT_SCHEMA_MAJOR",
    "JOURNEY_CONTRACT_SCHEMA_VERSION",
    "JOURNEY_SCHEMA_NAMES",
    "RESOLUTION_ACTIONS",
    "RESOLUTION_ACTOR_TYPES",
    "ClinicalArtifact",
    "ClinicalFact",
    "ConflictSet",
    "ContractGraphDiagnostics",
    "DatasetSnapshot",
    "EvidenceLocator",
    "JourneyContractError",
    "JourneyContractGraphError",
    "JourneySchemaVersionError",
    "ResolutionEvent",
    "canonical_digest",
    "canonical_json",
    "compute_derivation_hash",
    "derived_opaque_id",
    "load_all_journey_schemas",
    "load_journey_schema",
    "new_opaque_id",
    "sha256_digest",
    "validate_contract_graph",
]


class JourneyContractError(ValueError):
    """Base error for malformed Journey contract data."""


class JourneySchemaVersionError(JourneyContractError):
    """Raised when a payload uses an unsupported schema-major version."""


class JourneyContractGraphError(JourneyContractError):
    """Raised when contract references form an invalid derivation graph."""


class _JourneyRecord:
    """Shared deterministic serialization behavior for Journey records."""

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Any:
        """Deserialize a record from a dictionary."""

        raise NotImplementedError

    def to_json(self, *, indent: int | None = None) -> str:
        """Return deterministic JSON for this record."""

        return canonical_json(self.to_dict(), indent=indent)

    @property
    def canonical_hash(self) -> str:
        """Return a digest of the complete serialized record."""

        return canonical_digest(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> Any:
        """Deserialize a record from JSON."""

        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            raise JourneyContractError("Journey record must be valid JSON") from None
        return cls.from_dict(payload)


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    """Serialize JSON data with stable ordering and finite numeric values."""

    if isinstance(value, _JourneyRecord):
        value = value.to_dict()
    return json.dumps(
        _plain_json(_freeze_json(value, "value")),
        allow_nan=False,
        ensure_ascii=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
        sort_keys=True,
    )


def sha256_digest(value: bytes | bytearray | memoryview | str) -> str:
    """Return a normalized SHA-256 digest for bytes or text."""

    if isinstance(value, str):
        payload = value.encode("utf-8")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
    else:
        raise TypeError("sha256_digest requires bytes or text")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def canonical_digest(value: Any) -> str:
    """Return a SHA-256 digest of canonical JSON data."""

    return sha256_digest(canonical_json(value))


def new_opaque_id(kind: str) -> str:
    """Return a random value-free identifier with a controlled kind prefix."""

    prefix = _controlled_prefix(kind)
    return f"{prefix}_{secrets.token_urlsafe(18)}"


def derived_opaque_id(kind: str, *materials: Any) -> str:
    """Return a deterministic opaque identifier derived from canonical inputs."""

    prefix = _controlled_prefix(kind)
    digest = hashlib.sha256(
        canonical_json({"kind": prefix, "materials": list(materials)}).encode("utf-8")
    ).hexdigest()
    return f"{prefix}_{digest[:32]}"


def compute_derivation_hash(
    component: str,
    component_version: str,
    input_ids: Iterable[str],
    *,
    configuration: Mapping[str, Any] | None = None,
) -> str:
    """Hash one deterministic derivation description."""

    component_name = _controlled(component, "component")
    version = _required_text(component_version, "component_version", max_length=128)
    normalized_inputs = _opaque_ids(input_ids, "input_ids", minimum=1)
    return canonical_digest(
        {
            "component": component_name,
            "component_version": version,
            "configuration": _plain_json(
                _freeze_mapping(configuration or {}, "configuration")
            ),
            "input_ids": list(normalized_inputs),
            "schema_version": JOURNEY_CONTRACT_SCHEMA_VERSION,
        }
    )


def _controlled_prefix(value: Any) -> str:
    if type(value) is not str or _ID_PREFIX_RE.fullmatch(value) is None:
        raise JourneyContractError("identifier kind must be a lowercase token")
    return value


def _controlled(value: Any, field_name: str) -> str:
    if type(value) is not str or _CONTROLLED_RE.fullmatch(value) is None:
        raise JourneyContractError(
            f"{field_name} must be a lowercase controlled identifier"
        )
    return value


def _required_text(value: Any, field_name: str, *, max_length: int = 1024) -> str:
    if type(value) is not str or not value or len(value) > max_length:
        raise JourneyContractError(f"{field_name} must be non-empty bounded text")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise JourneyContractError(f"{field_name} contains a control character")
    return value


def _optional_text(
    value: Any,
    field_name: str,
    *,
    max_length: int = 1024,
) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name, max_length=max_length)


def _opaque_id(value: Any, field_name: str) -> str:
    if type(value) is not str or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise JourneyContractError(f"{field_name} must be an opaque identifier")
    return value


def _optional_opaque_id(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _opaque_id(value, field_name)


def _digest(value: Any, field_name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise JourneyContractError(f"{field_name} must be a SHA-256 digest")
    normalized = value.lower()
    return normalized if normalized.startswith("sha256:") else f"sha256:{normalized}"


def _optional_digest(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _digest(value, field_name)


def _integer(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise JourneyContractError(f"{field_name} must be an integer >= {minimum}")
    return value


def _confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JourneyContractError("confidence must be a number between 0 and 1")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise JourneyContractError("confidence must be a number between 0 and 1")
    return result


def _timestamp(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name, max_length=64)
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        raise JourneyContractError(
            f"{field_name} must be an ISO-8601 timestamp"
        ) from None
    if parsed.tzinfo is None:
        raise JourneyContractError(f"{field_name} must include a timezone")
    return text


def _schema_version(value: Any) -> str:
    if type(value) is int:
        if value != JOURNEY_CONTRACT_SCHEMA_MAJOR:
            raise JourneySchemaVersionError("unsupported Journey schema major version")
        return JOURNEY_CONTRACT_SCHEMA_VERSION
    if type(value) is not str:
        raise JourneySchemaVersionError("Journey schema version must be semantic")
    match = _SCHEMA_VERSION_RE.fullmatch(value.strip())
    if match is None:
        raise JourneySchemaVersionError("Journey schema version must be semantic")
    major, minor, patch = (int(part or 0) for part in match.groups())
    if major != JOURNEY_CONTRACT_SCHEMA_MAJOR:
        raise JourneySchemaVersionError("unsupported Journey schema major version")
    return f"{major}.{minor}.{patch}"


def _freeze_json(value: Any, field_name: str) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise JourneyContractError(f"{field_name} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise JourneyContractError(f"{field_name} object keys must be strings")
            normalized[key] = _freeze_json(item, field_name)
        return MappingProxyType(dict(sorted(normalized.items())))
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return tuple(_freeze_json(item, field_name) for item in value)
    raise JourneyContractError(f"{field_name} must contain JSON-compatible values")


def _freeze_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise JourneyContractError(f"{field_name} must be an object")
    frozen = _freeze_json(value, field_name)
    assert isinstance(frozen, Mapping)
    return frozen


def _extension_mapping(
    value: Any,
    known_fields: frozenset[str],
) -> Mapping[str, Any]:
    extensions = _freeze_mapping(value, "extensions")
    if set(extensions).intersection(known_fields | {"schema_version"}):
        raise JourneyContractError("extensions collide with known contract fields")
    return extensions


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _opaque_ids(
    values: Any,
    field_name: str,
    *,
    minimum: int = 0,
) -> tuple[str, ...]:
    if values is None:
        values = ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise JourneyContractError(f"{field_name} must be an identifier sequence")
    normalized = tuple(sorted(_opaque_id(value, field_name) for value in values))
    if len(normalized) < minimum:
        raise JourneyContractError(
            f"{field_name} must contain at least {minimum} identifier(s)"
        )
    if len(set(normalized)) != len(normalized):
        raise JourneyContractError(f"{field_name} identifiers must be unique")
    return normalized


def _controlled_values(values: Any, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise JourneyContractError(f"{field_name} must be a sequence")
    normalized = tuple(sorted(_controlled(value, field_name) for value in values))
    if len(set(normalized)) != len(normalized):
        raise JourneyContractError(f"{field_name} entries must be unique")
    return normalized


def _payload(
    value: Any,
    *,
    known_fields: frozenset[str],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise JourneyContractError("Journey record must be an object")
    if any(type(key) is not str for key in value):
        raise JourneyContractError("Journey record keys must be strings")
    data = dict(value)
    data["schema_version"] = _schema_version(
        data.get("schema_version", JOURNEY_CONTRACT_SCHEMA_VERSION)
    )
    extensions = {
        key: item
        for key, item in data.items()
        if key not in known_fields and key != "schema_version"
    }
    return data, _freeze_mapping(extensions, "extensions")


def _merged_payload(
    known: Mapping[str, Any],
    extensions: Mapping[str, Any],
) -> dict[str, Any]:
    collision = set(known).intersection(extensions)
    if collision:
        raise JourneyContractError("extensions collide with known contract fields")
    payload = _plain_json(extensions)
    payload.update(_plain_json(known))
    return payload


def _normalize_location(location_type: str, location: Any) -> Mapping[str, Any]:
    value = dict(_freeze_mapping(location, "location"))

    def exact_keys(required: set[str], optional: set[str] = set()) -> None:
        if not required.issubset(value):
            raise JourneyContractError(
                f"{location_type} location is missing a required field"
            )
        if set(value) - required - optional:
            raise JourneyContractError(
                f"{location_type} location contains an unsupported field"
            )

    if location_type == "text_span":
        exact_keys({"start", "end"})
        start = _integer(value["start"], "location.start")
        end = _integer(value["end"], "location.end")
        if end <= start:
            raise JourneyContractError("text span must satisfy 0 <= start < end")
        value = {"start": start, "end": end}
    elif location_type == "json_pointer":
        exact_keys({"pointer"})
        pointer = value["pointer"]
        if type(pointer) is not str or (pointer and not pointer.startswith("/")):
            raise JourneyContractError("JSON pointer must be empty or start with '/'")
        if _JSON_POINTER_BAD_ESCAPE_RE.search(pointer):
            raise JourneyContractError("JSON pointer contains an invalid escape")
        value = {"pointer": pointer}
    elif location_type == "message_field":
        exact_keys({"path"})
        path = _required_text(value["path"], "location.path", max_length=256)
        if _MESSAGE_FIELD_RE.fullmatch(path) is None:
            raise JourneyContractError("message field path has an invalid format")
        value = {"path": path}
    elif location_type == "page_box":
        exact_keys(
            {"page", "box", "coordinate_space"},
            {"page_width", "page_height"},
        )
        page = _integer(value["page"], "location.page", minimum=1)
        coordinate_space = value["coordinate_space"]
        if coordinate_space not in {"normalized", "pixels", "points"}:
            raise JourneyContractError("page box coordinate_space is unsupported")
        box = value["box"]
        if (
            isinstance(box, (str, bytes))
            or not isinstance(box, Sequence)
            or len(box) != 4
        ):
            raise JourneyContractError("page box must contain four numbers")
        coordinates: list[float] = []
        for item in box:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise JourneyContractError("page box must contain four numbers")
            number = float(item)
            if not math.isfinite(number) or number < 0:
                raise JourneyContractError(
                    "page box coordinates must be finite and non-negative"
                )
            coordinates.append(number)
        x0, y0, x1, y1 = coordinates
        if x1 <= x0 or y1 <= y0:
            raise JourneyContractError("page box must have positive width and height")
        page_width = value.get("page_width")
        page_height = value.get("page_height")
        if (page_width is None) != (page_height is None):
            raise JourneyContractError(
                "page box width and height must be supplied together"
            )
        if coordinate_space == "normalized" and max(coordinates) > 1.0:
            raise JourneyContractError("normalized page box coordinates must be <= 1")
        if page_width is not None:
            if isinstance(page_width, bool) or not isinstance(page_width, (int, float)):
                raise JourneyContractError("page dimensions must be positive numbers")
            if isinstance(page_height, bool) or not isinstance(
                page_height, (int, float)
            ):
                raise JourneyContractError("page dimensions must be positive numbers")
            width = float(page_width)
            height = float(page_height)
            if (
                not math.isfinite(width)
                or not math.isfinite(height)
                or width <= 0
                or height <= 0
            ):
                raise JourneyContractError("page dimensions must be positive numbers")
            if x1 > width or y1 > height:
                raise JourneyContractError("page box exceeds declared page dimensions")
        value = {
            "box": coordinates,
            "coordinate_space": coordinate_space,
            "page": page,
        }
        if page_width is not None:
            assert page_height is not None
            value["page_width"] = float(page_width)
            value["page_height"] = float(page_height)
    elif location_type == "dicom_element":
        exact_keys({"study_uid", "series_uid", "instance_uid", "tag"})
        normalized: dict[str, str] = {}
        for key in ("study_uid", "series_uid", "instance_uid"):
            uid = _required_text(value[key], f"location.{key}", max_length=64)
            if _DICOM_UID_RE.fullmatch(uid) is None:
                raise JourneyContractError("DICOM UID has an invalid format")
            normalized[key] = uid
        tag = _required_text(value["tag"], "location.tag", max_length=9)
        if _DICOM_TAG_RE.fullmatch(tag) is None:
            raise JourneyContractError("DICOM tag has an invalid format")
        compact_tag = tag.replace(",", "").upper()
        normalized["tag"] = f"{compact_tag[:4]},{compact_tag[4:]}"
        value = normalized
    elif location_type == "table_cell":
        exact_keys({"row", "column"}, {"sheet"})
        value = {
            "column": _integer(value["column"], "location.column", minimum=1),
            "row": _integer(value["row"], "location.row", minimum=1),
        }
        if "sheet" in location:
            value["sheet"] = _required_text(
                location["sheet"], "location.sheet", max_length=128
            )
    else:
        raise JourneyContractError("evidence location_type is unsupported")
    return _freeze_mapping(value, "location")


def _relative_file_hashes(value: Any, field_name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise JourneyContractError(f"{field_name} must be an object")
    normalized: dict[str, str] = {}
    for path, digest in value.items():
        if type(path) is not str or not path or path.startswith(("/", "\\")):
            raise JourneyContractError(f"{field_name} keys must be relative paths")
        parts = re.split(r"[/\\]", path)
        if any(
            part in {"", ".", ".."} or _SAFE_PATH_COMPONENT_RE.fullmatch(part) is None
            for part in parts
        ):
            raise JourneyContractError(f"{field_name} keys must be safe relative paths")
        normalized["/".join(parts)] = _digest(digest, field_name)
    return MappingProxyType(dict(sorted(normalized.items())))


def _controlled_mapping(value: Any, field_name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise JourneyContractError(f"{field_name} must be an object")
    normalized = {
        _controlled(key, field_name): _required_text(item, field_name, max_length=256)
        for key, item in value.items()
    }
    return MappingProxyType(dict(sorted(normalized.items())))


def _digest_mapping(value: Any, field_name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise JourneyContractError(f"{field_name} must be an object")
    normalized = {
        _controlled(key, field_name): _digest(item, field_name)
        for key, item in value.items()
    }
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True, slots=True)
class ClinicalArtifact(_JourneyRecord):
    """Immutable metadata for one ingested or derived clinical artifact."""

    artifact_id: str
    artifact_type: str
    media_type: str
    content_hash: str
    byte_size: int
    source_id: str
    recorded_at: str
    subject_id: str | None = None
    encounter_id: str | None = None
    parent_artifact_ids: tuple[str, ...] = ()
    derivation_hash: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = JOURNEY_CONTRACT_SCHEMA_VERSION
    extensions: Mapping[str, Any] = field(default_factory=dict, repr=False)

    _KNOWN_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "artifact_id",
            "artifact_type",
            "media_type",
            "content_hash",
            "byte_size",
            "source_id",
            "recorded_at",
            "subject_id",
            "encounter_id",
            "parent_artifact_ids",
            "derivation_hash",
            "attributes",
            "schema_version",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "artifact_id", _opaque_id(self.artifact_id, "artifact_id")
        )
        object.__setattr__(
            self, "artifact_type", _controlled(self.artifact_type, "artifact_type")
        )
        media_type = _required_text(
            self.media_type, "media_type", max_length=255
        ).lower()
        if _MEDIA_TYPE_RE.fullmatch(media_type) is None:
            raise JourneyContractError("media_type must be a valid media type")
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(
            self, "content_hash", _digest(self.content_hash, "content_hash")
        )
        object.__setattr__(self, "byte_size", _integer(self.byte_size, "byte_size"))
        object.__setattr__(self, "source_id", _opaque_id(self.source_id, "source_id"))
        object.__setattr__(
            self, "recorded_at", _timestamp(self.recorded_at, "recorded_at")
        )
        object.__setattr__(
            self, "subject_id", _optional_opaque_id(self.subject_id, "subject_id")
        )
        object.__setattr__(
            self, "encounter_id", _optional_opaque_id(self.encounter_id, "encounter_id")
        )
        parents = _opaque_ids(self.parent_artifact_ids, "parent_artifact_ids")
        if self.artifact_id in parents:
            raise JourneyContractError("artifact cannot derive from itself")
        object.__setattr__(self, "parent_artifact_ids", parents)
        derivation_hash = _optional_digest(self.derivation_hash, "derivation_hash")
        if parents and derivation_hash is None:
            raise JourneyContractError("derived artifacts require derivation_hash")
        object.__setattr__(self, "derivation_hash", derivation_hash)
        object.__setattr__(
            self, "attributes", _freeze_mapping(self.attributes, "attributes")
        )
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(
            self, "extensions", _extension_mapping(self.extensions, self._KNOWN_FIELDS)
        )

    def to_dict(self) -> dict[str, Any]:
        return _merged_payload(
            {
                "artifact_id": self.artifact_id,
                "artifact_type": self.artifact_type,
                "attributes": self.attributes,
                "byte_size": self.byte_size,
                "content_hash": self.content_hash,
                "derivation_hash": self.derivation_hash,
                "encounter_id": self.encounter_id,
                "media_type": self.media_type,
                "parent_artifact_ids": self.parent_artifact_ids,
                "recorded_at": self.recorded_at,
                "schema_version": self.schema_version,
                "source_id": self.source_id,
                "subject_id": self.subject_id,
            },
            self.extensions,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClinicalArtifact":
        data, extensions = _payload(payload, known_fields=cls._KNOWN_FIELDS)
        try:
            return cls(
                artifact_id=data["artifact_id"],
                artifact_type=data["artifact_type"],
                media_type=data["media_type"],
                content_hash=data["content_hash"],
                byte_size=data["byte_size"],
                source_id=data["source_id"],
                recorded_at=data["recorded_at"],
                subject_id=data.get("subject_id"),
                encounter_id=data.get("encounter_id"),
                parent_artifact_ids=tuple(data.get("parent_artifact_ids") or ()),
                derivation_hash=data.get("derivation_hash"),
                attributes=data.get("attributes") or {},
                schema_version=data["schema_version"],
                extensions=extensions,
            )
        except KeyError:
            raise JourneyContractError(
                "clinical artifact is missing a required field"
            ) from None


@dataclass(frozen=True, slots=True)
class EvidenceLocator(_JourneyRecord):
    """A modality-aware pointer from a derived record to immutable evidence."""

    locator_id: str
    artifact_id: str
    location_type: str
    location: Mapping[str, Any]
    transform: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = JOURNEY_CONTRACT_SCHEMA_VERSION
    extensions: Mapping[str, Any] = field(default_factory=dict, repr=False)

    _KNOWN_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "locator_id",
            "artifact_id",
            "location_type",
            "location",
            "transform",
            "schema_version",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "locator_id", _opaque_id(self.locator_id, "locator_id")
        )
        object.__setattr__(
            self, "artifact_id", _opaque_id(self.artifact_id, "artifact_id")
        )
        location_type = _controlled(self.location_type, "location_type")
        if location_type not in EVIDENCE_LOCATION_TYPES:
            raise JourneyContractError("evidence location_type is unsupported")
        object.__setattr__(self, "location_type", location_type)
        object.__setattr__(
            self, "location", _normalize_location(location_type, self.location)
        )
        object.__setattr__(
            self, "transform", _freeze_mapping(self.transform, "transform")
        )
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(
            self, "extensions", _extension_mapping(self.extensions, self._KNOWN_FIELDS)
        )

    def to_dict(self) -> dict[str, Any]:
        return _merged_payload(
            {
                "artifact_id": self.artifact_id,
                "location": self.location,
                "location_type": self.location_type,
                "locator_id": self.locator_id,
                "schema_version": self.schema_version,
                "transform": self.transform,
            },
            self.extensions,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceLocator":
        data, extensions = _payload(payload, known_fields=cls._KNOWN_FIELDS)
        try:
            return cls(
                locator_id=data["locator_id"],
                artifact_id=data["artifact_id"],
                location_type=data["location_type"],
                location=data["location"],
                transform=data.get("transform") or {},
                schema_version=data["schema_version"],
                extensions=extensions,
            )
        except KeyError:
            raise JourneyContractError(
                "evidence locator is missing a required field"
            ) from None


@dataclass(frozen=True, slots=True)
class ClinicalFact(_JourneyRecord):
    """One evidence-bound clinical fact independent of a model family."""

    fact_id: str
    subject_id: str
    fact_type: str
    value: Any
    status: str
    evidence_ids: tuple[str, ...]
    derivation_hash: str
    encounter_id: str | None = None
    parent_fact_ids: tuple[str, ...] = ()
    effective_time: Mapping[str, Any] = field(default_factory=dict)
    unit: str | None = None
    confidence: float | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = JOURNEY_CONTRACT_SCHEMA_VERSION
    extensions: Mapping[str, Any] = field(default_factory=dict, repr=False)

    _KNOWN_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "fact_id",
            "subject_id",
            "fact_type",
            "value",
            "status",
            "evidence_ids",
            "derivation_hash",
            "encounter_id",
            "parent_fact_ids",
            "effective_time",
            "unit",
            "confidence",
            "attributes",
            "schema_version",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact_id", _opaque_id(self.fact_id, "fact_id"))
        object.__setattr__(
            self, "subject_id", _opaque_id(self.subject_id, "subject_id")
        )
        object.__setattr__(self, "fact_type", _controlled(self.fact_type, "fact_type"))
        object.__setattr__(self, "value", _freeze_json(self.value, "value"))
        object.__setattr__(self, "status", _controlled(self.status, "status"))
        object.__setattr__(
            self,
            "evidence_ids",
            _opaque_ids(self.evidence_ids, "evidence_ids", minimum=1),
        )
        object.__setattr__(
            self, "derivation_hash", _digest(self.derivation_hash, "derivation_hash")
        )
        object.__setattr__(
            self, "encounter_id", _optional_opaque_id(self.encounter_id, "encounter_id")
        )
        parents = _opaque_ids(self.parent_fact_ids, "parent_fact_ids")
        if self.fact_id in parents:
            raise JourneyContractError("fact cannot derive from itself")
        object.__setattr__(self, "parent_fact_ids", parents)
        object.__setattr__(
            self,
            "effective_time",
            _freeze_mapping(self.effective_time, "effective_time"),
        )
        object.__setattr__(
            self, "unit", _optional_text(self.unit, "unit", max_length=128)
        )
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(
            self, "attributes", _freeze_mapping(self.attributes, "attributes")
        )
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(
            self, "extensions", _extension_mapping(self.extensions, self._KNOWN_FIELDS)
        )

    def to_dict(self) -> dict[str, Any]:
        return _merged_payload(
            {
                "attributes": self.attributes,
                "confidence": self.confidence,
                "derivation_hash": self.derivation_hash,
                "effective_time": self.effective_time,
                "encounter_id": self.encounter_id,
                "evidence_ids": self.evidence_ids,
                "fact_id": self.fact_id,
                "fact_type": self.fact_type,
                "parent_fact_ids": self.parent_fact_ids,
                "schema_version": self.schema_version,
                "status": self.status,
                "subject_id": self.subject_id,
                "unit": self.unit,
                "value": self.value,
            },
            self.extensions,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClinicalFact":
        data, extensions = _payload(payload, known_fields=cls._KNOWN_FIELDS)
        try:
            return cls(
                fact_id=data["fact_id"],
                subject_id=data["subject_id"],
                fact_type=data["fact_type"],
                value=data["value"],
                status=data["status"],
                evidence_ids=tuple(data["evidence_ids"]),
                derivation_hash=data["derivation_hash"],
                encounter_id=data.get("encounter_id"),
                parent_fact_ids=tuple(data.get("parent_fact_ids") or ()),
                effective_time=data.get("effective_time") or {},
                unit=data.get("unit"),
                confidence=data.get("confidence"),
                attributes=data.get("attributes") or {},
                schema_version=data["schema_version"],
                extensions=extensions,
            )
        except KeyError:
            raise JourneyContractError(
                "clinical fact is missing a required field"
            ) from None


@dataclass(frozen=True, slots=True)
class ConflictSet(_JourneyRecord):
    """A reviewable set of facts that cannot all be treated as current truth."""

    conflict_id: str
    subject_id: str
    conflict_type: str
    fact_ids: tuple[str, ...]
    status: str
    detected_by: str
    derivation_hash: str
    evidence_ids: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = JOURNEY_CONTRACT_SCHEMA_VERSION
    extensions: Mapping[str, Any] = field(default_factory=dict, repr=False)

    _KNOWN_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "conflict_id",
            "subject_id",
            "conflict_type",
            "fact_ids",
            "status",
            "detected_by",
            "derivation_hash",
            "evidence_ids",
            "attributes",
            "schema_version",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "conflict_id", _opaque_id(self.conflict_id, "conflict_id")
        )
        object.__setattr__(
            self, "subject_id", _opaque_id(self.subject_id, "subject_id")
        )
        object.__setattr__(
            self, "conflict_type", _controlled(self.conflict_type, "conflict_type")
        )
        object.__setattr__(
            self, "fact_ids", _opaque_ids(self.fact_ids, "fact_ids", minimum=2)
        )
        if self.status not in CONFLICT_STATUSES:
            raise JourneyContractError("conflict status is unsupported")
        object.__setattr__(
            self, "detected_by", _controlled(self.detected_by, "detected_by")
        )
        object.__setattr__(
            self, "derivation_hash", _digest(self.derivation_hash, "derivation_hash")
        )
        object.__setattr__(
            self, "evidence_ids", _opaque_ids(self.evidence_ids, "evidence_ids")
        )
        object.__setattr__(
            self, "attributes", _freeze_mapping(self.attributes, "attributes")
        )
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(
            self, "extensions", _extension_mapping(self.extensions, self._KNOWN_FIELDS)
        )

    def to_dict(self) -> dict[str, Any]:
        return _merged_payload(
            {
                "attributes": self.attributes,
                "conflict_id": self.conflict_id,
                "conflict_type": self.conflict_type,
                "derivation_hash": self.derivation_hash,
                "detected_by": self.detected_by,
                "evidence_ids": self.evidence_ids,
                "fact_ids": self.fact_ids,
                "schema_version": self.schema_version,
                "status": self.status,
                "subject_id": self.subject_id,
            },
            self.extensions,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConflictSet":
        data, extensions = _payload(payload, known_fields=cls._KNOWN_FIELDS)
        try:
            return cls(
                conflict_id=data["conflict_id"],
                subject_id=data["subject_id"],
                conflict_type=data["conflict_type"],
                fact_ids=tuple(data["fact_ids"]),
                status=data["status"],
                detected_by=data["detected_by"],
                derivation_hash=data["derivation_hash"],
                evidence_ids=tuple(data.get("evidence_ids") or ()),
                attributes=data.get("attributes") or {},
                schema_version=data["schema_version"],
                extensions=extensions,
            )
        except KeyError:
            raise JourneyContractError(
                "conflict set is missing a required field"
            ) from None


@dataclass(frozen=True, slots=True)
class ResolutionEvent(_JourneyRecord):
    """An append-only policy or human decision over a conflict set."""

    resolution_id: str
    conflict_id: str
    action: str
    actor_type: str
    policy_id: str
    policy_version: str
    occurred_at: str
    rationale_code: str
    derivation_hash: str
    selected_fact_ids: tuple[str, ...] = ()
    rejected_fact_ids: tuple[str, ...] = ()
    supersedes_resolution_id: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = JOURNEY_CONTRACT_SCHEMA_VERSION
    extensions: Mapping[str, Any] = field(default_factory=dict, repr=False)

    _KNOWN_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "resolution_id",
            "conflict_id",
            "action",
            "actor_type",
            "policy_id",
            "policy_version",
            "occurred_at",
            "rationale_code",
            "derivation_hash",
            "selected_fact_ids",
            "rejected_fact_ids",
            "supersedes_resolution_id",
            "attributes",
            "schema_version",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "resolution_id", _opaque_id(self.resolution_id, "resolution_id")
        )
        object.__setattr__(
            self, "conflict_id", _opaque_id(self.conflict_id, "conflict_id")
        )
        if self.action not in RESOLUTION_ACTIONS:
            raise JourneyContractError("resolution action is unsupported")
        if self.actor_type not in RESOLUTION_ACTOR_TYPES:
            raise JourneyContractError("resolution actor_type is unsupported")
        object.__setattr__(self, "policy_id", _controlled(self.policy_id, "policy_id"))
        object.__setattr__(
            self,
            "policy_version",
            _required_text(self.policy_version, "policy_version", max_length=128),
        )
        object.__setattr__(
            self, "occurred_at", _timestamp(self.occurred_at, "occurred_at")
        )
        object.__setattr__(
            self, "rationale_code", _controlled(self.rationale_code, "rationale_code")
        )
        object.__setattr__(
            self, "derivation_hash", _digest(self.derivation_hash, "derivation_hash")
        )
        selected = _opaque_ids(self.selected_fact_ids, "selected_fact_ids")
        rejected = _opaque_ids(self.rejected_fact_ids, "rejected_fact_ids")
        if set(selected).intersection(rejected):
            raise JourneyContractError("selected and rejected facts must be disjoint")
        if self.action in {"select", "reject", "merge"} and not (selected or rejected):
            raise JourneyContractError(
                "resolution action requires selected or rejected facts"
            )
        object.__setattr__(self, "selected_fact_ids", selected)
        object.__setattr__(self, "rejected_fact_ids", rejected)
        supersedes = _optional_opaque_id(
            self.supersedes_resolution_id, "supersedes_resolution_id"
        )
        if supersedes == self.resolution_id:
            raise JourneyContractError("resolution cannot supersede itself")
        object.__setattr__(self, "supersedes_resolution_id", supersedes)
        object.__setattr__(
            self, "attributes", _freeze_mapping(self.attributes, "attributes")
        )
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(
            self, "extensions", _extension_mapping(self.extensions, self._KNOWN_FIELDS)
        )

    def to_dict(self) -> dict[str, Any]:
        return _merged_payload(
            {
                "action": self.action,
                "actor_type": self.actor_type,
                "attributes": self.attributes,
                "conflict_id": self.conflict_id,
                "derivation_hash": self.derivation_hash,
                "occurred_at": self.occurred_at,
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "rationale_code": self.rationale_code,
                "rejected_fact_ids": self.rejected_fact_ids,
                "resolution_id": self.resolution_id,
                "schema_version": self.schema_version,
                "selected_fact_ids": self.selected_fact_ids,
                "supersedes_resolution_id": self.supersedes_resolution_id,
            },
            self.extensions,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResolutionEvent":
        data, extensions = _payload(payload, known_fields=cls._KNOWN_FIELDS)
        try:
            return cls(
                resolution_id=data["resolution_id"],
                conflict_id=data["conflict_id"],
                action=data["action"],
                actor_type=data["actor_type"],
                policy_id=data["policy_id"],
                policy_version=data["policy_version"],
                occurred_at=data["occurred_at"],
                rationale_code=data["rationale_code"],
                derivation_hash=data["derivation_hash"],
                selected_fact_ids=tuple(data.get("selected_fact_ids") or ()),
                rejected_fact_ids=tuple(data.get("rejected_fact_ids") or ()),
                supersedes_resolution_id=data.get("supersedes_resolution_id"),
                attributes=data.get("attributes") or {},
                schema_version=data["schema_version"],
                extensions=extensions,
            )
        except KeyError:
            raise JourneyContractError(
                "resolution event is missing a required field"
            ) from None


@dataclass(frozen=True, slots=True)
class DatasetSnapshot(_JourneyRecord):
    """Immutable manifest for a governed dataset snapshot."""

    snapshot_id: str
    dataset_id: str
    created_at: str
    query_hash: str
    schema_hash: str
    manifest_hash: str
    record_count: int
    source_artifact_ids: tuple[str, ...] = ()
    source_fact_ids: tuple[str, ...] = ()
    parent_snapshot_ids: tuple[str, ...] = ()
    file_hashes: Mapping[str, str] = field(default_factory=dict)
    split_hashes: Mapping[str, str] = field(default_factory=dict)
    component_versions: Mapping[str, str] = field(default_factory=dict)
    license_tags: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = JOURNEY_CONTRACT_SCHEMA_VERSION
    extensions: Mapping[str, Any] = field(default_factory=dict, repr=False)

    _KNOWN_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "snapshot_id",
            "dataset_id",
            "created_at",
            "query_hash",
            "schema_hash",
            "manifest_hash",
            "record_count",
            "source_artifact_ids",
            "source_fact_ids",
            "parent_snapshot_ids",
            "file_hashes",
            "split_hashes",
            "component_versions",
            "license_tags",
            "attributes",
            "schema_version",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "snapshot_id", _opaque_id(self.snapshot_id, "snapshot_id")
        )
        object.__setattr__(
            self, "dataset_id", _opaque_id(self.dataset_id, "dataset_id")
        )
        object.__setattr__(
            self, "created_at", _timestamp(self.created_at, "created_at")
        )
        object.__setattr__(self, "query_hash", _digest(self.query_hash, "query_hash"))
        object.__setattr__(
            self, "schema_hash", _digest(self.schema_hash, "schema_hash")
        )
        object.__setattr__(
            self, "manifest_hash", _digest(self.manifest_hash, "manifest_hash")
        )
        object.__setattr__(
            self, "record_count", _integer(self.record_count, "record_count")
        )
        artifacts = _opaque_ids(self.source_artifact_ids, "source_artifact_ids")
        facts = _opaque_ids(self.source_fact_ids, "source_fact_ids")
        parents = _opaque_ids(self.parent_snapshot_ids, "parent_snapshot_ids")
        if self.snapshot_id in parents:
            raise JourneyContractError("snapshot cannot derive from itself")
        if not (artifacts or facts or parents):
            raise JourneyContractError(
                "snapshot requires at least one source reference"
            )
        object.__setattr__(self, "source_artifact_ids", artifacts)
        object.__setattr__(self, "source_fact_ids", facts)
        object.__setattr__(self, "parent_snapshot_ids", parents)
        object.__setattr__(
            self, "file_hashes", _relative_file_hashes(self.file_hashes, "file_hashes")
        )
        object.__setattr__(
            self, "split_hashes", _digest_mapping(self.split_hashes, "split_hashes")
        )
        object.__setattr__(
            self,
            "component_versions",
            _controlled_mapping(self.component_versions, "component_versions"),
        )
        object.__setattr__(
            self, "license_tags", _controlled_values(self.license_tags, "license_tags")
        )
        object.__setattr__(
            self, "attributes", _freeze_mapping(self.attributes, "attributes")
        )
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        object.__setattr__(
            self, "extensions", _extension_mapping(self.extensions, self._KNOWN_FIELDS)
        )

    def to_dict(self) -> dict[str, Any]:
        return _merged_payload(
            {
                "attributes": self.attributes,
                "component_versions": self.component_versions,
                "created_at": self.created_at,
                "dataset_id": self.dataset_id,
                "file_hashes": self.file_hashes,
                "license_tags": self.license_tags,
                "manifest_hash": self.manifest_hash,
                "parent_snapshot_ids": self.parent_snapshot_ids,
                "query_hash": self.query_hash,
                "record_count": self.record_count,
                "schema_hash": self.schema_hash,
                "schema_version": self.schema_version,
                "snapshot_id": self.snapshot_id,
                "source_artifact_ids": self.source_artifact_ids,
                "source_fact_ids": self.source_fact_ids,
                "split_hashes": self.split_hashes,
            },
            self.extensions,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetSnapshot":
        data, extensions = _payload(payload, known_fields=cls._KNOWN_FIELDS)
        try:
            return cls(
                snapshot_id=data["snapshot_id"],
                dataset_id=data["dataset_id"],
                created_at=data["created_at"],
                query_hash=data["query_hash"],
                schema_hash=data["schema_hash"],
                manifest_hash=data["manifest_hash"],
                record_count=data["record_count"],
                source_artifact_ids=tuple(data.get("source_artifact_ids") or ()),
                source_fact_ids=tuple(data.get("source_fact_ids") or ()),
                parent_snapshot_ids=tuple(data.get("parent_snapshot_ids") or ()),
                file_hashes=data.get("file_hashes") or {},
                split_hashes=data.get("split_hashes") or {},
                component_versions=data.get("component_versions") or {},
                license_tags=tuple(data.get("license_tags") or ()),
                attributes=data.get("attributes") or {},
                schema_version=data["schema_version"],
                extensions=extensions,
            )
        except KeyError:
            raise JourneyContractError(
                "dataset snapshot is missing a required field"
            ) from None


@dataclass(frozen=True, slots=True)
class ContractGraphDiagnostics:
    """Value-free counts for one validated contract graph."""

    artifact_count: int
    evidence_count: int
    fact_count: int
    conflict_count: int
    resolution_count: int
    snapshot_count: int
    edge_count: int

    @property
    def node_count(self) -> int:
        """Return the total number of records in the graph."""

        return (
            self.artifact_count
            + self.evidence_count
            + self.fact_count
            + self.conflict_count
            + self.resolution_count
            + self.snapshot_count
        )

    def to_dict(self) -> dict[str, int | bool]:
        """Return value-free validation diagnostics."""

        return {
            "artifact_count": self.artifact_count,
            "conflict_count": self.conflict_count,
            "edge_count": self.edge_count,
            "evidence_count": self.evidence_count,
            "fact_count": self.fact_count,
            "node_count": self.node_count,
            "resolution_count": self.resolution_count,
            "snapshot_count": self.snapshot_count,
            "valid": True,
        }


def validate_contract_graph(
    *,
    artifacts: Iterable[ClinicalArtifact] = (),
    evidence: Iterable[EvidenceLocator] = (),
    facts: Iterable[ClinicalFact] = (),
    conflicts: Iterable[ConflictSet] = (),
    resolutions: Iterable[ResolutionEvent] = (),
    snapshots: Iterable[DatasetSnapshot] = (),
) -> ContractGraphDiagnostics:
    """Validate references and cycles across Journey contract records.

    Errors deliberately report only record categories and counts, never IDs or
    clinical values.
    """

    artifact_index = _record_index(artifacts, "artifact_id", "artifact")
    evidence_index = _record_index(evidence, "locator_id", "evidence")
    fact_index = _record_index(facts, "fact_id", "fact")
    conflict_index = _record_index(conflicts, "conflict_id", "conflict")
    resolution_index = _record_index(resolutions, "resolution_id", "resolution")
    snapshot_index = _record_index(snapshots, "snapshot_id", "snapshot")

    all_ids = [
        *artifact_index,
        *evidence_index,
        *fact_index,
        *conflict_index,
        *resolution_index,
        *snapshot_index,
    ]
    if len(set(all_ids)) != len(all_ids):
        raise JourneyContractGraphError("record identifiers must be globally unique")

    edge_count = 0
    artifact_graph: dict[str, tuple[str, ...]] = {}
    for artifact_id, artifact in artifact_index.items():
        _require_references(
            artifact.parent_artifact_ids, artifact_index, "artifact parent"
        )
        artifact_graph[artifact_id] = artifact.parent_artifact_ids
        edge_count += len(artifact.parent_artifact_ids)

    for locator in evidence_index.values():
        _require_references((locator.artifact_id,), artifact_index, "evidence artifact")
        edge_count += 1

    fact_graph: dict[str, tuple[str, ...]] = {}
    for fact_id, fact in fact_index.items():
        _require_references(fact.evidence_ids, evidence_index, "fact evidence")
        _require_references(fact.parent_fact_ids, fact_index, "fact parent")
        fact_graph[fact_id] = fact.parent_fact_ids
        edge_count += len(fact.evidence_ids) + len(fact.parent_fact_ids)

    for conflict in conflict_index.values():
        _require_references(conflict.fact_ids, fact_index, "conflict fact")
        _require_references(conflict.evidence_ids, evidence_index, "conflict evidence")
        if any(
            fact_index[fact_id].subject_id != conflict.subject_id
            for fact_id in conflict.fact_ids
        ):
            raise JourneyContractGraphError(
                "conflict facts must belong to the conflict subject"
            )
        edge_count += len(conflict.fact_ids) + len(conflict.evidence_ids)

    resolution_graph: dict[str, tuple[str, ...]] = {}
    for resolution_id, resolution in resolution_index.items():
        _require_references(
            (resolution.conflict_id,), conflict_index, "resolution conflict"
        )
        conflict = conflict_index[resolution.conflict_id]
        resolved_facts = set(
            resolution.selected_fact_ids + resolution.rejected_fact_ids
        )
        if not resolved_facts.issubset(conflict.fact_ids):
            raise JourneyContractGraphError(
                "resolution facts must belong to the referenced conflict"
            )
        parent = (
            (resolution.supersedes_resolution_id,)
            if resolution.supersedes_resolution_id is not None
            else ()
        )
        _require_references(parent, resolution_index, "superseded resolution")
        if parent and resolution_index[parent[0]].conflict_id != resolution.conflict_id:
            raise JourneyContractGraphError(
                "superseded resolution must reference the same conflict"
            )
        resolution_graph[resolution_id] = parent
        edge_count += 1 + len(resolved_facts) + len(parent)

    snapshot_graph: dict[str, tuple[str, ...]] = {}
    for snapshot_id, snapshot in snapshot_index.items():
        _require_references(
            snapshot.source_artifact_ids, artifact_index, "snapshot artifact"
        )
        _require_references(snapshot.source_fact_ids, fact_index, "snapshot fact")
        _require_references(
            snapshot.parent_snapshot_ids, snapshot_index, "snapshot parent"
        )
        snapshot_graph[snapshot_id] = snapshot.parent_snapshot_ids
        edge_count += (
            len(snapshot.source_artifact_ids)
            + len(snapshot.source_fact_ids)
            + len(snapshot.parent_snapshot_ids)
        )

    _require_acyclic(artifact_graph, "artifact")
    _require_acyclic(fact_graph, "fact")
    _require_acyclic(resolution_graph, "resolution")
    _require_acyclic(snapshot_graph, "snapshot")

    return ContractGraphDiagnostics(
        artifact_count=len(artifact_index),
        evidence_count=len(evidence_index),
        fact_count=len(fact_index),
        conflict_count=len(conflict_index),
        resolution_count=len(resolution_index),
        snapshot_count=len(snapshot_index),
        edge_count=edge_count,
    )


def _record_index(
    records: Iterable[Any],
    field_name: str,
    category: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for record in records:
        try:
            record_id = getattr(record, field_name)
        except AttributeError:
            raise JourneyContractGraphError(
                f"{category} graph contains an invalid record"
            ) from None
        if record_id in result:
            raise JourneyContractGraphError(
                f"{category} graph contains a duplicate identifier"
            )
        result[record_id] = record
    return result


def _require_references(
    references: Iterable[str],
    index: Mapping[str, Any],
    category: str,
) -> None:
    if any(reference not in index for reference in references):
        raise JourneyContractGraphError(f"{category} reference is missing")


def _require_acyclic(graph: Mapping[str, Sequence[str]], category: str) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise JourneyContractGraphError(f"{category} derivation graph has a cycle")
        if node in visited:
            return
        visiting.add(node)
        for parent in graph.get(node, ()):
            visit(parent)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)


def load_journey_schema(name: str) -> dict[str, Any]:
    """Load one bundled Journey JSON Schema by logical name."""

    normalized = name.removeprefix("journey_").removesuffix(".schema.json")
    if normalized not in JOURNEY_SCHEMA_NAMES:
        raise KeyError("unknown Journey schema")
    resource = resources.files(JOURNEY_SCHEMA_PACKAGE).joinpath(
        f"journey_{normalized}.schema.json"
    )
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_all_journey_schemas() -> dict[str, dict[str, Any]]:
    """Load all bundled Journey JSON Schemas."""

    return {name: load_journey_schema(name) for name in JOURNEY_SCHEMA_NAMES}
