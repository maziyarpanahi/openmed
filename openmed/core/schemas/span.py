"""Canonical span record and schema-version helpers."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.labels import CANONICAL_LABELS, policy_label_for

CURRENT_SCHEMA_VERSION = 1
SCHEMA_PACKAGE = "openmed.core.schemas.json"
PACKAGED_SCHEMA_ROOT = "openmed"
PACKAGED_SCHEMA_SUFFIX = ".schema.json"
SCHEMA_NAMES = (
    "span",
    "action",
    "audit",
    "attestation",
    "entity",
    "relation",
    "code",
    "provenance",
    "service_job_record",
    "service_webhook_payload",
    "service_error_envelope",
)
ACTION_KEEP = "keep"
ACTION_FORMAT_PRESERVE = "format_preserve"
ACTION_VALUES = (
    ACTION_KEEP,
    "redact",
    "replace",
    "mask",
    "hash",
    ACTION_FORMAT_PRESERVE,
)
_TEXT_HASH_RE = re.compile(r"^hmac-sha256:[0-9a-f]{64}$")
_SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_LABEL_VERSION_RE = re.compile(r"^(?P<label>.*?)v(?P<major>0|[1-9]\d*)$")
_ID_SEMVER_RE = re.compile(r"/((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))$")


@dataclass(frozen=True)
class OpenMedSpan:
    """Immutable span contract shared across OpenMed privacy layers."""

    doc_id: str
    start: int
    end: int
    text_hash: str
    entity_type: str
    canonical_label: str
    policy_label: str | None = None
    regulatory_tags: Sequence[str] = field(default_factory=tuple)
    score: float | None = None
    detector: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    action: str = ACTION_KEEP
    replacement: str | None = None
    reversible_id: str | None = None
    section: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = CURRENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {CURRENT_SCHEMA_VERSION}, "
                f"got {self.schema_version!r}"
            )
        if not self.doc_id:
            raise ValueError("doc_id must be non-empty")
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError("start and end must be integers")
        if self.start < 0 or self.end < self.start:
            raise ValueError("start/end must be non-negative offsets with end >= start")
        if not _TEXT_HASH_RE.fullmatch(self.text_hash):
            raise ValueError("text_hash must be hmac-sha256:<64 lowercase hex chars>")
        if self.canonical_label not in CANONICAL_LABELS:
            raise ValueError(
                f"canonical_label must be in CANONICAL_LABELS: {self.canonical_label!r}"
            )
        if not self.entity_type:
            raise ValueError("entity_type must be non-empty")
        if self.action not in ACTION_VALUES:
            raise ValueError(f"action must be one of {ACTION_VALUES!r}")
        if self.score is not None and not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")

        object.__setattr__(
            self, "regulatory_tags", tuple(map(str, self.regulatory_tags))
        )
        object.__setattr__(self, "evidence", _plain_mapping(self.evidence))
        object.__setattr__(self, "metadata", _plain_mapping(self.metadata))
        if self.policy_label is None:
            object.__setattr__(
                self, "policy_label", policy_label_for(self.canonical_label)
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this span to a JSON-compatible dictionary."""

        return {
            "schema_version": self.schema_version,
            "doc_id": self.doc_id,
            "start": self.start,
            "end": self.end,
            "text_hash": self.text_hash,
            "entity_type": self.entity_type,
            "canonical_label": self.canonical_label,
            "policy_label": self.policy_label,
            "regulatory_tags": list(self.regulatory_tags),
            "score": self.score,
            "detector": self.detector,
            "evidence": copy.deepcopy(dict(self.evidence)),
            "action": self.action,
            "replacement": self.replacement,
            "reversible_id": self.reversible_id,
            "section": self.section,
            "metadata": copy.deepcopy(dict(self.metadata)),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize this span to deterministic JSON."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OpenMedSpan":
        """Deserialize a span from a JSON-compatible dictionary."""

        data = dict(payload)
        return cls(
            doc_id=str(data["doc_id"]),
            start=int(data["start"]),
            end=int(data["end"]),
            text_hash=str(data["text_hash"]),
            entity_type=str(data["entity_type"]),
            canonical_label=str(data["canonical_label"]),
            policy_label=data.get("policy_label"),
            regulatory_tags=tuple(data.get("regulatory_tags") or ()),
            score=data.get("score"),
            detector=data.get("detector"),
            evidence=data.get("evidence") or {},
            action=str(data.get("action") or ACTION_KEEP),
            replacement=data.get("replacement"),
            reversible_id=data.get("reversible_id"),
            section=data.get("section"),
            metadata=data.get("metadata") or {},
            schema_version=int(data.get("schema_version", CURRENT_SCHEMA_VERSION)),
        )

    @classmethod
    def from_json(cls, text: str) -> "OpenMedSpan":
        """Deserialize a span from JSON text."""

        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class SchemaDriftResult:
    """Result of comparing one schema against a saved fingerprint snapshot.

    Versions keep the form the schema declares: an integer for the core
    records, a semver string such as ``"1.2.0"`` or a versioned label for
    others, and ``None`` when a schema declares no version at all.
    """

    schema_name: str
    schema_version: int | str | None
    snapshot_version: int | str | None
    fingerprint: str
    snapshot_fingerprint: str | None
    version_bumped: bool
    breaking_change: bool
    removed_required: tuple[str, ...] = ()
    removed_properties: tuple[str, ...] = ()


def _resolve_hmac_secret(secret: str | bytes | None) -> bytes:
    """Resolve an explicit key or a private key for one operation's lifetime."""
    if secret is None:
        from secrets import token_bytes

        return token_bytes(32)
    if isinstance(secret, str):
        key = secret.encode("utf-8")
    elif isinstance(secret, (bytes, bytearray)):
        key = bytes(secret)
    else:
        raise TypeError("hmac_secret must be a string or bytes")
    if not key:
        raise ValueError("hmac_secret must be non-empty")
    return key


def hmac_text_hash(surface: str | bytes, secret: str | bytes) -> str:
    """Return the safe loggable HMAC-SHA256 hash for source surface text."""

    payload = surface.encode("utf-8") if isinstance(surface, str) else surface
    if secret is None:
        raise TypeError("hmac_text_hash requires an explicit key")
    key = _resolve_hmac_secret(secret)
    digest = hmac.new(key, payload, hashlib.sha256).hexdigest()
    return f"hmac-sha256:{digest}"


def current_schema_version() -> int:
    """Return the current schema version."""

    return CURRENT_SCHEMA_VERSION


def load_schema(name: str) -> dict[str, Any]:
    """Load one bundled JSON Schema by logical name."""

    schema_name = _normalise_schema_name(name)
    resource = resources.files(SCHEMA_PACKAGE).joinpath(f"{schema_name}.schema.json")
    with resource.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if "schema_version" not in schema:
        raise ValueError(f"{schema_name} schema is missing schema_version")
    return schema


def load_all_schemas() -> dict[str, dict[str, Any]]:
    """Load every bundled JSON Schema."""

    return {name: load_schema(name) for name in SCHEMA_NAMES}


def load_schema_bundle() -> dict[str, Any]:
    """Return current schema_version plus parsed bundled schemas."""

    return {
        "schema_version": current_schema_version(),
        "schemas": load_all_schemas(),
    }


def schema_fingerprint(schema: Mapping[str, Any]) -> str:
    """Return a deterministic SHA256 fingerprint for a JSON Schema."""

    payload = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def build_schema_snapshot(
    schemas: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build the drift snapshot format from parsed schemas."""

    return {
        name: _schema_state(schema)
        for name, schema in sorted(schemas.items(), key=lambda item: item[0])
    }


def load_schema_snapshot(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the committed schema fingerprint snapshot."""

    if path is None:
        resource = resources.files(SCHEMA_PACKAGE).joinpath("schema-fingerprints.json")
        with resource.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_schema_drift(
    schema_name: str,
    current_schema: Mapping[str, Any] | None = None,
    *,
    snapshot: Mapping[str, Mapping[str, Any]] | None = None,
) -> SchemaDriftResult:
    """Compare one schema against a saved snapshot.

    A field removal without a schema_version bump is reported as a breaking
    change. Gate wiring that turns this result into CI failure is owned by a
    later task.
    """

    name = _normalise_schema_name(schema_name)
    schema = dict(current_schema or load_schema(name))
    saved = snapshot or load_schema_snapshot()
    return _drift_result(name, schema, saved.get(name))


def compare_all_schema_drift(
    *,
    schemas: Mapping[str, Mapping[str, Any]] | None = None,
    snapshot: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, SchemaDriftResult]:
    """Compare all current schemas against the committed snapshot."""

    current = schemas or load_all_schemas()
    saved = snapshot or load_schema_snapshot()
    return {
        name: compare_schema_drift(name, schema, snapshot=saved)
        for name, schema in current.items()
    }


def schema_version_of(schema: Mapping[str, Any]) -> int | str | None:
    """Return the version a JSON Schema declares for its records.

    Bundled schemas declare their version in one of four ways, checked in this
    order: a top-level ``schema_version`` keyword, a ``const`` on the
    ``schema_version`` property, an ``enum`` of accepted ``schema_version``
    values (the highest one is the current version), or a trailing semver
    segment in ``$id`` such as ``.../decision-request/1.2.0``.

    Args:
        schema: A parsed JSON Schema document.

    Returns:
        The declared version as an integer, a semver string, or a versioned
        label such as ``"openmed.reviewer_disagreement.v1"``; ``None`` when the
        schema declares no version.
    """

    if "schema_version" in schema:
        return _version_value(schema["schema_version"])
    properties = schema.get("properties")
    declared = (
        properties.get("schema_version") if isinstance(properties, Mapping) else None
    )
    if isinstance(declared, Mapping):
        if "const" in declared:
            return _version_value(declared["const"])
        values = declared.get("enum")
        if isinstance(values, Sequence) and not isinstance(values, str) and values:
            return _highest_version(values)
    schema_id = schema.get("$id")
    if isinstance(schema_id, str):
        match = _ID_SEMVER_RE.search(schema_id)
        if match:
            return match.group(1)
    return None


def schema_version_bumped(
    current: int | str | None,
    saved: int | str | None,
) -> bool:
    """Return whether ``current`` is a newer schema version than ``saved``.

    Integers compare numerically, semver strings compare by major, minor and
    patch (so ``"1.10.0"`` is newer than ``"1.9.0"``), and versioned labels
    ending in ``vN`` compare by ``N`` when their prefix matches. Versions that
    cannot be ordered count as bumped only when they differ. A schema that
    declares no version, before or after, is never bumped.

    Args:
        current: Version declared by the current schema.
        saved: Version recorded in the snapshot.

    Returns:
        ``True`` when the current version supersedes the saved one.
    """

    if current is None or saved is None:
        return False
    current_key = _version_key(current)
    saved_key = _version_key(saved)
    if (
        current_key is not None
        and saved_key is not None
        and current_key[0] == saved_key[0]
    ):
        return current_key[1] > saved_key[1]
    return current != saved


def packaged_schema_paths() -> tuple[str, ...]:
    """Return the path of every JSON Schema file bundled in the package.

    The package tree is walked through :mod:`importlib.resources`, so the list
    is the same for a source checkout and an installed wheel. Paths are POSIX
    and relative to the distribution root (for example
    ``openmed/core/config.schema.json``); they never include a local
    installation prefix.

    Returns:
        Sorted package paths of every ``*.schema.json`` file.
    """

    found: list[str] = []
    pending = [(resources.files(PACKAGED_SCHEMA_ROOT), PACKAGED_SCHEMA_ROOT)]
    while pending:
        node, prefix = pending.pop()
        for child in node.iterdir():
            name = child.name
            if child.is_dir():
                if name != "__pycache__" and not name.startswith("."):
                    pending.append((child, f"{prefix}/{name}"))
            elif name.endswith(PACKAGED_SCHEMA_SUFFIX):
                found.append(f"{prefix}/{name}")
    return tuple(sorted(found))


def load_packaged_schema(path: str) -> dict[str, Any]:
    """Load one bundled JSON Schema by its package path.

    Args:
        path: An entry of :func:`packaged_schema_paths`.

    Returns:
        The parsed schema document.

    Raises:
        KeyError: If ``path`` is not a bundled schema file.
    """

    if path not in packaged_schema_paths():
        raise KeyError(f"unknown packaged schema: {path!r}")
    parts = path.split("/")[1:]
    resource = resources.files(PACKAGED_SCHEMA_ROOT).joinpath(*parts)
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_packaged_schemas() -> dict[str, dict[str, Any]]:
    """Load every bundled JSON Schema keyed by its package path."""

    return {path: load_packaged_schema(path) for path in packaged_schema_paths()}


def build_packaged_schema_snapshot(
    schemas: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the drift snapshot for every bundled JSON Schema.

    Each entry records the declared version, the fingerprint, and the record
    field names (``required`` and ``properties``). The entries for the core
    records are identical to those in ``schema-fingerprints.json``.

    Args:
        schemas: Parsed schemas keyed by package path. Defaults to every
            bundled schema.

    Returns:
        One snapshot entry per schema, keyed by package path.
    """

    current = load_packaged_schemas() if schemas is None else schemas
    return build_schema_snapshot(current)


def compare_packaged_schema_drift(
    path: str,
    current_schema: Mapping[str, Any] | None = None,
    *,
    snapshot: Mapping[str, Mapping[str, Any]],
) -> SchemaDriftResult:
    """Compare one bundled JSON Schema against a saved snapshot.

    Removing a required or declared property without a newer
    ``schema_version`` (see :func:`schema_version_bumped`) is reported as a
    breaking change. Adding properties never is.

    Args:
        path: Package path of the schema, as in :func:`packaged_schema_paths`.
        current_schema: Schema to compare. Defaults to the bundled file.
        snapshot: Snapshot built by :func:`build_packaged_schema_snapshot`.

    Returns:
        The drift result for ``path``.
    """

    schema = (
        dict(current_schema)
        if current_schema is not None
        else load_packaged_schema(path)
    )
    return _drift_result(path, schema, snapshot.get(path))


def compare_all_packaged_schema_drift(
    *,
    snapshot: Mapping[str, Mapping[str, Any]],
    schemas: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, SchemaDriftResult]:
    """Compare every bundled JSON Schema against a saved snapshot.

    Args:
        snapshot: Snapshot built by :func:`build_packaged_schema_snapshot`.
        schemas: Parsed schemas keyed by package path. Defaults to every
            bundled schema.

    Returns:
        One drift result per current schema, keyed by package path. Snapshot
        entries without a current schema are not included; callers that need
        to report removed schemas compare the key sets.
    """

    current = load_packaged_schemas() if schemas is None else schemas
    return {
        path: compare_packaged_schema_drift(path, schema, snapshot=snapshot)
        for path, schema in sorted(current.items(), key=lambda item: item[0])
    }


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), sort_keys=True))


def _normalise_schema_name(name: str) -> str:
    schema_name = name.removesuffix(".schema.json").removesuffix(".schema")
    if schema_name not in SCHEMA_NAMES:
        raise KeyError(f"unknown schema: {name!r}")
    return schema_name


def _schema_state(schema: Mapping[str, Any]) -> dict[str, Any]:
    required, properties = _record_fields(schema)
    return {
        "schema_version": schema_version_of(schema),
        "fingerprint": schema_fingerprint(schema),
        "required": required,
        "properties": properties,
    }


def _drift_result(
    name: str,
    schema: Mapping[str, Any],
    saved_state: Mapping[str, Any] | None,
) -> SchemaDriftResult:
    state = _schema_state(schema)
    snapshot_state = dict(saved_state or {})
    snapshot_version = snapshot_state.get("schema_version")
    version_bumped = schema_version_bumped(state["schema_version"], snapshot_version)
    removed_required = tuple(
        sorted(set(snapshot_state.get("required", ())) - set(state["required"]))
    )
    removed_properties = tuple(
        sorted(set(snapshot_state.get("properties", ())) - set(state["properties"]))
    )
    breaking_change = bool(
        not version_bumped
        and (removed_required or removed_properties)
        and snapshot_state
    )
    return SchemaDriftResult(
        schema_name=name,
        schema_version=state["schema_version"],
        snapshot_version=snapshot_version,
        fingerprint=str(state["fingerprint"]),
        snapshot_fingerprint=snapshot_state.get("fingerprint"),
        version_bumped=version_bumped,
        breaking_change=breaking_change,
        removed_required=removed_required,
        removed_properties=removed_properties,
    )


def _record_fields(schema: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Return the sorted required and declared field names of a record schema.

    Object schemas contribute their own ``properties``. An array-rooted schema
    contributes its item schema, and a schema whose root is a ``oneOf`` or
    ``anyOf`` of record variants contributes each variant's fields qualified
    by the variant name (``definition.measure_id``). Local ``$ref`` pointers
    are followed.
    """

    root = _resolve_local_ref(schema, schema)
    items = root.get("items")
    if root.get("type") == "array" and isinstance(items, Mapping):
        root = _resolve_local_ref(schema, items)
    required = {str(item) for item in root.get("required") or ()}
    properties = {str(key) for key in root.get("properties") or {}}
    if not properties:
        for keyword in ("oneOf", "anyOf"):
            variants = root.get(keyword)
            if not isinstance(variants, Sequence) or isinstance(variants, str):
                continue
            for index, variant in enumerate(variants):
                if not isinstance(variant, Mapping):
                    continue
                ref = variant.get("$ref")
                label = (
                    ref.rsplit("/", 1)[-1]
                    if isinstance(ref, str) and ref.startswith("#/")
                    else f"{keyword}[{index}]"
                )
                target = _resolve_local_ref(schema, variant)
                required |= {f"{label}.{item}" for item in target.get("required") or ()}
                properties |= {
                    f"{label}.{key}" for key in target.get("properties") or {}
                }
    return sorted(required), sorted(properties)


def _resolve_local_ref(
    schema: Mapping[str, Any], node: Mapping[str, Any]
) -> Mapping[str, Any]:
    seen: set[str] = set()
    while isinstance(node.get("$ref"), str):
        ref = node["$ref"]
        if not ref.startswith("#/") or ref in seen:
            break
        seen.add(ref)
        target: Any = schema
        for part in ref[2:].split("/"):
            if not isinstance(target, Mapping) or part not in target:
                return node
            target = target[part]
        if not isinstance(target, Mapping):
            break
        node = target
    return node


def _version_value(value: Any) -> int | str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, str)):
        return value
    return None


def _highest_version(values: Sequence[Any]) -> int | str | None:
    candidates = [_version_value(value) for value in values]
    versions = [value for value in candidates if value is not None]
    if not versions:
        return None
    keys = [_version_key(value) for value in versions]
    if all(key is not None for key in keys) and len({key[0] for key in keys}) == 1:
        return max(zip(keys, versions, strict=True), key=lambda pair: pair[0][1])[1]
    return versions[-1]


def _version_key(version: int | str) -> tuple[str, tuple[int, ...]] | None:
    if isinstance(version, bool):
        return None
    if isinstance(version, int):
        return ("", (version,))
    text = version.strip()
    if text.isascii() and text.isdigit():
        return ("", (int(text),))
    semver = _SEMVER_RE.match(text)
    if semver:
        return ("", tuple(int(part) for part in semver.groups()))
    label = _LABEL_VERSION_RE.match(text)
    if label:
        return (label.group("label"), (int(label.group("major")),))
    return None


__all__ = [
    "ACTION_KEEP",
    "ACTION_FORMAT_PRESERVE",
    "ACTION_VALUES",
    "CURRENT_SCHEMA_VERSION",
    "PACKAGED_SCHEMA_ROOT",
    "PACKAGED_SCHEMA_SUFFIX",
    "SCHEMA_NAMES",
    "OpenMedSpan",
    "SchemaDriftResult",
    "build_packaged_schema_snapshot",
    "build_schema_snapshot",
    "compare_all_packaged_schema_drift",
    "compare_all_schema_drift",
    "compare_packaged_schema_drift",
    "compare_schema_drift",
    "current_schema_version",
    "hmac_text_hash",
    "load_all_schemas",
    "load_packaged_schema",
    "load_packaged_schemas",
    "load_schema",
    "load_schema_bundle",
    "load_schema_snapshot",
    "packaged_schema_paths",
    "schema_fingerprint",
    "schema_version_bumped",
    "schema_version_of",
]
