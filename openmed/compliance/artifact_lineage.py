"""Deterministic, PHI-safe lineage manifests for privacy artifacts.

Privacy transformations often produce several reviewable artifacts from one
input: for example, a redaction result, a schema-constrained export, and a
policy evidence report.  :class:`ArtifactLineageManifest` records how those
artifacts relate without retaining the source payloads.  Nodes contain only
typed SHA-256 digests, controlled metadata, and references to parent digests.

The manifest is deliberately local-only.  It performs no discovery, network
access, or external validation.  Each node hash commits to its type, parents,
transformation, policy fingerprint, and schema version.  Verification then
reports deterministic aggregate counts for missing parents, cycles, duplicate
hashes, and stale hashes without returning the offending values.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from openmed.core.audit import stable_hash

ARTIFACT_LINEAGE_SCHEMA_VERSION: Final = "openmed.compliance.artifact_lineage.v1"
LINEAGE_NODE_SCHEMA_VERSION: Final = "openmed.compliance.artifact_lineage.node.v1"
EMPTY_POLICY_FINGERPRINT: Final = stable_hash({"policy": None})

_DIGEST_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX_DIGEST_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE: Final = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_MAX_SCHEMA_VERSION: Final = 1_000_000
_MISSING: Final = object()

__all__ = [
    "ARTIFACT_LINEAGE_SCHEMA_VERSION",
    "EMPTY_POLICY_FINGERPRINT",
    "LINEAGE_NODE_SCHEMA_VERSION",
    "ArtifactLineageDiagnostics",
    "ArtifactLineageError",
    "ArtifactLineageManifest",
    "ArtifactLineageNode",
    "ArtifactLineageParent",
    "ArtifactLineageValidationError",
    "ArtifactManifest",
    "ArtifactNode",
    "LineageDiagnostics",
    "LineageNode",
    "LineageParent",
    "build_artifact_lineage_manifest",
    "build_manifest",
    "compute_artifact_hash",
    "compute_policy_fingerprint",
    "load_artifact_lineage_manifest",
    "verify_artifact_lineage",
    "verify_manifest",
]


class ArtifactLineageError(ValueError):
    """Base error for malformed or unsafe lineage metadata."""


class ArtifactLineageValidationError(ArtifactLineageError):
    """Raised when a lineage value cannot be represented safely."""


SchemaVersion = str | int


def _required_identifier(value: Any, field_name: str) -> str:
    """Validate a controlled metadata identifier without echoing its value."""

    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ArtifactLineageValidationError(
            f"{field_name} must be a lowercase metadata identifier"
        )
    return value


def _required_digest(value: Any, field_name: str) -> str:
    """Validate and canonicalize a SHA-256 digest without echoing its value."""

    if not isinstance(value, str):
        raise ArtifactLineageValidationError(f"{field_name} must be a SHA-256 digest")
    digest = value.lower()
    if _HEX_DIGEST_RE.fullmatch(digest) is not None:
        digest = f"sha256:{digest}"
    if _DIGEST_RE.fullmatch(digest) is None:
        raise ArtifactLineageValidationError(f"{field_name} must be a SHA-256 digest")
    return digest


def _required_schema_version(value: Any, field_name: str) -> SchemaVersion:
    """Validate an integer or controlled string schema version."""

    if type(value) is int:
        if 0 <= value <= _MAX_SCHEMA_VERSION:
            return value
    elif isinstance(value, str) and _IDENTIFIER_RE.fullmatch(value) is not None:
        return value
    raise ArtifactLineageValidationError(
        f"{field_name} must be a bounded integer or metadata identifier"
    )


def _first_value(
    payload: Mapping[str, Any],
    names: tuple[str, ...],
    *,
    default: Any = _MISSING,
) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    if default is not _MISSING:
        return default
    raise ArtifactLineageValidationError("lineage metadata is missing a required field")


@dataclass(frozen=True, order=True)
class ArtifactLineageParent:
    """A typed digest reference to one parent artifact."""

    parent_type: str
    parent_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parent_type",
            _required_identifier(self.parent_type, "parent_type"),
        )
        object.__setattr__(
            self,
            "parent_hash",
            _required_digest(self.parent_hash, "parent_hash"),
        )

    @property
    def artifact_type(self) -> str:
        """Return the referenced artifact type."""

        return self.parent_type

    @property
    def artifact_hash(self) -> str:
        """Return the referenced artifact digest."""

        return self.parent_hash

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArtifactLineageParent":
        """Create a parent reference from its JSON-compatible representation."""

        if not isinstance(payload, Mapping):
            raise ArtifactLineageValidationError("parent reference must be an object")
        return cls(
            parent_type=_first_value(payload, ("parent_type", "artifact_type", "type")),
            parent_hash=_first_value(payload, ("parent_hash", "artifact_hash", "hash")),
        )

    def to_dict(self) -> dict[str, str]:
        """Return the stable, raw-payload-free parent representation."""

        return {"type": self.parent_type, "hash": self.parent_hash}


def _coerce_parent(value: Any) -> ArtifactLineageParent:
    if isinstance(value, ArtifactLineageParent):
        return value
    if isinstance(value, Mapping):
        return ArtifactLineageParent.from_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 2:
            return ArtifactLineageParent(parent_type=value[0], parent_hash=value[1])
    raise ArtifactLineageValidationError("parent reference must be typed metadata")


def _normalise_parents(value: Any) -> tuple[ArtifactLineageParent, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        values = (
            {"type": parent_type, "hash": parent_hash}
            for parent_type, parent_hash in value.items()
        )
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        values = tuple(value)
    else:
        raise ArtifactLineageValidationError("parents must be typed metadata")

    parents = tuple(sorted(_coerce_parent(item) for item in values))
    if len(set(parents)) != len(parents):
        raise ArtifactLineageValidationError("parent references must be unique")
    return parents


def _resolve_transformation(
    transformation: Any = None,
    *,
    transformation_name: Any = None,
    transform: Any = None,
) -> str:
    candidates = [
        value
        for value in (transformation, transformation_name, transform)
        if value is not None
    ]
    if not candidates:
        return "record"
    first = _required_identifier(candidates[0], "transformation")
    if any(
        _required_identifier(candidate, "transformation") != first
        for candidate in candidates[1:]
    ):
        raise ArtifactLineageValidationError("transformation aliases must agree")
    return first


def _node_hash_material(
    *,
    artifact_type: str,
    parents: tuple[ArtifactLineageParent, ...],
    transformation: str,
    policy_fingerprint: str,
    schema_version: SchemaVersion,
) -> dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "parents": [parent.to_dict() for parent in parents],
        "policy_fingerprint": policy_fingerprint,
        "schema_version": schema_version,
        "transformation": transformation,
    }


@dataclass(frozen=True)
class ArtifactLineageNode:
    """One privacy artifact and the metadata committed by its digest.

    ``artifact_hash`` is a lineage commitment produced by
    :meth:`create`/``compute_artifact_hash``.  It is not a source payload and
    must not be replaced with a raw identifier or source text.
    """

    artifact_hash: str
    artifact_type: str
    transformation: str
    policy_fingerprint: str = EMPTY_POLICY_FINGERPRINT
    schema_version: SchemaVersion = LINEAGE_NODE_SCHEMA_VERSION
    parents: tuple[ArtifactLineageParent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_hash",
            _required_digest(self.artifact_hash, "artifact_hash"),
        )
        object.__setattr__(
            self,
            "artifact_type",
            _required_identifier(self.artifact_type, "artifact_type"),
        )
        object.__setattr__(
            self,
            "transformation",
            _required_identifier(self.transformation, "transformation"),
        )
        object.__setattr__(
            self,
            "policy_fingerprint",
            _required_digest(self.policy_fingerprint, "policy_fingerprint"),
        )
        object.__setattr__(
            self,
            "schema_version",
            _required_schema_version(self.schema_version, "schema_version"),
        )
        object.__setattr__(self, "parents", _normalise_parents(self.parents))

    @property
    def transformation_name(self) -> str:
        """Return the canonical transformation name."""

        return self.transformation

    @property
    def parent_hashes(self) -> tuple[str, ...]:
        """Return parent digests in canonical typed-reference order."""

        return tuple(parent.parent_hash for parent in self.parents)

    def hash_material(self) -> dict[str, Any]:
        """Return the fields committed by ``artifact_hash``."""

        return _node_hash_material(
            artifact_type=self.artifact_type,
            parents=self.parents,
            transformation=self.transformation,
            policy_fingerprint=self.policy_fingerprint,
            schema_version=self.schema_version,
        )

    @property
    def expected_artifact_hash(self) -> str:
        """Return the digest expected for the node's current metadata."""

        return stable_hash(self.hash_material())

    @property
    def hash_matches(self) -> bool:
        """Return whether the node digest commits to its current metadata."""

        return self.artifact_hash == self.expected_artifact_hash

    @classmethod
    def create(
        cls,
        *,
        artifact_type: str,
        parents: Iterable[ArtifactLineageParent | Mapping[str, Any]] = (),
        transformation: str | None = None,
        transformation_name: str | None = None,
        transform: str | None = None,
        policy_fingerprint: str = EMPTY_POLICY_FINGERPRINT,
        schema_version: SchemaVersion = LINEAGE_NODE_SCHEMA_VERSION,
    ) -> "ArtifactLineageNode":
        """Create a node and derive its deterministic commitment hash."""

        normalised_type = _required_identifier(artifact_type, "artifact_type")
        normalised_parents = _normalise_parents(parents)
        normalised_transformation = _resolve_transformation(
            transformation,
            transformation_name=transformation_name,
            transform=transform,
        )
        normalised_policy = _required_digest(
            policy_fingerprint,
            "policy_fingerprint",
        )
        normalised_schema = _required_schema_version(schema_version, "schema_version")
        artifact_hash = compute_artifact_hash(
            normalised_type,
            normalised_parents,
            normalised_transformation,
            normalised_policy,
            normalised_schema,
        )
        return cls(
            artifact_hash=artifact_hash,
            artifact_type=normalised_type,
            transformation=normalised_transformation,
            policy_fingerprint=normalised_policy,
            schema_version=normalised_schema,
            parents=normalised_parents,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArtifactLineageNode":
        """Load a node while rejecting unknown or payload-bearing fields."""

        if not isinstance(payload, Mapping):
            raise ArtifactLineageValidationError("lineage node must be an object")
        allowed = {
            "artifact_hash",
            "artifact_type",
            "hash",
            "type",
            "parents",
            "parent_hashes",
            "policy_fingerprint",
            "schema_version",
            "transformation",
            "transformation_name",
            "transform",
        }
        if any(key not in allowed for key in payload):
            raise ArtifactLineageValidationError(
                "lineage node contains unsupported fields"
            )

        artifact_type = _first_value(payload, ("artifact_type", "type"))
        parent_values = _first_value(
            payload,
            ("parents", "parent_hashes"),
            default=(),
        )
        transformation = _first_value(
            payload,
            ("transformation", "transformation_name", "transform"),
            default=None,
        )
        policy_fingerprint = payload.get(
            "policy_fingerprint",
            EMPTY_POLICY_FINGERPRINT,
        )
        schema_version = payload.get("schema_version", LINEAGE_NODE_SCHEMA_VERSION)
        artifact_hash = _first_value(
            payload,
            ("artifact_hash", "hash"),
            default=None,
        )
        if artifact_hash is None:
            return cls.create(
                artifact_type=artifact_type,
                parents=parent_values,
                transformation=transformation,
                policy_fingerprint=policy_fingerprint,
                schema_version=schema_version,
            )
        return cls(
            artifact_hash=artifact_hash,
            artifact_type=artifact_type,
            transformation=_resolve_transformation(transformation),
            policy_fingerprint=policy_fingerprint,
            schema_version=schema_version,
            parents=parent_values,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, raw-payload-free node."""

        return {
            "artifact_hash": self.artifact_hash,
            "artifact_type": self.artifact_type,
            "parents": [parent.to_dict() for parent in self.parents],
            "policy_fingerprint": self.policy_fingerprint,
            "schema_version": self.schema_version,
            "transformation": self.transformation,
        }


def compute_artifact_hash(
    artifact_type: str,
    parents: Iterable[ArtifactLineageParent | Mapping[str, Any]] = (),
    transformation: str | None = None,
    policy_fingerprint: str = EMPTY_POLICY_FINGERPRINT,
    schema_version: SchemaVersion = LINEAGE_NODE_SCHEMA_VERSION,
    *,
    transformation_name: str | None = None,
    transform: str | None = None,
) -> str:
    """Compute a node commitment from metadata and typed parent digests."""

    normalised_type = _required_identifier(artifact_type, "artifact_type")
    normalised_parents = _normalise_parents(parents)
    normalised_transformation = _resolve_transformation(
        transformation,
        transformation_name=transformation_name,
        transform=transform,
    )
    normalised_policy = _required_digest(policy_fingerprint, "policy_fingerprint")
    normalised_schema = _required_schema_version(schema_version, "schema_version")
    return stable_hash(
        _node_hash_material(
            artifact_type=normalised_type,
            parents=normalised_parents,
            transformation=normalised_transformation,
            policy_fingerprint=normalised_policy,
            schema_version=normalised_schema,
        )
    )


@dataclass(frozen=True)
class ArtifactLineageDiagnostics:
    """Counts-only result of validating a lineage manifest."""

    node_count: int
    parent_reference_count: int
    cycle_count: int
    missing_parent_count: int
    hash_mismatch_count: int
    duplicate_hash_count: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "node_count",
            "parent_reference_count",
            "cycle_count",
            "missing_parent_count",
            "hash_mismatch_count",
            "duplicate_hash_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")

    @property
    def valid(self) -> bool:
        """Return whether all deterministic integrity counts are zero."""

        return not any(
            (
                self.cycle_count,
                self.missing_parent_count,
                self.hash_mismatch_count,
                self.duplicate_hash_count,
            )
        )

    @property
    def is_valid(self) -> bool:
        """Alias for :attr:`valid`."""

        return self.valid

    @property
    def ok(self) -> bool:
        """Alias for :attr:`valid`."""

        return self.valid

    @property
    def cycles(self) -> int:
        """Return the cycle count under a short diagnostic name."""

        return self.cycle_count

    @property
    def missing_parents(self) -> int:
        """Return the missing-parent count under a short diagnostic name."""

        return self.missing_parent_count

    @property
    def hash_mismatches(self) -> int:
        """Return the hash-mismatch count under a short diagnostic name."""

        return self.hash_mismatch_count

    def to_dict(self) -> dict[str, Any]:
        """Return only counts and the aggregate validity flag."""

        return {
            "cycle_count": self.cycle_count,
            "duplicate_hash_count": self.duplicate_hash_count,
            "hash_mismatch_count": self.hash_mismatch_count,
            "missing_parent_count": self.missing_parent_count,
            "node_count": self.node_count,
            "parent_reference_count": self.parent_reference_count,
            "valid": self.valid,
        }


def _manifest_hash_material(
    *,
    schema_version: str,
    nodes: tuple[ArtifactLineageNode, ...],
) -> dict[str, Any]:
    return {
        "nodes": [node.to_dict() for node in nodes],
        "schema_version": schema_version,
    }


def _node_sort_key(node: ArtifactLineageNode) -> tuple[Any, ...]:
    return (
        node.artifact_hash,
        node.artifact_type,
        node.transformation,
        str(node.schema_version),
        node.policy_fingerprint,
        tuple((parent.parent_type, parent.parent_hash) for parent in node.parents),
    )


@dataclass(frozen=True)
class ArtifactLineageManifest:
    """A canonical collection of privacy-artifact lineage nodes."""

    nodes: tuple[ArtifactLineageNode, ...] = ()
    schema_version: str = ARTIFACT_LINEAGE_SCHEMA_VERSION
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        normalised_schema = _required_identifier(
            self.schema_version,
            "manifest schema_version",
        )
        normalised_nodes = tuple(
            sorted(
                (
                    node
                    if isinstance(node, ArtifactLineageNode)
                    else ArtifactLineageNode.from_mapping(node)
                    for node in self.nodes
                ),
                key=_node_sort_key,
            )
        )
        object.__setattr__(self, "schema_version", normalised_schema)
        object.__setattr__(self, "nodes", normalised_nodes)
        if self.manifest_hash:
            normalised_hash = _required_digest(self.manifest_hash, "manifest_hash")
        else:
            normalised_hash = stable_hash(
                _manifest_hash_material(
                    schema_version=normalised_schema,
                    nodes=normalised_nodes,
                )
            )
        object.__setattr__(self, "manifest_hash", normalised_hash)

    @property
    def expected_manifest_hash(self) -> str:
        """Return the hash expected for the current canonical manifest."""

        return stable_hash(
            _manifest_hash_material(
                schema_version=self.schema_version,
                nodes=self.nodes,
            )
        )

    @property
    def hash_matches(self) -> bool:
        """Return whether the manifest hash commits to the current nodes."""

        return self.manifest_hash == self.expected_manifest_hash

    def verify(self) -> ArtifactLineageDiagnostics:
        """Return deterministic counts-only integrity diagnostics."""

        by_hash: dict[str, list[ArtifactLineageNode]] = {}
        for node in self.nodes:
            by_hash.setdefault(node.artifact_hash, []).append(node)

        duplicate_hash_count = sum(
            max(0, len(candidates) - 1) for candidates in by_hash.values()
        )
        parent_reference_count = sum(len(node.parents) for node in self.nodes)
        missing_parent_count = 0
        hash_mismatch_count = sum(not node.hash_matches for node in self.nodes)

        for node in self.nodes:
            for parent in node.parents:
                candidates = by_hash.get(parent.parent_hash)
                if not candidates:
                    missing_parent_count += 1
                    continue
                typed_matches = [
                    candidate
                    for candidate in candidates
                    if candidate.artifact_type == parent.parent_type
                ]
                if len(typed_matches) != 1:
                    hash_mismatch_count += 1

        if not self.hash_matches:
            hash_mismatch_count += 1

        return ArtifactLineageDiagnostics(
            node_count=len(self.nodes),
            parent_reference_count=parent_reference_count,
            cycle_count=_count_cycles(self.nodes, by_hash),
            missing_parent_count=missing_parent_count,
            hash_mismatch_count=hash_mismatch_count,
            duplicate_hash_count=duplicate_hash_count,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArtifactLineageManifest":
        """Load a manifest from a strict, JSON-compatible mapping."""

        if not isinstance(payload, Mapping):
            raise ArtifactLineageValidationError("lineage manifest must be an object")
        allowed = {"nodes", "artifacts", "schema_version", "manifest_hash"}
        if any(key not in allowed for key in payload):
            raise ArtifactLineageValidationError(
                "lineage manifest contains unsupported fields"
            )
        raw_nodes = _first_value(payload, ("nodes", "artifacts"), default=())
        if not isinstance(raw_nodes, Sequence) or isinstance(
            raw_nodes,
            (str, bytes),
        ):
            raise ArtifactLineageValidationError("lineage nodes must be an array")
        return cls(
            nodes=tuple(ArtifactLineageNode.from_mapping(item) for item in raw_nodes),
            schema_version=payload.get(
                "schema_version", ARTIFACT_LINEAGE_SCHEMA_VERSION
            ),
            manifest_hash=payload.get("manifest_hash", ""),
        )

    @classmethod
    def from_json(cls, value: str) -> "ArtifactLineageManifest":
        """Parse a JSON manifest without exposing parser input in exceptions."""

        try:
            payload = json.loads(value)
        except (TypeError, ValueError):
            raise ArtifactLineageValidationError(
                "lineage manifest JSON is invalid"
            ) from None
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical, raw-payload-free manifest."""

        return {
            "manifest_hash": self.manifest_hash,
            "nodes": [node.to_dict() for node in self.nodes],
            "schema_version": self.schema_version,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the manifest deterministically as JSON."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write a local JSON manifest and return its path."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path


def _count_cycles(
    nodes: tuple[ArtifactLineageNode, ...],
    by_hash: Mapping[str, list[ArtifactLineageNode]],
) -> int:
    """Count strongly connected components that contain a directed cycle."""

    adjacency: dict[str, set[str]] = {artifact_hash: set() for artifact_hash in by_hash}
    for node in nodes:
        candidates = by_hash[node.artifact_hash]
        if candidates[0] is not node:
            continue
        for parent in node.parents:
            typed_matches = [
                candidate
                for candidate in by_hash.get(parent.parent_hash, ())
                if candidate.artifact_type == parent.parent_type
            ]
            if len(typed_matches) == 1:
                adjacency[node.artifact_hash].add(parent.parent_hash)

    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    next_index = 0
    cycle_count = 0

    def strong_connect(vertex: str) -> None:
        nonlocal next_index, cycle_count
        indices[vertex] = next_index
        lowlinks[vertex] = next_index
        next_index += 1
        stack.append(vertex)
        on_stack.add(vertex)

        for neighbour in sorted(adjacency[vertex]):
            if neighbour not in indices:
                strong_connect(neighbour)
                lowlinks[vertex] = min(lowlinks[vertex], lowlinks[neighbour])
            elif neighbour in on_stack:
                lowlinks[vertex] = min(lowlinks[vertex], indices[neighbour])

        if lowlinks[vertex] != indices[vertex]:
            return
        component: list[str] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == vertex:
                break
        if len(component) > 1 or vertex in adjacency[vertex]:
            cycle_count += 1

    for vertex in sorted(adjacency):
        if vertex not in indices:
            strong_connect(vertex)
    return cycle_count


def build_artifact_lineage_manifest(
    nodes: Iterable[ArtifactLineageNode | Mapping[str, Any]],
    *,
    schema_version: str = ARTIFACT_LINEAGE_SCHEMA_VERSION,
) -> ArtifactLineageManifest:
    """Build a canonical local manifest from nodes or safe mappings."""

    return ArtifactLineageManifest(
        nodes=tuple(
            node
            if isinstance(node, ArtifactLineageNode)
            else ArtifactLineageNode.from_mapping(node)
            for node in nodes
        ),
        schema_version=schema_version,
    )


def verify_artifact_lineage(
    manifest: ArtifactLineageManifest | Mapping[str, Any],
) -> ArtifactLineageDiagnostics:
    """Verify a manifest or mapping and return counts-only diagnostics."""

    resolved = (
        manifest
        if isinstance(manifest, ArtifactLineageManifest)
        else ArtifactLineageManifest.from_mapping(manifest)
    )
    return resolved.verify()


def compute_policy_fingerprint(policy_metadata: Any) -> str:
    """Hash JSON-compatible policy metadata without returning the metadata."""

    try:
        return stable_hash(policy_metadata)
    except (TypeError, ValueError, OverflowError):
        raise ArtifactLineageValidationError(
            "policy metadata must be deterministically JSON serializable"
        ) from None


def load_artifact_lineage_manifest(path: str | Path) -> ArtifactLineageManifest:
    """Load a local JSON manifest without including file contents in errors."""

    try:
        payload = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise ArtifactLineageValidationError(
            "lineage manifest could not be read"
        ) from None
    return ArtifactLineageManifest.from_json(payload)


# Descriptive aliases keep the public surface easy to discover while the
# canonical names remain explicit in serialized manifests.
LineageParent = ArtifactLineageParent
LineageNode = ArtifactLineageNode
ArtifactNode = ArtifactLineageNode
ArtifactManifest = ArtifactLineageManifest
LineageDiagnostics = ArtifactLineageDiagnostics
build_manifest = build_artifact_lineage_manifest
verify_manifest = verify_artifact_lineage
