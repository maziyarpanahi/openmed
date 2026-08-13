"""Deterministic, non-destructive deletion impact planning.

The planner consumes a local manifest of owned artifacts and their dependency
links.  It follows reverse dependency links so a deletion candidate's caches,
maps, and evidence artifacts that depend on it are included in the impact
preview.  Public serialization contains counts and integrity digests only;
raw identifiers, paths, and manifest fields outside the safe schema are never
copied into a report.

Planning is always a dry-run.  Actual deletion is deliberately delegated to an
injected callback and requires an exact confirmation token derived from the
plan.  This module performs no network or filesystem operation unless a local
JSON manifest is explicitly loaded by the caller.
"""

from __future__ import annotations

import hmac
import json
import re
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

from openmed.core.audit import hash_text, stable_hash

__all__ = [
    "ConfirmationRequiredError",
    "DeletionArtifact",
    "DeletionExecutionError",
    "DeletionExecutionResult",
    "DeletionPlanError",
    "DeletionImpactPlan",
    "build_deletion_plan",
    "execute_deletion_plan",
    "load_deletion_manifest",
    "plan_deletion_impact",
]

_SCHEMA_VERSION = 1
_HASH_RE = re.compile(r"^(?:sha256|hmac-sha256):[0-9a-f]{64}$")
_BARE_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_HASH_PREFIXES = frozenset({"sha256", "hmac-sha256"})
_MISSING = object()

ManifestInput: TypeAlias = Mapping[str, Any] | Sequence[Any] | Path
DeletionExecutor: TypeAlias = Callable[["DeletionArtifact"], None]


class DeletionPlanError(ValueError):
    """Raised when a deletion manifest or plan cannot be used safely."""


class ConfirmationRequiredError(DeletionPlanError):
    """Raised when execution is missing the plan-specific confirmation."""


class DeletionExecutionError(RuntimeError):
    """Raised when an injected deletion callback fails."""


def _canonical_hash(value: Any) -> str:
    """Return a safe artifact reference without retaining raw input values."""

    if not isinstance(value, str):
        raise DeletionPlanError("artifact references must be strings")
    reference = value.strip()
    if not reference:
        raise DeletionPlanError("artifact references must not be empty")

    if _HASH_RE.fullmatch(reference):
        return reference.lower()
    if _BARE_HASH_RE.fullmatch(reference):
        return f"sha256:{reference.lower()}"

    prefix = reference.partition(":")[0].lower()
    if prefix in _HASH_PREFIXES:
        raise DeletionPlanError("artifact references must contain a valid digest")

    # Accept opaque local identifiers at the boundary, but only retain their
    # digest.  This keeps callers from accidentally putting a path or PHI into
    # a plan object while preserving deterministic matching between links.
    return hash_text(reference)


def _safe_label(value: Any, *, default: str) -> str:
    """Validate a report-safe manifest label without echoing invalid input."""

    if value is None:
        return default
    if not isinstance(value, str):
        raise DeletionPlanError("manifest labels must be strings")
    label = value.strip().lower()
    if not _LABEL_RE.fullmatch(label):
        raise DeletionPlanError("manifest labels must be short lowercase tokens")
    return label


def _field(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return _MISSING


def _dependency_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        value = _field(
            value,
            ("artifact_hash", "hash", "digest", "artifact_digest"),
        )
    if value is _MISSING:
        raise DeletionPlanError("dependency links must contain artifact references")
    return value


def _normalize_dependencies(value: Any) -> tuple[str, ...]:
    if value is None or value is _MISSING:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise DeletionPlanError("dependency links must be a sequence")
    try:
        dependencies = {_canonical_hash(_dependency_value(item)) for item in value}
    except TypeError as exc:
        raise DeletionPlanError("dependency links must be a sequence") from None
    return tuple(sorted(dependencies))


def _normalize_owned(value: Any) -> bool:
    if value is _MISSING or value is None:
        return True
    if type(value) is not bool:
        raise DeletionPlanError("manifest ownership must be boolean")
    return value


@dataclass(frozen=True, slots=True)
class DeletionArtifact:
    """Privacy-safe manifest entry for one deletable artifact.

    ``artifact_hash`` accepts an existing SHA-256/HMAC-SHA-256 digest or an
    opaque local reference.  Opaque references are immediately replaced with a
    SHA-256 digest.  ``dependencies`` point from this artifact to the hashes it
    needs; the planner follows those links in reverse when computing impact.
    """

    artifact_hash: str
    kind: str = "artifact"
    retention_class: str = "unspecified"
    dependencies: tuple[str, ...] = ()
    owned: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_hash", _canonical_hash(self.artifact_hash))
        object.__setattr__(
            self,
            "kind",
            _safe_label(self.kind, default="artifact"),
        )
        object.__setattr__(
            self,
            "retention_class",
            _safe_label(self.retention_class, default="unspecified"),
        )
        object.__setattr__(
            self,
            "dependencies",
            _normalize_dependencies(self.dependencies),
        )
        object.__setattr__(self, "owned", _normalize_owned(self.owned))

    @property
    def digest(self) -> str:
        """Return the canonical artifact digest."""

        return self.artifact_hash

    @property
    def artifact_type(self) -> str:
        """Return the normalized kind name."""

        return self.kind

    def to_dict(self) -> dict[str, Any]:
        """Return the safe manifest representation."""

        return {
            "artifact_hash": self.artifact_hash,
            "kind": self.kind,
            "retention_class": self.retention_class,
            "dependencies": list(self.dependencies),
            "owned": self.owned,
        }


@dataclass(frozen=True, slots=True)
class DeletionExecutionResult:
    """Counts-only result from an explicitly confirmed execution callback."""

    plan_digest: str
    requested_count: int
    deleted_count: int
    dry_run: bool = False
    executed: bool = True

    @property
    def failed_count(self) -> int:
        """Return the number of failed callbacks represented by this result."""

        return self.requested_count - self.deleted_count

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, counts-only result."""

        return {
            "plan_digest": self.plan_digest,
            "requested_count": self.requested_count,
            "deleted_count": self.deleted_count,
            "failed_count": self.failed_count,
            "dry_run": self.dry_run,
            "executed": self.executed,
            "raw_values_included": False,
        }

    def to_json(self) -> str:
        """Serialize the result with stable JSON formatting."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class DeletionImpactPlan:
    """Deterministic dry-run impact plan for a set of deletion candidates."""

    manifest_digest: str
    plan_digest: str
    target_count: int
    affected_count: int
    owned_affected_count: int
    unowned_affected_count: int
    blocked_target_count: int
    unresolved_dependency_count: int
    dependency_edge_count: int
    _kind_counts: tuple[tuple[str, int], ...] = field(repr=False, compare=False)
    _retention_counts: tuple[tuple[str, int], ...] = field(
        repr=False,
        compare=False,
    )
    _entries: tuple[DeletionArtifact, ...] = field(repr=False, compare=False)
    _target_hashes: tuple[str, ...] = field(repr=False, compare=False)
    _affected_hashes: tuple[str, ...] = field(repr=False, compare=False)
    dry_run: bool = True

    @property
    def counts_by_kind(self) -> dict[str, int]:
        """Return counts grouped by safe artifact kind."""

        return dict(self._kind_counts)

    @property
    def counts_by_retention_class(self) -> dict[str, int]:
        """Return counts grouped by safe retention class."""

        return dict(self._retention_counts)

    @property
    def target_hashes(self) -> tuple[str, ...]:
        """Return canonical target hashes for an injected executor."""

        return self._target_hashes

    @property
    def affected_hashes(self) -> tuple[str, ...]:
        """Return canonical hashes in the computed impact closure."""

        return self._affected_hashes

    @property
    def affected_artifacts(self) -> tuple[DeletionArtifact, ...]:
        """Return safe manifest entries in deterministic impact order."""

        affected = set(self._affected_hashes)
        return tuple(
            entry for entry in self._entries if entry.artifact_hash in affected
        )

    @property
    def confirmation_token(self) -> str:
        """Return the exact token required by :func:`execute_deletion_plan`."""

        return f"confirm:{self.plan_digest}"

    @property
    def summary(self) -> dict[str, Any]:
        """Return counts-only, privacy-safe plan metadata."""

        return {
            "schema_version": _SCHEMA_VERSION,
            "dry_run": self.dry_run,
            "manifest_digest": self.manifest_digest,
            "plan_digest": self.plan_digest,
            "target_count": self.target_count,
            "affected_count": self.affected_count,
            "owned_affected_count": self.owned_affected_count,
            "unowned_affected_count": self.unowned_affected_count,
            "blocked_target_count": self.blocked_target_count,
            "unresolved_dependency_count": self.unresolved_dependency_count,
            "dependency_edge_count": self.dependency_edge_count,
            "affected_by_kind": self.counts_by_kind,
            "affected_by_retention_class": self.counts_by_retention_class,
            "confirmation_required": True,
            "raw_values_included": False,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic counts-only report."""

        return self.summary

    def to_json(self) -> str:
        """Serialize the plan with stable JSON formatting."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    def to_markdown(self) -> str:
        """Render a counts-only human-readable dry-run report."""

        lines = [
            "# Deletion impact plan",
            "",
            "This is a non-destructive dry-run; no artifact was deleted.",
            "",
            "| Metric | Count |",
            "|---|---:|",
            f"| Target artifacts | {self.target_count} |",
            f"| Affected artifacts | {self.affected_count} |",
            f"| Owned affected artifacts | {self.owned_affected_count} |",
            f"| Unowned affected artifacts | {self.unowned_affected_count} |",
            f"| Blocked targets | {self.blocked_target_count} |",
            f"| Unresolved dependency links | {self.unresolved_dependency_count} |",
            f"| Dependency links | {self.dependency_edge_count} |",
            "",
            "## Affected artifacts by kind",
            "",
        ]
        lines.extend(f"- `{kind}`: {count}" for kind, count in self._kind_counts)
        if not self._kind_counts:
            lines.append("- (none)")
        lines.extend(["", "## Affected artifacts by retention class", ""])
        lines.extend(
            f"- `{retention_class}`: {count}"
            for retention_class, count in self._retention_counts
        )
        if not self._retention_counts:
            lines.append("- (none)")
        lines.extend(
            [
                "",
                f"Manifest digest: `{self.manifest_digest}`",
                f"Plan digest: `{self.plan_digest}`",
                "",
                "Execution requires the exact plan confirmation token and an "
                "injected deletion callback.",
            ]
        )
        return "\n".join(lines)

    def execute(
        self,
        *,
        confirmation: str | None = None,
        confirm: str | None = None,
        executor: DeletionExecutor | None = None,
        delete_fn: DeletionExecutor | None = None,
    ) -> DeletionExecutionResult:
        """Execute target callbacks after explicit plan confirmation."""

        return execute_deletion_plan(
            self,
            confirmation=confirmation,
            confirm=confirm,
            executor=executor,
            delete_fn=delete_fn,
        )


def _parse_artifact(value: Any) -> DeletionArtifact:
    if isinstance(value, DeletionArtifact):
        return value
    if not isinstance(value, Mapping):
        raise DeletionPlanError("manifest entries must be mappings")

    artifact_hash = _field(
        value,
        ("artifact_hash", "hash", "digest", "artifact_digest"),
    )
    if artifact_hash is _MISSING:
        raise DeletionPlanError("manifest entries require an artifact hash")

    kind = _field(value, ("kind", "artifact_type", "type", "category"))
    retention_class = _field(value, ("retention_class", "retention"))
    dependencies = _field(
        value,
        ("dependencies", "dependency_links", "depends_on", "links"),
    )
    owned = _field(value, ("owned", "is_owned"))
    return DeletionArtifact(
        artifact_hash=artifact_hash,
        kind="artifact" if kind is _MISSING else kind,
        retention_class=(
            "unspecified" if retention_class is _MISSING else retention_class
        ),
        dependencies=() if dependencies is _MISSING else dependencies,
        owned=owned,
    )


def _parse_manifest_payload(payload: Any) -> tuple[DeletionArtifact, ...]:
    if isinstance(payload, Mapping):
        entries = _field(payload, ("artifacts", "entries", "manifest"))
        if entries is _MISSING:
            entries = [payload]
    else:
        entries = payload

    if isinstance(entries, (str, bytes)) or not isinstance(entries, Iterable):
        raise DeletionPlanError("manifest artifacts must be a sequence")
    try:
        parsed = tuple(
            sorted(
                (_parse_artifact(item) for item in entries),
                key=lambda item: item.artifact_hash,
            )
        )
    except TypeError:
        raise DeletionPlanError("manifest artifacts must be a sequence") from None
    if not parsed:
        raise DeletionPlanError("manifest must contain at least one artifact")
    if len({entry.artifact_hash for entry in parsed}) != len(parsed):
        raise DeletionPlanError("manifest artifact hashes must be unique")
    return parsed


def load_deletion_manifest(path: str | Path) -> tuple[DeletionArtifact, ...]:
    """Load and normalize one local JSON deletion manifest.

    The loader is intentionally local-only.  Read errors and malformed content
    use content-free exceptions so a sensitive path or document cannot escape
    through an error message.
    """

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise DeletionPlanError("deletion manifest could not be loaded") from None
    return _parse_manifest_payload(payload)


def _normalize_manifest(manifest: ManifestInput) -> tuple[DeletionArtifact, ...]:
    if isinstance(manifest, Path):
        return load_deletion_manifest(manifest)
    if isinstance(manifest, (str, bytes)):
        raise DeletionPlanError("manifest must be a mapping, sequence, or local path")
    return _parse_manifest_payload(manifest)


def _normalize_targets(value: Iterable[str] | str | None) -> tuple[str, ...]:
    if value is None:
        raise DeletionPlanError("at least one deletion target is required")
    if isinstance(value, str):
        values: Iterable[str] = (value,)
    else:
        if isinstance(value, bytes) or not isinstance(value, Iterable):
            raise DeletionPlanError("deletion targets must be a sequence")
        values = value
    try:
        targets = tuple(sorted({_canonical_hash(item) for item in values}))
    except TypeError:
        raise DeletionPlanError("deletion targets must be a sequence") from None
    if not targets:
        raise DeletionPlanError("at least one deletion target is required")
    return targets


def plan_deletion_impact(
    manifest: ManifestInput,
    artifact_hashes: Iterable[str] | str | None = None,
    *,
    targets: Iterable[str] | str | None = None,
    dry_run: bool = True,
) -> DeletionImpactPlan:
    """Build a deterministic, non-destructive deletion impact plan.

    Args:
        manifest: A manifest mapping, a sequence of entries, or a local JSON
            path.  Entries use ``artifact_hash``, ``kind``,
            ``retention_class``, ``dependencies``, and ``owned`` fields.
        artifact_hashes: One or more target artifact hashes.  Opaque local
            references are hashed at the boundary and are never reported.
        targets: Keyword alias for ``artifact_hashes``.
        dry_run: Must remain ``True``.  Execution is a separate confirmed
            operation through :func:`execute_deletion_plan`.

    Returns:
        A counts-only :class:`DeletionImpactPlan`.  Reverse dependency links
        are followed transitively, so dependents of each target are included
        in the impact closure.
    """

    if dry_run is not True:
        raise DeletionPlanError(
            "planning is always a dry-run; use explicit execution separately"
        )
    if artifact_hashes is not None and targets is not None:
        raise DeletionPlanError("provide deletion targets only once")

    entries = _normalize_manifest(manifest)
    requested = _normalize_targets(
        artifact_hashes if artifact_hashes is not None else targets
    )
    by_hash = {entry.artifact_hash: entry for entry in entries}
    if any(target not in by_hash for target in requested):
        raise DeletionPlanError("every deletion target must be in the manifest")

    dependents: defaultdict[str, set[str]] = defaultdict(set)
    for entry in entries:
        for dependency in entry.dependencies:
            if dependency in by_hash:
                dependents[dependency].add(entry.artifact_hash)

    affected = set(requested)
    pending = deque(requested)
    while pending:
        dependency = pending.popleft()
        for dependent in sorted(dependents.get(dependency, ())):
            if dependent not in affected:
                affected.add(dependent)
                pending.append(dependent)

    affected_entries = tuple(
        entry for entry in entries if entry.artifact_hash in affected
    )
    target_entries = tuple(by_hash[target] for target in requested)
    kind_counts = _count_labels(entry.kind for entry in affected_entries)
    retention_counts = _count_labels(
        entry.retention_class for entry in affected_entries
    )
    unresolved_dependency_count = sum(
        dependency not in by_hash
        for entry in affected_entries
        for dependency in entry.dependencies
    )
    dependency_edge_count = sum(len(entry.dependencies) for entry in affected_entries)

    manifest_digest = stable_hash(
        {
            "schema_version": _SCHEMA_VERSION,
            "artifacts": [entry.to_dict() for entry in entries],
        }
    )
    plan_digest = stable_hash(
        {
            "schema_version": _SCHEMA_VERSION,
            "manifest_digest": manifest_digest,
            "targets": list(requested),
        }
    )
    return DeletionImpactPlan(
        manifest_digest=manifest_digest,
        plan_digest=plan_digest,
        target_count=len(target_entries),
        affected_count=len(affected_entries),
        owned_affected_count=sum(entry.owned for entry in affected_entries),
        unowned_affected_count=sum(not entry.owned for entry in affected_entries),
        blocked_target_count=sum(not entry.owned for entry in target_entries),
        unresolved_dependency_count=unresolved_dependency_count,
        dependency_edge_count=dependency_edge_count,
        _kind_counts=kind_counts,
        _retention_counts=retention_counts,
        _entries=entries,
        _target_hashes=requested,
        _affected_hashes=tuple(sorted(affected)),
    )


def _count_labels(labels: Iterable[str]) -> tuple[tuple[str, int], ...]:
    counts: defaultdict[str, int] = defaultdict(int)
    for label in labels:
        counts[label] += 1
    return tuple(sorted(counts.items()))


def execute_deletion_plan(
    plan: DeletionImpactPlan,
    *,
    confirmation: str | None = None,
    confirm: str | None = None,
    executor: DeletionExecutor | None = None,
    delete_fn: DeletionExecutor | None = None,
) -> DeletionExecutionResult:
    """Execute target callbacks only after exact explicit confirmation.

    The planner never deletes impacted dependents automatically.  The injected
    callback is invoked for the explicitly requested target entries in stable
    hash order.  Unowned targets and incomplete dependency closures fail closed
    before the first callback is invoked.
    """

    if not isinstance(plan, DeletionImpactPlan):
        raise DeletionPlanError("execution requires a deletion impact plan")
    if confirmation is not None and confirm is not None:
        raise ConfirmationRequiredError("provide confirmation only once")
    supplied_confirmation = confirmation if confirmation is not None else confirm
    if not isinstance(supplied_confirmation, str) or not hmac.compare_digest(
        supplied_confirmation,
        plan.confirmation_token,
    ):
        raise ConfirmationRequiredError(
            "explicit confirmation is required before deletion"
        )
    if executor is not None and delete_fn is not None:
        raise DeletionPlanError("provide a deletion callback only once")
    callback = executor if executor is not None else delete_fn
    if callback is None:
        raise DeletionPlanError("an injected deletion callback is required")
    if plan.blocked_target_count or plan.unresolved_dependency_count:
        raise DeletionPlanError("deletion plan failed manifest safety checks")

    deleted_count = 0
    target_by_hash = {entry.artifact_hash: entry for entry in plan._entries}
    try:
        for artifact_hash in plan.target_hashes:
            callback(target_by_hash[artifact_hash])
            deleted_count += 1
    except Exception:
        raise DeletionExecutionError("deletion callback failed") from None
    return DeletionExecutionResult(
        plan_digest=plan.plan_digest,
        requested_count=plan.target_count,
        deleted_count=deleted_count,
    )


build_deletion_plan = plan_deletion_impact
