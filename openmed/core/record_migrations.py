"""Registry of stored-record migrators keyed by schema id and version.

Stored OpenMed records (review packets, decision results, budget ledgers and
similar) carry a version field. This module answers one question for all of
them: how is a record brought from version A to version B? Each registered
migrator is one directed edge ``(schema_id, from_version, to_version)``; it
declares whether it is lossless and names a before/after fixture pair under
``tests/fixtures/migrations/``.

Chains are resolved deterministically. Registering an edge twice, or an edge
that would close a cycle, fails. A request with more than one route between
two versions is refused as ambiguous rather than guessed.

Migrations run in memory on deep copies and never touch the caller's record.
Reports are value-free: they list JSON Pointer field paths that were added,
moved, changed or dropped, plus versions, counts and SHA-256 digests, and never
field values. List indexes are collapsed to ``*`` so a report does not depend
on how many items a record holds.

Example:
    >>> registry = MigratorRegistry()
    >>> _ = registry.register_schema("example_note", current_version="2")
    >>> _ = registry.register_migrator(
    ...     "example_note",
    ...     "1",
    ...     "2",
    ...     lambda record: {**record, "schema_version": "2", "extensions": {}},
    ...     lossless=True,
    ...     fixture="example_note/1-to-2",
    ... )
    >>> result = registry.migrate_record(
    ...     {"schema_version": "1"}, schema_id="example_note"
    ... )
    >>> result.report.added
    ('/extensions',)
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import re
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from .errors import ConfigurationError, InputError

RECORD_MIGRATION_REPORT_SCHEMA_VERSION: Final = "1.0.0"
"""Version of the value-free report produced by :func:`migrate_record`."""

DEFAULT_VERSION_FIELD: Final = "schema_version"
"""Top-level record field that carries the record version by default."""

MIGRATION_FIXTURE_ROOT: Final = "tests/fixtures/migrations"
"""Repository-relative directory holding every edge's fixture pair."""

REFUSAL_REASONS: Final = (
    "ambiguous_path",
    "ambiguous_schema",
    "invalid_record",
    "invalid_version",
    "migrator_failed",
    "migrator_output_invalid",
    "no_path",
    "unknown_schema",
    "version_mismatch",
)
"""Controlled reason codes carried by :class:`MigrationRefusal`."""

REGISTRATION_REASONS: Final = (
    "cyclic_edge",
    "duplicate_edge",
    "duplicate_schema",
    "invalid_edge",
    "invalid_fixture",
    "invalid_schema",
    "unknown_schema",
)
"""Controlled reason codes carried by :class:`MigratorRegistrationError`."""

Migrator = Callable[[dict[str, Any]], Mapping[str, Any]]
"""A function that returns a new record one version step later."""

RecordMatcher = Callable[[Mapping[str, Any]], bool]
"""A predicate that recognises records of one schema by their shape."""

_SCHEMA_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
_FIXTURE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}$")
_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_MAX_DIFF_DEPTH = 48
_BUILTIN_REGISTRATIONS: tuple[str, ...] = ()


class MigratorRegistrationError(ConfigurationError):
    """A migrator or schema registration was rejected.

    Attributes:
        reason: Controlled reason code from :data:`REGISTRATION_REASONS`.
        schema_id: Schema id of the rejected registration, when it is safe.
        from_version: Source version of the rejected edge, when it is safe.
        to_version: Target version of the rejected edge, when it is safe.
    """

    code = "record_migrator_registration_error"

    def __init__(
        self,
        reason: str,
        *,
        schema_id: str | None = None,
        from_version: str | None = None,
        to_version: str | None = None,
    ) -> None:
        self.reason = reason
        self.schema_id = schema_id
        self.from_version = from_version
        self.to_version = to_version
        super().__init__(
            f"record migrator registration rejected ({reason})"
            f"{_describe(schema_id, from_version, to_version)}",
            details={
                "reason": reason,
                "schema_id": schema_id,
                "from_version": from_version,
                "to_version": to_version,
            },
        )


@dataclass(frozen=True, slots=True)
class MigrationRefusal:
    """Typed, value-free reason why a record cannot be migrated.

    Attributes:
        reason: Controlled reason code from :data:`REFUSAL_REASONS`.
        schema_id: Schema id, or ``None`` when it is unknown or unsafe to echo.
        from_version: Source version, or ``None`` when unknown or unsafe.
        to_version: Target version, or ``None`` when unknown or unsafe.
        error_type: Exception class name when a migrator raised, else ``None``.
    """

    reason: str
    schema_id: str | None = None
    from_version: str | None = None
    to_version: str | None = None
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the refusal as a JSON-ready mapping of codes and versions.

        Returns:
            A mapping with ``reason``, ``schema_id``, ``from_version`` and
            ``to_version`` keys, plus ``error_type`` when a migrator raised.
        """

        payload: dict[str, Any] = {
            "reason": self.reason,
            "schema_id": self.schema_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
        }
        if self.error_type is not None:
            payload["error_type"] = self.error_type
        return payload


class RecordMigrationRefused(InputError):
    """A record migration was refused; see :attr:`refusal` for the reason.

    The message and details contain only the reason code, the schema id and
    the versions. Record values and migrator exception messages are never
    included.

    Attributes:
        refusal: The typed refusal.
    """

    code = "record_migration_refused"

    def __init__(self, refusal: MigrationRefusal) -> None:
        self.refusal = refusal
        where = _describe(refusal.schema_id, refusal.from_version, refusal.to_version)
        super().__init__(
            f"record migration refused ({refusal.reason}){where}",
            details=refusal.to_dict(),
        )


@dataclass(frozen=True, slots=True)
class RecordSchema:
    """A record family the registry can migrate.

    Attributes:
        schema_id: Stable schema id, normally the packaged schema name.
        current_version: Version records are migrated to by default.
        version_field: Top-level record field holding the version.
        version_when_missing: Version assumed for records without the version
            field, or ``None`` to refuse such records.
        matcher: Optional predicate used by :meth:`MigratorRegistry.identify`.
    """

    schema_id: str
    current_version: str
    version_field: str = DEFAULT_VERSION_FIELD
    version_when_missing: str | None = None
    matcher: RecordMatcher | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Return the schema description without the matcher.

        Returns:
            A JSON-ready mapping of the schema id, versions and version field.
        """

        return {
            "schema_id": self.schema_id,
            "current_version": self.current_version,
            "version_field": self.version_field,
            "version_when_missing": self.version_when_missing,
            "identifiable": self.matcher is not None,
        }


@dataclass(frozen=True, slots=True)
class MigrationEdge:
    """One registered migrator from one version to the next.

    Attributes:
        schema_id: Schema id the edge belongs to.
        from_version: Version the migrator accepts.
        to_version: Version the migrator produces.
        migrator: Function that returns the migrated record.
        lossless: Whether the step preserves every value of its input.
        fixture: Fixture directory relative to :data:`MIGRATION_FIXTURE_ROOT`.
    """

    schema_id: str
    from_version: str
    to_version: str
    migrator: Migrator = field(repr=False, compare=False)
    lossless: bool = True
    fixture: str = ""

    def fixture_pair(self, repo_root: str | Path) -> tuple[Path, Path]:
        """Return the ``before.json`` and ``after.json`` paths of the fixture.

        Args:
            repo_root: Repository root that contains ``tests/fixtures``.

        Returns:
            The before and after fixture paths. Existence is not checked.
        """

        directory = Path(repo_root) / MIGRATION_FIXTURE_ROOT / self.fixture
        return directory / "before.json", directory / "after.json"

    def to_dict(self) -> dict[str, Any]:
        """Return the edge description without the migrator function.

        Returns:
            A JSON-ready mapping of the schema id, versions, lossless flag and
            fixture name.
        """

        return {
            "schema_id": self.schema_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "lossless": self.lossless,
            "fixture": self.fixture,
        }


@dataclass(frozen=True, slots=True)
class FieldMove:
    """A field whose value moved from one path to another.

    Attributes:
        from_path: JSON Pointer of the field in the input record.
        to_path: JSON Pointer of the field in the migrated record.
    """

    from_path: str
    to_path: str

    def to_dict(self) -> dict[str, str]:
        """Return both paths without the value that moved.

        Returns:
            A mapping with ``from`` and ``to`` JSON Pointers.
        """

        return {"from": self.from_path, "to": self.to_path}


@dataclass(frozen=True, slots=True)
class FieldChanges:
    """Value-free field paths that differ between two records.

    Attributes:
        added: Paths present only in the newer record.
        dropped: Paths present only in the older record.
        moved: Values that left one path and appeared at another.
        changed: Paths present in both records with different values.
    """

    added: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    moved: tuple[FieldMove, ...] = ()
    changed: tuple[str, ...] = ()

    def counts(self) -> dict[str, int]:
        """Return how many paths fall into each category.

        Returns:
            A mapping of ``added``, ``dropped``, ``moved`` and ``changed`` to
            their path counts.
        """

        return {
            "added": len(self.added),
            "changed": len(self.changed),
            "dropped": len(self.dropped),
            "moved": len(self.moved),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the field paths as JSON-ready lists.

        Returns:
            A mapping of each category to its sorted path list.
        """

        return {
            "added": list(self.added),
            "changed": list(self.changed),
            "dropped": list(self.dropped),
            "moved": [move.to_dict() for move in self.moved],
        }


@dataclass(frozen=True, slots=True)
class MigrationStepReport:
    """Value-free report for one edge of a migration chain.

    Attributes:
        from_version: Version before the step.
        to_version: Version after the step.
        lossless: The edge's declared lossless flag.
        fixture: The edge's fixture name.
        fields: Field paths the step added, dropped, moved or changed.
    """

    from_version: str
    to_version: str
    lossless: bool
    fixture: str
    fields: FieldChanges

    def to_dict(self) -> dict[str, Any]:
        """Return the step report as a JSON-ready mapping.

        Returns:
            Versions, the lossless flag, the fixture name and field paths.
        """

        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "lossless": self.lossless,
            "fixture": self.fixture,
            **self.fields.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RecordMigrationReport:
    """Value-free report for one record migration.

    Attributes:
        schema_id: Schema id of the record.
        source_version: Version of the input record.
        target_version: Version of the migrated record.
        steps: One report per edge, in chain order.
        fields: Field paths that differ between input and output overall.
        source_digest: SHA-256 digest of the canonical input record.
        target_digest: SHA-256 digest of the canonical migrated record.
    """

    schema_id: str
    source_version: str
    target_version: str
    steps: tuple[MigrationStepReport, ...]
    fields: FieldChanges
    source_digest: str
    target_digest: str

    @property
    def status(self) -> str:
        """Return ``current`` when no step ran, otherwise ``migrated``."""

        return "migrated" if self.steps else "current"

    @property
    def lossy(self) -> bool:
        """Return whether any step in the chain is declared lossy."""

        return any(not step.lossless for step in self.steps)

    @property
    def added(self) -> tuple[str, ...]:
        """Return the paths added across the whole chain."""

        return self.fields.added

    @property
    def dropped(self) -> tuple[str, ...]:
        """Return the paths dropped across the whole chain."""

        return self.fields.dropped

    @property
    def moved(self) -> tuple[FieldMove, ...]:
        """Return the values moved across the whole chain."""

        return self.fields.moved

    @property
    def changed(self) -> tuple[str, ...]:
        """Return the paths whose values changed across the whole chain."""

        return self.fields.changed

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic, value-free report.

        Returns:
            A JSON-ready mapping of ids, versions, status, the lossy flag,
            digests, counts, field paths and per-step reports.
        """

        return {
            "report_schema_version": RECORD_MIGRATION_REPORT_SCHEMA_VERSION,
            "schema_id": self.schema_id,
            "source_version": self.source_version,
            "target_version": self.target_version,
            "status": self.status,
            "lossy": self.lossy,
            "source_digest": self.source_digest,
            "target_digest": self.target_digest,
            "counts": {**self.fields.counts(), "steps": len(self.steps)},
            **self.fields.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class RecordMigrationResult:
    """A migrated record and its value-free report.

    The record is excluded from ``repr`` so accidental logging cannot expose
    its values.

    Attributes:
        record: The migrated record, a new JSON-compatible mapping.
        report: The value-free migration report.
    """

    record: dict[str, Any] = field(repr=False)
    report: RecordMigrationReport

    def __iter__(self) -> Iterator[Any]:
        """Allow unpacking as ``record, report``."""

        yield self.record
        yield self.report


@dataclass(frozen=True, slots=True)
class RecordIdentity:
    """The schema id and version a record was identified as.

    Attributes:
        schema_id: Registered schema id.
        version: Normalised record version.
    """

    schema_id: str
    version: str

    def to_dict(self) -> dict[str, str]:
        """Return the identity as a JSON-ready mapping.

        Returns:
            A mapping with ``schema_id`` and ``version``.
        """

        return {"schema_id": self.schema_id, "version": self.version}


class MigratorRegistry:
    """Thread-safe registry of record schemas and their migration edges.

    Most callers use the module-level functions, which act on
    :func:`default_registry`. Tests and embedders can build an isolated
    registry instead.
    """

    def __init__(self) -> None:
        """Create an empty registry."""

        self._lock = threading.RLock()
        self._schemas: dict[str, RecordSchema] = {}
        self._edges: dict[str, dict[str, dict[str, MigrationEdge]]] = {}

    def register_schema(
        self,
        schema_id: str,
        *,
        current_version: str | int,
        version_field: str = DEFAULT_VERSION_FIELD,
        version_when_missing: str | int | None = None,
        matcher: RecordMatcher | None = None,
    ) -> RecordSchema:
        """Register a record family before its migrators.

        Args:
            schema_id: Lower-case schema id, normally the packaged schema name.
            current_version: Version records migrate to by default.
            version_field: Top-level field that holds the record version.
            version_when_missing: Version assumed for records without the
                version field, or ``None`` to refuse them.
            matcher: Optional predicate that recognises records of this
                schema for :meth:`identify`.

        Returns:
            The registered schema.

        Raises:
            MigratorRegistrationError: If the id, versions or field are
                malformed, or the schema is already registered.
        """

        safe_id = _safe_schema_id(schema_id)
        current = _version_text(current_version)
        missing = (
            None
            if version_when_missing is None
            else _version_text(version_when_missing)
        )
        if (
            safe_id is None
            or current is None
            or (version_when_missing is not None and missing is None)
            or not isinstance(version_field, str)
            or _FIELD_RE.fullmatch(version_field) is None
            or (matcher is not None and not callable(matcher))
        ):
            raise MigratorRegistrationError("invalid_schema", schema_id=safe_id)
        schema = RecordSchema(
            schema_id=safe_id,
            current_version=current,
            version_field=version_field,
            version_when_missing=missing,
            matcher=matcher,
        )
        with self._lock:
            if safe_id in self._schemas:
                raise MigratorRegistrationError("duplicate_schema", schema_id=safe_id)
            self._schemas[safe_id] = schema
            self._edges[safe_id] = {}
        return schema

    def register_migrator(
        self,
        schema_id: str,
        from_version: str | int,
        to_version: str | int,
        migrator: Migrator,
        *,
        lossless: bool,
        fixture: str,
    ) -> MigrationEdge:
        """Register one migration edge for a registered schema.

        Args:
            schema_id: Registered schema id.
            from_version: Version the migrator accepts.
            to_version: Version the migrator produces.
            migrator: Function taking a deep-copied record and returning the
                migrated record. It must not perform I/O.
            lossless: Whether the step preserves every input value.
            fixture: Fixture directory relative to
                :data:`MIGRATION_FIXTURE_ROOT` holding ``before.json`` and
                ``after.json``.

        Returns:
            The registered edge.

        Raises:
            MigratorRegistrationError: If the schema is unknown, the edge is
                malformed, already registered, or would close a cycle.
        """

        safe_id = _safe_schema_id(schema_id)
        source = _version_text(from_version)
        target = _version_text(to_version)

        def reject(reason: str) -> MigratorRegistrationError:
            return MigratorRegistrationError(
                reason, schema_id=safe_id, from_version=source, to_version=target
            )

        if (
            source is None
            or target is None
            or not callable(migrator)
            or not isinstance(lossless, bool)
        ):
            raise reject("invalid_edge")
        if not _valid_fixture(fixture):
            raise reject("invalid_fixture")
        with self._lock:
            if safe_id is None or safe_id not in self._schemas:
                raise reject("unknown_schema")
            graph = self._edges[safe_id]
            if target in graph.get(source, {}):
                raise reject("duplicate_edge")
            if source == target or _reaches(graph, target, source):
                raise reject("cyclic_edge")
            edge = MigrationEdge(
                schema_id=safe_id,
                from_version=source,
                to_version=target,
                migrator=migrator,
                lossless=lossless,
                fixture=fixture,
            )
            graph.setdefault(source, {})[target] = edge
        return edge

    def schemas(self) -> tuple[RecordSchema, ...]:
        """Return every registered schema sorted by schema id.

        Returns:
            The registered schemas.
        """

        with self._lock:
            return tuple(self._schemas[key] for key in sorted(self._schemas))

    def schema(self, schema_id: str) -> RecordSchema:
        """Return one registered schema.

        Args:
            schema_id: Registered schema id.

        Returns:
            The registered schema.

        Raises:
            RecordMigrationRefused: With reason ``unknown_schema`` when the id
                is not registered.
        """

        safe_id = _safe_schema_id(schema_id)
        with self._lock:
            schema = self._schemas.get(safe_id) if safe_id is not None else None
        if schema is None:
            raise RecordMigrationRefused(
                MigrationRefusal("unknown_schema", schema_id=safe_id)
            )
        return schema

    def edges(self, schema_id: str | None = None) -> tuple[MigrationEdge, ...]:
        """Return registered edges in deterministic order.

        Args:
            schema_id: Optional schema id to filter by.

        Returns:
            Edges sorted by schema id, source version and target version.
        """

        with self._lock:
            selected = (
                [schema_id] if schema_id is not None else sorted(self._edges.keys())
            )
            result: list[MigrationEdge] = []
            for key in selected:
                graph = self._edges.get(key, {})
                for source in sorted(graph):
                    for target in sorted(graph[source]):
                        result.append(graph[source][target])
            return tuple(result)

    def migration_path(
        self,
        schema_id: str,
        from_version: str | int,
        to_version: str | int | None = None,
    ) -> tuple[MigrationEdge, ...]:
        """Resolve the unique chain of edges between two versions.

        Args:
            schema_id: Registered schema id.
            from_version: Version of the stored record.
            to_version: Target version; the schema's current version when
                omitted.

        Returns:
            The edges to apply in order. The tuple is empty when the record is
            already at the target version.

        Raises:
            RecordMigrationRefused: With reason ``unknown_schema``,
                ``invalid_version``, ``no_path`` or ``ambiguous_path``.
        """

        schema = self.schema(schema_id)
        source = _version_text(from_version)
        target = (
            schema.current_version if to_version is None else _version_text(to_version)
        )
        if source is None or target is None:
            raise RecordMigrationRefused(
                MigrationRefusal(
                    "invalid_version",
                    schema_id=schema.schema_id,
                    from_version=source,
                    to_version=target,
                )
            )
        if source == target:
            return ()
        with self._lock:
            routes = _routes(self._edges[schema.schema_id], source, target)
        if not routes:
            reason = "no_path"
        elif len(routes) > 1:
            reason = "ambiguous_path"
        else:
            return routes[0]
        raise RecordMigrationRefused(
            MigrationRefusal(
                reason,
                schema_id=schema.schema_id,
                from_version=source,
                to_version=target,
            )
        )

    def record_version(self, schema_id: str, record: Mapping[str, Any]) -> str:
        """Return the normalised version a record declares for a schema.

        Args:
            schema_id: Registered schema id.
            record: Stored record.

        Returns:
            The record version as text.

        Raises:
            RecordMigrationRefused: With reason ``unknown_schema``,
                ``invalid_record`` or ``invalid_version``.
        """

        schema = self.schema(schema_id)
        if not isinstance(record, Mapping):
            raise RecordMigrationRefused(
                MigrationRefusal("invalid_record", schema_id=schema.schema_id)
            )
        return _record_version(schema, record)

    def identify(self, record: Mapping[str, Any]) -> RecordIdentity:
        """Identify a record's schema id and version with registered matchers.

        Args:
            record: Stored record.

        Returns:
            The single matching schema id and the record's version.

        Raises:
            RecordMigrationRefused: With reason ``invalid_record`` for a
                non-mapping, ``unknown_schema`` when no matcher accepts the
                record, ``ambiguous_schema`` when several do, or
                ``invalid_version``.
        """

        if not isinstance(record, Mapping):
            raise RecordMigrationRefused(MigrationRefusal("invalid_record"))
        matches = [
            schema
            for schema in self.schemas()
            if schema.matcher is not None and _matches(schema.matcher, record)
        ]
        if not matches:
            version = _version_text(record.get(DEFAULT_VERSION_FIELD))
            raise RecordMigrationRefused(
                MigrationRefusal("unknown_schema", from_version=version)
            )
        if len(matches) > 1:
            raise RecordMigrationRefused(MigrationRefusal("ambiguous_schema"))
        schema = matches[0]
        return RecordIdentity(
            schema_id=schema.schema_id, version=_record_version(schema, record)
        )

    def migrate_record(
        self,
        record: Mapping[str, Any],
        *,
        schema_id: str | None = None,
        to_version: str | int | None = None,
    ) -> RecordMigrationResult:
        """Migrate one record in memory and report changed field paths.

        The input is copied before any migrator runs, so it is never modified.

        Args:
            record: Stored JSON-compatible record.
            schema_id: Registered schema id; identified with :meth:`identify`
                when omitted.
            to_version: Target version; the schema's current version when
                omitted.

        Returns:
            The migrated record and its value-free report. When the record is
            already at the target version the report has no steps and status
            ``current``.

        Raises:
            RecordMigrationRefused: When the record cannot be read, no unique
                path exists, a migrator fails, or a migrator returns a record
                that is not JSON-compatible or not at the edge's target version.
        """

        if schema_id is None:
            schema = self.schema(self.identify(record).schema_id)
        else:
            schema = self.schema(schema_id)
        payload = _json_copy(record)
        if payload is None:
            raise RecordMigrationRefused(
                MigrationRefusal("invalid_record", schema_id=schema.schema_id)
            )
        source_version = _record_version(schema, payload)
        chain = self.migration_path(schema.schema_id, source_version, to_version)
        current = payload
        steps: list[MigrationStepReport] = []
        for edge in chain:
            migrated = _run_edge(schema, edge, current)
            steps.append(
                MigrationStepReport(
                    from_version=edge.from_version,
                    to_version=edge.to_version,
                    lossless=edge.lossless,
                    fixture=edge.fixture,
                    fields=diff_field_paths(current, migrated),
                )
            )
            current = migrated
        report = RecordMigrationReport(
            schema_id=schema.schema_id,
            source_version=source_version,
            target_version=chain[-1].to_version if chain else source_version,
            steps=tuple(steps),
            fields=diff_field_paths(payload, current),
            source_digest=record_digest(payload),
            target_digest=record_digest(current),
        )
        return RecordMigrationResult(record=current, report=report)


_DEFAULT_REGISTRY: MigratorRegistry | None = None
_DEFAULT_LOCK = threading.Lock()


def default_registry() -> MigratorRegistry:
    """Return the process-wide registry with the built-in migrators loaded.

    Built-in registrations load lazily on first use; the modules that own the
    migrators are imported only when a migrator runs.

    Returns:
        The shared registry.
    """

    global _DEFAULT_REGISTRY
    with _DEFAULT_LOCK:
        if _DEFAULT_REGISTRY is None:
            registry = MigratorRegistry()
            for module_name in _BUILTIN_REGISTRATIONS:
                module = importlib.import_module(module_name)
                module.register_record_migrations(registry)
            _DEFAULT_REGISTRY = registry
        return _DEFAULT_REGISTRY


def register_schema(
    schema_id: str,
    *,
    current_version: str | int,
    version_field: str = DEFAULT_VERSION_FIELD,
    version_when_missing: str | int | None = None,
    matcher: RecordMatcher | None = None,
    registry: MigratorRegistry | None = None,
) -> RecordSchema:
    """Register a record family in the default or a given registry.

    Args:
        schema_id: Lower-case schema id, normally the packaged schema name.
        current_version: Version records migrate to by default.
        version_field: Top-level field that holds the record version.
        version_when_missing: Version assumed for records without the version
            field, or ``None`` to refuse them.
        matcher: Optional predicate that recognises records of this schema.
        registry: Registry to use instead of :func:`default_registry`.

    Returns:
        The registered schema.

    Raises:
        MigratorRegistrationError: If the registration is malformed or the
            schema id is already registered.
    """

    return (registry or default_registry()).register_schema(
        schema_id,
        current_version=current_version,
        version_field=version_field,
        version_when_missing=version_when_missing,
        matcher=matcher,
    )


def register_migrator(
    schema_id: str,
    from_version: str | int,
    to_version: str | int,
    migrator: Migrator,
    *,
    lossless: bool,
    fixture: str,
    registry: MigratorRegistry | None = None,
) -> MigrationEdge:
    """Register one migration edge in the default or a given registry.

    Args:
        schema_id: Registered schema id.
        from_version: Version the migrator accepts.
        to_version: Version the migrator produces.
        migrator: Function taking a deep-copied record and returning the
            migrated record.
        lossless: Whether the step preserves every input value.
        fixture: Fixture directory relative to :data:`MIGRATION_FIXTURE_ROOT`.
        registry: Registry to use instead of :func:`default_registry`.

    Returns:
        The registered edge.

    Raises:
        MigratorRegistrationError: If the schema is unknown, or the edge is
            malformed, duplicated, or would close a cycle.
    """

    return (registry or default_registry()).register_migrator(
        schema_id,
        from_version,
        to_version,
        migrator,
        lossless=lossless,
        fixture=fixture,
    )


def migration_path(
    schema_id: str,
    from_version: str | int,
    to_version: str | int | None = None,
    *,
    registry: MigratorRegistry | None = None,
) -> tuple[MigrationEdge, ...]:
    """Resolve the unique chain of edges between two versions of a schema.

    Args:
        schema_id: Registered schema id.
        from_version: Version of the stored record.
        to_version: Target version; the schema's current version when omitted.
        registry: Registry to use instead of :func:`default_registry`.

    Returns:
        The edges to apply in order; empty when the versions are equal.

    Raises:
        RecordMigrationRefused: With reason ``unknown_schema``,
            ``invalid_version``, ``no_path`` or ``ambiguous_path``.
    """

    return (registry or default_registry()).migration_path(
        schema_id, from_version, to_version
    )


def identify_record(
    record: Mapping[str, Any],
    *,
    registry: MigratorRegistry | None = None,
) -> RecordIdentity:
    """Identify a record's schema id and version with registered matchers.

    Args:
        record: Stored record.
        registry: Registry to use instead of :func:`default_registry`.

    Returns:
        The matching schema id and the record version.

    Raises:
        RecordMigrationRefused: When no single registered schema matches.
    """

    return (registry or default_registry()).identify(record)


def migrate_record(
    record: Mapping[str, Any],
    *,
    schema_id: str | None = None,
    to_version: str | int | None = None,
    registry: MigratorRegistry | None = None,
) -> RecordMigrationResult:
    """Migrate one record in memory and report changed field paths.

    Args:
        record: Stored JSON-compatible record. It is never modified.
        schema_id: Registered schema id; identified from the record's shape
            when omitted.
        to_version: Target version; the schema's current version when omitted.
        registry: Registry to use instead of :func:`default_registry`.

    Returns:
        The migrated record and its value-free report.

    Raises:
        RecordMigrationRefused: When the record cannot be migrated.
    """

    return (registry or default_registry()).migrate_record(
        record, schema_id=schema_id, to_version=to_version
    )


def record_digest(record: Any) -> str:
    """Return the SHA-256 digest of a record's canonical JSON form.

    Args:
        record: JSON-compatible value.

    Returns:
        ``sha256:`` followed by 64 lowercase hex characters.
    """

    return "sha256:" + hashlib.sha256(_canonical(record)).hexdigest()


def diff_field_paths(before: Any, after: Any) -> FieldChanges:
    """Return the value-free field paths that differ between two records.

    Paths are JSON Pointers. List indexes are collapsed to ``*``. A value that
    disappears from one path and reappears unchanged at exactly one other path
    is reported as moved; empty containers, empty strings, booleans and
    ``null`` are never matched as moves because they are too common to tell
    apart.

    Args:
        before: The older JSON-compatible record.
        after: The newer JSON-compatible record.

    Returns:
        Sorted added, dropped, moved and changed paths.
    """

    added: list[tuple[Any, ...]] = []
    dropped: list[tuple[Any, ...]] = []
    changed: list[tuple[Any, ...]] = []
    _walk(before, after, (), added, dropped, changed, 0)
    moves = _detect_moves(before, after, dropped, added)
    move_sources = {source for source, _ in moves}
    move_targets = {target for _, target in moves}
    return FieldChanges(
        added=_collapse(path for path in added if path not in move_targets),
        dropped=_collapse(path for path in dropped if path not in move_sources),
        moved=tuple(
            sorted(
                {
                    FieldMove(_pointer(source), _pointer(target))
                    for source, target in moves
                },
                key=lambda move: (move.from_path, move.to_path),
            )
        ),
        changed=_collapse(changed),
    )


def _run_edge(
    schema: RecordSchema, edge: MigrationEdge, record: dict[str, Any]
) -> dict[str, Any]:
    def refuse(reason: str, error_type: str | None = None) -> RecordMigrationRefused:
        return RecordMigrationRefused(
            MigrationRefusal(
                reason,
                schema_id=edge.schema_id,
                from_version=edge.from_version,
                to_version=edge.to_version,
                error_type=error_type,
            )
        )

    try:
        output = edge.migrator(copy.deepcopy(record))
    except Exception as exc:
        raise refuse("migrator_failed", type(exc).__name__) from None
    migrated = _json_copy(output) if isinstance(output, Mapping) else None
    if migrated is None:
        raise refuse("migrator_output_invalid")
    try:
        version = _record_version(schema, migrated)
    except RecordMigrationRefused:
        raise refuse("version_mismatch") from None
    if version != edge.to_version:
        raise refuse("version_mismatch")
    return migrated


def _record_version(schema: RecordSchema, record: Mapping[str, Any]) -> str:
    if schema.version_field not in record:
        if schema.version_when_missing is not None:
            return schema.version_when_missing
        raise RecordMigrationRefused(
            MigrationRefusal("invalid_version", schema_id=schema.schema_id)
        )
    version = _version_text(record.get(schema.version_field))
    if version is None:
        raise RecordMigrationRefused(
            MigrationRefusal("invalid_version", schema_id=schema.schema_id)
        )
    return version


def _routes(
    graph: Mapping[str, Mapping[str, MigrationEdge]], source: str, target: str
) -> list[tuple[MigrationEdge, ...]]:
    """Return up to two routes from ``source`` to ``target`` in an acyclic graph."""

    memo: dict[str, list[tuple[MigrationEdge, ...]]] = {}

    def visit(version: str) -> list[tuple[MigrationEdge, ...]]:
        if version == target:
            return [()]
        if version in memo:
            return memo[version]
        found: list[tuple[MigrationEdge, ...]] = []
        for following in sorted(graph.get(version, {})):
            for route in visit(following):
                found.append((graph[version][following], *route))
                if len(found) > 1:
                    break
            if len(found) > 1:
                break
        memo[version] = found
        return found

    return visit(source)


def _reaches(
    graph: Mapping[str, Mapping[str, MigrationEdge]], start: str, goal: str
) -> bool:
    seen = {start}
    pending = [start]
    while pending:
        version = pending.pop()
        if version == goal:
            return True
        for following in graph.get(version, {}):
            if following not in seen:
                seen.add(following)
                pending.append(following)
    return False


def _matches(matcher: RecordMatcher, record: Mapping[str, Any]) -> bool:
    try:
        return matcher(record) is True
    except Exception:
        return False


def _walk(
    before: Any,
    after: Any,
    path: tuple[Any, ...],
    added: list[tuple[Any, ...]],
    dropped: list[tuple[Any, ...]],
    changed: list[tuple[Any, ...]],
    depth: int,
) -> None:
    if depth >= _MAX_DIFF_DEPTH:
        if _canonical(before) != _canonical(after):
            changed.append(path)
        return
    if isinstance(before, dict) and isinstance(after, dict):
        for key in before.keys() - after.keys():
            dropped.append((*path, key))
        for key in after.keys() - before.keys():
            added.append((*path, key))
        for key in before.keys() & after.keys():
            _walk(
                before[key],
                after[key],
                (*path, key),
                added,
                dropped,
                changed,
                depth + 1,
            )
        return
    if isinstance(before, list) and isinstance(after, list):
        shared = min(len(before), len(after))
        for index in range(shared):
            _walk(
                before[index],
                after[index],
                (*path, index),
                added,
                dropped,
                changed,
                depth + 1,
            )
        for index in range(shared, len(before)):
            dropped.append((*path, index))
        for index in range(shared, len(after)):
            added.append((*path, index))
        return
    if type(before) is not type(after) or before != after:
        changed.append(path)


def _detect_moves(
    before: Any,
    after: Any,
    dropped: list[tuple[Any, ...]],
    added: list[tuple[Any, ...]],
) -> list[tuple[tuple[Any, ...], tuple[Any, ...]]]:
    sources: dict[str, list[tuple[Any, ...]]] = {}
    targets: dict[str, list[tuple[Any, ...]]] = {}
    for root in dropped:
        for path, value in _subtree(_lookup(before, root), root):
            if not _trivial(value):
                sources.setdefault(record_digest(value), []).append(path)
    for root in added:
        for path, value in _subtree(_lookup(after, root), root):
            if not _trivial(value):
                targets.setdefault(record_digest(value), []).append(path)
    candidates = sorted(
        (
            (paths[0], targets[digest][0])
            for digest, paths in sources.items()
            if len(paths) == 1 and len(targets.get(digest, ())) == 1
        ),
        key=lambda pair: (
            len(pair[0]),
            _pointer(pair[0], collapse=False),
            _pointer(pair[1], collapse=False),
        ),
    )
    moves: list[tuple[tuple[Any, ...], tuple[Any, ...]]] = []
    for source, target in candidates:
        if any(
            _nested(source, taken_source) or _nested(target, taken_target)
            for taken_source, taken_target in moves
        ):
            continue
        moves.append((source, target))
    return moves


def _subtree(
    value: Any, path: tuple[Any, ...], depth: int = 0
) -> Iterator[tuple[tuple[Any, ...], Any]]:
    yield path, value
    if depth >= _MAX_DIFF_DEPTH:
        return
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _subtree(value[key], (*path, key), depth + 1)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _subtree(item, (*path, index), depth + 1)


def _lookup(value: Any, path: tuple[Any, ...]) -> Any:
    for segment in path:
        value = value[segment]
    return value


def _nested(path: tuple[Any, ...], other: tuple[Any, ...]) -> bool:
    shorter, longer = sorted((path, other), key=len)
    return longer[: len(shorter)] == shorter


def _trivial(value: Any) -> bool:
    return value is None or isinstance(value, bool) or value in ("", [], {})


def _collapse(paths: Any) -> tuple[str, ...]:
    return tuple(sorted({_pointer(path) for path in paths}))


def _pointer(path: tuple[Any, ...], *, collapse: bool = True) -> str:
    if not path:
        return ""
    segments = (
        ("*" if collapse else str(segment))
        if isinstance(segment, int)
        else str(segment).replace("~", "~0").replace("/", "~1")
        for segment in path
    )
    return "/" + "/".join(segments)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_copy(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        copied = json.loads(_canonical(dict(value)))
    except (TypeError, ValueError, RecursionError):
        return None
    return copied if isinstance(copied, dict) else None


def _version_text(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value) if value >= 0 else None
    if isinstance(value, str) and _VERSION_RE.fullmatch(value):
        return value
    return None


def _safe_schema_id(value: Any) -> str | None:
    if isinstance(value, str) and _SCHEMA_ID_RE.fullmatch(value):
        return value
    return None


def _valid_fixture(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    segments = value.split("/")
    return all(
        _FIXTURE_SEGMENT_RE.fullmatch(segment) and segment not in {".", ".."}
        for segment in segments
    )


def _describe(
    schema_id: str | None, from_version: str | None, to_version: str | None
) -> str:
    parts = []
    if schema_id is not None:
        parts.append(f"schema {schema_id}")
    if from_version is not None:
        parts.append(f"from {from_version}")
    if to_version is not None:
        parts.append(f"to {to_version}")
    return f": {' '.join(parts)}" if parts else ""


__all__ = [
    "DEFAULT_VERSION_FIELD",
    "MIGRATION_FIXTURE_ROOT",
    "RECORD_MIGRATION_REPORT_SCHEMA_VERSION",
    "REFUSAL_REASONS",
    "REGISTRATION_REASONS",
    "FieldChanges",
    "FieldMove",
    "MigrationEdge",
    "MigrationRefusal",
    "MigrationStepReport",
    "Migrator",
    "MigratorRegistrationError",
    "MigratorRegistry",
    "RecordIdentity",
    "RecordMatcher",
    "RecordMigrationRefused",
    "RecordMigrationReport",
    "RecordMigrationResult",
    "RecordSchema",
    "default_registry",
    "diff_field_paths",
    "identify_record",
    "migrate_record",
    "migration_path",
    "record_digest",
    "register_migrator",
    "register_schema",
]
