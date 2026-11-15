"""Release-wide registry of durable service schemas and dry-run migration plans.

Every durable schema a service instance owns is registered here once: the
owning engine, the current version, the source versions this binary upgrades
from, what the previous release's binary does with the current schema, the
migration trigger and command, the deployment posture, the recovery point a
rollback needs, and a rollback class.  Components that open pull requests will
add are registered as ``not_present`` placeholders so the release evidence
names them and their owners must classify them when they land.

Rollback classes, from least to most severe:

* ``reversible``: a tested in-place down path exists.  Nothing uses it today;
  destructive downgrade is never implied.
* ``backward_compatible``: the previous release's binary keeps working
  against the current schema (or never reads the component at all).
* ``restore_required``: forward-only.  Rolling back means restoring a verified
  recovery point taken before the upgrade
  (:func:`openmed.service.recovery.create_recovery_point`).
* ``release_blocking``: irreversible.  Neither the previous binary nor a
  recovery point can bring the previous state back, so the release gate
  refuses it.

The module also observes live state read-only and builds a versioned,
value-free :class:`MigrationPlan` (a dry run) that names source and target
versions, ordered actions, compatibility classes, required recovery points,
and preflight checks for free space, backup readiness, encryption-key
availability, version compatibility, cross-component consistency, and
deployment posture.  Observations, plans, and errors carry only versions,
categories, aggregate counts, schema-metadata digests, and controlled reason
codes: never paths, record contents, identifiers, or key material.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from openmed.__about__ import __version__
from openmed.core.audit_chain import CHAIN_VERSION
from openmed.core.registry_slots import REGISTRY_STATE_SCHEMA_VERSION
from openmed.service.recovery import (
    JOB_STORE_LAYOUT_VERSION,
    RECOVERY_FORMAT,
    RECOVERY_FORMAT_VERSION,
    RECOVERY_METADATA_DIRECTORY,
    ServiceStateSources,
)
from openmed.structured.store.migrations import MIGRATIONS
from openmed.structured.store.postgres import POSTGRES_MIGRATIONS

__all__ = [
    "AUDIT_CHAIN",
    "AUDIT_CONTINUITY",
    "JOB_STORE",
    "JOURNEY_ARTIFACT_MARKER",
    "JOURNEY_POSTGRES",
    "JOURNEY_SQLITE",
    "MODEL_ACTIVATION",
    "MODEL_REGISTRY",
    "PLAN_SCHEMA",
    "PREVIOUS_RELEASE",
    "PROTECTED_PAYLOADS",
    "RECOVERY_MANIFEST",
    "REGISTRY_SCHEMA",
    "RELEASED_CHECKSUMS",
    "ComponentObservation",
    "DeploymentPosture",
    "DurableSchema",
    "MigrationPlan",
    "MigrationStep",
    "MigrationTrigger",
    "ObservedState",
    "PlannedAction",
    "PreflightCheck",
    "Presence",
    "PreviousBinary",
    "RecoveryPointKind",
    "RollbackClass",
    "durable_schemas",
    "get_schema",
    "observe_artifact_marker",
    "observe_audit_chains",
    "observe_job_store",
    "observe_journey_postgres",
    "observe_journey_sqlite",
    "observe_model_registry",
    "observe_recovery_point",
    "observe_service_state",
    "plan_service_migrations",
    "registry_digest",
    "registry_document",
    "registry_findings",
]

REGISTRY_SCHEMA = "openmed.service.durable-schemas.v1"
PLAN_SCHEMA = "openmed.service.migration-plan.v1"
PREVIOUS_RELEASE = "3.2.0"
"""Release whose binary a rollback returns to; compatibility is judged on it."""

JOURNEY_SQLITE = "journey_sqlite"
JOURNEY_POSTGRES = "journey_postgres"
JOURNEY_ARTIFACT_MARKER = "journey_artifact_marker"
JOB_STORE = "job_store"
MODEL_REGISTRY = "model_registry"
AUDIT_CHAIN = "audit_chain"
RECOVERY_MANIFEST = "recovery_manifest"
PROTECTED_PAYLOADS = "protected_payload_store"
MODEL_ACTIVATION = "model_activation_journal"
AUDIT_CONTINUITY = "audit_continuity"

RELEASED_CHECKSUMS: Mapping[str, Mapping[int, str]] = {
    JOURNEY_SQLITE: {
        1: "a82e022ed86d5372803ad9ca0d66a5833b492b720d9c6a942336168a0de07956",
        2: "d5ffd6536f534ebfd979d4d29a2af8999df86ef088942db5f26006d11d918064",
        3: "4391d0cd413dcfd2965ce7043a7a49c19185ccdd6b1c7ea6ac2f73e37cd5f7d1",
    },
    JOURNEY_POSTGRES: {
        1: "f4256b1fd10cbf993e742c6c21e47fbc16df2d3a3df57d890ffab9b3ded76e12",
        2: "8fd41077190f346b3c6b09a80a5aa048ddd1194bde4ba5742ea1321800ea2c6f",
        3: "ca3dcc63c7c61395b9f07cc1bad9d27a1100285ff659b49504bc1c1a77ee0e38",
    },
}
"""Checksums of the SQL migrations shipped in :data:`PREVIOUS_RELEASE`.

Released migrations are immutable: editing one makes every deployed store
drift.  :func:`registry_findings` fails the release when a released step's
checksum changes or disappears.
"""

_MARKER_NAME = "encryption.json"
_MARKER_FIELDS = frozenset(
    {"envelope_version", "key_purpose", "key_verifiers", "schema_version", "state"}
)
_MARKER_SCHEMA_VERSION = 1
_ENCRYPTION_STATES = frozenset({"plaintext", "migrating", "encrypted"})
_DESCRIPTOR_NAME = "recovery-point.json"
_STAGING_DESCRIPTOR = "staging.json"
_ACTIVATION_RECORD = "activation.json"
_JOURNEY_DATABASE = "journey.sqlite3"
_ARTIFACTS = "artifacts"
_MAX_DOCUMENT_BYTES = 256 * 1024 * 1024
_MAX_DESCRIPTOR_BYTES = 64 * 1024
_SCHEMA_RE = re.compile(r"[a-z][a-z0-9_]{0,62}\Z")


class Presence(str, Enum):
    """Whether a registered component exists in this codebase."""

    PRESENT = "present"
    NOT_PRESENT = "not_present"


class RollbackClass(str, Enum):
    """How a schema change can be rolled back; see the module docstring."""

    REVERSIBLE = "reversible"
    BACKWARD_COMPATIBLE = "backward_compatible"
    RESTORE_REQUIRED = "restore_required"
    RELEASE_BLOCKING = "release_blocking"


_SEVERITY = {
    RollbackClass.REVERSIBLE: 0,
    RollbackClass.BACKWARD_COMPATIBLE: 1,
    RollbackClass.RESTORE_REQUIRED: 2,
    RollbackClass.RELEASE_BLOCKING: 3,
}


class MigrationTrigger(str, Enum):
    """What applies a component's forward migration."""

    AUTOMATIC_ON_OPEN = "automatic_on_open"
    EXPLICIT_COMMAND = "explicit_command"
    NONE = "none"


class DeploymentPosture(str, Enum):
    """Expected downtime or mixed-version posture while migrating."""

    STOP_WRITERS = "stop_writers"
    MIXED_VERSION_SAFE = "mixed_version_safe"


class RecoveryPointKind(str, Enum):
    """The recovery point a restore-based rollback depends on."""

    SERVICE_RECOVERY_POINT = "service_recovery_point"
    OPERATOR_DATABASE_BACKUP = "operator_database_backup"
    OPERATOR_FILE_COPY = "operator_file_copy"
    NOT_REQUIRED = "not_required"


class PreviousBinary(str, Enum):
    """What the previous release's binary does with the current schema."""

    READS = "reads"
    REFUSES = "refuses"
    IGNORES = "ignores"


class ObservedState(str, Enum):
    """Read-only classification of one component's persisted schema."""

    ABSENT = "absent"
    CURRENT = "current"
    UPGRADE_PENDING = "upgrade_pending"
    EXPLICIT_MIGRATION_REQUIRED = "explicit_migration_required"
    NEWER_THAN_BINARY = "newer_than_binary"
    DRIFTED = "drifted"
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"
    NOT_INSPECTED = "not_inspected"
    NOT_PRESENT = "not_present"


_REFUSING_STATES = frozenset(
    {
        ObservedState.NEWER_THAN_BINARY,
        ObservedState.DRIFTED,
        ObservedState.INVALID,
        ObservedState.UNAVAILABLE,
    }
)


@dataclass(frozen=True)
class MigrationStep:
    """One ordered schema version of a component.

    Attributes:
        version: Schema version this step produces.
        name: Controlled step name.
        checksum: SHA-256 of the step's statements for SQL engines, else
            ``None``.
        released: Whether :data:`PREVIOUS_RELEASE` (or earlier) shipped it.
        rollback_class: Classification of a step new in this release;
            ``None`` for released steps and for unclassified new steps.
    """

    version: int
    name: str
    checksum: str | None
    released: bool
    rollback_class: RollbackClass | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, value-free mapping."""

        return {
            "checksum": self.checksum,
            "name": self.name,
            "released": self.released,
            "rollback_class": (
                None if self.rollback_class is None else self.rollback_class.value
            ),
            "version": self.version,
        }


@dataclass(frozen=True)
class DurableSchema:
    """Registry entry for one durable service schema.

    Attributes:
        component: Stable component identifier.
        owner: Dotted name of the module or class that owns the native
            migration engine.
        presence: Whether the component exists in this codebase.
        recovery_category: Recovery-point component category, when the
            component is captured by service recovery points.
        current_version: Version this binary writes.
        source_versions: Versions this binary upgrades from or reads; ``0``
            means the state does not exist yet (a clean install).
        readable_versions: Versions this binary reads without migrating.
        restorable_versions: Source versions a service recovery point can
            capture and restore.
        previous_release_version: Version :data:`PREVIOUS_RELEASE` wrote, or
            ``None`` when the component is new in this release.
        previous_binary: What the previous release's binary does with the
            current version.
        previous_binary_code: Controlled code the previous binary reports
            when it refuses the current version.
        trigger: What applies the forward migration.
        command: Fixed, path-free description of the migration command.
        posture: Expected downtime or mixed-version posture.
        recovery_point: Recovery point a restore-based rollback needs.
        refusal_codes: Controlled codes the native engine reports for
            ``drift`` and ``unknown_version``.
        steps: Ordered schema versions with their classification.
        tracking_issue: Issue that will add a ``not_present`` component.
        pending_module: Module that will host a ``not_present`` component;
            its arrival without a registration fails the registry tests.
    """

    component: str
    owner: str
    presence: Presence = Presence.PRESENT
    recovery_category: str | None = None
    current_version: int | None = None
    source_versions: tuple[int, ...] = ()
    readable_versions: tuple[int, ...] = ()
    restorable_versions: tuple[int, ...] = ()
    previous_release_version: int | None = None
    previous_binary: PreviousBinary | None = None
    previous_binary_code: str | None = None
    trigger: MigrationTrigger | None = None
    command: str | None = None
    posture: DeploymentPosture | None = None
    recovery_point: RecoveryPointKind | None = None
    refusal_codes: Mapping[str, str] = field(default_factory=dict)
    steps: tuple[MigrationStep, ...] = ()
    tracking_issue: int | None = None
    pending_module: str | None = None

    @property
    def new_steps(self) -> tuple[MigrationStep, ...]:
        """Return the steps this release adds."""

        return tuple(step for step in self.steps if not step.released)

    @property
    def rollback_class(self) -> RollbackClass | None:
        """Return the rollback class of the previous-to-current transition.

        ``None`` means the component is not present or is unclassified, which
        the release gate refuses for a present component.
        """

        if self.presence is not Presence.PRESENT or self.previous_binary is None:
            return None
        new = self.new_steps
        if any(step.rollback_class is None for step in new):
            return None
        if not new:
            if self.previous_binary is PreviousBinary.REFUSES:
                return None
            return RollbackClass.BACKWARD_COMPATIBLE
        worst = max(
            (step.rollback_class for step in new if step.rollback_class is not None),
            key=lambda item: _SEVERITY[item],
        )
        if worst is RollbackClass.RESTORE_REQUIRED and self.recovery_point not in {
            RecoveryPointKind.SERVICE_RECOVERY_POINT,
            RecoveryPointKind.OPERATOR_DATABASE_BACKUP,
        }:
            return RollbackClass.RELEASE_BLOCKING
        if (
            _SEVERITY[worst] < _SEVERITY[RollbackClass.RESTORE_REQUIRED]
            and self.previous_binary is PreviousBinary.REFUSES
        ):
            return None
        return worst

    def compatibility_window(self) -> dict[str, Any]:
        """Return the declared compatibility window as a value-free mapping."""

        return {
            "previous_binary": (
                None if self.previous_binary is None else self.previous_binary.value
            ),
            "previous_binary_code": self.previous_binary_code,
            "previous_release": PREVIOUS_RELEASE,
            "previous_release_version": self.previous_release_version,
            "readable_versions": list(self.readable_versions),
            "source_versions": list(self.source_versions),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, value-free mapping."""

        rollback = self.rollback_class
        return {
            "command": self.command,
            "compatibility_window": self.compatibility_window(),
            "component": self.component,
            "current_version": self.current_version,
            "owner": self.owner,
            "pending_module": self.pending_module,
            "posture": None if self.posture is None else self.posture.value,
            "presence": self.presence.value,
            "recovery_category": self.recovery_category,
            "recovery_point": (
                None if self.recovery_point is None else self.recovery_point.value
            ),
            "refusal_codes": dict(sorted(self.refusal_codes.items())),
            "restorable_versions": list(self.restorable_versions),
            "rollback_class": None if rollback is None else rollback.value,
            "steps": [step.to_dict() for step in self.steps],
            "tracking_issue": self.tracking_issue,
            "trigger": None if self.trigger is None else self.trigger.value,
        }


# Classification of every step new in this release.  A new step without an
# entry here is unclassified and blocks the release gate.
_NEW_STEP_CLASSES: Mapping[tuple[str, int], RollbackClass] = {
    # Store encryption state: the 3.2.0 engine refuses schema 4.
    (JOURNEY_SQLITE, 4): RollbackClass.RESTORE_REQUIRED,
    # The 3.2.0 artifact reader cannot verify sealed blobs.
    (JOURNEY_ARTIFACT_MARKER, 1): RollbackClass.RESTORE_REQUIRED,
    # The 3.2.0 binary never reads recovery points.
    (RECOVERY_MANIFEST, 1): RollbackClass.BACKWARD_COMPATIBLE,
}


def _sql_steps(
    component: str, chain: Sequence[Any], previous: int
) -> tuple[MigrationStep, ...]:
    return tuple(
        MigrationStep(
            version=migration.version,
            name=migration.name,
            checksum=migration.checksum,
            released=migration.version <= previous,
            rollback_class=(
                None
                if migration.version <= previous
                else _NEW_STEP_CLASSES.get((component, migration.version))
            ),
        )
        for migration in chain
    )


def _document_steps(
    component: str, names: Sequence[tuple[int, str]], previous: int | None
) -> tuple[MigrationStep, ...]:
    return tuple(
        MigrationStep(
            version=version,
            name=name,
            checksum=None,
            released=previous is not None and version <= previous,
            rollback_class=(
                None
                if previous is not None and version <= previous
                else _NEW_STEP_CLASSES.get((component, version))
            ),
        )
        for version, name in names
    )


def durable_schemas() -> tuple[DurableSchema, ...]:
    """Return every registered durable schema in a stable order.

    SQL steps are derived from the native migration chains at call time, so
    a migration added without a classification shows up as unclassified.

    Returns:
        Registry entries, present components first.
    """

    sqlite_latest = MIGRATIONS[-1].version
    postgres_latest = POSTGRES_MIGRATIONS[-1].version
    sqlite_previous = 3
    postgres_previous = 3
    return (
        DurableSchema(
            component=JOURNEY_SQLITE,
            owner="openmed.structured.store.local.SQLiteJourneyStore",
            recovery_category="journey_metadata",
            current_version=sqlite_latest,
            source_versions=tuple(range(0, sqlite_latest + 1)),
            readable_versions=(sqlite_latest,),
            restorable_versions=tuple(range(1, sqlite_latest + 1)),
            previous_release_version=sqlite_previous,
            previous_binary=(
                PreviousBinary.REFUSES
                if sqlite_latest > sqlite_previous
                else PreviousBinary.READS
            ),
            previous_binary_code=(
                "schema_unsupported" if sqlite_latest > sqlite_previous else None
            ),
            trigger=MigrationTrigger.AUTOMATIC_ON_OPEN,
            command="automatic when LocalJourneyStore or SQLiteJourneyStore opens",
            posture=DeploymentPosture.STOP_WRITERS,
            recovery_point=RecoveryPointKind.SERVICE_RECOVERY_POINT,
            refusal_codes={
                "drift": "schema_unsupported",
                "unknown_version": "schema_unsupported",
            },
            steps=_sql_steps(JOURNEY_SQLITE, MIGRATIONS, sqlite_previous),
        ),
        DurableSchema(
            component=JOURNEY_POSTGRES,
            owner="openmed.structured.store.postgres.PostgresJourneyStore",
            current_version=postgres_latest,
            source_versions=tuple(range(0, postgres_latest + 1)),
            readable_versions=(postgres_latest,),
            previous_release_version=postgres_previous,
            previous_binary=(
                PreviousBinary.REFUSES
                if postgres_latest > postgres_previous
                else PreviousBinary.READS
            ),
            previous_binary_code=(
                "schema_unsupported" if postgres_latest > postgres_previous else None
            ),
            trigger=MigrationTrigger.EXPLICIT_COMMAND,
            command=(
                "python -m openmed.deploy.journey migrate "
                "(Helm journey-migration-job pre-install/pre-upgrade hook)"
            ),
            posture=DeploymentPosture.MIXED_VERSION_SAFE,
            recovery_point=RecoveryPointKind.OPERATOR_DATABASE_BACKUP,
            refusal_codes={
                "drift": "schema_unsupported",
                "unknown_version": "schema_unsupported",
            },
            steps=_sql_steps(JOURNEY_POSTGRES, POSTGRES_MIGRATIONS, postgres_previous),
        ),
        DurableSchema(
            component=JOURNEY_ARTIFACT_MARKER,
            owner="openmed.structured.store.local.LocalArtifactStore",
            recovery_category="journey_encryption",
            current_version=_MARKER_SCHEMA_VERSION,
            source_versions=(0, _MARKER_SCHEMA_VERSION),
            readable_versions=(_MARKER_SCHEMA_VERSION,),
            restorable_versions=(0, _MARKER_SCHEMA_VERSION),
            previous_release_version=0,
            previous_binary=PreviousBinary.REFUSES,
            previous_binary_code="artifact_integrity_failed",
            trigger=MigrationTrigger.AUTOMATIC_ON_OPEN,
            command=(
                "automatic when the artifact store opens; existing plaintext "
                "blobs need LocalJourneyStore.migrate_to_encrypted"
            ),
            posture=DeploymentPosture.STOP_WRITERS,
            recovery_point=RecoveryPointKind.SERVICE_RECOVERY_POINT,
            refusal_codes={
                "drift": "encryption_state_invalid",
                "unknown_version": "encryption_state_invalid",
            },
            steps=_document_steps(
                JOURNEY_ARTIFACT_MARKER, ((1, "encryption_marker"),), 0
            ),
        ),
        DurableSchema(
            component=JOB_STORE,
            owner="openmed.service.jobs.LocalJobStore",
            recovery_category="job_metadata",
            current_version=JOB_STORE_LAYOUT_VERSION,
            source_versions=(0, JOB_STORE_LAYOUT_VERSION),
            readable_versions=(JOB_STORE_LAYOUT_VERSION,),
            restorable_versions=(JOB_STORE_LAYOUT_VERSION,),
            previous_release_version=1,
            previous_binary=PreviousBinary.READS,
            trigger=MigrationTrigger.AUTOMATIC_ON_OPEN,
            command="automatic when the async job queue starts",
            posture=DeploymentPosture.STOP_WRITERS,
            recovery_point=RecoveryPointKind.SERVICE_RECOVERY_POINT,
            refusal_codes={
                "drift": "job_store_unknown_fields",
                "unknown_version": "job_store_schema_unsupported",
            },
            steps=_document_steps(JOB_STORE, ((1, "jobs_object"),), 1),
        ),
        DurableSchema(
            component=MODEL_REGISTRY,
            owner="openmed.core.registry_slots",
            recovery_category="model_registry",
            current_version=REGISTRY_STATE_SCHEMA_VERSION,
            source_versions=(1, REGISTRY_STATE_SCHEMA_VERSION),
            readable_versions=(REGISTRY_STATE_SCHEMA_VERSION,),
            restorable_versions=(REGISTRY_STATE_SCHEMA_VERSION,),
            previous_release_version=2,
            previous_binary=PreviousBinary.READS,
            trigger=MigrationTrigger.EXPLICIT_COMMAND,
            command="python scripts/release/registry_ctl.py migrate",
            posture=DeploymentPosture.STOP_WRITERS,
            recovery_point=RecoveryPointKind.SERVICE_RECOVERY_POINT,
            refusal_codes={
                "drift": "registry_state_invalid",
                "explicit": "registry_migration_required",
                "unknown_version": "registry_schema_unsupported",
            },
            steps=_document_steps(
                MODEL_REGISTRY, ((1, "family_keyed"), (2, "slot_keyed")), 2
            ),
        ),
        DurableSchema(
            component=AUDIT_CHAIN,
            owner="openmed.core.audit_chain.AuditChain",
            recovery_category="audit_chain",
            current_version=CHAIN_VERSION,
            source_versions=(0, CHAIN_VERSION),
            readable_versions=(CHAIN_VERSION,),
            restorable_versions=(CHAIN_VERSION,),
            previous_release_version=1,
            previous_binary=PreviousBinary.READS,
            trigger=MigrationTrigger.NONE,
            command="none; version 1 is the only audit-chain format",
            posture=DeploymentPosture.MIXED_VERSION_SAFE,
            recovery_point=RecoveryPointKind.SERVICE_RECOVERY_POINT,
            refusal_codes={
                "drift": "audit_chain_invalid",
                "unknown_version": "audit_chain_version_unsupported",
            },
            steps=_document_steps(AUDIT_CHAIN, ((1, "hash_chain"),), 1),
        ),
        DurableSchema(
            component=RECOVERY_MANIFEST,
            owner="openmed.service.recovery",
            current_version=RECOVERY_FORMAT_VERSION,
            source_versions=(0, RECOVERY_FORMAT_VERSION),
            readable_versions=(RECOVERY_FORMAT_VERSION,),
            previous_release_version=None,
            previous_binary=PreviousBinary.IGNORES,
            trigger=MigrationTrigger.NONE,
            command="written by openmed recovery backup; never migrated in place",
            posture=DeploymentPosture.MIXED_VERSION_SAFE,
            recovery_point=RecoveryPointKind.NOT_REQUIRED,
            refusal_codes={
                "drift": "manifest_authentication_failed",
                "unknown_version": "format_unsupported",
            },
            steps=_document_steps(RECOVERY_MANIFEST, ((1, "recovery_point_v1"),), None),
        ),
        DurableSchema(
            component=PROTECTED_PAYLOADS,
            owner="openmed.service.job_payloads",
            presence=Presence.NOT_PRESENT,
            tracking_issue=4918,
            pending_module="openmed.service.job_payloads",
        ),
        DurableSchema(
            component=MODEL_ACTIVATION,
            owner="openmed.service.model_activation",
            presence=Presence.NOT_PRESENT,
            tracking_issue=5066,
            pending_module="openmed.service.model_activation",
        ),
        DurableSchema(
            component=AUDIT_CONTINUITY,
            owner="openmed.core.audit_continuity",
            presence=Presence.NOT_PRESENT,
            tracking_issue=3599,
            pending_module="openmed.core.audit_continuity",
        ),
    )


def get_schema(component: str) -> DurableSchema:
    """Return one registry entry.

    Args:
        component: Component identifier.

    Returns:
        The registry entry.

    Raises:
        KeyError: If the component is not registered.
    """

    for schema in durable_schemas():
        if schema.component == component:
            return schema
    raise KeyError("component is not registered")


def registry_findings(
    schemas: Sequence[DurableSchema] | None = None,
) -> tuple[str, ...]:
    """Return controlled codes for every unclassified or drifted registry path.

    A finding names an unclassified rollback class, an unclassified new step,
    a released step whose checksum changed, or a released step that no longer
    exists.  An empty result means every migration and rollback path is
    classified.

    Args:
        schemas: Registry entries; defaults to :func:`durable_schemas`.

    Returns:
        Sorted ``component:finding`` codes.
    """

    selected = durable_schemas() if schemas is None else tuple(schemas)
    findings: set[str] = set()
    for schema in selected:
        if schema.presence is not Presence.PRESENT:
            continue
        if schema.rollback_class is None:
            findings.add(f"{schema.component}:rollback_unclassified")
        for step in schema.new_steps:
            if step.rollback_class is None:
                findings.add(f"{schema.component}:step_{step.version}_unclassified")
        released = RELEASED_CHECKSUMS.get(schema.component, {})
        by_version = {step.version: step for step in schema.steps}
        for version, checksum in released.items():
            step = by_version.get(version)
            if step is None:
                findings.add(f"{schema.component}:released_step_{version}_missing")
            elif step.checksum != checksum:
                findings.add(f"{schema.component}:released_step_{version}_changed")
    return tuple(sorted(findings))


def registry_document() -> dict[str, Any]:
    """Return the value-free compatibility matrix of the registry.

    Returns:
        A JSON-safe mapping with every component and any findings.
    """

    schemas = durable_schemas()
    return {
        "components": [schema.to_dict() for schema in schemas],
        "findings": list(registry_findings(schemas)),
        "previous_release": PREVIOUS_RELEASE,
        "release": str(__version__),
        "schema": REGISTRY_SCHEMA,
    }


def registry_digest() -> str:
    """Return the SHA-256 digest of the canonical registry document."""

    return _digest(_canonical_json(registry_document()))


# ---------------------------------------------------------------------------
# Read-only observation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentObservation:
    """Read-only, value-free view of one component's persisted schema.

    Attributes:
        component: Component identifier.
        state: Classification of the persisted schema.
        version: Observed schema version; ``0`` when no state exists yet.
        code: Controlled reason code for a refusing or pending state.
        digest: Digest of schema metadata only (versions, step names,
            checksums, table names), never of records.
        counts: Aggregate counters, for example record or revision counts.
        attributes: Controlled categories, for example an encryption state.
    """

    component: str
    state: ObservedState
    version: int | None = None
    code: str | None = None
    digest: str | None = None
    counts: Mapping[str, int] = field(default_factory=dict)
    attributes: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, value-free mapping."""

        return {
            "attributes": dict(sorted(self.attributes.items())),
            "code": self.code,
            "component": self.component,
            "counts": dict(sorted(self.counts.items())),
            "digest": self.digest,
            "state": self.state.value,
            "version": self.version,
        }


def _classify_chain(
    rows: Sequence[tuple[int, str, str]], chain: Sequence[Any]
) -> tuple[ObservedState, int, str | None]:
    """Mirror the native engines' order: newer first, then drift."""

    latest = chain[-1].version
    known = {item.version: (item.name, item.checksum) for item in chain}
    versions = [row[0] for row in rows]
    if not versions:
        return ObservedState.UPGRADE_PENDING, 0, "migration_pending"
    highest = max(versions)
    if highest > latest:
        return ObservedState.NEWER_THAN_BINARY, highest, "schema_newer_than_binary"
    if any(known.get(version) != (name, checksum) for version, name, checksum in rows):
        return ObservedState.DRIFTED, highest, "schema_drifted"
    if versions != list(range(1, highest + 1)):
        return ObservedState.DRIFTED, highest, "schema_history_gap"
    if highest < latest:
        return ObservedState.UPGRADE_PENDING, highest, "migration_pending"
    return ObservedState.CURRENT, highest, None


def _schema_digest(rows: Sequence[tuple[int, str, str]], tables: Sequence[str]) -> str:
    # An empty history table is the engines' bootstrap, not a schema version.
    visible = sorted(
        table
        for table in tables
        if table != "sqlite_sequence" and (rows or table != "schema_migrations")
    )
    return _digest(
        _canonical_json(
            {
                "migrations": [list(row) for row in rows],
                "tables": visible,
            }
        )
    )


def observe_journey_sqlite(database: Path | None) -> ComponentObservation:
    """Observe a local Journey SQLite store without migrating or creating it.

    Args:
        database: Path of ``journey.sqlite3``; ``None`` when not configured.

    Returns:
        The observation.
    """

    if database is None or not database.exists():
        return ComponentObservation(
            JOURNEY_SQLITE, ObservedState.ABSENT, 0, digest=_schema_digest((), ())
        )
    if database.is_symlink() or not database.is_file():
        return ComponentObservation(
            JOURNEY_SQLITE, ObservedState.INVALID, None, "journey_store_invalid"
        )
    try:
        # ``mode=rw`` never creates a database; only SELECTs run.
        connection = sqlite3.connect(
            f"{database.resolve().as_uri()}?mode=rw",
            uri=True,
            timeout=5.0,
            isolation_level=None,
        )
    except (OSError, sqlite3.Error, ValueError):
        return ComponentObservation(
            JOURNEY_SQLITE, ObservedState.UNAVAILABLE, None, "journey_store_unavailable"
        )
    try:
        tables = sorted(
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        )
        rows: tuple[tuple[int, str, str], ...] = ()
        if "schema_migrations" in tables:
            rows = tuple(
                (int(row[0]), str(row[1]), str(row[2]))
                for row in connection.execute(
                    "SELECT version, name, checksum FROM schema_migrations "
                    "ORDER BY version"
                )
            )
        revision = 0
        if "store_revisions" in tables:
            value = connection.execute(
                "SELECT MAX(revision) FROM store_revisions"
            ).fetchone()[0]
            revision = 0 if value is None else int(value)
        encryption = "none"
        if "store_encryption" in tables:
            row = connection.execute(
                "SELECT state FROM store_encryption WHERE singleton = 1"
            ).fetchone()
            if row is not None and str(row[0]) in _ENCRYPTION_STATES:
                encryption = str(row[0])
    except (sqlite3.Error, ValueError, TypeError):
        return ComponentObservation(
            JOURNEY_SQLITE, ObservedState.UNAVAILABLE, None, "journey_store_unavailable"
        )
    finally:
        connection.close()
    if "schema_migrations" not in tables and set(tables) - {"sqlite_sequence"}:
        return ComponentObservation(
            JOURNEY_SQLITE, ObservedState.INVALID, None, "schema_history_missing"
        )
    state, version, code = _classify_chain(rows, MIGRATIONS)
    return ComponentObservation(
        JOURNEY_SQLITE,
        state,
        version,
        code,
        _schema_digest(rows, tables),
        {"revisions": revision},
        {"encryption_state": encryption},
    )


def observe_journey_postgres(
    connection: Any | None, schema: str = "openmed_journey"
) -> ComponentObservation:
    """Observe a PostgreSQL Journey schema with SELECT statements only.

    Args:
        connection: Dedicated DB-API connection, or ``None`` when not
            supplied.  The caller keeps ownership and closes it.
        schema: Controlled PostgreSQL schema identifier.

    Returns:
        The observation.

    Raises:
        ValueError: If ``schema`` is not a controlled identifier.
    """

    if _SCHEMA_RE.fullmatch(schema) is None:
        raise ValueError("PostgreSQL schema must be a controlled identifier")
    if connection is None:
        return ComponentObservation(
            JOURNEY_POSTGRES,
            ObservedState.NOT_INSPECTED,
            None,
            "postgres_connection_not_supplied",
        )
    try:
        connection.autocommit = True
        tables = sorted(
            str(row[0])
            for row in _pg_rows(
                connection,
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = %s ORDER BY table_name",
                (schema,),
            )
        )
        rows: tuple[tuple[int, str, str], ...] = ()
        revision = 0
        if "schema_migrations" in tables:
            rows = tuple(
                (int(row[0]), str(row[1]), str(row[2]))
                for row in _pg_rows(
                    connection,
                    f"SELECT version, name, checksum FROM {schema}.schema_migrations "  # noqa: S608
                    "ORDER BY version",
                )
            )
        if "store_revisions" in tables:
            found = _pg_rows(
                connection,
                f"SELECT MAX(revision) FROM {schema}.store_revisions",  # noqa: S608
            )
            revision = 0 if not found or found[0][0] is None else int(found[0][0])
    except Exception:
        return ComponentObservation(
            JOURNEY_POSTGRES,
            ObservedState.UNAVAILABLE,
            None,
            "postgres_inspection_failed",
        )
    if not tables:
        return ComponentObservation(
            JOURNEY_POSTGRES, ObservedState.ABSENT, 0, digest=_schema_digest((), ())
        )
    if "schema_migrations" not in tables:
        return ComponentObservation(
            JOURNEY_POSTGRES, ObservedState.INVALID, None, "schema_history_missing"
        )
    state, version, code = _classify_chain(rows, POSTGRES_MIGRATIONS)
    return ComponentObservation(
        JOURNEY_POSTGRES,
        state,
        version,
        code,
        _schema_digest(rows, tables),
        {"revisions": revision},
    )


def _pg_rows(
    connection: Any, sql: str, parameters: Sequence[Any] = ()
) -> list[Sequence[Any]]:
    cursor = connection.cursor()
    try:
        cursor.execute(sql, tuple(parameters))
        return list(cursor.fetchall())
    finally:
        cursor.close()


def observe_artifact_marker(artifacts: Path | None) -> ComponentObservation:
    """Observe the Journey artifact store's encryption marker read-only.

    Args:
        artifacts: Artifact store root (``journey/artifacts``), or ``None``.

    Returns:
        The observation.  Version ``0`` is the marker-free layout of the
        previous release.
    """

    if artifacts is None or not artifacts.exists():
        return ComponentObservation(JOURNEY_ARTIFACT_MARKER, ObservedState.ABSENT, 0)
    if artifacts.is_symlink() or not artifacts.is_dir():
        return ComponentObservation(
            JOURNEY_ARTIFACT_MARKER,
            ObservedState.INVALID,
            None,
            "artifact_store_invalid",
        )
    blobs = _count_blobs(artifacts)
    marker = artifacts / _MARKER_NAME
    if not marker.exists():
        return ComponentObservation(
            JOURNEY_ARTIFACT_MARKER,
            ObservedState.UPGRADE_PENDING,
            0,
            "legacy_plaintext_artifacts" if blobs else "encryption_marker_pending",
            _digest(_canonical_json({"marker": None})),
            {"blobs": blobs},
            {"layout": "legacy_plaintext" if blobs else "empty"},
        )
    document = _read_json(marker, _MAX_DESCRIPTOR_BYTES)
    if (
        not isinstance(document, dict)
        or type(document.get("schema_version")) is not int
    ):
        return ComponentObservation(
            JOURNEY_ARTIFACT_MARKER,
            ObservedState.INVALID,
            None,
            "encryption_state_invalid",
            counts={"blobs": blobs},
        )
    version = int(document["schema_version"])
    digest = _digest(_canonical_json({"fields": sorted(document), "version": version}))
    if version > _MARKER_SCHEMA_VERSION:
        return ComponentObservation(
            JOURNEY_ARTIFACT_MARKER,
            ObservedState.NEWER_THAN_BINARY,
            version,
            "encryption_state_invalid",
            digest,
            {"blobs": blobs},
        )
    state = document.get("state")
    if (
        version != _MARKER_SCHEMA_VERSION
        or set(document) != _MARKER_FIELDS
        or state not in _ENCRYPTION_STATES
        or not isinstance(document.get("key_verifiers"), dict)
    ):
        return ComponentObservation(
            JOURNEY_ARTIFACT_MARKER,
            ObservedState.DRIFTED,
            version,
            "encryption_state_invalid",
            digest,
            {"blobs": blobs},
        )
    return ComponentObservation(
        JOURNEY_ARTIFACT_MARKER,
        ObservedState.CURRENT,
        version,
        None,
        digest,
        {"blobs": blobs},
        {"encryption_state": str(state)},
    )


def _count_blobs(artifacts: Path) -> int:
    blob_root = artifacts / "blobs" / "sha256"
    if blob_root.is_symlink() or not blob_root.is_dir():
        return 0
    count = 0
    for prefix in blob_root.iterdir():
        if prefix.is_dir() and not prefix.is_symlink():
            count += sum(
                1
                for item in prefix.iterdir()
                if item.is_file() and not item.name.startswith(".")
            )
    return count


def observe_job_store(path: Path | None) -> ComponentObservation:
    """Observe the async job metadata store read-only.

    Args:
        path: Job store JSON file, or ``None``.

    Returns:
        The observation.  A file without ``schema_version`` is the
        version-1 layout the previous release wrote.
    """

    if path is None or not path.exists():
        return ComponentObservation(JOB_STORE, ObservedState.ABSENT, 0)
    document = _read_json(path, _MAX_DOCUMENT_BYTES)
    if not isinstance(document, dict) or not isinstance(document.get("jobs"), dict):
        return ComponentObservation(
            JOB_STORE, ObservedState.INVALID, None, "job_store_invalid"
        )
    records = len(document["jobs"])
    version = document.get("schema_version", JOB_STORE_LAYOUT_VERSION)
    layout = "explicit" if "schema_version" in document else "legacy"
    digest = _digest(_canonical_json({"fields": sorted(document), "version": version}))
    if type(version) is not int or version < 1:
        return ComponentObservation(
            JOB_STORE, ObservedState.INVALID, None, "job_store_invalid"
        )
    if version > JOB_STORE_LAYOUT_VERSION:
        return ComponentObservation(
            JOB_STORE,
            ObservedState.NEWER_THAN_BINARY,
            version,
            "job_store_schema_unsupported",
            digest,
            {"records": records},
        )
    if set(document) - {"jobs", "schema_version"}:
        return ComponentObservation(
            JOB_STORE,
            ObservedState.DRIFTED,
            version,
            "job_store_unknown_fields",
            digest,
            {"records": records},
        )
    return ComponentObservation(
        JOB_STORE,
        ObservedState.CURRENT,
        version,
        None,
        digest,
        {"records": records},
        {"layout": layout},
    )


def observe_model_registry(path: Path | None) -> ComponentObservation:
    """Observe model-registry slot state read-only.

    Args:
        path: Registry state JSON file, or ``None``.

    Returns:
        The observation.
    """

    from openmed.core.registry_errors import RegistryStateError
    from openmed.core.registry_slots import load_registry_state

    if path is None or not path.exists():
        return ComponentObservation(MODEL_REGISTRY, ObservedState.ABSENT, 0)
    document = _read_json(path, _MAX_DOCUMENT_BYTES)
    version = document.get("schema_version") if isinstance(document, dict) else None
    if type(version) is not int:
        return ComponentObservation(
            MODEL_REGISTRY, ObservedState.INVALID, None, "registry_state_invalid"
        )
    assert isinstance(document, dict)
    container = "slots" if version >= 2 else "families"
    entries = document.get(container)
    count = len(entries) if isinstance(entries, dict) else 0
    digest = _digest(_canonical_json({"fields": sorted(document), "version": version}))
    if version == 1:
        return ComponentObservation(
            MODEL_REGISTRY,
            ObservedState.EXPLICIT_MIGRATION_REQUIRED,
            1,
            "registry_migration_required",
            digest,
            {"entries": count},
        )
    if version > REGISTRY_STATE_SCHEMA_VERSION:
        return ComponentObservation(
            MODEL_REGISTRY,
            ObservedState.NEWER_THAN_BINARY,
            version,
            "registry_schema_unsupported",
            digest,
            {"entries": count},
        )
    if version != REGISTRY_STATE_SCHEMA_VERSION:
        return ComponentObservation(
            MODEL_REGISTRY, ObservedState.INVALID, version, "registry_state_invalid"
        )
    try:
        load_registry_state(path)
    except RegistryStateError:
        return ComponentObservation(
            MODEL_REGISTRY,
            ObservedState.DRIFTED,
            version,
            "registry_state_invalid",
            digest,
            {"entries": count},
        )
    return ComponentObservation(
        MODEL_REGISTRY,
        ObservedState.CURRENT,
        version,
        None,
        digest,
        {"entries": count},
    )


def observe_audit_chains(paths: Sequence[Path]) -> ComponentObservation:
    """Observe audit-chain files read-only and verify their hash chains.

    Args:
        paths: Audit-chain JSON files.

    Returns:
        The aggregate observation across every chain.
    """

    from openmed.core.audit_chain import AuditChain

    if not paths:
        return ComponentObservation(AUDIT_CHAIN, ObservedState.ABSENT, 0)
    entries = 0
    versions: list[Any] = []
    for path in paths:
        document = _read_json(path, _MAX_DOCUMENT_BYTES)
        version = document.get("version") if isinstance(document, dict) else None
        versions.append(version)
        if type(version) is int and version > CHAIN_VERSION:
            return ComponentObservation(
                AUDIT_CHAIN,
                ObservedState.NEWER_THAN_BINARY,
                version,
                "audit_chain_version_unsupported",
                counts={"chains": len(paths)},
            )
        try:
            chain = AuditChain.from_dict(document if isinstance(document, dict) else {})
            valid = chain.verify().valid
        except (ValueError, TypeError, AttributeError, KeyError, RecursionError):
            valid = False
        if not valid:
            return ComponentObservation(
                AUDIT_CHAIN,
                ObservedState.DRIFTED,
                version if type(version) is int else None,
                "audit_chain_invalid",
                counts={"chains": len(paths)},
            )
        entries += len(chain.entries)
    return ComponentObservation(
        AUDIT_CHAIN,
        ObservedState.CURRENT,
        CHAIN_VERSION,
        None,
        _digest(_canonical_json({"versions": versions})),
        {"chains": len(paths), "entries": entries},
    )


def observe_recovery_point(point: Path | None) -> ComponentObservation:
    """Observe a recovery point's cleartext descriptor (no key needed).

    Args:
        point: Recovery-point directory, or ``None``.

    Returns:
        The observation of the recovery format version.
    """

    if point is None:
        return ComponentObservation(
            RECOVERY_MANIFEST,
            ObservedState.NOT_INSPECTED,
            None,
            "recovery_point_not_supplied",
        )
    descriptor = point / _DESCRIPTOR_NAME
    if not descriptor.exists():
        return ComponentObservation(RECOVERY_MANIFEST, ObservedState.ABSENT, 0)
    document = _read_json(descriptor, _MAX_DESCRIPTOR_BYTES)
    if (
        not isinstance(document, dict)
        or document.get("format") != RECOVERY_FORMAT
        or type(document.get("format_version")) is not int
    ):
        return ComponentObservation(
            RECOVERY_MANIFEST, ObservedState.INVALID, None, "descriptor_invalid"
        )
    version = int(document["format_version"])
    digest = _digest(_canonical_json({"fields": sorted(document), "version": version}))
    if version > RECOVERY_FORMAT_VERSION:
        return ComponentObservation(
            RECOVERY_MANIFEST,
            ObservedState.NEWER_THAN_BINARY,
            version,
            "format_unsupported",
            digest,
        )
    if version != RECOVERY_FORMAT_VERSION:
        return ComponentObservation(
            RECOVERY_MANIFEST, ObservedState.INVALID, version, "format_unsupported"
        )
    return ComponentObservation(
        RECOVERY_MANIFEST,
        ObservedState.CURRENT,
        version,
        None,
        digest,
        {"recovery_points": 1},
        {"kind": str(document.get("kind"))[:16]},
    )


def observe_service_state(
    sources: ServiceStateSources,
    *,
    recovery_point: Path | None = None,
    postgres_connection: Any | None = None,
    postgres_schema: str = "openmed_journey",
) -> tuple[ComponentObservation, ...]:
    """Observe every registered component read-only.

    Args:
        sources: Locations of the service's durable state.
        recovery_point: Optional recovery point to observe.
        postgres_connection: Optional dedicated DB-API connection for the
            PostgreSQL Journey schema; only SELECT statements run.
        postgres_schema: PostgreSQL schema identifier.

    Returns:
        One observation per registered component, in registry order.
    """

    journey = sources.journey_root
    observers: dict[str, Callable[[], ComponentObservation]] = {
        JOURNEY_SQLITE: lambda: observe_journey_sqlite(
            None if journey is None else journey / _JOURNEY_DATABASE
        ),
        JOURNEY_POSTGRES: lambda: observe_journey_postgres(
            postgres_connection, postgres_schema
        ),
        JOURNEY_ARTIFACT_MARKER: lambda: observe_artifact_marker(
            None if journey is None else journey / _ARTIFACTS
        ),
        JOB_STORE: lambda: observe_job_store(sources.job_store),
        MODEL_REGISTRY: lambda: observe_model_registry(sources.registry_state),
        AUDIT_CHAIN: lambda: observe_audit_chains(sources.audit_chains),
        RECOVERY_MANIFEST: lambda: observe_recovery_point(recovery_point),
    }
    observations = []
    for schema in durable_schemas():
        observer = observers.get(schema.component)
        if schema.presence is not Presence.PRESENT or observer is None:
            observations.append(
                ComponentObservation(schema.component, ObservedState.NOT_PRESENT)
            )
            continue
        observations.append(observer())
    return tuple(observations)


# ---------------------------------------------------------------------------
# Dry-run migration plan and preflight
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedAction:
    """One ordered, value-free action of a migration plan.

    Attributes:
        order: Position in the plan.
        component: Component identifier.
        action: ``none``, ``migrate``, ``explicit_migrate``, or ``refuse``.
        source_version: Observed version.
        target_version: Version after the action.
        rollback_class: Rollback class of this transition.
        recovery_point: Recovery point the rollback needs, if any.
        posture: Expected downtime or mixed-version posture.
        trigger: What applies the migration.
        code: Controlled reason code.
    """

    order: int
    component: str
    action: str
    source_version: int | None
    target_version: int | None
    rollback_class: str | None
    recovery_point: str | None
    posture: str | None
    trigger: str | None
    code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, value-free mapping."""

        return {
            "action": self.action,
            "code": self.code,
            "component": self.component,
            "order": self.order,
            "posture": self.posture,
            "recovery_point": self.recovery_point,
            "rollback_class": self.rollback_class,
            "source_version": self.source_version,
            "target_version": self.target_version,
            "trigger": self.trigger,
        }


@dataclass(frozen=True)
class PreflightCheck:
    """Value-free result of one preflight check.

    Attributes:
        name: Check name.
        passed: Whether the check allows admission.
        code: Controlled outcome code.
        details: Aggregate counters or controlled categories.
    """

    name: str
    passed: bool
    code: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, value-free mapping."""

        return {
            "code": self.code,
            "details": dict(sorted(self.details.items())),
            "name": self.name,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class MigrationPlan:
    """Versioned dry-run migration plan; computing it never mutates state.

    Attributes:
        actions: Ordered actions, one per present component.
        checks: Preflight checks.
        observations: Read-only component observations.
        admission: ``ready`` when nothing is pending and every check passed,
            ``migration_required`` when actions are pending, ``blocked`` when
            a check failed.  Readiness is false unless ``ready``.
        release: Version of this binary.
        schema: Plan document schema identifier.
    """

    actions: tuple[PlannedAction, ...]
    checks: tuple[PreflightCheck, ...]
    observations: tuple[ComponentObservation, ...]
    admission: str
    release: str = str(__version__)
    schema: str = PLAN_SCHEMA

    @property
    def ready(self) -> bool:
        """Return whether the service may admit traffic on this state."""

        return self.admission == "ready"

    @property
    def blocking_codes(self) -> tuple[str, ...]:
        """Return the codes of failed checks."""

        return tuple(check.code for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, value-free mapping."""

        return {
            "actions": [action.to_dict() for action in self.actions],
            "admission": self.admission,
            "checks": [check.to_dict() for check in self.checks],
            "observations": [item.to_dict() for item in self.observations],
            "previous_release": PREVIOUS_RELEASE,
            "ready": self.ready,
            "release": self.release,
            "schema": self.schema,
        }


def _transition_class(
    schema: DurableSchema, source: int
) -> tuple[RollbackClass | None, RecoveryPointKind | None]:
    current = schema.current_version or 0
    pending = [step for step in schema.steps if source < step.version <= current]
    classes: list[RollbackClass] = []
    for step in pending:
        if step.released:
            # Every native engine refuses versions newer than it knows, so a
            # binary older than this step cannot read the upgraded state.
            classes.append(RollbackClass.RESTORE_REQUIRED)
        elif step.rollback_class is not None:
            classes.append(step.rollback_class)
        else:
            return None, None
    if not classes:
        return RollbackClass.BACKWARD_COMPATIBLE, None
    worst = max(classes, key=lambda item: _SEVERITY[item])
    if worst is not RollbackClass.RESTORE_REQUIRED:
        return worst, None
    if (
        schema.recovery_point is RecoveryPointKind.SERVICE_RECOVERY_POINT
        and source not in schema.restorable_versions
    ):
        return worst, RecoveryPointKind.OPERATOR_FILE_COPY
    return worst, schema.recovery_point


def _actions(
    observations: Sequence[ComponentObservation],
) -> tuple[PlannedAction, ...]:
    schemas = {schema.component: schema for schema in durable_schemas()}
    actions = []
    for order, observation in enumerate(observations):
        schema = schemas[observation.component]
        common = {
            "order": order,
            "component": observation.component,
            "posture": None if schema.posture is None else schema.posture.value,
            "trigger": None if schema.trigger is None else schema.trigger.value,
        }
        state = observation.state
        if state in _REFUSING_STATES:
            actions.append(
                PlannedAction(
                    action="refuse",
                    source_version=observation.version,
                    target_version=None,
                    rollback_class=None,
                    recovery_point=None,
                    code=observation.code,
                    **common,
                )
            )
            continue
        if state in {
            ObservedState.UPGRADE_PENDING,
            ObservedState.EXPLICIT_MIGRATION_REQUIRED,
        }:
            source = observation.version or 0
            rollback, point = _transition_class(schema, source)
            actions.append(
                PlannedAction(
                    action=(
                        "explicit_migrate"
                        if state is ObservedState.EXPLICIT_MIGRATION_REQUIRED
                        or schema.trigger is MigrationTrigger.EXPLICIT_COMMAND
                        else "migrate"
                    ),
                    source_version=source,
                    target_version=schema.current_version,
                    rollback_class=None if rollback is None else rollback.value,
                    recovery_point=None if point is None else point.value,
                    code=observation.code,
                    **common,
                )
            )
            continue
        actions.append(
            PlannedAction(
                action="none",
                source_version=observation.version,
                target_version=observation.version,
                rollback_class=None,
                recovery_point=None,
                code=observation.code,
                **common,
            )
        )
    return tuple(actions)


def _state_bytes(sources: ServiceStateSources) -> int:
    total = 0
    roots: list[Path] = []
    if sources.journey_root is not None:
        roots.append(sources.journey_root)
    for item in (sources.job_store, sources.registry_state, *sources.audit_chains):
        if item is not None:
            roots.append(item)
    for root in roots:
        try:
            if root.is_file():
                total += root.stat().st_size
            elif root.is_dir():
                total += sum(
                    path.stat().st_size
                    for path in root.rglob("*")
                    if path.is_file() and not path.is_symlink()
                )
        except OSError:
            continue
    return total


def _anchor(sources: ServiceStateSources, state_root: Path | None) -> Path | None:
    candidates = [state_root, sources.journey_root, sources.job_store]
    candidates.extend([sources.registry_state, *sources.audit_chains])
    for candidate in candidates:
        if candidate is None:
            continue
        for path in (candidate, *candidate.parents):
            if path.is_dir():
                return path
    return None


def _free_space_check(
    sources: ServiceStateSources,
    state_root: Path | None,
    disk_usage: Callable[[Path], Any],
) -> PreflightCheck:
    required = 2 * _state_bytes(sources)
    anchor = _anchor(sources, state_root)
    if anchor is None:
        return PreflightCheck(
            "free_space", True, "no_local_state", {"required_bytes": 0}
        )
    try:
        available = int(disk_usage(anchor).free)
    except OSError:
        return PreflightCheck("free_space", False, "free_space_unavailable")
    details = {"available_bytes": available, "required_bytes": required}
    if available < required:
        return PreflightCheck("free_space", False, "insufficient_free_space", details)
    return PreflightCheck("free_space", True, "sufficient_free_space", details)


def _encryption_check(
    observations: Mapping[str, ComponentObservation],
    journey_key_provider: Any | None,
) -> PreflightCheck:
    from openmed.structured.store import JOURNEY_KEY_PURPOSE

    states = {
        observations[name].attributes.get("encryption_state", "none")
        for name in (JOURNEY_SQLITE, JOURNEY_ARTIFACT_MARKER)
        if name in observations
    }
    if not states & {"encrypted", "migrating"}:
        return PreflightCheck("encryption_keys", True, "not_required")
    if journey_key_provider is None:
        return PreflightCheck(
            "encryption_keys",
            True,
            "required_unverified",
            {"category": "caller_owned_provider"},
        )
    try:
        journey_key_provider.active_key(JOURNEY_KEY_PURPOSE)
    except Exception:
        return PreflightCheck("encryption_keys", False, "required_unavailable")
    return PreflightCheck("encryption_keys", True, "required_supplied")


def _cross_component_check(
    observations: Mapping[str, ComponentObservation],
) -> PreflightCheck:
    database = observations.get(JOURNEY_SQLITE)
    marker = observations.get(JOURNEY_ARTIFACT_MARKER)
    if database is None or marker is None:
        return PreflightCheck("cross_component", True, "consistent")
    db_state = database.attributes.get("encryption_state", "none")
    marker_state = marker.attributes.get("encryption_state")
    if (
        db_state in _ENCRYPTION_STATES
        and marker_state in _ENCRYPTION_STATES
        and (db_state == "plaintext") != (marker_state == "plaintext")
    ):
        return PreflightCheck(
            "cross_component", False, "journey_encryption_state_mismatch"
        )
    if (
        db_state in {"encrypted", "migrating"}
        and marker.state is ObservedState.UPGRADE_PENDING
        and marker.counts.get("blobs", 0) > 0
    ):
        return PreflightCheck(
            "cross_component", False, "journey_encryption_metadata_missing"
        )
    return PreflightCheck("cross_component", True, "consistent")


def _recovery_state_check(state_root: Path | None) -> PreflightCheck:
    if state_root is None:
        return PreflightCheck("recovery_state", True, "not_restored")
    metadata = state_root / RECOVERY_METADATA_DIRECTORY
    if not (metadata / _STAGING_DESCRIPTOR).exists():
        return PreflightCheck("recovery_state", True, "not_restored")
    if not (metadata / _ACTIVATION_RECORD).exists():
        return PreflightCheck("recovery_state", False, "recovery_activation_incomplete")
    return PreflightCheck("recovery_state", True, "recovery_activated")


_RECEIPT_CATEGORIES = {
    "journey_metadata": JOURNEY_SQLITE,
    "journey_encryption": JOURNEY_ARTIFACT_MARKER,
    "job_metadata": JOB_STORE,
    "model_registry": MODEL_REGISTRY,
    "audit_chain": AUDIT_CHAIN,
}


def _backup_check(
    actions: Sequence[PlannedAction],
    observations: Mapping[str, ComponentObservation],
    recovery_point: Path | None,
    backup_key_provider: Any | None,
) -> PreflightCheck:
    needed = sorted(
        action.component
        for action in actions
        if action.recovery_point == RecoveryPointKind.SERVICE_RECOVERY_POINT.value
    )
    if not needed:
        return PreflightCheck("backup_readiness", True, "not_required")
    details: dict[str, Any] = {"components": ",".join(needed)}
    if recovery_point is None:
        return PreflightCheck(
            "backup_readiness", False, "recovery_point_required", details
        )
    if backup_key_provider is None:
        return PreflightCheck(
            "backup_readiness", False, "recovery_point_unverified", details
        )
    from openmed.service.recovery import verify_recovery_point

    receipt = verify_recovery_point(recovery_point, key_provider=backup_key_provider)
    if not receipt.completed:
        details["reason"] = receipt.reason or "recovery_failed"
        return PreflightCheck(
            "backup_readiness", False, "recovery_point_unverified", details
        )
    captured = {
        _RECEIPT_CATEGORIES[item.category]: item.schema_version
        for item in receipt.components
        if item.category in _RECEIPT_CATEGORIES
    }
    for component in needed:
        live = observations.get(component)
        if live is not None and not live.version:
            continue  # Nothing to capture yet, for example no marker file.
        if component not in captured:
            return PreflightCheck(
                "backup_readiness", False, "recovery_point_incomplete", details
            )
    for component, version in captured.items():
        live = observations.get(component)
        if live is None or live.version is None:
            continue
        if version != live.version:
            return PreflightCheck(
                "backup_readiness", False, "recovery_point_version_mismatch", details
            )
    journey = (receipt.boundary or {}).get("journey") or {}
    database = observations.get(JOURNEY_SQLITE)
    if database is not None and database.state is not ObservedState.ABSENT:
        captured_revision = journey.get("revision") or 0
        if captured_revision != database.counts.get("revisions", 0):
            return PreflightCheck(
                "backup_readiness", False, "recovery_point_stale", details
            )
    return PreflightCheck("backup_readiness", True, "recovery_point_verified", details)


def plan_service_migrations(
    sources: ServiceStateSources,
    *,
    state_root: Path | None = None,
    recovery_point: Path | None = None,
    backup_key_provider: Any | None = None,
    journey_key_provider: Any | None = None,
    postgres_connection: Any | None = None,
    postgres_schema: str = "openmed_journey",
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> MigrationPlan:
    """Build a dry-run migration plan and run every preflight check.

    Nothing is migrated, created, or written.  Unknown, newer, or drifted
    versions are refused before any mutation could happen, readiness stays
    false while a migration or recovery activation is pending, and a
    restore-required migration needs a verified recovery point whose
    component versions and Journey revision match the live state.

    Args:
        sources: Locations of the service's durable state.
        state_root: Optional canonical state root, used to detect an
            unfinished recovery activation.
        recovery_point: Optional recovery point proposed as the rollback.
        backup_key_provider: Caller-owned provider for the backup keys.
        journey_key_provider: Optional caller-owned Journey key provider,
            checked only for an active key.
        postgres_connection: Optional dedicated DB-API connection for a
            PostgreSQL Journey schema; only SELECT statements run.
        postgres_schema: PostgreSQL schema identifier.
        disk_usage: Free-space probe; defaults to :func:`shutil.disk_usage`.

    Returns:
        The plan.
    """

    observations = observe_service_state(
        sources,
        recovery_point=recovery_point,
        postgres_connection=postgres_connection,
        postgres_schema=postgres_schema,
    )
    by_component = {item.component: item for item in observations}
    actions = _actions(observations)
    refused = [action for action in actions if action.action == "refuse"]
    pending = [
        action for action in actions if action.action in {"migrate", "explicit_migrate"}
    ]
    unclassified = [action for action in pending if action.rollback_class is None]
    if refused:
        compatibility = PreflightCheck(
            "version_compatibility",
            False,
            refused[0].code or "schema_incompatible",
            {"refused": len(refused)},
        )
    elif unclassified:
        compatibility = PreflightCheck(
            "version_compatibility", False, "migration_unclassified"
        )
    else:
        compatibility = PreflightCheck(
            "version_compatibility",
            True,
            "upgrade_pending" if pending else "current",
            {"pending": len(pending)},
        )
    posture = (
        DeploymentPosture.STOP_WRITERS.value
        if any(
            action.posture == DeploymentPosture.STOP_WRITERS.value for action in pending
        )
        else DeploymentPosture.MIXED_VERSION_SAFE.value
    )
    checks = (
        compatibility,
        _cross_component_check(by_component),
        _recovery_state_check(state_root),
        _backup_check(actions, by_component, recovery_point, backup_key_provider),
        _encryption_check(by_component, journey_key_provider),
        _free_space_check(sources, state_root, disk_usage),
        PreflightCheck(
            "deployment_posture",
            True,
            "downtime_required" if posture == "stop_writers" else "online",
            {"posture": posture},
        ),
    )
    if not all(check.passed for check in checks):
        admission = "blocked"
    elif pending:
        admission = "migration_required"
    else:
        admission = "ready"
    return MigrationPlan(
        actions=actions,
        checks=checks,
        observations=observations,
        admission=admission,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path, limit: int) -> Any:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        with open(path, "rb") as handle:
            data = handle.read(limit + 1)
        if len(data) > limit:
            return None
        return json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        return None


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()
