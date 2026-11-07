"""Unit tests for the durable-schema registry, observers, and dry-run plans."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from openmed.service import schema_registry
from openmed.service.recovery import (
    JOB_STORE_LAYOUT_VERSION,
    RECOVERY_METADATA_DIRECTORY,
    ServiceStateSources,
)
from openmed.service.schema_registry import (
    AUDIT_CHAIN,
    JOB_STORE,
    JOURNEY_ARTIFACT_MARKER,
    JOURNEY_POSTGRES,
    JOURNEY_SQLITE,
    MODEL_REGISTRY,
    RECOVERY_MANIFEST,
    RELEASED_CHECKSUMS,
    DurableSchema,
    MigrationStep,
    ObservedState,
    Presence,
    PreviousBinary,
    RecoveryPointKind,
    RollbackClass,
    durable_schemas,
    get_schema,
    observe_artifact_marker,
    observe_audit_chains,
    observe_job_store,
    observe_journey_postgres,
    observe_journey_sqlite,
    observe_model_registry,
    observe_recovery_point,
    plan_service_migrations,
    registry_digest,
    registry_document,
    registry_findings,
)
from openmed.structured.store import JourneyStoreEncryption, SQLiteJourneyStore
from openmed.structured.store.migrations import MIGRATIONS
from openmed.structured.store.postgres import POSTGRES_MIGRATIONS

PLAINTEXT_DEV = JourneyStoreEncryption.development_plaintext()


@pytest.fixture(autouse=True)
def _development_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENMED_PROFILE", raising=False)


def test_registry_names_every_durable_schema_once() -> None:
    components = [schema.component for schema in durable_schemas()]

    assert components == [
        JOURNEY_SQLITE,
        JOURNEY_POSTGRES,
        JOURNEY_ARTIFACT_MARKER,
        JOB_STORE,
        MODEL_REGISTRY,
        AUDIT_CHAIN,
        RECOVERY_MANIFEST,
        "protected_payload_store",
        "model_activation_journal",
        "audit_continuity",
    ]
    for schema in durable_schemas():
        if schema.presence is Presence.PRESENT:
            assert schema.owner and schema.command
            assert schema.trigger is not None and schema.posture is not None
            assert schema.recovery_point is not None
            assert schema.rollback_class is not None
            assert schema.current_version in schema.source_versions
            assert set(schema.readable_versions) <= set(schema.source_versions)
            assert {"drift", "unknown_version"} <= set(schema.refusal_codes)
        else:
            assert schema.tracking_issue in {4918, 5066, 3599}
            assert schema.rollback_class is None


def test_pending_components_must_register_when_their_module_lands() -> None:
    for schema in durable_schemas():
        if schema.presence is Presence.NOT_PRESENT and schema.pending_module:
            # When this fails, register the component as present, classify
            # it, and add a rehearsal adapter.
            assert importlib.util.find_spec(schema.pending_module) is None


def test_released_migrations_are_immutable_and_new_steps_are_classified() -> None:
    assert registry_findings() == ()
    sqlite = get_schema(JOURNEY_SQLITE)
    for version, checksum in RELEASED_CHECKSUMS[JOURNEY_SQLITE].items():
        assert MIGRATIONS[version - 1].checksum == checksum
    for version, checksum in RELEASED_CHECKSUMS[JOURNEY_POSTGRES].items():
        assert POSTGRES_MIGRATIONS[version - 1].checksum == checksum
    assert [step.version for step in sqlite.new_steps] == [4]
    assert sqlite.new_steps[0].rollback_class is RollbackClass.RESTORE_REQUIRED
    assert sqlite.rollback_class is RollbackClass.RESTORE_REQUIRED
    assert get_schema(JOURNEY_POSTGRES).new_steps == ()
    assert get_schema(JOURNEY_POSTGRES).rollback_class is (
        RollbackClass.BACKWARD_COMPATIBLE
    )


def test_registry_document_is_value_free_and_stable() -> None:
    document = registry_document()

    assert json.loads(json.dumps(document)) == document
    assert document["findings"] == []
    assert registry_digest() == registry_digest()
    assert registry_digest().startswith("sha256:")
    text = json.dumps(document)
    assert "/Users" not in text and "\\\\" not in text
    with pytest.raises(KeyError):
        get_schema("unknown")


def test_job_store_version_matches_the_recovery_layout() -> None:
    pytest.importorskip("fastapi")
    from openmed.service.jobs import JOB_STORE_SCHEMA_VERSION

    assert JOB_STORE_SCHEMA_VERSION == JOB_STORE_LAYOUT_VERSION
    assert get_schema(JOB_STORE).current_version == JOB_STORE_SCHEMA_VERSION


def _schema(**changes: Any) -> DurableSchema:
    base = {
        "component": "synthetic",
        "owner": "synthetic.owner",
        "current_version": 2,
        "previous_release_version": 1,
        "previous_binary": PreviousBinary.REFUSES,
        "recovery_point": RecoveryPointKind.SERVICE_RECOVERY_POINT,
        "steps": (
            MigrationStep(1, "one", None, True),
            MigrationStep(2, "two", None, False, RollbackClass.RESTORE_REQUIRED),
        ),
    }
    base.update(changes)
    return DurableSchema(**base)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, RollbackClass.RESTORE_REQUIRED),
        (
            {"recovery_point": RecoveryPointKind.NOT_REQUIRED},
            RollbackClass.RELEASE_BLOCKING,
        ),
        (
            {"steps": (MigrationStep(1, "one", None, True),)},
            None,
        ),
        (
            {
                "steps": (MigrationStep(1, "one", None, True),),
                "previous_binary": PreviousBinary.READS,
            },
            RollbackClass.BACKWARD_COMPATIBLE,
        ),
        (
            {"steps": (MigrationStep(2, "two", None, False),)},
            None,
        ),
        (
            {
                "steps": (
                    MigrationStep(2, "two", None, False, RollbackClass.REVERSIBLE),
                ),
            },
            None,
        ),
        (
            {
                "steps": (
                    MigrationStep(2, "two", None, False, RollbackClass.REVERSIBLE),
                ),
                "previous_binary": PreviousBinary.READS,
            },
            RollbackClass.REVERSIBLE,
        ),
        ({"presence": Presence.NOT_PRESENT}, None),
    ],
)
def test_rollback_class_derivation(
    changes: dict[str, Any], expected: RollbackClass | None
) -> None:
    schema = _schema(**changes)

    assert schema.rollback_class is expected
    findings = registry_findings([schema])
    if expected is None and schema.presence is Presence.PRESENT:
        assert "synthetic:rollback_unclassified" in findings


def _write_history(database: Path, rows: list[tuple[int, str, str]]) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        connection.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, "
            "name TEXT NOT NULL, checksum TEXT NOT NULL)"
        )
        known = {migration.version: migration for migration in MIGRATIONS}
        for version, name, checksum in rows:
            migration = known.get(version)
            if migration is not None:
                for statement in migration.statements:
                    connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations VALUES (?, ?, ?)",
                (version, name, checksum),
            )
    finally:
        connection.close()


def _dump(database: Path) -> str:
    connection = sqlite3.connect(database)
    try:
        return "\n".join(connection.iterdump())
    finally:
        connection.close()


_HISTORY = st.integers(min_value=0, max_value=len(MIGRATIONS)).flatmap(
    lambda length: st.tuples(
        st.just(length),
        st.sampled_from(["clean", "checksum", "name", "future"]),
        st.integers(min_value=1, max_value=max(1, length)),
    )
)


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(case=_HISTORY)
def test_observer_agrees_with_the_native_engine_and_refusal_never_mutates(
    tmp_path_factory: pytest.TempPathFactory, case: tuple[int, str, int]
) -> None:
    length, perturbation, target = case
    rows = [(item.version, item.name, item.checksum) for item in MIGRATIONS[:length]]
    if perturbation == "checksum" and rows:
        version, name, _checksum = rows[target - 1]
        rows[target - 1] = (version, name, "0" * 64)
    elif perturbation == "name" and rows:
        version, _name, checksum = rows[target - 1]
        rows[target - 1] = (version, "renamed", checksum)
    elif perturbation == "future":
        rows.append((len(MIGRATIONS) + target, "future", "f" * 64))
    database = tmp_path_factory.mktemp("history") / "journey.sqlite3"
    _write_history(database, rows)
    observed = observe_journey_sqlite(database)
    before = _dump(database)

    opened = SQLiteJourneyStore.open(database, encryption=PLAINTEXT_DEV)

    if observed.state in {ObservedState.CURRENT, ObservedState.UPGRADE_PENDING}:
        assert opened.ok, (observed, opened.code)
        assert opened.value.schema_version == MIGRATIONS[-1].version
        opened.value.close()
    else:
        assert observed.state in {
            ObservedState.DRIFTED,
            ObservedState.NEWER_THAN_BINARY,
        }
        assert not opened.ok
        assert opened.code == "schema_unsupported"
        assert _dump(database) == before


def test_observers_report_versions_without_values(tmp_path: Path) -> None:
    assert observe_journey_sqlite(tmp_path / "missing.sqlite3").state is (
        ObservedState.ABSENT
    )
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": {"job-1": {"label": "SECRET"}}}))
    legacy = observe_job_store(jobs)
    assert (legacy.state, legacy.version, legacy.attributes["layout"]) == (
        ObservedState.CURRENT,
        1,
        "legacy",
    )
    assert "SECRET" not in json.dumps(legacy.to_dict())
    jobs.write_text(json.dumps({"jobs": {}, "schema_version": 2}))
    assert observe_job_store(jobs).code == "job_store_schema_unsupported"
    jobs.write_text(json.dumps({"jobs": {}, "shadow": {}}))
    assert observe_job_store(jobs).code == "job_store_unknown_fields"
    jobs.write_text("{not json")
    assert observe_job_store(jobs).state is ObservedState.INVALID

    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"schema_version": 1, "families": {}}))
    assert observe_model_registry(registry).state is (
        ObservedState.EXPLICIT_MIGRATION_REQUIRED
    )
    registry.write_text(json.dumps({"schema_version": 3, "slots": {}}))
    assert observe_model_registry(registry).code == "registry_schema_unsupported"
    registry.write_text(json.dumps({"schema_version": 2, "slots": {}}))
    assert observe_model_registry(registry).state is ObservedState.CURRENT

    chain = tmp_path / "chain-0000.json"
    chain.write_text(json.dumps({"version": 2}))
    assert observe_audit_chains([chain]).code == "audit_chain_version_unsupported"
    chain.write_text(json.dumps({"version": 1}))
    assert observe_audit_chains([chain]).code == "audit_chain_invalid"
    assert observe_audit_chains([]).state is ObservedState.ABSENT

    artifacts = tmp_path / "artifacts"
    assert observe_artifact_marker(artifacts).state is ObservedState.ABSENT
    artifacts.mkdir()
    assert observe_artifact_marker(artifacts).version == 0
    (artifacts / "encryption.json").write_text(json.dumps({"schema_version": 2}))
    assert observe_artifact_marker(artifacts).state is (ObservedState.NEWER_THAN_BINARY)

    assert observe_recovery_point(None).state is ObservedState.NOT_INSPECTED
    point = tmp_path / "point"
    point.mkdir()
    (point / "recovery-point.json").write_text(
        json.dumps({"format": "openmed.service.recovery-point", "format_version": 2})
    )
    assert observe_recovery_point(point).code == "format_unsupported"
    assert observe_journey_postgres(None).state is ObservedState.NOT_INSPECTED
    with pytest.raises(ValueError):
        observe_journey_postgres(None, schema="bad-schema")


def test_postgres_observer_reports_failure_without_details() -> None:
    class Broken:
        autocommit = False

        def cursor(self) -> Any:
            raise RuntimeError("postgresql://user:secret@host/db")

    observation = observe_journey_postgres(Broken())

    assert observation.state is ObservedState.UNAVAILABLE
    assert observation.code == "postgres_inspection_failed"
    assert "secret" not in json.dumps(observation.to_dict())


def _journey_root(root: Path, version: int) -> Path:
    database = root / "journey" / "journey.sqlite3"
    _write_history(
        database,
        [(item.version, item.name, item.checksum) for item in MIGRATIONS[:version]],
    )
    return database


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_plan_refuses_newer_state_before_any_mutation(tmp_path: Path) -> None:
    root = tmp_path / "state"
    _journey_root(root, len(MIGRATIONS))
    jobs = root / "jobs" / "service-jobs.json"
    jobs.parent.mkdir(parents=True)
    jobs.write_text(json.dumps({"jobs": {}, "schema_version": 9}))
    before = _tree(root)

    plan = plan_service_migrations(
        ServiceStateSources.from_state_root(root), state_root=root
    )

    assert plan.admission == "blocked"
    assert plan.blocking_codes == ("job_store_schema_unsupported",)
    refused = next(item for item in plan.actions if item.component == JOB_STORE)
    assert refused.action == "refuse"
    assert _tree(root) == before
    assert json.loads(json.dumps(plan.to_dict()))["ready"] is False


def test_plan_detects_cross_component_and_recovery_state(tmp_path: Path) -> None:
    root = tmp_path / "state"
    database = _journey_root(root, len(MIGRATIONS))
    connection = sqlite3.connect(database, isolation_level=None)
    connection.execute(
        "INSERT INTO store_encryption(singleton, state) VALUES (1, 'encrypted')"
    )
    connection.close()
    artifacts = root / "journey" / "artifacts"
    artifacts.mkdir()
    (artifacts / "encryption.json").write_text(
        json.dumps(
            {
                "envelope_version": None,
                "key_purpose": None,
                "key_verifiers": {},
                "schema_version": 1,
                "state": "plaintext",
            }
        )
    )
    sources = ServiceStateSources.from_state_root(root)
    mismatch = plan_service_migrations(sources, state_root=root)
    assert mismatch.blocking_codes == ("journey_encryption_state_mismatch",)

    (artifacts / "encryption.json").unlink()
    metadata = root / RECOVERY_METADATA_DIRECTORY
    metadata.mkdir()
    (metadata / "staging.json").write_text("{}")
    staged = plan_service_migrations(sources, state_root=root)
    assert "recovery_activation_incomplete" in staged.blocking_codes
    (metadata / "activation.json").write_text("{}")
    activated = plan_service_migrations(sources, state_root=root)
    assert "recovery_activation_incomplete" not in activated.blocking_codes
    encryption = next(
        item for item in activated.checks if item.name == "encryption_keys"
    )
    assert encryption.code == "required_unverified"


def test_plan_checks_free_space_and_keys(tmp_path: Path) -> None:
    root = tmp_path / "state"
    _journey_root(root, 3)
    sources = ServiceStateSources.from_state_root(root)

    plan = plan_service_migrations(
        sources,
        state_root=root,
        disk_usage=lambda path: SimpleNamespace(free=0),
    )

    checks = {item.name: item for item in plan.checks}
    assert checks["free_space"].code == "insufficient_free_space"
    assert checks["free_space"].details["required_bytes"] > 0
    assert checks["deployment_posture"].code == "downtime_required"
    assert checks["version_compatibility"].code == "upgrade_pending"
    assert checks["encryption_keys"].code == "not_required"
    assert plan.admission == "blocked"

    class NoKeys:
        def active_key(self, purpose: str) -> Any:
            raise RuntimeError("no key")

    current = tmp_path / "current"
    database = _journey_root(current, len(MIGRATIONS))
    connection = sqlite3.connect(database, isolation_level=None)
    connection.execute(
        "INSERT INTO store_encryption(singleton, state) VALUES (1, 'encrypted')"
    )
    connection.close()
    keyed = plan_service_migrations(
        ServiceStateSources.from_state_root(current),
        journey_key_provider=NoKeys(),
    )
    codes = {item.name: item.code for item in keyed.checks}
    assert codes["encryption_keys"] == "required_unavailable"
    assert keyed.admission == "blocked"


def test_plan_names_explicit_and_unrestorable_migrations(tmp_path: Path) -> None:
    root = tmp_path / "state"
    registry = root / "registry" / "registry_state.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({"schema_version": 1, "families": {}}))

    plan = plan_service_migrations(
        ServiceStateSources.from_state_root(root), state_root=root
    )

    action = next(item for item in plan.actions if item.component == MODEL_REGISTRY)
    assert (action.action, action.source_version, action.target_version) == (
        "explicit_migrate",
        1,
        2,
    )
    assert action.recovery_point == "operator_file_copy"
    assert action.trigger == "explicit_command"
    assert plan.admission == "migration_required"
    assert not plan.ready
