"""Tests for the record migrator registry keyed by schema id and version."""

from __future__ import annotations

import copy
import dataclasses
import doctest
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from openmed.core import record_migration_catalog, record_migrations
from openmed.core.errors import ConfigurationError, InputError
from openmed.core.record_migrations import (
    MIGRATION_FIXTURE_ROOT,
    REFUSAL_REASONS,
    REGISTRATION_REASONS,
    FieldMove,
    MigratorRegistrationError,
    MigratorRegistry,
    RecordMigrationRefused,
    diff_field_paths,
    record_digest,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "synthetic_visit_note"
SENTINELS = (
    "SYNTHETIC-AUTHOR-SENTINEL",
    "SYNTHETIC-BODY-SENTINEL",
    "SYNTHETIC-SECTION-SENTINEL",
    "SYNTHETIC-TAG-SENTINEL",
)


def _fixture(name: str, side: str) -> dict[str, Any]:
    path = ROOT / MIGRATION_FIXTURE_ROOT / name / f"{side}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _add_provenance(record: dict[str, Any]) -> dict[str, Any]:
    record["provenance"] = {"author": record.pop("author")}
    record["extensions"] = {}
    for section in record["sections"]:
        section["kind"] = "free_text"
    record["schema_version"] = "1.1.0"
    return record


def _count_tags(record: dict[str, Any]) -> dict[str, Any]:
    record["tag_count"] = len(record.pop("tags"))
    record["schema_version"] = "2.0.0"
    return record


def _is_visit_note(record: Any) -> bool:
    return "note_id" in record and "sections" in record


def _registry() -> MigratorRegistry:
    registry = MigratorRegistry()
    registry.register_schema(SCHEMA, current_version="2.0.0", matcher=_is_visit_note)
    registry.register_migrator(
        SCHEMA,
        "1.0.0",
        "1.1.0",
        _add_provenance,
        lossless=True,
        fixture=f"{SCHEMA}/1.0.0-to-1.1.0",
    )
    registry.register_migrator(
        SCHEMA,
        "1.1.0",
        "2.0.0",
        _count_tags,
        lossless=False,
        fixture=f"{SCHEMA}/1.1.0-to-2.0.0",
    )
    return registry


def _assert_value_free(text: str) -> None:
    for sentinel in SENTINELS:
        assert sentinel not in text


def test_two_step_chain_migrates_fixture_and_reports_field_paths_only() -> None:
    registry = _registry()
    record = _fixture(f"{SCHEMA}/1.0.0-to-1.1.0", "before")
    original = copy.deepcopy(record)

    result = registry.migrate_record(record, schema_id=SCHEMA)

    assert record == original
    assert result.record == _fixture(f"{SCHEMA}/1.1.0-to-2.0.0", "after")
    report = result.report
    assert report.status == "migrated"
    assert report.lossy is True
    assert report.source_version == "1.0.0"
    assert report.target_version == "2.0.0"
    assert report.added == (
        "/extensions",
        "/provenance",
        "/sections/*/kind",
        "/tag_count",
    )
    assert report.dropped == ("/tags",)
    assert report.moved == (FieldMove("/author", "/provenance/author"),)
    assert report.changed == ("/schema_version",)
    assert [(step.from_version, step.to_version) for step in report.steps] == [
        ("1.0.0", "1.1.0"),
        ("1.1.0", "2.0.0"),
    ]
    assert report.steps[0].lossless is True
    assert report.steps[0].fields.moved == (FieldMove("/author", "/provenance/author"),)
    assert report.steps[1].lossless is False
    assert report.steps[1].fields.dropped == ("/tags",)
    assert report.source_digest == record_digest(original)
    assert report.target_digest == record_digest(result.record)

    payload = report.to_dict()
    assert payload["counts"] == {
        "added": 4,
        "changed": 1,
        "dropped": 1,
        "moved": 1,
        "steps": 2,
    }
    assert payload["moved"] == [{"from": "/author", "to": "/provenance/author"}]
    rendered = json.dumps(payload, sort_keys=True)
    _assert_value_free(rendered)
    _assert_value_free(repr(result))
    assert json.loads(rendered) == payload


def test_each_synthetic_edge_migrates_its_before_fixture_to_its_after_fixture() -> None:
    registry = _registry()

    for edge in registry.edges(SCHEMA):
        before_path, after_path = edge.fixture_pair(ROOT)
        before = json.loads(before_path.read_text(encoding="utf-8"))
        after = json.loads(after_path.read_text(encoding="utf-8"))

        result = registry.migrate_record(
            before, schema_id=SCHEMA, to_version=edge.to_version
        )

        assert result.record == after
        assert [step.fixture for step in result.report.steps] == [edge.fixture]


def test_registering_two_migrators_for_the_same_edge_fails() -> None:
    registry = _registry()

    with pytest.raises(MigratorRegistrationError) as caught:
        registry.register_migrator(
            SCHEMA,
            "1.0.0",
            "1.1.0",
            _add_provenance,
            lossless=True,
            fixture=f"{SCHEMA}/duplicate",
        )

    assert caught.value.reason == "duplicate_edge"
    assert isinstance(caught.value, ConfigurationError)
    assert caught.value.to_dict()["details"] == {
        "reason": "duplicate_edge",
        "schema_id": SCHEMA,
        "from_version": "1.0.0",
        "to_version": "1.1.0",
    }
    assert len(registry.edges(SCHEMA)) == 2


def test_request_with_no_path_returns_typed_refusal_with_ids_only() -> None:
    registry = _registry()
    record = _fixture(f"{SCHEMA}/1.0.0-to-1.1.0", "before")

    with pytest.raises(RecordMigrationRefused) as caught:
        registry.migrate_record(record, schema_id=SCHEMA, to_version="9.0.0")

    refusal = caught.value.refusal
    assert refusal.to_dict() == {
        "reason": "no_path",
        "schema_id": SCHEMA,
        "from_version": "1.0.0",
        "to_version": "9.0.0",
    }
    assert isinstance(caught.value, InputError)
    assert caught.value.code == "record_migration_refused"
    assert str(caught.value) == (
        f"record migration refused (no_path): schema {SCHEMA} from 1.0.0 to 9.0.0"
    )
    assert caught.value.details == refusal.to_dict()
    _assert_value_free(str(caught.value) + repr(caught.value.details))


def test_downgrade_without_registered_edge_is_refused_as_no_path() -> None:
    registry = _registry()

    with pytest.raises(RecordMigrationRefused) as caught:
        registry.migration_path(SCHEMA, "2.0.0", "1.0.0")

    assert caught.value.refusal.reason == "no_path"


def test_unknown_schema_is_refused_with_id_only() -> None:
    registry = _registry()

    with pytest.raises(RecordMigrationRefused) as caught:
        registry.migration_path("unregistered_schema", "1.0.0", "2.0.0")

    assert caught.value.refusal.to_dict() == {
        "reason": "unknown_schema",
        "schema_id": "unregistered_schema",
        "from_version": None,
        "to_version": None,
    }


def test_unsafe_identifiers_are_never_echoed() -> None:
    registry = _registry()
    record = _fixture(f"{SCHEMA}/1.0.0-to-1.1.0", "before")
    record["schema_version"] = "SYNTHETIC-AUTHOR-SENTINEL <unsafe>"

    with pytest.raises(RecordMigrationRefused) as unknown:
        registry.migration_path("SYNTHETIC-AUTHOR-SENTINEL <unsafe>", "1.0.0")
    with pytest.raises(RecordMigrationRefused) as invalid:
        registry.migrate_record(record, schema_id=SCHEMA)

    assert unknown.value.refusal.schema_id is None
    assert invalid.value.refusal.to_dict() == {
        "reason": "invalid_version",
        "schema_id": SCHEMA,
        "from_version": None,
        "to_version": None,
    }
    _assert_value_free(str(unknown.value) + str(invalid.value))


def test_chain_resolution_is_deterministic_and_empty_when_current() -> None:
    registry = _registry()

    first = registry.migration_path(SCHEMA, "1.0.0")
    second = registry.migration_path(SCHEMA, "1.0.0", "2.0.0")

    assert first == second
    assert [edge.to_dict() for edge in first] == [
        {
            "schema_id": SCHEMA,
            "from_version": "1.0.0",
            "to_version": "1.1.0",
            "lossless": True,
            "fixture": f"{SCHEMA}/1.0.0-to-1.1.0",
        },
        {
            "schema_id": SCHEMA,
            "from_version": "1.1.0",
            "to_version": "2.0.0",
            "lossless": False,
            "fixture": f"{SCHEMA}/1.1.0-to-2.0.0",
        },
    ]
    assert registry.migration_path(SCHEMA, "2.0.0") == ()


def test_shortcut_edge_makes_the_longer_request_ambiguous() -> None:
    registry = _registry()
    registry.register_migrator(
        SCHEMA,
        "1.0.0",
        "2.0.0",
        lambda record: _count_tags(_add_provenance(record)),
        lossless=False,
        fixture=f"{SCHEMA}/1.0.0-to-2.0.0",
    )

    with pytest.raises(RecordMigrationRefused) as caught:
        registry.migration_path(SCHEMA, "1.0.0", "2.0.0")

    assert caught.value.refusal.to_dict() == {
        "reason": "ambiguous_path",
        "schema_id": SCHEMA,
        "from_version": "1.0.0",
        "to_version": "2.0.0",
    }
    assert len(registry.migration_path(SCHEMA, "1.0.0", "1.1.0")) == 1


@pytest.mark.parametrize(
    ("source", "target"),
    [("2.0.0", "1.0.0"), ("1.1.0", "1.0.0"), ("1.1.0", "1.1.0")],
)
def test_edges_that_close_a_cycle_are_refused(source: str, target: str) -> None:
    registry = _registry()

    with pytest.raises(MigratorRegistrationError) as caught:
        registry.register_migrator(
            SCHEMA,
            source,
            target,
            lambda record: record,
            lossless=True,
            fixture=f"{SCHEMA}/cycle",
        )

    assert caught.value.reason == "cyclic_edge"
    assert len(registry.edges(SCHEMA)) == 2


def test_current_record_reports_current_without_steps() -> None:
    registry = _registry()
    record = _fixture(f"{SCHEMA}/1.1.0-to-2.0.0", "after")

    result = registry.migrate_record(record, schema_id=SCHEMA)

    assert result.record == record
    assert result.record is not record
    assert result.report.status == "current"
    assert result.report.lossy is False
    assert result.report.steps == ()
    assert result.report.to_dict()["counts"] == {
        "added": 0,
        "changed": 0,
        "dropped": 0,
        "moved": 0,
        "steps": 0,
    }
    assert result.report.source_digest == result.report.target_digest


def test_migrator_exceptions_become_value_free_refusals() -> None:
    registry = MigratorRegistry()
    registry.register_schema(SCHEMA, current_version="2")

    def explode(record: dict[str, Any]) -> dict[str, Any]:
        raise ValueError(f"bad value {record['note']}")

    registry.register_migrator(SCHEMA, "1", "2", explode, lossless=True, fixture="x")

    with pytest.raises(RecordMigrationRefused) as caught:
        registry.migrate_record(
            {"schema_version": "1", "note": "SYNTHETIC-BODY-SENTINEL"},
            schema_id=SCHEMA,
        )

    assert caught.value.refusal.to_dict() == {
        "reason": "migrator_failed",
        "schema_id": SCHEMA,
        "from_version": "1",
        "to_version": "2",
        "error_type": "ValueError",
    }
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    _assert_value_free(str(caught.value) + repr(caught.value.details))


@pytest.mark.parametrize(
    ("migrator", "reason"),
    [
        (lambda record: {**record, "schema_version": "3"}, "version_mismatch"),
        (lambda record: {"note": "kept"}, "version_mismatch"),
        (lambda record: ["not", "a", "mapping"], "migrator_output_invalid"),
        (
            lambda record: {**record, "schema_version": "2", "value": object()},
            "migrator_output_invalid",
        ),
    ],
)
def test_migrator_output_is_checked(migrator: Any, reason: str) -> None:
    registry = MigratorRegistry()
    registry.register_schema(SCHEMA, current_version="2")
    registry.register_migrator(SCHEMA, "1", "2", migrator, lossless=True, fixture="x")

    with pytest.raises(RecordMigrationRefused) as caught:
        registry.migrate_record({"schema_version": "1"}, schema_id=SCHEMA)

    assert caught.value.refusal.reason == reason


def test_migrators_receive_copies_and_never_mutate_the_input() -> None:
    registry = MigratorRegistry()
    registry.register_schema(SCHEMA, current_version="2")

    def mutate(record: dict[str, Any]) -> dict[str, Any]:
        record["items"].append("added")
        record["schema_version"] = "2"
        return record

    registry.register_migrator(SCHEMA, "1", "2", mutate, lossless=True, fixture="x")
    record = {"schema_version": "1", "items": ["kept"]}

    result = registry.migrate_record(record, schema_id=SCHEMA)

    assert record == {"schema_version": "1", "items": ["kept"]}
    assert result.record == {"schema_version": "2", "items": ["kept", "added"]}
    assert result.report.added == ("/items/*",)


def test_non_json_records_are_refused() -> None:
    registry = _registry()

    with pytest.raises(RecordMigrationRefused) as caught:
        registry.migrate_record(
            {"schema_version": "1.0.0", "value": float("nan")}, schema_id=SCHEMA
        )

    assert caught.value.refusal.reason == "invalid_record"


def test_identify_uses_registered_matchers() -> None:
    registry = _registry()
    registry.register_schema(
        "synthetic_ledger",
        current_version="1",
        matcher=lambda record: "entries" in record,
    )
    registry.register_schema(
        "synthetic_broken_matcher",
        current_version="1",
        matcher=lambda record: record["missing"] == 1,
    )
    record = _fixture(f"{SCHEMA}/1.0.0-to-1.1.0", "before")

    identity = registry.identify(record)
    result = registry.migrate_record(record)

    assert identity.to_dict() == {"schema_id": SCHEMA, "version": "1.0.0"}
    assert result.report.schema_id == SCHEMA
    with pytest.raises(RecordMigrationRefused) as unknown:
        registry.identify({"schema_version": "7.0.0", "other": "SYNTHETIC-BODY"})
    assert unknown.value.refusal.to_dict() == {
        "reason": "unknown_schema",
        "schema_id": None,
        "from_version": "7.0.0",
        "to_version": None,
    }
    with pytest.raises(RecordMigrationRefused) as ambiguous:
        registry.identify({**record, "entries": []})
    assert ambiguous.value.refusal.reason == "ambiguous_schema"
    with pytest.raises(RecordMigrationRefused) as invalid:
        registry.identify(["not", "a", "record"])  # type: ignore[arg-type]
    assert invalid.value.refusal.reason == "invalid_record"


def test_integer_versions_and_missing_versions_are_normalised() -> None:
    registry = MigratorRegistry()
    registry.register_schema(
        "synthetic_counter", current_version=2, version_when_missing=0
    )
    registry.register_migrator(
        "synthetic_counter",
        0,
        1,
        lambda record: {**record, "schema_version": 1},
        lossless=True,
        fixture="synthetic_counter/0-to-1",
    )
    registry.register_migrator(
        "synthetic_counter",
        1,
        2,
        lambda record: {**record, "schema_version": 2, "count": 0},
        lossless=True,
        fixture="synthetic_counter/1-to-2",
    )

    result = registry.migrate_record({"value": 3}, schema_id="synthetic_counter")

    assert result.record == {"value": 3, "schema_version": 2, "count": 0}
    assert result.report.source_version == "0"
    assert result.report.target_version == "2"
    assert result.report.added == ("/count", "/schema_version")
    assert registry.record_version("synthetic_counter", {"schema_version": 1}) == "1"
    assert registry.schema("synthetic_counter").to_dict() == {
        "schema_id": "synthetic_counter",
        "current_version": "2",
        "version_field": "schema_version",
        "version_when_missing": "0",
        "identifiable": False,
    }


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"schema_id": "Bad Id"}, "invalid_schema"),
        ({"current_version": "1 0"}, "invalid_schema"),
        ({"current_version": True}, "invalid_schema"),
        ({"version_field": "nested/field"}, "invalid_schema"),
        ({"version_when_missing": "-1"}, "invalid_schema"),
        ({"matcher": "not callable"}, "invalid_schema"),
        ({"schema_id": SCHEMA}, "duplicate_schema"),
    ],
)
def test_invalid_schema_registrations_are_rejected(
    kwargs: dict[str, Any], reason: str
) -> None:
    registry = _registry()
    arguments: dict[str, Any] = {
        "schema_id": "another_schema",
        "current_version": "1",
        **kwargs,
    }
    schema_id = arguments.pop("schema_id")

    with pytest.raises(MigratorRegistrationError) as caught:
        registry.register_schema(schema_id, **arguments)

    assert caught.value.reason == reason


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        ((SCHEMA, "2.0.0", "3.0.0", "not callable"), "invalid_edge"),
        ((SCHEMA, "2.0.0", "bad version", lambda record: record), "invalid_edge"),
        (("not_registered", "1", "2", lambda record: record), "unknown_schema"),
    ],
)
def test_invalid_edges_are_rejected(arguments: tuple[Any, ...], reason: str) -> None:
    registry = _registry()

    with pytest.raises(MigratorRegistrationError) as caught:
        registry.register_migrator(*arguments, lossless=True, fixture="x")

    assert caught.value.reason == reason


@pytest.mark.parametrize(
    "fixture", ["", "../escape", "/absolute", "a//b", "a/./b", "spaces here"]
)
def test_fixture_names_must_stay_under_the_fixture_root(fixture: str) -> None:
    registry = _registry()

    with pytest.raises(MigratorRegistrationError) as caught:
        registry.register_migrator(
            SCHEMA,
            "2.0.0",
            "3.0.0",
            lambda record: record,
            lossless=True,
            fixture=fixture,
        )

    assert caught.value.reason == "invalid_fixture"


def test_lossless_flag_must_be_boolean() -> None:
    registry = _registry()

    with pytest.raises(MigratorRegistrationError) as caught:
        registry.register_migrator(
            SCHEMA,
            "2.0.0",
            "3.0.0",
            lambda record: record,
            lossless="yes",  # type: ignore[arg-type]
            fixture="x",
        )

    assert caught.value.reason == "invalid_edge"


def test_field_diff_escapes_pointers_collapses_indexes_and_skips_trivial_moves() -> (
    None
):
    before = {
        "a/b": 1,
        "tilde~key": "same",
        "rows": [{"x": 1}, {"x": 2}],
        "flag": True,
        "empty": [],
        "twin_one": "duplicate",
        "twin_two": "duplicate",
    }
    after = {
        "a/b": 2,
        "tilde~key": "same",
        "rows": [{"x": 1, "y": 0}, {"x": 3, "y": 0}],
        "moved_flag": True,
        "moved_empty": [],
        "twin_three": "duplicate",
    }

    changes = diff_field_paths(before, after)

    assert changes.changed == ("/a~1b", "/rows/*/x")
    assert changes.added == (
        "/moved_empty",
        "/moved_flag",
        "/rows/*/y",
        "/twin_three",
    )
    assert changes.dropped == ("/empty", "/flag", "/twin_one", "/twin_two")
    assert changes.moved == ()


def test_field_diff_reports_moves_out_of_nested_containers() -> None:
    before = {"extensions": {"questions": ["q1", "q2"]}, "schema_version": "1"}
    after = {"extensions": {}, "questions": ["q1", "q2"], "schema_version": "2"}

    changes = diff_field_paths(before, after)

    assert changes.moved == (FieldMove("/extensions/questions", "/questions"),)
    assert changes.added == ()
    assert changes.dropped == ()
    assert changes.changed == ("/schema_version",)


def test_reason_code_catalogues_are_stable() -> None:
    assert REFUSAL_REASONS == tuple(sorted(REFUSAL_REASONS))
    assert REGISTRATION_REASONS == tuple(sorted(REGISTRATION_REASONS))


def test_module_functions_use_an_injected_or_default_registry() -> None:
    registry = _registry()
    record = _fixture(f"{SCHEMA}/1.0.0-to-1.1.0", "before")

    assert record_migrations.default_registry() is (
        record_migrations.default_registry()
    )
    assert record_migrations.identify_record(record, registry=registry).schema_id == (
        SCHEMA
    )
    assert len(record_migrations.migration_path(SCHEMA, "1.0.0", registry=registry))
    result = record_migrations.migrate_record(record, registry=registry)
    assert result.report.target_version == "2.0.0"
    record_migrations.register_schema(
        "synthetic_extra", current_version="2", registry=registry
    )
    edge = record_migrations.register_migrator(
        "synthetic_extra",
        "1",
        "2",
        lambda item: {**item, "schema_version": "2"},
        lossless=True,
        fixture="synthetic_extra/1-to-2",
        registry=registry,
    )
    assert edge in registry.edges("synthetic_extra")
    assert [schema.schema_id for schema in registry.schemas()] == [
        "synthetic_extra",
        SCHEMA,
    ]


def test_default_registry_loads_builtin_registrations_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[MigratorRegistry] = []

    class Builtins:
        @staticmethod
        def register_record_migrations(registry: MigratorRegistry) -> None:
            calls.append(registry)
            registry.register_schema("synthetic_builtin", current_version="1")

    monkeypatch.setattr(record_migrations, "_DEFAULT_REGISTRY", None)
    monkeypatch.setattr(
        record_migrations, "_BUILTIN_REGISTRATIONS", ("synthetic.builtins",)
    )
    monkeypatch.setattr(
        record_migrations.importlib, "import_module", lambda name: Builtins
    )

    first = record_migrations.default_registry()
    second = record_migrations.default_registry()

    assert first is second
    assert calls == [first]
    assert first.schema("synthetic_builtin").current_version == "1"


def test_module_docstring_example_runs() -> None:
    result = doctest.testmod(record_migrations, optionflags=doctest.ELLIPSIS)

    assert result.attempted > 0
    assert result.failed == 0


# Built-in registrations -----------------------------------------------------

BUILTIN_EDGES = {
    ("clinical_review_packet", "1.0.0", "1.1.0"),
    ("clinical_review_packet_mapping", "1", "2"),
    ("decision_result", "1.0.0", "1.1.0"),
    ("decision_result", "1.1.0", "1.2.0"),
    ("dp_budget_ledger", "unversioned", "openmed.dp_budget_ledger.v1"),
}
EXPECTED_BUILTIN_FIELDS = {
    "clinical_review_packet/1.0.0-to-1.1.0": {
        "added": [
            "/compatibility_policy",
            "/extensions",
            "/transition_ids",
            "/transitions/*/compatibility_policy",
        ],
        "changed": ["/schema_version", "/transitions/*/schema_version"],
        "dropped": [],
        "moved": [],
    },
    "clinical_review_packet_mapping/1-to-2": {
        "added": ["/safety/privacy_scan_required"],
        "changed": ["/schema_version"],
        "dropped": [],
        "moved": [],
    },
    "decision_result/1.0.0-to-1.1.0": {
        "added": ["/questions"],
        "changed": ["/schema_version"],
        "dropped": [],
        "moved": [
            {"from": "/extensions/expected_level", "to": "/expected_level"},
            {"from": "/extensions/level_scores", "to": "/level_scores"},
        ],
    },
    "decision_result/1.1.0-to-1.2.0": {
        "added": [],
        "changed": ["/schema_version"],
        "dropped": [],
        "moved": [],
    },
    "dp_budget_ledger/unversioned-to-v1": {
        "added": ["/entries", "/schema_version"],
        "changed": [],
        "dropped": ["/compositions", "/ledger", "/policies"],
        "moved": [],
    },
}
# Values present in the built-in fixtures that must never reach a report.
BUILTIN_VALUE_CANARIES = (
    "reviewpacket_aaaaaaaaaaaaaaaa",
    "fact_aaaaaaaaaaaaaaaa",
    "reviewer_assigned",
    "release-a",
    "Synthetic urgency",
    "assistive review evidence",
    "NAME",
    "2026-09-21",
)


def _builtin_registry() -> MigratorRegistry:
    registry = MigratorRegistry()
    record_migration_catalog.register_record_migrations(registry)
    return registry


def _builtin_edges() -> list[record_migrations.MigrationEdge]:
    return list(_builtin_registry().edges())


def _load_pair(edge: record_migrations.MigrationEdge) -> tuple[Any, Any]:
    before_path, after_path = edge.fixture_pair(ROOT)
    return (
        json.loads(before_path.read_text(encoding="utf-8")),
        json.loads(after_path.read_text(encoding="utf-8")),
    )


def test_default_registry_holds_exactly_the_builtin_edges() -> None:
    edges = record_migrations.default_registry().edges()

    assert {(edge.schema_id, edge.from_version, edge.to_version) for edge in edges} == (
        BUILTIN_EDGES
    )
    assert {edge.fixture for edge in edges} == set(EXPECTED_BUILTIN_FIELDS)


@pytest.mark.parametrize(
    "edge", _builtin_edges(), ids=lambda edge: edge.fixture.replace("/", ":")
)
def test_each_builtin_edge_migrates_its_before_fixture_to_exactly_its_after_fixture(
    edge: record_migrations.MigrationEdge,
) -> None:
    registry = _builtin_registry()
    before, after = _load_pair(edge)
    original = copy.deepcopy(before)

    result = registry.migrate_record(
        before, schema_id=edge.schema_id, to_version=edge.to_version
    )

    assert before == original
    assert result.record == after
    assert result.report.source_version == edge.from_version
    assert result.report.target_version == edge.to_version
    assert [step.fixture for step in result.report.steps] == [edge.fixture]
    payload = result.report.to_dict()
    assert {
        key: payload[key] for key in ("added", "changed", "dropped", "moved")
    } == EXPECTED_BUILTIN_FIELDS[edge.fixture]
    rendered = json.dumps(payload, sort_keys=True) + repr(result)
    for canary in BUILTIN_VALUE_CANARIES:
        assert canary not in rendered
    current = registry.migrate_record(
        after, schema_id=edge.schema_id, to_version=edge.to_version
    )
    assert current.report.status == "current"


def test_lossy_builtin_edges_are_marked_lossy_and_list_dropped_paths() -> None:
    registry = _builtin_registry()

    for edge in registry.edges():
        before, _ = _load_pair(edge)
        report = registry.migrate_record(
            before, schema_id=edge.schema_id, to_version=edge.to_version
        ).report
        assert report.lossy is (not edge.lossless)
        if edge.lossless:
            assert report.dropped == ()
        else:
            assert report.dropped
            assert report.steps[0].fields.dropped == report.dropped
    assert [edge.schema_id for edge in registry.edges() if not edge.lossless] == [
        "dp_budget_ledger"
    ]


def test_builtin_chains_reach_the_current_version_of_every_supported_version() -> None:
    from openmed.clinical.review_packet_mapping_migrations import (
        REVIEW_PACKET_SCHEMA_VERSION,
        SUPPORTED_REVIEW_PACKET_SCHEMA_VERSIONS,
    )
    from openmed.clinical.review_packet_migrations import (
        SUPPORTED_REVIEW_PACKET_VERSIONS,
    )
    from openmed.clinical.review_transitions import (
        CLINICAL_REVIEW_PACKET_SCHEMA_VERSION,
    )
    from openmed.risk.budget_migration import BudgetLedgerSnapshot
    from openmed.structured.decision import (
        DECISION_SCHEMA_VERSION,
        decision_result_schema,
    )

    registry = _builtin_registry()
    ledger_version = {
        item.name: item.default for item in dataclasses.fields(BudgetLedgerSnapshot)
    }["schema_version"]
    current = {
        "clinical_review_packet": (
            CLINICAL_REVIEW_PACKET_SCHEMA_VERSION,
            SUPPORTED_REVIEW_PACKET_VERSIONS,
        ),
        "clinical_review_packet_mapping": (
            str(REVIEW_PACKET_SCHEMA_VERSION),
            tuple(str(item) for item in SUPPORTED_REVIEW_PACKET_SCHEMA_VERSIONS),
        ),
        "decision_result": (
            DECISION_SCHEMA_VERSION,
            tuple(decision_result_schema()["properties"]["schema_version"]["enum"]),
        ),
        "dp_budget_ledger": (ledger_version, ("unversioned", ledger_version)),
    }

    assert {schema.schema_id for schema in registry.schemas()} == set(current)
    for schema_id, (version, supported) in current.items():
        assert registry.schema(schema_id).current_version == version
        for source in supported:
            chain = registry.migration_path(schema_id, source)
            assert (chain[-1].to_version if chain else source) == version


def test_builtin_matchers_identify_their_fixtures() -> None:
    registry = _builtin_registry()

    for edge in registry.edges():
        if registry.schema(edge.schema_id).matcher is None:
            continue
        for record in _load_pair(edge):
            assert registry.identify(record).schema_id == edge.schema_id
    mapping_edge = registry.edges("clinical_review_packet_mapping")[0]
    with pytest.raises(RecordMigrationRefused) as caught:
        registry.identify(_load_pair(mapping_edge)[0])
    assert caught.value.refusal.reason == "unknown_schema"
    with pytest.raises(RecordMigrationRefused):
        registry.identify(_fixture(f"{SCHEMA}/1.0.0-to-1.1.0", "before"))


def test_builtin_migration_failures_are_value_free_refusals() -> None:
    registry = _builtin_registry()
    edge = registry.edges("clinical_review_packet")[0]
    before, _ = _load_pair(edge)
    before["fact_ids"] = ["SYNTHETIC-PRIVATE-CANARY"]

    with pytest.raises(RecordMigrationRefused) as caught:
        registry.migrate_record(before, schema_id="clinical_review_packet")

    assert caught.value.refusal.to_dict() == {
        "reason": "migrator_failed",
        "schema_id": "clinical_review_packet",
        "from_version": "1.0.0",
        "to_version": "1.1.0",
        "error_type": "BuiltinMigrationError",
    }
    assert "CANARY" not in str(caught.value) + repr(caught.value.details)


def test_builtin_budget_migration_is_checked_by_the_budget_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openmed.risk import budget_migration

    registry = _builtin_registry()
    edge = registry.edges("dp_budget_ledger")[0]
    before, _ = _load_pair(edge)
    calls: list[str] = []
    original = budget_migration.enforce_budget_migration

    def spy(previous: Any, current: Any) -> Any:
        calls.append(current["schema_version"])
        return original(previous, current)

    monkeypatch.setattr(budget_migration, "enforce_budget_migration", spy)

    registry.migrate_record(before, schema_id="dp_budget_ledger")

    assert calls == ["openmed.dp_budget_ledger.v1"]


def test_builtin_registrations_do_not_import_the_owning_packages() -> None:
    script = (
        "import sys\n"
        "from openmed.core.record_migrations import default_registry\n"
        "registry = default_registry()\n"
        "assert len(registry.edges()) == 5\n"
        "owners = ('openmed.clinical', 'openmed.structured', 'openmed.risk')\n"
        "print(sorted(name for name in sys.modules if name.startswith(owners)))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    env.pop("HF_TOKEN", None)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=True,
        cwd=ROOT,
        env=env,
        text=True,
    )

    assert completed.stdout.strip() == "[]"
