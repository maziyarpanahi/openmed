"""Bounded, value-free diagnostic and recovery bundle contracts.

Every fixture is synthetic and offline.  The frozen state plants sentinels in
configuration values, file and directory names, store payloads, queue records,
environment variables and exception messages; the bundle must contain none.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import socket
import sqlite3
import stat
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openmed.core.doctor import value_free_health_checks
from openmed.service import diagnostics
from openmed.service.diagnostics import (
    CATEGORIES,
    FINDING_CODES,
    PARTIAL_RESULT_CODES,
    RECOVERY_ACTIONS,
    RECOVERY_KINDS,
    RECOVERY_RULES,
    DiagnosticContext,
    DiagnosticLimits,
    DiagnosticRequest,
    DiagnosticRequestError,
    DiagnosticSources,
    collect_diagnostic_bundle,
    read_diagnostic_bundle,
    verify_diagnostic_bundle,
    write_diagnostic_bundle,
)
from openmed.service.metrics import PrometheusMetricsRegistry
from openmed.service.operational import (
    OperationalCategory,
    OperationalEvent,
    OperationalState,
)
from tests.fixtures.diagnostic_state import (
    SENTINEL_KEY,
    SENTINEL_MRN,
    SENTINEL_NAME,
    SENTINEL_TOKEN,
    FrozenState,
    assert_no_sentinels,
    build_frozen_state,
    plant_sentinel_environment,
    registry_document,
    snapshot_tree,
)

ROOT = Path(__file__).resolve().parents[3]
DOCS_PAGE = ROOT / "docs" / "recovery" / "diagnostic-bundle.md"
FIXED_NOW = datetime(2026, 11, 7, 10, 41, 13, tzinfo=timezone.utc)


def _sources(state: FrozenState, **overrides: Any) -> DiagnosticSources:
    values: dict[str, Any] = {
        "config_path": state.config_path,
        "profiles_dir": state.profiles_dir,
        "jobs_store_path": state.jobs_store_path,
        "journey_store_root": state.journey_store_root,
        "registry_state_path": state.registry_state_path,
        "audit_chain_path": state.audit_chain_path,
        "ambient_defaults": False,
    }
    values.update(overrides)
    return DiagnosticSources(**values)


def _context(**overrides: Any) -> DiagnosticContext:
    values: dict[str, Any] = {"wall_clock": lambda: FIXED_NOW}
    values.update(overrides)
    return DiagnosticContext(**values)


def _request(*categories: str, **limits: int) -> DiagnosticRequest:
    return DiagnosticRequest(
        categories=categories or CATEGORIES, limits=DiagnosticLimits(**limits)
    )


def _entry(bundle: Any, category: str) -> dict[str, Any]:
    document = bundle.document if hasattr(bundle, "document") else bundle
    return next(item for item in document["entries"] if item["category"] == category)


@pytest.fixture
def state(tmp_path: Path) -> FrozenState:
    return build_frozen_state(tmp_path)


class _FakeRuntime:
    """In-process runtime double exposing only the documented seams."""

    preload_models = (f"OpenMed/{SENTINEL_NAME}",)
    max_resident_models = 2

    def served_model_ids(self) -> tuple[str, ...]:
        return (f"OpenMed/{SENTINEL_MRN}", "OpenMed/public-model")

    def model_policy_report(self) -> dict[str, Any]:
        return {"explicit": False, "findings": ["served_models_not_explicit"]}

    def loaded_models(self) -> dict[str, Any]:
        return {"models": {f"OpenMed/{SENTINEL_NAME}": {"path": "/x"}}}


def _live_app() -> SimpleNamespace:
    metrics = PrometheusMetricsRegistry()
    metrics.request_started()
    metrics.request_finished(
        route=f"/v1/{SENTINEL_TOKEN}", status_code=500, duration_seconds=0.1
    )
    metrics.request_finished(route="/v1/deid", status_code=200, duration_seconds=0.1)
    metrics.record_operational_event(
        OperationalEvent(
            category=OperationalCategory.STORE,
            operation="write",
            state=OperationalState.FAILURE,
            count=3,
        )
    )
    metrics.set_circuit_breaker_state_counts({"closed": 1, "open": 1, "half_open": 0})
    return SimpleNamespace(
        state=SimpleNamespace(
            ready=False,
            shutting_down=False,
            inflight=2,
            metrics=metrics,
            runtime=_FakeRuntime(),
        )
    )


# ---------------------------------------------------------------------------
# Privacy: full sentinel scan
# ---------------------------------------------------------------------------


def test_full_bundle_sentinel_scan_covers_every_source(
    state: FrozenState,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plant_sentinel_environment(monkeypatch)
    caplog.set_level(logging.DEBUG)

    def backup_provider(section: Any) -> dict[str, Any]:
        section.finding("backup_stale")
        section.finding(SENTINEL_TOKEN)
        section.data[SENTINEL_TOKEN] = SENTINEL_TOKEN
        return {
            "state": "present",
            "backup_count": 2,
            "latest_verification": SENTINEL_NAME,
            "latest_manifest_digest": str(state.root),
            "label": SENTINEL_NAME,
            SENTINEL_TOKEN: 1,
            "Nested Key": {"x": SENTINEL_MRN},
        }

    def activation_provider(section: Any) -> dict[str, Any]:
        raise SystemExit(f"{SENTINEL_NAME} {SENTINEL_KEY} {state.root}")

    bundle = collect_diagnostic_bundle(
        sources=_sources(state),
        context=_context(
            app=_live_app(),
            providers={
                "backup": backup_provider,
                "model_activation": activation_provider,
            },
        ),
    )

    payload = bundle.to_bytes().decode("ascii")
    assert_no_sentinels(payload, str(state.root), str(state.root.parent))
    assert_no_sentinels(repr(bundle), str(state.root))
    assert_no_sentinels(repr(_sources(state)), str(state.root))
    for record in caplog.records:
        assert_no_sentinels(record.getMessage(), str(state.root))
    captured = capsys.readouterr()
    assert_no_sentinels(captured.out + captured.err, str(state.root))
    assert "/" not in json.dumps(
        [entry["data"] for entry in bundle.document["entries"]]
    )

    backup = _entry(bundle, "backup")
    assert backup["data"] == {
        "backup_count": 2,
        "source": "provider",
        "state": "present",
    }
    assert backup["findings"] == ["backup_stale"]
    assert backup["omissions"] == {
        "unknown_code": 1,
        "unknown_key": 4,
        "unsafe_value": 2,
    }
    activation = _entry(bundle, "model_activation")
    assert activation["status"] == "failed"
    assert activation["reason_codes"] == ["collector_failed"]
    configuration = _entry(bundle, "configuration")
    assert configuration["findings"] == ["persisted_credential"]
    assert configuration["data"]["profile"] == {
        "class": "custom_file",
        "source_class": "environment",
    }
    assert configuration["data"]["secret_bearing_settings_present"] == [
        "auth_api_keys",
        "otlp_headers",
    ]
    assert configuration["data"]["unrecognized_openmed_variable_count"] == 2
    assert configuration["omissions"]["unknown_key"] == 2
    assert configuration["data"]["keys"]["hf_token"]["conflict_category"] == (
        "not_reported"
    )
    assert verify_diagnostic_bundle(payload).valid


def test_bundle_is_never_restorable_state(state: FrozenState) -> None:
    bundle = collect_diagnostic_bundle(sources=_sources(state), context=_context())
    header = bundle.document["header"]
    assert header["restorable_state"] is False
    serialized = bundle.to_bytes()
    assert b"payload_json" not in serialized
    assert b"text_hash" not in serialized
    assert len(serialized) < DiagnosticLimits().max_bundle_bytes


def test_collection_never_mutates_sources_or_opens_sockets(
    state: FrozenState, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = snapshot_tree(state.root)
    before_listing = sorted(str(path) for path in state.root.rglob("*"))
    attempts: list[str] = []

    def no_network(*args: Any, **kwargs: Any) -> None:
        attempts.append("network")
        raise OSError("network disabled for this test")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    bundle = collect_diagnostic_bundle(sources=_sources(state), context=_context())

    assert attempts == []
    assert snapshot_tree(state.root) == before
    assert sorted(str(path) for path in state.root.rglob("*")) == before_listing
    assert _entry(bundle, "journey_store")["data"]["wal_state"] == "absent"


def test_live_wal_store_is_read_without_creating_files(state: FrozenState) -> None:
    database = state.journey_store_root / "journey.sqlite3"
    writer = sqlite3.connect(database, isolation_level=None)
    try:
        writer.execute("PRAGMA journal_mode = WAL")
        writer.execute(
            "INSERT INTO store_revisions(revision, committed_at) VALUES (2, 'x')"
        )
        sidecars = sorted(path.name for path in database.parent.iterdir())
        bundle = collect_diagnostic_bundle(
            _request("journey_store"), sources=_sources(state), context=_context()
        )
        assert sorted(path.name for path in database.parent.iterdir()) == sidecars
    finally:
        writer.close()
    data = _entry(bundle, "journey_store")["data"]
    assert data["wal_state"] == "present"
    assert data["record_counts"]["store_revisions"] == 2
    assert data["latest_revision"] == 2


def test_section_and_source_repr_hide_paths(state: FrozenState) -> None:
    rendered = repr(_sources(state)) + repr(_context(app=_live_app()))
    assert str(state.root) not in rendered


# ---------------------------------------------------------------------------
# Limits: enforced before and during collection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"max_files": 257}, "limit_above_maximum"),
        ({"max_bundle_bytes": 4 * 1024**2}, "limit_above_maximum"),
        ({"max_elapsed_ms": 10_000_000}, "limit_above_maximum"),
        ({"max_files": 0}, "limit_below_minimum"),
        ({"max_bundle_bytes": 1024}, "limit_below_minimum"),
        ({"max_files": "3"}, "limit_invalid"),
        ({"max_depth": True}, "limit_invalid"),
        ({"max_total_bytes": 1.5}, "limit_invalid"),
    ],
)
def test_limits_are_validated_before_collection(
    overrides: dict[str, Any], code: str
) -> None:
    with pytest.raises(DiagnosticRequestError) as raised:
        DiagnosticLimits(**overrides)
    assert raised.value.code == code
    assert str(raised.value) == code


def test_default_and_maximum_limits_are_documented_values() -> None:
    defaults = DiagnosticLimits().to_dict()
    maximum = DiagnosticLimits.maximum().to_dict()
    assert set(defaults) == set(maximum)
    assert all(defaults[name] <= maximum[name] for name in defaults)
    page = DOCS_PAGE.read_text(encoding="utf-8")
    for name in defaults:
        assert f"`{name}`" in page


@pytest.mark.parametrize(
    ("categories", "code"),
    [
        (("environment_dump",), "category_unknown"),
        (("filesystem",), "category_unknown"),
        (("jobs", "jobs"), "category_duplicate"),
        ((), "category_empty"),
        ("jobs", "request_invalid"),
    ],
)
def test_requests_cannot_select_arbitrary_capture(categories: Any, code: str) -> None:
    with pytest.raises(DiagnosticRequestError) as raised:
        DiagnosticRequest(categories=categories)
    assert raised.value.code == code


def test_providers_are_limited_to_documented_categories() -> None:
    with pytest.raises(DiagnosticRequestError) as raised:
        DiagnosticContext(providers={"jobs": lambda section: {}})
    assert raised.value.code == "provider_category_unsupported"
    with pytest.raises(DiagnosticRequestError) as raised:
        DiagnosticContext(providers={"backup": "not callable"})
    assert raised.value.code == "provider_invalid"


def test_categories_are_normalized_to_canonical_order(state: FrozenState) -> None:
    request = DiagnosticRequest(categories=("disk", "runtime", "jobs"))
    assert request.categories == ("runtime", "jobs", "disk")
    bundle = collect_diagnostic_bundle(
        request, sources=_sources(state), context=_context()
    )
    assert [item["category"] for item in bundle.document["entries"]] == [
        "runtime",
        "jobs",
        "disk",
    ]


def test_file_count_limit_is_enforced_during_collection(state: FrozenState) -> None:
    bundle = collect_diagnostic_bundle(
        _request("configuration", "jobs", max_files=1),
        sources=_sources(state),
        context=_context(),
    )
    jobs = _entry(bundle, "jobs")
    assert jobs["status"] == "partial"
    assert jobs["reason_codes"] == ["file_limit_reached"]
    assert jobs["data"] == {}
    assert "rerun_with_larger_limits" in bundle.recovery_actions


def test_per_file_byte_limit_refuses_oversized_sources(state: FrozenState) -> None:
    size = state.jobs_store_path.stat().st_size
    bundle = collect_diagnostic_bundle(
        _request("jobs", max_file_bytes=size - 1),
        sources=_sources(state),
        context=_context(),
    )
    jobs = _entry(bundle, "jobs")
    assert jobs["status"] == "partial"
    assert jobs["reason_codes"] == ["file_too_large"]
    assert "record_count" not in jobs["data"]


def test_total_byte_limit_spans_all_sources(state: FrozenState) -> None:
    config_size = state.config_path.stat().st_size
    jobs_size = state.jobs_store_path.stat().st_size
    bundle = collect_diagnostic_bundle(
        _request(
            "configuration",
            "jobs",
            max_file_bytes=max(config_size, jobs_size),
            max_total_bytes=config_size + jobs_size - 1,
        ),
        sources=_sources(state),
        context=_context(),
    )
    assert _entry(bundle, "configuration")["status"] == "complete"
    jobs = _entry(bundle, "jobs")
    assert jobs["reason_codes"] == ["total_bytes_limit_reached"]


def test_directory_depth_limit_stops_recursion(state: FrozenState) -> None:
    bundle = collect_diagnostic_bundle(
        _request("journey_store", max_depth=2),
        sources=_sources(state),
        context=_context(),
    )
    journey = _entry(bundle, "journey_store")
    assert journey["status"] == "partial"
    assert "depth_limit_reached" in journey["reason_codes"]
    assert journey["data"]["blobs"]["blob_count"] == 0
    assert journey["data"]["migration_state"] == "current"


def test_directory_entry_limit_stops_scans(state: FrozenState) -> None:
    bundle = collect_diagnostic_bundle(
        _request("journey_store", max_dir_entries=2),
        sources=_sources(state),
        context=_context(),
    )
    journey = _entry(bundle, "journey_store")
    assert "entry_limit_reached" in journey["reason_codes"]


def test_source_json_nesting_is_bounded(state: FrozenState) -> None:
    nested: Any = "leaf"
    for _ in range(diagnostics.MAX_SOURCE_JSON_NESTING + 2):
        nested = [nested]
    state.jobs_store_path.write_text(json.dumps({"jobs": {"a": {"x": nested}}}))
    bundle = collect_diagnostic_bundle(
        _request("jobs"), sources=_sources(state), context=_context()
    )
    assert _entry(bundle, "jobs")["reason_codes"] == ["depth_limit_reached"]


def test_item_limit_truncates_lists(state: FrozenState) -> None:
    state.registry_state_path.write_text(json.dumps(registry_document(extra_slots=3)))
    bundle = collect_diagnostic_bundle(
        _request("model_registry", max_items=2),
        sources=_sources(state),
        context=_context(),
    )
    registry = _entry(bundle, "model_registry")
    assert registry["reason_codes"] == ["item_limit_reached"]
    assert registry["data"]["slot_count"] == 4
    assert len(registry["data"]["slots"]) == 2


def test_bundle_byte_limit_drops_oversized_sections(state: FrozenState) -> None:
    def large(section: Any) -> dict[str, Any]:
        digests = [diagnostics.identifier_digest("model", str(i)) for i in range(4_000)]
        return {"served_model_digests": digests}

    limit = 32 * 1024
    bundle = collect_diagnostic_bundle(
        _request(
            "runtime", "model_activation", max_bundle_bytes=limit, max_items=4_096
        ),
        sources=_sources(state),
        context=_context(providers={"model_activation": large}),
    )
    assert len(bundle.to_bytes()) <= limit
    activation = _entry(bundle, "model_activation")
    assert activation["status"] == "partial"
    assert activation["reason_codes"] == ["section_too_large"]
    assert activation["data"] == {}
    assert _entry(bundle, "runtime")["data"]
    assert verify_diagnostic_bundle(bundle.to_bytes()).valid


# ---------------------------------------------------------------------------
# Unavailable, corrupt, slow, cancelled and oversized collectors
# ---------------------------------------------------------------------------


def test_unconfigured_and_absent_sources_yield_stable_codes(tmp_path: Path) -> None:
    sources = DiagnosticSources(
        jobs_store_path=tmp_path / "missing.json",
        registry_state_path=tmp_path / "missing-registry.json",
        ambient_defaults=False,
    )
    bundle = collect_diagnostic_bundle(sources=sources, context=_context())
    expected = {
        "jobs": ["source_absent"],
        "journey_store": ["source_not_configured"],
        "model_registry": ["source_absent"],
        "audit_continuity": ["source_not_configured"],
        "backup": ["provider_not_registered"],
        "operational_events": ["not_in_process"],
    }
    for category, codes in expected.items():
        entry = _entry(bundle, category)
        assert entry["status"] == "unavailable", category
        assert entry["reason_codes"] == codes, category
    assert _entry(bundle, "readiness")["reason_codes"] == ["not_in_process"]


def test_corrupt_sources_are_reported_without_content(state: FrozenState) -> None:
    state.jobs_store_path.write_text(f'{{"jobs": "{SENTINEL_NAME}"')
    (state.journey_store_root / "journey.sqlite3").write_bytes(
        b"not a database " + SENTINEL_MRN.encode()
    )
    state.registry_state_path.write_text(
        json.dumps({"schema_version": 2, "slots": {SENTINEL_NAME: {}}})
    )
    state.audit_chain_path.write_text(json.dumps({"format": SENTINEL_NAME}))
    bundle = collect_diagnostic_bundle(sources=_sources(state), context=_context())
    assert_no_sentinels(bundle.to_bytes().decode("ascii"), str(state.root))

    jobs = _entry(bundle, "jobs")
    assert (jobs["status"], jobs["reason_codes"]) == ("partial", ["source_corrupt"])
    assert jobs["findings"] == ["job_store_corrupt"]
    journey = _entry(bundle, "journey_store")
    assert "source_corrupt" in journey["reason_codes"]
    assert "journey_store_corrupt" in journey["findings"]
    registry = _entry(bundle, "model_registry")
    assert registry["findings"] == ["registry_state_invalid"]
    audit = _entry(bundle, "audit_continuity")
    assert audit["data"]["state"] == "invalid"
    assert audit["findings"] == ["audit_chain_invalid"]
    actions = bundle.recovery_actions
    for action in (
        "quarantine_job_store",
        "quarantine_journey_store",
        "restore_from_backup",
        "restore_registry_state",
        "escalate_audit_discontinuity",
    ):
        assert action in actions


def test_unsafe_source_types_are_refused(state: FrozenState) -> None:
    state.jobs_store_path.unlink()
    state.jobs_store_path.mkdir()
    bundle = collect_diagnostic_bundle(
        _request("jobs"), sources=_sources(state), context=_context()
    )
    assert _entry(bundle, "jobs")["reason_codes"] == ["source_unsafe"]


def test_tampered_audit_chain_reports_broken_continuity(state: FrozenState) -> None:
    document = json.loads(state.audit_chain_path.read_text())
    document["entries"][0]["report_hash"] = "sha256:" + "0" * 64
    state.audit_chain_path.write_text(json.dumps(document))
    bundle = collect_diagnostic_bundle(
        _request("audit_continuity"), sources=_sources(state), context=_context()
    )
    audit = _entry(bundle, "audit_continuity")
    assert audit["data"]["state"] == "broken"
    assert audit["data"]["result"] == "mutation_detected"
    assert audit["data"]["failed_entry_index"] == 0
    assert audit["findings"] == ["audit_chain_broken"]


def test_registry_v1_state_requires_migration(state: FrozenState) -> None:
    state.registry_state_path.write_text(json.dumps({"schema_version": 1}))
    bundle = collect_diagnostic_bundle(
        _request("model_registry"), sources=_sources(state), context=_context()
    )
    registry = _entry(bundle, "model_registry")
    assert registry["findings"] == ["registry_migration_required"]
    assert registry["reason_codes"] == ["schema_unsupported"]
    assert "rerun_migrations" in bundle.recovery_actions


def test_journey_migration_states_are_classified(state: FrozenState) -> None:
    database = state.journey_store_root / "journey.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("DELETE FROM schema_migrations WHERE version = 3")
    connection.execute(
        "UPDATE schema_migrations SET checksum = 'changed' WHERE version = 1"
    )
    connection.commit()
    connection.close()
    bundle = collect_diagnostic_bundle(
        _request("journey_store"), sources=_sources(state), context=_context()
    )
    journey = _entry(bundle, "journey_store")
    assert journey["data"]["migration_state"] == "checksum_mismatch"
    assert journey["data"]["migrations"][0] == {
        "state": "checksum_mismatch",
        "version": 1,
    }
    assert "migration_checksum_mismatch" in journey["findings"]


def test_slow_collector_times_out_and_later_sections_run(state: FrozenState) -> None:
    release = threading.Event()

    def slow(section: Any) -> dict[str, Any]:
        release.wait(30)
        return {"state": SENTINEL_TOKEN}

    try:
        bundle = collect_diagnostic_bundle(
            _request("backup", "disk", section_timeout_ms=50),
            sources=_sources(state),
            context=_context(providers={"backup": slow}),
        )
    finally:
        release.set()
    backup = _entry(bundle, "backup")
    assert (backup["status"], backup["reason_codes"]) == (
        "timeout",
        ["section_timeout"],
    )
    assert backup["data"] == {}
    assert _entry(bundle, "disk")["status"] == "complete"
    assert "rerun_with_larger_limits" in bundle.recovery_actions


def test_elapsed_budget_exhaustion_skips_remaining_sections(
    state: FrozenState,
) -> None:
    clock = {"now": 0.0}

    def advance(section: Any) -> dict[str, Any]:
        clock["now"] += 120.0
        section.check()
        return {}

    bundle = collect_diagnostic_bundle(
        _request("model_activation", "backup", "disk", max_elapsed_ms=60_000),
        sources=_sources(state),
        context=_context(
            monotonic=lambda: clock["now"],
            providers={"model_activation": advance},
        ),
    )
    assert _entry(bundle, "model_activation")["reason_codes"] == ["section_timeout"]
    for category in ("backup", "disk"):
        entry = _entry(bundle, category)
        assert entry["status"] == "timeout"
        assert entry["reason_codes"] == ["elapsed_budget_exhausted"]


def test_long_sqlite_queries_are_interrupted_at_the_deadline(
    tmp_path: Path,
) -> None:
    database = tmp_path / "slow.sqlite3"
    sqlite3.connect(database).close()
    interrupted = threading.Event()
    finished = threading.Event()

    def endless_query(section: Any) -> dict[str, Any]:
        connection = section.open_sqlite(database)
        try:
            diagnostics._guard_sql(
                section,
                lambda: connection.execute(
                    "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
                    "SELECT COUNT(*) FROM c"
                ).fetchone(),
            )
        except Exception:
            interrupted.set()
            raise
        finally:
            connection.close()
            finished.set()
        return {}

    bundle = collect_diagnostic_bundle(
        _request("backup", section_timeout_ms=100),
        sources=DiagnosticSources(ambient_defaults=False),
        context=_context(providers={"backup": endless_query}),
    )
    assert finished.wait(10)
    assert interrupted.is_set()
    assert _entry(bundle, "backup")["reason_codes"] == ["section_timeout"]


def test_journey_collector_checks_its_deadline_between_queries(
    state: FrozenState,
) -> None:
    clock = {"now": 0.0}

    def ticking() -> float:
        clock["now"] += 0.05
        return clock["now"]

    bundle = collect_diagnostic_bundle(
        _request("journey_store", section_timeout_ms=1_000),
        sources=_sources(state),
        context=_context(monotonic=ticking),
    )
    journey = _entry(bundle, "journey_store")
    assert journey["status"] == "timeout"
    assert journey["reason_codes"] == ["section_timeout"]


def test_cancellation_stops_collection_with_stable_codes(state: FrozenState) -> None:
    cancel = threading.Event()

    def cancel_now(section: Any) -> dict[str, Any]:
        cancel.set()
        section.check()
        return {}

    bundle = collect_diagnostic_bundle(
        _request("runtime", "model_activation", "backup", "disk"),
        sources=_sources(state),
        context=_context(
            cancel_event=cancel, providers={"model_activation": cancel_now}
        ),
    )
    assert _entry(bundle, "runtime")["status"] == "complete"
    for category in ("model_activation", "backup", "disk"):
        entry = _entry(bundle, category)
        assert entry["status"] == "cancelled"
        assert entry["reason_codes"] == ["collection_cancelled"]
    assert "retry_collection" in bundle.recovery_actions


def test_every_emitted_code_is_from_the_stable_vocabulary(state: FrozenState) -> None:
    bundle = collect_diagnostic_bundle(
        sources=_sources(state), context=_context(app=_live_app())
    )
    for entry in bundle.document["entries"]:
        assert set(entry["reason_codes"]) <= PARTIAL_RESULT_CODES
        assert set(entry["findings"]) <= FINDING_CODES
        assert set(entry["omissions"]) <= diagnostics.OMISSION_REASONS


# ---------------------------------------------------------------------------
# In-process evidence
# ---------------------------------------------------------------------------


def test_live_service_readiness_metrics_and_activation(state: FrozenState) -> None:
    bundle = collect_diagnostic_bundle(
        _request("readiness", "operational_events", "model_activation"),
        sources=_sources(state),
        context=_context(app=_live_app()),
    )
    readiness = _entry(bundle, "readiness")
    assert readiness["status"] == "complete"
    assert readiness["data"]["process"] == {
        "inflight_requests": 2,
        "state": "not_ready",
    }
    assert readiness["findings"] == ["process_not_ready"]
    assert set(readiness["data"]["service_settings"].values()) == {"valid"}

    events = _entry(bundle, "operational_events")
    assert events["data"]["operational"] == {"store": {"write": {"failure": 3}}}
    assert events["data"]["request_status_classes"] == {
        "status_2xx": 1,
        "status_5xx": 1,
    }
    assert set(events["findings"]) == {
        "circuit_breaker_open",
        "operational_failures_observed",
        "server_errors_observed",
    }

    activation = _entry(bundle, "model_activation")
    assert activation["status"] == "complete"
    assert activation["data"]["source"] == "in_process"
    assert activation["data"]["served_model_count"] == 2
    assert activation["data"]["resident_model_count"] == 1
    assert all(
        value.startswith("sha256:")
        for value in activation["data"]["served_model_digests"]
    )
    assert_no_sentinels(bundle.to_bytes().decode("ascii"))


def test_invalid_service_settings_are_reported_by_group(
    state: FrozenState, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_SIZE", SENTINEL_NAME)
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_WORKERS", "0")
    bundle = collect_diagnostic_bundle(
        _request("readiness"), sources=_sources(state), context=_context()
    )
    readiness = _entry(bundle, "readiness")
    assert readiness["data"]["service_settings"]["batching"] == "invalid"
    assert readiness["data"]["service_settings"]["jobs"] == "invalid"
    assert readiness["findings"] == ["service_setting_invalid"]
    assert_no_sentinels(bundle.to_bytes().decode("ascii"))


def test_metrics_diagnostic_snapshot_is_value_free() -> None:
    snapshot = _live_app().state.metrics.diagnostic_snapshot()
    assert snapshot["request_status_classes"] == {"status_2xx": 1, "status_5xx": 1}
    assert snapshot["operational"] == {"store": {"write": {"failure": 3}}}
    assert snapshot["circuit_breaker"] == {"closed": 1, "open": 1, "half_open": 0}
    assert SENTINEL_TOKEN not in json.dumps(snapshot)


def test_doctor_value_free_health_checks_hide_environment_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environ = {
        "HF_TOKEN": SENTINEL_KEY,
        "HF_ENDPOINT": f"http://{SENTINEL_NAME}.example.test",
        "HTTPS_PROXY": f"http://user:{SENTINEL_KEY}@proxy.example.test",
        "OPENMED_OFFLINE": "1",
    }
    checks = {item["name"]: item for item in value_free_health_checks(environ)}
    assert checks["hf_token"]["code"] == "token_present"
    assert checks["hf_endpoint"] == {
        "name": "hf_endpoint",
        "status": "WARN",
        "code": "endpoint_not_https",
    }
    assert checks["proxy"]["code"] == "proxy_configured"
    assert checks["openmed_offline"]["code"] == "offline_enabled"
    assert all(set(item) == {"name", "status", "code"} for item in checks.values())
    assert_no_sentinels(json.dumps(list(checks.values())))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_repeated_collection_from_frozen_state_is_byte_identical(
    state: FrozenState, monkeypatch: pytest.MonkeyPatch
) -> None:
    plant_sentinel_environment(monkeypatch)
    first = collect_diagnostic_bundle(sources=_sources(state), context=_context())
    second = collect_diagnostic_bundle(sources=_sources(state), context=_context())
    assert first.to_bytes() == second.to_bytes()
    assert first.root_digest == second.root_digest


def test_only_the_documented_time_field_varies(state: FrozenState) -> None:
    first = collect_diagnostic_bundle(sources=_sources(state), context=_context())
    later = datetime(2026, 11, 7, 12, 5, tzinfo=timezone.utc)
    second = collect_diagnostic_bundle(
        sources=_sources(state), context=_context(wall_clock=lambda: later)
    )
    assert first.document["header"]["generated_at"] == "2026-11-07T10:00:00Z"
    assert second.document["header"]["generated_at"] == "2026-11-07T12:00:00Z"
    assert first.document["entries"] == second.document["entries"]
    assert first.document["recovery"] == second.document["recovery"]
    first_header = dict(first.document["header"], generated_at=None)
    second_header = dict(second.document["header"], generated_at=None)
    assert first_header == second_header
    assert first.root_digest != second.root_digest


# ---------------------------------------------------------------------------
# Offline verifier
# ---------------------------------------------------------------------------


@pytest.fixture
def document(state: FrozenState) -> dict[str, Any]:
    bundle = collect_diagnostic_bundle(sources=_sources(state), context=_context())
    return json.loads(bundle.to_bytes())


def _refresh_digests(document: dict[str, Any]) -> None:
    for entry in document["entries"]:
        entry["digest"] = diagnostics._entry_digest(entry)
    document["recovery"] = diagnostics._recovery(document["entries"])
    document["header"]["collection"] = diagnostics._collection_summary(
        document["entries"]
    )
    document["manifest"] = diagnostics._manifest(
        document["header"], document["entries"], document["recovery"]
    )


def test_verifier_accepts_untouched_bundle(document: dict[str, Any]) -> None:
    result = verify_diagnostic_bundle(document)
    assert result.valid
    assert result.reason_code == "bundle_valid"
    assert result.entry_count == len(CATEGORIES)
    assert verify_diagnostic_bundle(json.dumps(document)).valid


def test_verifier_detects_missing_entry(document: dict[str, Any]) -> None:
    document["entries"].pop(4)
    assert verify_diagnostic_bundle(document).reason_code == "entry_missing"


def test_verifier_detects_added_entry(document: dict[str, Any]) -> None:
    document["entries"].append(copy.deepcopy(document["entries"][0]))
    assert verify_diagnostic_bundle(document).reason_code == "entry_added"


def test_verifier_detects_reordered_entries(document: dict[str, Any]) -> None:
    entries = document["entries"]
    entries[1], entries[2] = entries[2], entries[1]
    result = verify_diagnostic_bundle(document)
    assert (result.reason_code, result.entry_index) == ("entry_reordered", 1)


def test_verifier_detects_modified_entry(document: dict[str, Any]) -> None:
    jobs = _entry(document, "jobs")
    jobs["data"]["record_count"] += 1
    result = verify_diagnostic_bundle(document)
    assert result.reason_code == "entry_modified"
    assert result.entry_index == CATEGORIES.index("jobs")


def test_verifier_detects_entry_with_recomputed_digest(
    document: dict[str, Any],
) -> None:
    jobs = _entry(document, "jobs")
    jobs["data"]["record_count"] += 1
    jobs["digest"] = diagnostics._entry_digest(jobs)
    assert verify_diagnostic_bundle(document).reason_code == "entry_modified"


def test_verifier_detects_modified_manifest(document: dict[str, Any]) -> None:
    document["manifest"]["entries"][0]["digest"] = "sha256:" + "0" * 64
    assert verify_diagnostic_bundle(document).reason_code == "manifest_modified"


def test_verifier_detects_modified_header(document: dict[str, Any]) -> None:
    document["header"]["limits"]["max_files"] = 1
    assert verify_diagnostic_bundle(document).reason_code == "header_modified"


def test_verifier_detects_modified_recovery(document: dict[str, Any]) -> None:
    document["recovery"]["actions"] = []
    assert verify_diagnostic_bundle(document).reason_code == "recovery_inconsistent"


def test_verifier_reapplies_value_safety_after_full_rehash(
    document: dict[str, Any],
) -> None:
    _entry(document, "jobs")["data"]["note"] = SENTINEL_NAME
    _refresh_digests(document)
    result = verify_diagnostic_bundle(document)
    assert result.reason_code == "unsafe_content"
    assert result.entry_index == CATEGORIES.index("jobs")


def test_verifier_accepts_consistent_rehash_of_safe_content(
    document: dict[str, Any],
) -> None:
    _entry(document, "jobs")["data"]["record_count"] = 99
    _refresh_digests(document)
    assert verify_diagnostic_bundle(document).valid


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda doc: doc.update(format="other"), "format_unsupported"),
        (lambda doc: doc.update(schema_version=2), "schema_version_unsupported"),
        (lambda doc: doc.update(extra=1), "unexpected_field"),
        (lambda doc: doc.pop("manifest"), "missing_field"),
        (lambda doc: doc["header"].update(restorable_state=True), "header_invalid"),
        (lambda doc: doc["manifest"].update(entry_count=1), "manifest_invalid"),
        (lambda doc: doc["entries"][0].update(status="fine"), "entry_invalid"),
    ],
)
def test_verifier_rejects_schema_violations(
    document: dict[str, Any], mutate: Any, code: str
) -> None:
    mutate(document)
    assert verify_diagnostic_bundle(document).reason_code == code


def test_verifier_rejects_non_json_and_oversized_input() -> None:
    assert verify_diagnostic_bundle(b"{not json").reason_code == "bundle_not_json"
    assert verify_diagnostic_bundle(b"[]").reason_code == "bundle_not_object"
    too_large = b" " * (DiagnosticLimits.maximum().max_bundle_bytes + 2)
    assert verify_diagnostic_bundle(too_large).reason_code == "bundle_too_large"


# ---------------------------------------------------------------------------
# Recovery guidance
# ---------------------------------------------------------------------------


def test_recovery_guidance_distinguishes_every_kind(state: FrozenState) -> None:
    state.jobs_store_path.write_text("{")
    connection = sqlite3.connect(state.journey_store_root / "journey.sqlite3")
    connection.execute("DELETE FROM schema_migrations WHERE version = 3")
    connection.commit()
    connection.close()
    document = json.loads(state.audit_chain_path.read_text())
    document["entry_count"] += 1
    state.audit_chain_path.write_text(json.dumps(document))
    bundle = collect_diagnostic_bundle(
        sources=_sources(state),
        context=_context(app=_live_app()),
    )
    actions = bundle.document["recovery"]["actions"]
    assert {item["kind"] for item in actions} == set(RECOVERY_KINDS)
    kinds = [RECOVERY_KINDS.index(item["kind"]) for item in actions]
    assert kinds == sorted(kinds)
    for item in actions:
        assert set(item) == {"action", "kind", "triggers", "sections", "doc"}
        assert item["doc"] == f"{diagnostics.DOCS_PAGE}#{item['action']}"
        assert set(item["triggers"]) <= PARTIAL_RESULT_CODES | FINDING_CODES
        assert set(item["sections"]) <= set(CATEGORIES)
    assert_no_sentinels(json.dumps(actions), str(state.root))


def test_every_finding_and_action_has_a_rule_and_documentation() -> None:
    assert set(RECOVERY_RULES) <= PARTIAL_RESULT_CODES | FINDING_CODES
    assert FINDING_CODES <= set(RECOVERY_RULES)
    used = {action for actions in RECOVERY_RULES.values() for action in actions}
    assert used == set(RECOVERY_ACTIONS)
    assert set(RECOVERY_ACTIONS.values()) == set(RECOVERY_KINDS)
    page = DOCS_PAGE.read_text(encoding="utf-8")
    headings = set(re.findall(r"^### ([a-z_]+)$", page, flags=re.MULTILINE))
    assert set(RECOVERY_ACTIONS) <= headings
    for code in PARTIAL_RESULT_CODES | FINDING_CODES:
        assert f"`{code}`" in page, code


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------


def test_bundle_file_is_owner_only_and_never_overwritten_silently(
    state: FrozenState, tmp_path: Path
) -> None:
    bundle = collect_diagnostic_bundle(
        _request("runtime"), sources=_sources(state), context=_context()
    )
    target = tmp_path / "bundle.json"
    write_diagnostic_bundle(bundle, target)
    if os.name == "posix":
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert read_diagnostic_bundle(target) == bundle.to_bytes()
    with pytest.raises(DiagnosticRequestError) as raised:
        write_diagnostic_bundle(bundle, target)
    assert raised.value.code == "output_exists"
    write_diagnostic_bundle(bundle, target, overwrite=True)
    assert verify_diagnostic_bundle(read_diagnostic_bundle(target)).valid
    assert sorted(path.name for path in tmp_path.iterdir() if path.is_file()) == [
        "bundle.json"
    ]


def test_jobs_store_defaults_match_the_service(monkeypatch: pytest.MonkeyPatch) -> None:
    from openmed.service import jobs

    assert diagnostics._JOBS_STORE_ENV_VAR == jobs.SERVICE_JOBS_STORE_ENV_VAR
    monkeypatch.delenv(jobs.SERVICE_JOBS_STORE_ENV_VAR, raising=False)
    assert DiagnosticSources().resolved_jobs_store_path() == jobs.parse_jobs_store_path(
        None
    )
    monkeypatch.setenv(jobs.SERVICE_JOBS_STORE_ENV_VAR, "/srv/jobs.json")
    assert DiagnosticSources().resolved_jobs_store_path() == Path("/srv/jobs.json")
