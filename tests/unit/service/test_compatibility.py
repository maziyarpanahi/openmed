"""Unit tests for the mixed-version service compatibility contract."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import grpc
import pytest
from fastapi.testclient import TestClient

from openmed.service import compatibility as compat
from openmed.service import jobs
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.auth import ROUTE_POLICIES, hash_api_key
from openmed.service.compatibility import (
    CONTRACT_HISTORY,
    CURRENT_CONTRACT,
    PEER_CONTRACT_HEADER,
    PEER_CONTRACT_METADATA_KEY,
    REASON_CODES,
    CompatibilityDescriptor,
    CompatibilityError,
    CompatibilityGate,
    CompatibilityState,
    IncompatibleStateError,
    RolloutEvidence,
    SchemaObservation,
    SchemaRange,
    WorkEnvelope,
    check_work,
    compatibility_matrix,
    evaluate_component,
    evaluate_persisted_state,
    job_store_observation,
    journey_store_observation,
    local_descriptor,
    model_generation_ref,
    negotiate_peer,
    peer_contract_decision,
    rollback_decision,
    validate_contract_history,
)
from openmed.service.proto.generated import openmed_pb2
from openmed.service.schemas import DeidentifyJobDocument, DeidentifyJobRequest
from openmed.structured.store import LATEST_MIGRATION_VERSION
from openmed.structured.store.postgres import LATEST_POSTGRES_MIGRATION_VERSION

LOOPBACK_BASE_URL = "http://127.0.0.1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TOKEN = re.compile(r"[a-z0-9_.\-/]+\Z")


class FakeLoader:
    """Model loader double; compatibility checks never load models."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, Any]:
        return {}


@pytest.fixture(autouse=True)
def service_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for name in (
        "OPENMED_SERVICE_AUTH_ENABLED",
        "OPENMED_SERVICE_AUTH_API_KEYS",
        "OPENMED_SERVICE_PRELOAD_MODELS",
        "OPENMED_SERVICE_SERVED_MODELS",
        "OPENMED_SERVICE_JOBS_WORKERS",
    ):
        monkeypatch.delenv(name, raising=False)
    store_path = tmp_path / "service-jobs.json"
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_STORE_PATH", str(store_path))
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    return store_path


def _contract(
    generation: int,
    minimum: int,
    *,
    jobs_range: tuple[int, int, int] = (1, 1, 1),
    capabilities: frozenset[str] = frozenset({"jobs.deidentify.v1"}),
) -> CompatibilityDescriptor:
    read_min, read_max, write = jobs_range
    return CompatibilityDescriptor(
        contract_generation=generation,
        min_peer_generation=minimum,
        durable_schemas={"jobs": SchemaRange(read_min, read_max, write)},
        capabilities=capabilities,
    )


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            item for key, inner in value.items() for item in (key, *_strings(inner))
        ]
    if isinstance(value, list):
        return [item for inner in value for item in _strings(inner)]
    return []


def _assert_value_free(payload: Any, *forbidden: str) -> None:
    for text in _strings(payload):
        assert _DIGEST.fullmatch(text) or _TOKEN.fullmatch(text), text
    serialized = json.dumps(payload)
    for item in forbidden:
        assert item not in serialized


# --- Published contract -----------------------------------------------------


def test_current_contract_declares_a_self_consistent_n_minus_one_window() -> None:
    assert CONTRACT_HISTORY[-1] is CURRENT_CONTRACT
    assert CURRENT_CONTRACT.contract_generation == compat.CONTRACT_GENERATION
    assert CURRENT_CONTRACT.min_peer_generation == compat.MIN_PEER_GENERATION
    assert CURRENT_CONTRACT.peer_contract == "1/1"
    assert CURRENT_CONTRACT.capabilities == compat.WORKER_CAPABILITIES
    validate_contract_history()


def test_durable_schema_table_matches_the_store_migration_chains() -> None:
    schemas = CURRENT_CONTRACT.durable_schemas
    assert schemas["jobs"].write == compat.JOB_STORE_SCHEMA_VERSION
    assert schemas["journey_local"] == SchemaRange(
        LATEST_MIGRATION_VERSION, LATEST_MIGRATION_VERSION, LATEST_MIGRATION_VERSION
    )
    assert schemas["journey_postgres"] == SchemaRange(
        LATEST_POSTGRES_MIGRATION_VERSION,
        LATEST_POSTGRES_MIGRATION_VERSION,
        LATEST_POSTGRES_MIGRATION_VERSION,
    )


def test_contract_history_rejects_gaps_wide_windows_and_incompatible_steps() -> None:
    one = _contract(1, 1)
    two = _contract(2, 1, jobs_range=(1, 2, 1))
    three = _contract(3, 2, jobs_range=(1, 2, 2))
    validate_contract_history((one, two))
    for history in ((), (one, three), (one, two, three), (one, _contract(2, 2))):
        with pytest.raises(CompatibilityError) as raised:
            validate_contract_history(history)
        assert raised.value.code == "contract_history_invalid"


def test_descriptor_round_trips_and_digest_is_canonical() -> None:
    descriptor = local_descriptor(
        SimpleNamespace(served_model_ids=lambda: ("OpenMed/synthetic-pii-v1",))
    )
    payload = descriptor.to_dict()
    assert CompatibilityDescriptor.from_dict(json.loads(json.dumps(payload))) == (
        descriptor
    )
    assert descriptor.digest() == CompatibilityDescriptor.from_dict(payload).digest()
    assert descriptor.digest() != CURRENT_CONTRACT.digest()
    assert payload["model_generations"] == [
        model_generation_ref("OpenMed/synthetic-pii-v1")
    ]
    _assert_value_free(payload, "OpenMed/synthetic-pii-v1")
    assert local_descriptor(SimpleNamespace()) is CURRENT_CONTRACT


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(schema_version="openmed.service.compatibility.v0"),
        lambda p: p.update(unexpected=True),
        lambda p: p.update(contract_generation=True),
        lambda p: p.update(contract_generation=3, min_peer_generation=1),
        lambda p: p.update(min_peer_generation=2),
        lambda p: p.update(capabilities=["Not A Token"]),
        lambda p: p.update(capabilities="jobs.deidentify.v1"),
        lambda p: p.update(model_generations=["OpenMed/synthetic"]),
        lambda p: p["durable_schemas"].update(jobs={"read": [2, 3], "write": 1}),
        lambda p: p["durable_schemas"].update(jobs={"read": [1], "write": 1}),
        lambda p: p["durable_schemas"].update({"../etc": {"read": [1, 1], "write": 1}}),
    ],
)
def test_descriptor_parser_rejects_anything_it_does_not_understand(mutate) -> None:
    payload = json.loads(json.dumps(CURRENT_CONTRACT.to_dict()))
    mutate(payload)
    with pytest.raises(CompatibilityError) as raised:
        CompatibilityDescriptor.from_dict(payload)
    assert raised.value.code == "peer_descriptor_unsupported"
    assert "etc" not in str(raised.value)
    assert negotiate_peer(CURRENT_CONTRACT, payload).code == (
        "peer_descriptor_unsupported"
    )


# --- Peer negotiation -------------------------------------------------------


def test_peer_negotiation_applies_the_newer_build_window() -> None:
    one = _contract(1, 1)
    two = _contract(2, 1, jobs_range=(1, 2, 1))
    three = _contract(3, 2, jobs_range=(1, 2, 2))
    cases = [
        (one, one, "peer_compatible"),
        (one, two, "peer_compatible"),
        (two, one, "peer_compatible"),
        (two, three, "peer_compatible"),
        (three, two, "peer_compatible"),
        (one, three, "peer_generation_outside_window"),
        (three, one, "peer_generation_outside_window"),
    ]
    for local, peer, code in cases:
        decision = negotiate_peer(local, peer)
        assert decision.code == code
        assert decision.compatible is (code == "peer_compatible")


def test_peer_negotiation_refuses_unreadable_shared_schemas() -> None:
    writes_two_early = _contract(2, 1, jobs_range=(1, 2, 2))
    decision = negotiate_peer(_contract(1, 1), writes_two_early)
    assert decision.to_dict() == {
        "compatible": False,
        "code": "peer_schema_outside_window",
        "component": "jobs",
    }
    other_store = CompatibilityDescriptor(
        contract_generation=1,
        min_peer_generation=1,
        durable_schemas={"other": SchemaRange(5, 5, 5)},
        capabilities=frozenset(),
    )
    assert negotiate_peer(_contract(1, 1), other_store).compatible


@pytest.mark.parametrize(
    ("values", "code"),
    [
        (["1/1"], "peer_compatible"),
        (["2/1"], "peer_compatible"),
        (["2/2"], "peer_generation_outside_window"),
        (["3/2"], "peer_generation_outside_window"),
        ([], "peer_contract_malformed"),
        (["1/1", "1/1"], "peer_contract_malformed"),
        ([""], "peer_contract_malformed"),
        (["1"], "peer_contract_malformed"),
        (["01/1"], "peer_contract_malformed"),
        (["1/2"], "peer_contract_malformed"),
        (["3/1"], "peer_contract_malformed"),
        (["a/b"], "peer_contract_malformed"),
        ([b"1/1"], "peer_contract_malformed"),
    ],
)
def test_peer_contract_token_decision(values: list[Any], code: str) -> None:
    assert peer_contract_decision(CURRENT_CONTRACT, values).code == code


# --- Work claims ------------------------------------------------------------


def _envelope(**changes: Any) -> dict[str, Any]:
    payload = WorkEnvelope.for_work(
        CURRENT_CONTRACT,
        components=("jobs",),
        required_capabilities={"jobs.deidentify.v1"},
        model_generation=model_generation_ref("OpenMed/synthetic-pii-v1"),
    ).to_dict()
    payload.update(changes)
    return payload


def test_work_claim_checks_generation_schema_capability_and_model() -> None:
    worker = CURRENT_CONTRACT.with_model_generations(
        [model_generation_ref("OpenMed/synthetic-pii-v1")]
    )
    other_model = model_generation_ref("OpenMed/synthetic-pii-v2")
    cases = [
        (_envelope(), "work_compatible", None),
        (
            _envelope(contract_generation=2, min_peer_generation=1),
            "work_compatible",
            None,
        ),
        (
            _envelope(contract_generation=3, min_peer_generation=2),
            "work_generation_outside_window",
            None,
        ),
        (_envelope(schema_versions={"jobs": 2}), "work_schema_outside_window", "jobs"),
        (
            _envelope(schema_versions={"ledger": 1}),
            "work_component_unknown",
            "ledger",
        ),
        (
            _envelope(required_capabilities=["jobs.deidentify.v2"]),
            "capability_missing",
            None,
        ),
        (_envelope(model_generation=other_model), "model_generation_unavailable", None),
        (_envelope(model_generation=None), "work_compatible", None),
        (None, "work_envelope_invalid", None),
        ({"contract_generation": 1}, "work_envelope_invalid", None),
        (_envelope(schema_versions={"jobs": True}), "work_envelope_invalid", None),
        (_envelope(schema_versions=[1]), "work_envelope_invalid", None),
        (
            _envelope(model_generation="OpenMed/synthetic"),
            "work_envelope_invalid",
            None,
        ),
    ]
    for envelope, code, component in cases:
        decision = check_work(worker, envelope)
        assert (decision.code, decision.component) == (code, component)
        assert decision.compatible is (code == "work_compatible")
        assert decision.code in REASON_CODES


def test_work_envelope_round_trips_without_values() -> None:
    envelope = WorkEnvelope.from_dict(_envelope())
    assert WorkEnvelope.from_dict(envelope.to_dict()) == envelope
    _assert_value_free(envelope.to_dict(), "synthetic-pii")


# --- Persisted state --------------------------------------------------------


def test_component_evaluation_blocks_unknown_newer_and_unmigrated_state() -> None:
    schema = SchemaRange(2, 3, 3)
    cases = [
        (SchemaObservation.absent(), CompatibilityState.ABSENT, "schema_absent"),
        (SchemaObservation.at(2), CompatibilityState.COMPATIBLE, "schema_compatible"),
        (SchemaObservation.at(3), CompatibilityState.COMPATIBLE, "schema_compatible"),
        (
            SchemaObservation.at(1),
            CompatibilityState.PENDING,
            "schema_migration_required",
        ),
        (
            SchemaObservation.at(4),
            CompatibilityState.INCOMPATIBLE,
            "schema_ahead_of_window",
        ),
        (SchemaObservation.unknown(), CompatibilityState.UNKNOWN, "schema_unknown"),
        (SchemaObservation.at(True), CompatibilityState.UNKNOWN, "schema_unknown"),
        (SchemaObservation.at("3"), CompatibilityState.UNKNOWN, "schema_unknown"),
    ]
    for observation, state, code in cases:
        result = evaluate_component(schema, observation)
        assert (result.state, result.code) == (state, code)
        assert result.usable is (state in {"absent", "compatible"})
    undeclared = evaluate_component(None, SchemaObservation.at(1))
    assert undeclared.state is CompatibilityState.UNKNOWN
    report = evaluate_persisted_state(
        CURRENT_CONTRACT,
        {"jobs": SchemaObservation.at(1), "journey_local": SchemaObservation.at(9)},
    )
    assert not report.ready
    assert set(report.blocking()) == {"journey_local"}
    assert report.to_dict() == {
        "state": "blocked",
        "components": {
            "jobs": {
                "state": "compatible",
                "code": "schema_compatible",
                "schema_version": 1,
            },
            "journey_local": {
                "state": "incompatible",
                "code": "schema_ahead_of_window",
                "schema_version": 9,
            },
        },
    }


def test_job_store_probe_reports_versions_without_reading_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "jobs.json"
    assert job_store_observation(path) == SchemaObservation.absent()
    cases = [
        ('{"jobs": {}}', SchemaObservation.at(1)),
        ('{"schema_version": 1, "jobs": {}}', SchemaObservation.at(1)),
        ('{"schema_version": 2, "jobs": {}}', SchemaObservation.at(2)),
        ('{"schema_version": true, "jobs": {}}', SchemaObservation.unknown()),
        ("{}", SchemaObservation.unknown()),
        ("[]", SchemaObservation.unknown()),
        ("{broken", SchemaObservation.unknown()),
    ]
    for contents, expected in cases:
        path.write_text(contents, encoding="utf-8")
        assert job_store_observation(path) == expected
    assert job_store_observation(tmp_path) == SchemaObservation.unknown()


def test_journey_store_probe_maps_schema_versions() -> None:
    assert journey_store_observation(SimpleNamespace(schema_version=3)) == (
        SchemaObservation.at(3)
    )
    assert journey_store_observation(SimpleNamespace(schema_version=0)) == (
        SchemaObservation.absent()
    )

    class Broken:
        @property
        def schema_version(self) -> int:
            raise RuntimeError("synthetic probe failure")

    assert journey_store_observation(Broken()) == SchemaObservation.unknown()


def test_gate_caches_reports_and_treats_failing_probes_as_unknown() -> None:
    now = [0.0]
    calls: list[int] = []
    versions = [1]

    def probe() -> SchemaObservation:
        calls.append(1)
        return SchemaObservation.at(versions[0])

    gate = CompatibilityGate(
        CURRENT_CONTRACT,
        {"jobs": probe},
        refresh_interval_seconds=5.0,
        clock=lambda: now[0],
    )
    assert gate.report().ready
    versions[0] = 2
    assert gate.report().ready and len(calls) == 1
    now[0] = 5.0
    assert not gate.report().ready and len(calls) == 2
    versions[0] = 1
    assert gate.report(force=True).ready and len(calls) == 3

    def explode() -> SchemaObservation:
        raise OSError("synthetic /var/lib/openmed failure")

    for failing in (explode, lambda: "compatible"):
        gate = CompatibilityGate(CURRENT_CONTRACT, {"jobs": failing})
        snapshot = gate.snapshot()
        assert snapshot["state"] == "blocked"
        assert snapshot["components"]["jobs"]["code"] == "schema_unknown"
        _assert_value_free(snapshot, "/var/lib")


def test_rollback_requires_peer_and_persisted_state_compatibility() -> None:
    one = _contract(1, 1, jobs_range=(1, 1, 1))
    two = _contract(2, 1, jobs_range=(1, 2, 1))
    three = _contract(3, 2, jobs_range=(1, 2, 2))
    allowed = rollback_decision(
        two, running=[three], observations={"jobs": SchemaObservation.at(2)}
    )
    assert allowed.code == "rollback_within_window"
    peer = rollback_decision(
        one, running=[three], observations={"jobs": SchemaObservation.at(1)}
    )
    assert peer.code == "peer_generation_outside_window"
    state = rollback_decision(
        one, running=[two], observations={"jobs": SchemaObservation.at(2)}
    )
    assert (state.code, state.component) == ("schema_ahead_of_window", "jobs")


def test_matrix_and_evidence_are_machine_readable_and_value_free() -> None:
    one = _contract(1, 1)
    two = _contract(2, 1, jobs_range=(1, 2, 1))
    three = _contract(3, 2, jobs_range=(1, 2, 2))
    matrix = compatibility_matrix((three, one, two))
    assert matrix["generations"] == [1, 2, 3]
    assert [
        (row["older"], row["newer"], row["code"]) for row in matrix["combinations"]
    ] == [
        (1, 1, "peer_compatible"),
        (1, 2, "peer_compatible"),
        (1, 3, "peer_generation_outside_window"),
        (2, 2, "peer_compatible"),
        (2, 3, "peer_compatible"),
        (3, 3, "peer_compatible"),
    ]
    _assert_value_free(matrix)
    assert compatibility_matrix()["generations"] == [1]

    evidence = RolloutEvidence("rolling_upgrade")
    evidence.register(one)
    evidence.record("peer_join", negotiate_peer(one, two), generations=[1, 2])
    evidence.count("jobs_completed", 2)
    document = evidence.to_dict()
    assert document["counts"] == {"jobs_completed": 2, "peer_compatible": 1}
    _assert_value_free(document)
    for bad in ("Free Text", "/etc/passwd", "a" * 40):
        with pytest.raises(CompatibilityError):
            evidence.count(bad)
    with pytest.raises(CompatibilityError):
        evidence.record(
            "step", compat.CompatibilityDecision(True, "made_up"), generations=[1]
        )
    with pytest.raises(CompatibilityError):
        evidence.record("step", negotiate_peer(one, two), generations=[True])


def test_cli_prints_descriptor_and_matrix(capsys: pytest.CaptureFixture[str]) -> None:
    assert compat.main([]) == 0
    descriptor = json.loads(capsys.readouterr().out)
    assert descriptor["descriptor_digest"] == CURRENT_CONTRACT.digest()
    assert descriptor["peer_contract"] == "1/1"
    assert compat.main(["matrix"]) == 0
    assert json.loads(capsys.readouterr().out) == compatibility_matrix()


# --- Job store and queue ----------------------------------------------------


def test_job_store_writes_its_schema_version_and_reads_the_legacy_layout(
    tmp_path: Path,
) -> None:
    path = tmp_path / "jobs.json"
    path.write_text('{"jobs": {"synthetic": {"id": "synthetic"}}}', encoding="utf-8")
    store = jobs.LocalJobStore(path)
    assert store.compatibility.code == "schema_compatible"
    store.create({"id": "synthetic-new", "status": "queued"})
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema_version"] == compat.JOB_STORE_SCHEMA_VERSION
    assert set(document["jobs"]) == {"synthetic", "synthetic-new"}
    assert jobs.LocalJobStore(tmp_path / "missing.json").compatibility.code == (
        "schema_absent"
    )


def test_job_store_outside_the_window_is_never_loaded_or_overwritten(
    tmp_path: Path,
) -> None:
    path = tmp_path / "jobs.json"
    contents = '{"schema_version": 2, "jobs": {"synthetic": {"id": "synthetic"}}}'
    path.write_text(contents, encoding="utf-8")
    store = jobs.LocalJobStore(path)
    assert store.compatibility.code == "schema_ahead_of_window"
    operations = [
        lambda: store.create({"id": "synthetic-new", "status": "queued"}),
        lambda: store.get("synthetic"),
        lambda: store.update("synthetic", status="running"),
        store.cleanup_expired,
    ]
    for operation in operations:
        with pytest.raises(IncompatibleStateError) as raised:
            operation()
        assert raised.value.to_details() == {
            "components": {
                "jobs": {"state": "incompatible", "reason": "schema_ahead_of_window"}
            }
        }
    assert path.read_text(encoding="utf-8") == contents
    path.write_text('{"schema_version": "2", "jobs": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema version is invalid"):
        jobs.LocalJobStore(path)


def test_job_store_refuses_to_replace_a_file_a_newer_build_rewrote(
    tmp_path: Path,
) -> None:
    path = tmp_path / "jobs.json"
    store = jobs.LocalJobStore(path)
    store.create({"id": "synthetic-a", "status": "queued"})
    newer = '{"schema_version": 2, "jobs": {}, "leases": {}}'
    path.write_text(newer, encoding="utf-8")
    with pytest.raises(IncompatibleStateError):
        store.update("synthetic-a", status="running")
    assert path.read_text(encoding="utf-8") == newer
    assert store._records["synthetic-a"]["status"] == "queued"
    with pytest.raises(IncompatibleStateError):
        store.get("synthetic-a")

    compatible_writer = jobs.LocalJobStore(tmp_path / "other.json")
    compatible_writer.create({"id": "synthetic-b", "status": "queued"})
    (tmp_path / "other.json").write_text(
        '{"schema_version": 1, "jobs": {}}', encoding="utf-8"
    )
    compatible_writer.update("synthetic-b", status="running")


def test_job_queue_stamps_and_checks_the_contract_before_claiming(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = SimpleNamespace(served_model_ids=lambda: ("OpenMed/synthetic-pii-v1",))
    store = jobs.LocalJobStore(tmp_path / "jobs.json")
    queue = jobs.DeidentifyJobQueue(runtime, store=store)
    payload = DeidentifyJobRequest(
        documents=[DeidentifyJobDocument(id="synthetic-0", text="synthetic text")],
        model_name="OpenMed/synthetic-pii-v1",
    )
    record = queue._new_record(payload)
    assert record["compatibility"] == {
        "contract_generation": 1,
        "min_peer_generation": 1,
        "schema_versions": {"jobs": 1},
        "required_capabilities": ["jobs.deidentify.v1"],
        "model_generation": model_generation_ref("OpenMed/synthetic-pii-v1"),
    }
    newer = dict(record["compatibility"], contract_generation=3, min_peer_generation=2)
    store.create({**record, "compatibility": newer})
    processed: list[str] = []
    monkeypatch.setattr(
        queue, "_deidentify_document", lambda *_: processed.append("document")
    )
    try:
        queue._run_job(jobs._JobWorkItem(record["id"], payload))
    finally:
        queue.shutdown()
    refused = store.get(record["id"])
    assert refused is not None
    assert processed == []
    assert refused["status"] == "failed"
    assert refused["started_at"] is None
    assert refused["error"] == {
        "type": "IncompatibleWork",
        "message": "Job was refused before processing",
        "code": "work_generation_outside_window",
    }


# --- REST and gRPC surfaces -------------------------------------------------


def test_compatibility_route_publishes_a_value_free_snapshot(
    service_env: Path,
) -> None:
    assert ROUTE_POLICIES[("GET", "/v1/compatibility")].scopes == (
        "compatibility:read",
    )
    app = create_app()
    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        response = client.get("/v1/compatibility")
    assert response.status_code == 200
    body = response.json()
    assert body["contract_generation"] == 1
    assert body["peer_contract"] == "1/1"
    assert body["state"] == "compatible"
    assert body["admission"] == "open"
    assert body["components"] == {"jobs": {"state": "absent", "code": "schema_absent"}}
    assert body["descriptor_digest"] == app.state.compatibility.descriptor.digest()
    assert body["model_generations"]
    _assert_value_free(body, str(service_env), "127.0.0.1", "localhost")
    assert "/v1/compatibility" not in app.openapi()["paths"]


def test_compatibility_route_requires_its_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        json.dumps(
            [
                {
                    "id": "synthetic-key",
                    "key_hash": hash_api_key("synthetic-secret"),
                    "principal": "synthetic-peer",
                    "scopes": ["compatibility:read"],
                },
                {
                    "id": "other-key",
                    "key_hash": hash_api_key("other-secret"),
                    "principal": "synthetic-client",
                    "scopes": ["jobs:read"],
                },
            ]
        ),
    )
    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        assert client.get("/v1/compatibility").status_code == 401
        denied = client.get("/v1/compatibility", headers={"X-API-Key": "other-secret"})
        allowed = client.get(
            "/v1/compatibility", headers={"X-API-Key": "synthetic-secret"}
        )
    assert denied.status_code == 403
    assert allowed.status_code == 200


def test_readiness_and_job_admission_fail_closed_on_newer_persisted_state(
    service_env: Path,
) -> None:
    contents = '{"schema_version": 2, "jobs": {}}'
    service_env.write_text(contents, encoding="utf-8")
    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        ready = client.get("/readyz")
        submitted = client.post(
            "/jobs", json={"documents": [{"id": "synthetic", "text": "synthetic"}]}
        )
        status = client.get("/jobs/synthetic")
        snapshot = client.get("/v1/compatibility").json()
    expected = {
        "components": {
            "jobs": {"state": "incompatible", "reason": "schema_ahead_of_window"}
        }
    }
    for response in (ready, submitted, status):
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "incompatible_state"
        assert response.json()["error"]["details"] == expected
    assert snapshot["state"] == "blocked"
    assert service_env.read_text(encoding="utf-8") == contents
    _assert_value_free(ready.json()["error"]["details"], str(service_env))


def test_readiness_follows_a_peer_migrating_the_shared_store(service_env: Path) -> None:
    app = create_app()
    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        assert client.get("/readyz").status_code == 200
        service_env.write_text('{"schema_version": 2, "jobs": {}}', encoding="utf-8")
        app.state.compatibility.report(force=True)
        assert client.get("/readyz").json()["error"]["code"] == "incompatible_state"
        service_env.write_text('{"schema_version": 1, "jobs": {}}', encoding="utf-8")
        app.state.compatibility.report(force=True)
        assert client.get("/readyz").status_code == 200


def test_rest_refuses_incompatible_peers_before_processing(service_env: Path) -> None:
    body = {"documents": [{"id": "synthetic", "text": "synthetic"}]}
    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        outside = client.post("/jobs", json=body, headers={PEER_CONTRACT_HEADER: "3/2"})
        malformed = client.post("/jobs", json=body, headers={PEER_CONTRACT_HEADER: "x"})
        probe = client.get("/readyz", headers={PEER_CONTRACT_HEADER: "x"})
        descriptor = client.get(
            "/v1/compatibility", headers={PEER_CONTRACT_HEADER: "3/2"}
        )
        assert not service_env.exists()
        accepted = client.post(
            "/jobs", json=body, headers={PEER_CONTRACT_HEADER: "2/1"}
        )
        client.app.state.job_queue.shutdown()
    assert outside.status_code == 409
    assert outside.json()["error"]["code"] == "peer_incompatible"
    assert outside.json()["error"]["details"] == {
        "reason": "peer_generation_outside_window",
        "peer_contract": "1/1",
    }
    assert malformed.status_code == 400
    assert malformed.json()["error"]["code"] == "peer_contract_malformed"
    assert probe.status_code == 200
    assert descriptor.status_code == 200
    assert accepted.status_code == 202
    assert accepted.json()["compatibility"]["contract_generation"] == 1


def test_grpc_refuses_incompatible_peers_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openmed.service.grpc_server import OpenMedGrpcServicer

    runtime = service_runtime.ServiceRuntime.from_env()

    class Aborted(Exception):
        pass

    class Context:
        def __init__(self, value: str) -> None:
            self.value = value
            self.aborted: tuple[Any, str] | None = None

        def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
            return ((PEER_CONTRACT_METADATA_KEY, self.value),)

        def abort(self, code: Any, details: str) -> None:
            self.aborted = (code, details)
            raise Aborted()

    servicer = OpenMedGrpcServicer(runtime)
    for value, expected in (
        (
            "3/2",
            (grpc.StatusCode.FAILED_PRECONDITION, "peer_generation_outside_window"),
        ),
        ("x", (grpc.StatusCode.INVALID_ARGUMENT, "peer_contract_malformed")),
    ):
        unary = Context(value)
        with pytest.raises(Aborted):
            servicer.Analyze(openmed_pb2.AnalyzeRequest(text="synthetic"), unary)
        assert unary.aborted == expected
        stream = Context(value)
        with pytest.raises(Aborted):
            list(
                servicer.StreamDeidentify(
                    openmed_pb2.DeidentifyStreamRequest(chunks=["synthetic"]),
                    stream,
                )
            )
        assert stream.aborted == expected
    assert runtime._loader is None


def test_reason_codes_are_controlled_tokens() -> None:
    for code in REASON_CODES:
        assert re.fullmatch(r"[a-z][a-z_]+", code)
    assert replace(CURRENT_CONTRACT) == CURRENT_CONTRACT
