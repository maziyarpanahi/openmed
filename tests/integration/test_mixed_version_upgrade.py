"""Offline N/N-1 mixed-version harness for zero-downtime service upgrades.

Two or three contract generations share one in-process job store and one
Journey store through versioned fakes. Every node makes its decisions with the
real ``openmed.service.compatibility`` contract: peer negotiation, work claims,
readiness on persisted state, and rollback checks. The fakes model the parts
that live in other components (leases with fencing tokens, checkpoints,
idempotency keys and webhook delivery identity) so the harness can prove that
no committed effect is lost or duplicated across a rolling upgrade, a rollback
inside the window, and a refusal outside it.

Generation 1 is the shipped :data:`CURRENT_CONTRACT`. Generation 2 is a
synthetic *expand* step (reads the next schema, still writes the current one,
adds a capability). Generation 3 is a synthetic *contract* step (writes the new
schema and drops generation 1 from its window).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pytest

from openmed.service.compatibility import (
    CONTRACT_HISTORY,
    CURRENT_CONTRACT,
    CompatibilityDecision,
    CompatibilityDescriptor,
    CompatibilityGate,
    IncompatibleStateError,
    RolloutEvidence,
    SchemaObservation,
    SchemaRange,
    WorkEnvelope,
    check_work,
    compatibility_matrix,
    model_generation_ref,
    negotiate_peer,
    peer_contract_decision,
    rollback_decision,
)

pytestmark = pytest.mark.integration

MODEL_V1 = "OpenMed/synthetic-pii-v1"
MODEL_V2 = "OpenMed/synthetic-pii-v2"
BASE_CAPABILITY = "jobs.deidentify.v1"
NEXT_CAPABILITY = "jobs.deidentify.v2"

OLD = CURRENT_CONTRACT
EXPAND = CompatibilityDescriptor(
    contract_generation=2,
    min_peer_generation=1,
    durable_schemas={
        **CURRENT_CONTRACT.durable_schemas,
        "jobs": SchemaRange(read_min=1, read_max=2, write=1),
        "journey_local": SchemaRange(read_min=3, read_max=4, write=3),
    },
    capabilities=CURRENT_CONTRACT.capabilities | {NEXT_CAPABILITY},
)
CONTRACT = CompatibilityDescriptor(
    contract_generation=3,
    min_peer_generation=2,
    durable_schemas={
        **CURRENT_CONTRACT.durable_schemas,
        "jobs": SchemaRange(read_min=1, read_max=2, write=2),
        "journey_local": SchemaRange(read_min=3, read_max=4, write=4),
    },
    capabilities=EXPAND.capabilities,
)
WINDOW = (OLD, EXPAND, CONTRACT)
SERVED = {1: (MODEL_V1,), 2: (MODEL_V1, MODEL_V2), 3: (MODEL_V1, MODEL_V2)}
SHARED_COMPONENTS = ("jobs", "journey_local")

_MATRIX = compatibility_matrix(WINDOW)
SUPPORTED = tuple(
    (row["older"], row["newer"]) for row in _MATRIX["combinations"] if row["compatible"]
)
UNSUPPORTED = tuple(
    (row["older"], row["newer"])
    for row in _MATRIX["combinations"]
    if not row["compatible"]
)
#: Every (API generation, worker generation) pair the harness runs, in both
#: directions, derived from the matrix so no supported pair can be skipped.
COMBINATIONS = tuple(
    sorted(
        {
            pair
            for older, newer in SUPPORTED
            for pair in ((older, newer), (newer, older))
        }
    )
)
BY_GENERATION = {item.contract_generation: item for item in WINDOW}

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TOKEN = re.compile(r"[a-z0-9_.\-]+\Z")


class Draining(RuntimeError):
    """Raised when a draining node is asked to admit new work."""


class PeerRefused(RuntimeError):
    """Raised when a node refuses an announced peer before processing."""

    def __init__(self, decision: CompatibilityDecision) -> None:
        super().__init__("peer refused")
        self.decision = decision


def _digest(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('ascii')).hexdigest()}"


@dataclass
class SharedState:
    """Versioned fake of the job store and Journey store all nodes share."""

    schemas: dict[str, int] = field(
        default_factory=lambda: {
            "jobs": OLD.durable_schemas["jobs"].write,
            "journey_local": OLD.durable_schemas["journey_local"].write,
        }
    )
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    idempotency: dict[str, str] = field(default_factory=dict)
    fence: int = 0
    effects: list[tuple[str, int]] = field(default_factory=list)
    artifacts: dict[str, int] = field(default_factory=dict)
    deliveries: Counter[str] = field(default_factory=Counter)
    mutations: int = 0

    def observe(self, component: str) -> SchemaObservation:
        return SchemaObservation.at(self.schemas[component])

    def digest(self) -> str:
        payload = {
            "schemas": self.schemas,
            "records": self.records,
            "idempotency": self.idempotency,
            "fence": self.fence,
            "effects": self.effects,
            "artifacts": self.artifacts,
            "deliveries": dict(self.deliveries),
        }
        return _digest(json.dumps(payload, sort_keys=True))

    def backup(self) -> "SharedState":
        """Model a verified backup of the shared stores (#3577)."""
        return copy.deepcopy(self)


@dataclass(frozen=True)
class Lease:
    job_id: str
    fence: int
    owner: str


class Node:
    """One API-and-worker process of a given contract generation."""

    def __init__(
        self,
        name: str,
        contract: CompatibilityDescriptor,
        shared: SharedState,
        evidence: RolloutEvidence,
    ) -> None:
        self.name = name
        self.descriptor = contract.with_model_generations(
            model_generation_ref(model)
            for model in SERVED[contract.contract_generation]
        )
        self.shared = shared
        self.evidence = evidence
        self.admitting = True
        self.alive = True
        self.open_streams = 0
        self.gate = CompatibilityGate(
            self.descriptor,
            {
                component: (lambda component=component: shared.observe(component))
                for component in SHARED_COMPONENTS
            },
            refresh_interval_seconds=0,
        )
        evidence.register(contract)

    @property
    def generation(self) -> int:
        return self.descriptor.contract_generation

    def ready(self) -> bool:
        return self.alive and self.admitting and self.gate.report().ready

    def accept_peer(self, peer: "Node") -> CompatibilityDecision:
        """Negotiate with a peer; the header and descriptor paths must agree."""
        full = negotiate_peer(self.descriptor, peer.descriptor.to_dict())
        header = peer_contract_decision(
            self.descriptor, [peer.descriptor.peer_contract]
        )
        assert header.compatible is full.compatible
        self.evidence.record(
            "peer_negotiation", full, generations=[peer.generation, self.generation]
        )
        return full

    def _require_writable(self) -> None:
        report = self.gate.report(force=True)
        for component, item in report.blocking().items():
            self.evidence.count("mutation_refused")
            raise IncompatibleStateError(component, item.code)

    def _write(self) -> None:
        self._require_writable()
        for component in SHARED_COMPONENTS:
            write = self.descriptor.durable_schemas[component].write
            self.shared.schemas[component] = max(self.shared.schemas[component], write)
        self.shared.mutations += 1

    def handle_request(self, via: "Node") -> int:
        """Serve one synchronous request a peer forwarded with its contract."""
        decision = self.accept_peer(via)
        if not decision.compatible:
            raise PeerRefused(decision)
        if not self.admitting:
            raise Draining()
        self.evidence.count("requests_completed")
        return self.generation

    def open_stream(self) -> int:
        if not self.admitting:
            raise Draining()
        self.open_streams += 1
        return self.generation

    def close_stream(self) -> None:
        self.open_streams -= 1
        self.evidence.count("streams_completed")

    def submit(
        self,
        key: str,
        *,
        documents: int = 2,
        model: str = MODEL_V1,
        capabilities: frozenset[str] = frozenset({BASE_CAPABILITY}),
        via: "Node | None" = None,
    ) -> str:
        if via is not None:
            decision = self.accept_peer(via)
            if not decision.compatible:
                raise PeerRefused(decision)
        if not self.admitting:
            raise Draining()
        self._require_writable()
        existing = self.shared.idempotency.get(key)
        if existing is not None:
            self.evidence.count("idempotent_replays")
            return existing
        job_id = f"job{len(self.shared.records):04d}"
        envelope = WorkEnvelope.for_work(
            self.descriptor,
            components=SHARED_COMPONENTS,
            required_capabilities=capabilities,
            model_generation=model_generation_ref(model),
        )
        self._write()
        self.shared.records[job_id] = {
            "status": "queued",
            "envelope": envelope.to_dict(),
            "documents": documents,
            "checkpoint": 0,
            "lease": None,
        }
        self.shared.idempotency[key] = job_id
        self.evidence.count("jobs_accepted")
        return job_id

    def claim(self) -> Lease | None:
        if not (self.alive and self.admitting):
            return None
        self._require_writable()
        for job_id, record in sorted(self.shared.records.items()):
            if record["status"] != "queued":
                continue
            decision = check_work(self.descriptor, record["envelope"])
            self.evidence.record(
                "work_claim",
                decision,
                generations=[
                    record["envelope"]["contract_generation"],
                    self.generation,
                ],
            )
            if not decision.compatible:
                continue
            self.shared.fence += 1
            record["lease"] = {"owner": self.name, "fence": self.shared.fence}
            record["status"] = "running"
            self._write()
            return Lease(job_id, self.shared.fence, self.name)
        return None

    def _holds(self, lease: Lease) -> bool:
        current = self.shared.records[lease.job_id]["lease"]
        if current is None or current["fence"] != lease.fence:
            self.evidence.count("stale_fence_rejected")
            return False
        return True

    def process(self, lease: Lease, *, limit: int | None = None) -> bool:
        """Commit documents from the checkpoint; return whether the job finished."""
        record = self.shared.records[lease.job_id]
        processed = 0
        while record["checkpoint"] < record["documents"]:
            if limit is not None and processed >= limit:
                return False
            if not self._holds(lease):
                return False
            self._write()
            self.shared.effects.append((lease.job_id, record["checkpoint"]))
            record["checkpoint"] += 1
            processed += 1
        if not self._holds(lease):
            return False
        self._finish(lease)
        return True

    def _finish(self, lease: Lease) -> None:
        record = self.shared.records[lease.job_id]
        self._write()
        # Journey artifact: content-addressed, so a replay is idempotent.
        artifact = _digest(f"{lease.job_id}:{record['documents']}")
        if artifact in self.shared.artifacts:
            self.evidence.count("artifact_replays_suppressed")
        else:
            self.shared.artifacts[artifact] = record["documents"]
        record["status"] = "done"
        record["lease"] = None
        self.evidence.count("jobs_completed")
        self.deliver_webhook(lease.job_id)

    def deliver_webhook(self, job_id: str) -> None:
        """Send the terminal webhook under a generation-independent identity."""
        delivery_id = _digest(f"{job_id}:job.done")
        if self.shared.deliveries[delivery_id]:
            self.evidence.count("webhook_replays_suppressed")
            return
        self._write()
        self.shared.deliveries[delivery_id] += 1

    def begin_drain(self) -> None:
        """Drop readiness and stop admitting requests, streams and claims."""
        self.admitting = False

    def drain(self, leases: tuple[Lease, ...] = ()) -> None:
        """Stop admission, checkpoint in-flight leases, then stop.

        This mirrors the drain coordinator phases (stop admission, finish or
        checkpoint in-flight work, release) without depending on it.
        """
        self.begin_drain()
        for lease in leases:
            record = self.shared.records[lease.job_id]
            if record["status"] == "running" and self._holds(lease):
                record["status"] = "queued"
                record["lease"] = None
                self._write()
                self.evidence.count("checkpointed")
        assert self.open_streams == 0
        self.alive = False


class Fleet:
    """Rolling-upgrade orchestrator over a shared state."""

    def __init__(self, shared: SharedState, evidence: RolloutEvidence) -> None:
        self.shared = shared
        self.evidence = evidence
        self.nodes: list[Node] = []
        self._names = 0

    def live(self) -> list[Node]:
        return [node for node in self.nodes if node.alive]

    def join(self, contract: CompatibilityDescriptor) -> Node:
        self._names += 1
        node = Node(f"node{self._names}", contract, self.shared, self.evidence)
        for peer in self.live():
            for decision in (node.accept_peer(peer), peer.accept_peer(node)):
                if not decision.compatible:
                    raise PeerRefused(decision)
        self.nodes.append(node)
        return node

    def assert_serving(self) -> None:
        ready = [node for node in self.live() if node.ready()]
        assert ready, "rolling upgrade lost readiness"
        for node in ready:
            for peer in ready:
                assert negotiate_peer(node.descriptor, peer.descriptor).compatible

    def run_until_idle(self) -> None:
        progressed = True
        while progressed:
            progressed = False
            for node in self.live():
                lease = node.claim()
                if lease is not None:
                    node.process(lease)
                    progressed = True


def _assert_exactly_once(shared: SharedState, job_ids: list[str]) -> None:
    for job_id in job_ids:
        record = shared.records[job_id]
        assert record["status"] == "done"
        effects = [effect for effect in shared.effects if effect[0] == job_id]
        assert effects == [(job_id, index) for index in range(record["documents"])]
        assert shared.deliveries[_digest(f"{job_id}:job.done")] == 1
    assert len(shared.effects) == len(set(shared.effects))


def _assert_value_free(document: Any) -> None:
    def strings(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            return [s for key, inner in value.items() for s in (key, *strings(inner))]
        if isinstance(value, list):
            return [s for inner in value for s in strings(inner)]
        return []

    for text in strings(document):
        assert _DIGEST.fullmatch(text) or _TOKEN.fullmatch(text), text
    serialized = json.dumps(document)
    for forbidden in ("synthetic-pii", "node1", "job0", "OpenMed/"):
        assert forbidden not in serialized


def test_harness_window_extends_the_published_contract() -> None:
    assert CONTRACT_HISTORY[-1] is OLD
    assert BY_GENERATION[1].digest() == CURRENT_CONTRACT.digest()
    assert SUPPORTED == ((1, 1), (1, 2), (2, 2), (2, 3), (3, 3))
    assert UNSUPPORTED == ((1, 3),)


@pytest.mark.parametrize(("api_generation", "worker_generation"), COMBINATIONS)
def test_every_supported_combination_completes_every_workflow(
    api_generation: int, worker_generation: int
) -> None:
    shared = SharedState()
    evidence = RolloutEvidence("supported_combination")
    fleet = Fleet(shared, evidence)
    api = fleet.join(BY_GENERATION[api_generation])
    worker = fleet.join(BY_GENERATION[worker_generation])

    assert worker.handle_request(via=api) == worker_generation
    assert worker.open_stream() == worker_generation
    worker.close_stream()

    job_ids = [
        api.submit("synthetic-key-a", via=None),
        worker.submit("synthetic-key-b", documents=3, via=api),
    ]
    assert worker.submit("synthetic-key-a", via=api) == job_ids[0]
    assert api.submit("synthetic-key-b") == job_ids[1]

    leases = []
    while (lease := worker.claim()) is not None:
        leases.append(lease)
        assert worker.process(lease)
    assert [lease.job_id for lease in leases] == job_ids
    for job_id in job_ids:
        worker.deliver_webhook(job_id)
        api.deliver_webhook(job_id)
    # Accepted work keeps the contract and model generation it was admitted
    # under through completion, whichever generation finished it.
    envelopes = [
        WorkEnvelope.from_dict(shared.records[job]["envelope"]) for job in job_ids
    ]
    assert [item.contract_generation for item in envelopes] == [
        api_generation,
        worker_generation,
    ]
    assert {item.model_generation for item in envelopes} == {
        model_generation_ref(MODEL_V1)
    }

    _assert_exactly_once(shared, job_ids)
    assert sorted(shared.artifacts.values()) == [2, 3]
    assert api.ready() and worker.ready()
    counts = evidence.to_dict()["counts"]
    assert counts["idempotent_replays"] == 2
    assert counts["webhook_replays_suppressed"] == 4
    assert counts["requests_completed"] == 1
    assert counts["streams_completed"] == 1
    assert "mutation_refused" not in counts
    _assert_value_free(evidence.to_dict())


def test_lease_fencing_survives_a_mixed_version_handoff() -> None:
    shared = SharedState()
    evidence = RolloutEvidence("lease_handoff")
    fleet = Fleet(shared, evidence)
    old, new = fleet.join(OLD), fleet.join(EXPAND)
    job_id = old.submit("synthetic-key", documents=4)
    stale = old.claim()
    assert stale is not None
    assert old.process(stale, limit=1) is False
    # The old worker stalls; its lease expires and the newer worker reclaims
    # the checkpoint with a higher fencing token.
    shared.records[job_id]["status"] = "queued"
    shared.records[job_id]["lease"] = None
    fresh = new.claim()
    assert fresh is not None and fresh.fence > stale.fence
    assert old.process(stale) is False
    assert new.process(fresh)
    _assert_exactly_once(shared, [job_id])
    assert evidence.to_dict()["counts"]["stale_fence_rejected"] >= 1


def test_unsupported_combinations_fail_before_mutation() -> None:
    for older, newer in UNSUPPORTED:
        shared = SharedState()
        evidence = RolloutEvidence("unsupported_combination")
        fleet = Fleet(shared, evidence)
        old = fleet.join(BY_GENERATION[older])
        with pytest.raises(PeerRefused) as refused:
            fleet.join(BY_GENERATION[newer])
        assert refused.value.decision.code == "peer_generation_outside_window"
        assert [node.generation for node in fleet.live()] == [older]

        # Even without a handshake, the newer build's work and state are
        # refused by the older build before it mutates anything.
        stray = Node("stray", BY_GENERATION[newer], shared, evidence)
        with pytest.raises(PeerRefused):
            old.submit("synthetic-key", via=stray)
        job_id = stray.submit("synthetic-key")
        before = (shared.digest(), shared.mutations)
        assert not old.ready()
        with pytest.raises(IncompatibleStateError) as blocked:
            old.claim()
        assert blocked.value.code == "schema_ahead_of_window"
        with pytest.raises(IncompatibleStateError):
            old.submit("synthetic-other")
        assert check_work(old.descriptor, shared.records[job_id]["envelope"]).code == (
            "work_generation_outside_window"
        )
        assert (shared.digest(), shared.mutations) == before


def test_new_api_work_an_old_worker_cannot_handle_waits_for_a_capable_worker() -> None:
    shared = SharedState()
    evidence = RolloutEvidence("capability_handoff")
    fleet = Fleet(shared, evidence)
    old, new = fleet.join(OLD), fleet.join(EXPAND)
    needs_v2 = new.submit(
        "synthetic-v2", capabilities=frozenset({BASE_CAPABILITY, NEXT_CAPABILITY})
    )
    pinned = new.submit("synthetic-model-v2", model=MODEL_V2)
    before = (shared.digest(), shared.mutations)
    assert old.claim() is None
    assert (shared.digest(), shared.mutations) == before
    codes = [step["code"] for step in evidence.steps if step["step"] == "work_claim"]
    assert codes == ["capability_missing", "model_generation_unavailable"]
    fleet.run_until_idle()
    _assert_exactly_once(shared, [needs_v2, pinned])
    assert shared.records[pinned]["envelope"]["model_generation"] == (
        model_generation_ref(MODEL_V2)
    )


def test_rolling_upgrade_drains_and_hands_off_without_lost_or_duplicate_work() -> None:
    shared = SharedState()
    evidence = RolloutEvidence("rolling_upgrade")
    fleet = Fleet(shared, evidence)
    old_a, old_b = fleet.join(OLD), fleet.join(OLD)
    job_ids = [old_a.submit(f"synthetic-{index}", documents=3) for index in range(3)]
    in_flight = old_a.claim()
    assert in_flight is not None and not old_a.process(in_flight, limit=1)
    stream_generation = old_b.open_stream()
    fleet.assert_serving()

    new_a = fleet.join(EXPAND)
    fleet.assert_serving()
    old_a.drain((in_flight,))
    assert not old_a.ready()
    with pytest.raises(Draining):
        old_a.submit("synthetic-late")
    fleet.assert_serving()
    assert old_a.process(in_flight) is False

    resumed = new_a.claim()
    assert resumed is not None and resumed.job_id == in_flight.job_id
    assert shared.records[resumed.job_id]["checkpoint"] == 1
    assert new_a.process(resumed)
    job_ids.append(
        new_a.submit(
            "synthetic-v2",
            capabilities=frozenset({BASE_CAPABILITY, NEXT_CAPABILITY}),
            via=old_b,
        )
    )
    job_ids.append(old_b.submit("synthetic-from-old", via=new_a))
    assert old_b.submit("synthetic-0", via=new_a) == job_ids[0]

    new_b = fleet.join(EXPAND)
    old_b.begin_drain()
    with pytest.raises(Draining):
        old_b.open_stream()
    # The stream admitted before the drain finishes on its own generation.
    assert stream_generation == 1
    old_b.close_stream()
    old_b.drain()
    fleet.assert_serving()
    fleet.run_until_idle()

    assert {node.generation for node in fleet.live()} == {2}
    assert new_b.ready()
    _assert_exactly_once(shared, job_ids)
    assert shared.schemas == {"jobs": 1, "journey_local": 3}
    document = evidence.to_dict()
    assert document["counts"]["checkpointed"] == 1
    assert document["counts"]["streams_completed"] == 1
    assert "mutation_refused" not in document["counts"]
    _assert_value_free(document)


def test_rollback_within_window_then_roll_forward_keeps_every_job() -> None:
    shared = SharedState()
    evidence = RolloutEvidence("rollback_within_window")
    fleet = Fleet(shared, evidence)
    new_a, new_b = fleet.join(EXPAND), fleet.join(EXPAND)
    plain = new_a.submit("synthetic-plain")
    needs_v2 = new_a.submit(
        "synthetic-v2", capabilities=frozenset({BASE_CAPABILITY, NEXT_CAPABILITY})
    )
    observations = {name: shared.observe(name) for name in SHARED_COMPONENTS}
    decision = rollback_decision(
        OLD, running=[new_a.descriptor], observations=observations
    )
    evidence.record("rollback_check", decision, generations=[2, 1])
    assert decision.code == "rollback_within_window"

    old_a = fleet.join(OLD)
    new_a.drain()
    old_b = fleet.join(OLD)
    new_b.drain()
    fleet.assert_serving()
    fleet.run_until_idle()
    assert shared.records[plain]["status"] == "done"
    assert shared.records[needs_v2]["status"] == "queued"
    assert shared.records[needs_v2]["lease"] is None

    forward = fleet.join(EXPAND)
    old_a.drain()
    old_b.drain()
    fleet.assert_serving()
    fleet.run_until_idle()
    assert forward.ready()
    _assert_exactly_once(shared, [plain, needs_v2])
    _assert_value_free(evidence.to_dict())


def test_rollback_outside_the_window_is_refused_until_a_backup_is_restored() -> None:
    shared = SharedState()
    evidence = RolloutEvidence("rollback_outside_window")
    fleet = Fleet(shared, evidence)
    expand_a, expand_b = fleet.join(EXPAND), fleet.join(EXPAND)
    first = expand_a.submit("synthetic-before")
    fleet.run_until_idle()
    backup = shared.backup()

    contract_a = fleet.join(CONTRACT)
    expand_a.drain()
    second = contract_a.submit("synthetic-after", via=expand_b)
    assert shared.schemas == {"jobs": 2, "journey_local": 4}
    fleet.run_until_idle()
    fleet.assert_serving()
    _assert_exactly_once(shared, [first, second])

    observations = {name: shared.observe(name) for name in SHARED_COMPONENTS}
    to_expand = rollback_decision(
        EXPAND, running=[contract_a.descriptor], observations=observations
    )
    to_old = rollback_decision(
        OLD, running=[expand_b.descriptor], observations=observations
    )
    evidence.record("rollback_check", to_expand, generations=[3, 2])
    evidence.record("rollback_check", to_old, generations=[3, 1])
    assert to_expand.code == "rollback_within_window"
    assert (to_old.code, to_old.component) == ("schema_ahead_of_window", "jobs")
    with pytest.raises(PeerRefused):
        fleet.join(OLD)

    newer_state = (shared.digest(), shared.mutations)
    stray_old = Node("stray", OLD, shared, evidence)
    assert not stray_old.ready()
    for attempt in (stray_old.claim, lambda: stray_old.submit("synthetic-x")):
        with pytest.raises(IncompatibleStateError):
            attempt()
    assert (shared.digest(), shared.mutations) == newer_state

    restored = backup.backup()
    recovered = Node("recovered", OLD, restored, evidence)
    assert recovered.ready()
    assert restored.records[first]["status"] == "done"
    assert second not in restored.records
    assert (shared.digest(), shared.mutations) == newer_state
    _assert_value_free(evidence.to_dict())


def test_published_matrix_contains_no_untested_supported_combination() -> None:
    published = compatibility_matrix()
    supported = {
        (row["older"], row["newer"])
        for row in published["combinations"]
        if row["compatible"]
    }
    assert supported <= set(SUPPORTED)
    for older, newer in supported:
        assert (older, newer) in COMBINATIONS and (newer, older) in COMBINATIONS
        for generation in (older, newer):
            assert published["descriptor_digests"][str(generation)] == (
                BY_GENERATION[generation].digest()
            )
    _assert_value_free(published)
    _assert_value_free(_MATRIX)
