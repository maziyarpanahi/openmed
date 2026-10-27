"""Offline tests for the fail-closed federated client round runner."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from openmed.training.federated_client_runner import (
    FEDERATED_CLIENT_RUNNER_SCHEMA_VERSION,
    FEDERATED_CLIENT_TERMINAL_STATES,
    FEDERATED_CLIENT_TRANSITIONS,
    FederatedCancellationMode,
    FederatedClientReasonCode,
    FederatedClientResourceLimits,
    FederatedClientRoundOffer,
    FederatedClientRoundState,
    FederatedClientRunner,
    FederatedClientRunnerError,
    FederatedEncodedUpdate,
    FederatedMemoryClass,
    FederatedTrainingStop,
)
from openmed.training.federated_round import FederatedRoundState
from openmed.training.federated_schedule import FederatedRoundSchedule
from openmed.training.federated_status import (
    FederatedRoundReasonCode,
    build_federated_round_status,
)
from openmed.training.federated_update_metadata import (
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdatePolicy,
)

C = FederatedClientRoundState
R = FederatedClientReasonCode
UTC = timezone.utc
SENTINELS = (
    "site-mercy-general-ward7",
    "/srv/hospital-a/charts/patient-123.jsonl",
    "Patient Jane Roe presented with chest pain",
    "tensor([0.4242, -1.3137])",
    "local_recall=0.4217",
)
MANIFEST = b'{"round": "synthetic-round-1", "task": "synthetic-adapter"}'
MODEL_DIGEST = "sha256:" + "a" * 64
PARAMETER = FederatedParameterMetadata(name="lora.a", shape=(2,), dtype="float32")
POLICY = FederatedUpdatePolicy(model_digest=MODEL_DIGEST, parameters=(PARAMETER,))
SCHEDULE = FederatedRoundSchedule(
    enrollment_starts_at=datetime(2026, 1, 1, tzinfo=UTC),
    update_submission_starts_at=datetime(2026, 1, 2, tzinfo=UTC),
    aggregation_starts_at=datetime(2026, 1, 3, tzinfo=UTC),
    evaluation_starts_at=datetime(2026, 1, 4, tzinfo=UTC),
    finishes_at=datetime(2026, 1, 5, tzinfo=UTC),
)
SUBMISSION_TIME = datetime(2026, 1, 2, 6, tzinfo=UTC)


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def make_offer(**overrides: Any) -> FederatedClientRoundOffer:
    manifest = overrides.pop("manifest", MANIFEST)
    arguments = {
        "round_ref": "synthetic-round-1",
        "manifest": manifest,
        "manifest_digest": (
            _digest(manifest) if isinstance(manifest, bytes) else MODEL_DIGEST
        ),
        "schedule": SCHEDULE,
        "update_policy": POLICY,
        "local_steps": 3,
        "required_memory_class": FederatedMemoryClass.SMALL,
    }
    arguments.update(overrides)
    return FederatedClientRoundOffer(**arguments)


class SimulatedCrash(BaseException):
    """Stands in for process death; runners must not swallow it."""


@dataclass
class Ports:
    """Synthetic fakes sharing one ordered call log."""

    calls: list[str] = field(default_factory=list)
    verified: object = True
    compatible: object = True
    permitted: object = True
    budget: object = True
    raise_in: set[str] = field(default_factory=set)
    crash_in: set[str] = field(default_factory=set)
    round_state: object = FederatedRoundState.COLLECTING
    now: datetime = SUBMISSION_TIME
    clock_step: timedelta = timedelta(0)
    steps: int = 3
    update: tuple[float, ...] = (0.25, -0.5)
    swallow_stop: bool = False
    cancel_at_step: int | None = None
    cancel_mode: FederatedCancellationMode = FederatedCancellationMode.ABORT
    cancellation: FederatedCancellationMode | None = None
    memory_breach_at_step: int | None = None
    encode_override: Any = None
    submit_failures: int = 0
    crash_after_submit: bool = False
    submitted: list[bytes] = field(default_factory=list)
    store: dict[str, bytes] = field(default_factory=dict)
    saved_states: list[str] = field(default_factory=list)
    save_failure_state: str | None = None
    controls: list[Any] = field(default_factory=list)

    def _enter(self, name: str) -> None:
        self.calls.append(name)
        if name in self.crash_in:
            raise SimulatedCrash()
        if name in self.raise_in:
            raise RuntimeError(" ".join(SENTINELS))

    # Manifest verification, capability, data-use and privacy budget ports.
    def verify(self, offer: FederatedClientRoundOffer) -> object:
        self._enter("verify")
        return self.verified

    def is_compatible(self, offer: FederatedClientRoundOffer) -> object:
        self._enter("capability")
        return self.compatible

    def permits(self, offer: FederatedClientRoundOffer) -> object:
        self._enter("data_use")
        return self.permitted

    def has_budget(self, offer: FederatedClientRoundOffer) -> object:
        self._enter("budget")
        return self.budget

    # Training, clipping, privacy mechanism and encoding ports.
    def train(self, offer: FederatedClientRoundOffer, control: Any) -> object:
        self._enter("train")
        self.controls.append(control)
        for index in range(1, self.steps + 1):
            if self.cancel_at_step == index:
                self.cancellation = self.cancel_mode
            if self.memory_breach_at_step == index:
                control.report_memory_breach()
            try:
                control.step()
            except FederatedTrainingStop:
                if not self.swallow_stop:
                    raise
        return {"weights": self.update, "note": SENTINELS[3]}

    def clip(self, update: object, offer: FederatedClientRoundOffer) -> object:
        self._enter("clip")
        return {"weights": tuple(min(1.0, max(-1.0, x)) for x in update["weights"])}

    def privatize(self, update: object, offer: FederatedClientRoundOffer) -> object:
        self._enter("privatize")
        return {"weights": tuple(x + 0.125 for x in update["weights"])}

    def encode(
        self, update: object, offer: FederatedClientRoundOffer
    ) -> FederatedEncodedUpdate:
        self._enter("encode")
        if self.encode_override is not None:
            return self.encode_override(update, offer)
        payload = json.dumps({"lora.a": list(update["weights"])}).encode()
        return encoded(payload)

    # Submission port.
    def get_round_state(self, offer: FederatedClientRoundOffer) -> object:
        self._enter("round_state")
        return self.round_state

    def submit(
        self, offer: FederatedClientRoundOffer, payload: bytes, update_digest: str
    ) -> None:
        self._enter("submit")
        assert update_digest == _digest(payload)
        if self.submit_failures:
            self.submit_failures -= 1
            raise OSError(" ".join(SENTINELS))
        self.submitted.append(payload)
        if self.crash_after_submit:
            self.crash_after_submit = False
            raise SimulatedCrash()

    # Clock, cancellation and durable store ports.
    def current_time(self) -> datetime:
        moment = self.now
        self.now = self.now + self.clock_step
        return moment

    def poll(self) -> FederatedCancellationMode | None:
        return self.cancellation

    def load(self, key: str) -> bytes | None:
        return self.store.get(key)

    def save(self, key: str, data: bytes) -> None:
        state = json.loads(data)["state"]
        if state == self.save_failure_state:
            raise OSError(SENTINELS[1])
        self.store[key] = data
        self.saved_states.append(state)


def encoded(payload: bytes, **overrides: Any) -> FederatedEncodedUpdate:
    metadata_arguments = {
        "model_digest": MODEL_DIGEST,
        "adapter_format": "dense",
        "parameters": (PARAMETER,),
        "total_elements": 2,
        "update_digest": _digest(payload),
        "clipped": True,
        "policy": overrides.pop("policy", POLICY),
    }
    metadata_arguments.update(overrides)
    return FederatedEncodedUpdate(
        payload=payload, metadata=FederatedUpdateMetadata(**metadata_arguments)
    )


class _Facade:
    """Binds the shared fake to the method names each port declares."""

    def __init__(self, ports: Ports, **methods: str):
        for public, private in methods.items():
            setattr(self, public, getattr(ports, private))


def make_runner(
    ports: Ports, limits: FederatedClientResourceLimits | None = None
) -> FederatedClientRunner:
    return FederatedClientRunner(
        manifest_verifier=_Facade(ports, verify="verify"),
        capability=_Facade(ports, is_compatible="is_compatible"),
        data_use_policy=_Facade(ports, permits="permits"),
        trainer=_Facade(ports, train="train"),
        clipper=_Facade(ports, clip="clip"),
        privacy_mechanism=_Facade(
            ports, has_budget="has_budget", privatize="privatize"
        ),
        encoder=_Facade(ports, encode="encode"),
        submitter=_Facade(ports, round_state="get_round_state", submit="submit"),
        clock=_Facade(ports, now="current_time"),
        store=_Facade(ports, load="load", save="save"),
        cancellation=_Facade(ports, poll="poll"),
        limits=limits
        or FederatedClientResourceLimits(
            max_wall_clock_seconds=3600,
            max_steps=10,
            memory_class=FederatedMemoryClass.MEDIUM,
        ),
    )


def _stored(ports: Ports) -> dict[str, Any]:
    (data,) = ports.store.values()
    return json.loads(data)


HAPPY_CALLS = [
    "round_state",
    "verify",
    "capability",
    "data_use",
    "budget",
    "train",
    "clip",
    "privatize",
    "encode",
    "round_state",
    "submit",
]


def test_round_completes_with_deterministic_transitions() -> None:
    ports = Ports()
    status = make_runner(ports).run_round(make_offer())

    assert status.state is C.SUBMITTED
    assert status.reason_code is None
    assert status.resumable is False
    assert ports.saved_states == [
        "received",
        "verified",
        "training",
        "sealed",
        "submitted",
    ]
    assert ports.calls == HAPPY_CALLS
    (payload,) = ports.submitted
    assert status.update_digest == _digest(payload)
    assert status.manifest_digest == _digest(MANIFEST)
    stored = _stored(ports)
    assert stored["update_digest"] == status.update_digest
    assert stored["payload"] is None


def test_identical_inputs_produce_byte_identical_runs() -> None:
    first, second = Ports(), Ports()
    first_status = make_runner(first).run_round(make_offer())
    second_status = make_runner(second).run_round(make_offer())

    assert first_status.to_json() == second_status.to_json()
    assert first.store == second.store
    assert first.submitted == second.submitted
    assert first.calls == second.calls


def test_terminal_round_rerun_calls_no_ports() -> None:
    ports = Ports()
    runner = make_runner(ports)
    status = runner.run_round(make_offer())
    calls = list(ports.calls)

    assert runner.run_round(make_offer()) == status
    assert runner.status(make_offer()) == status
    assert ports.calls == calls
    assert len(ports.submitted) == 1


def test_status_is_none_before_receipt() -> None:
    assert make_runner(Ports()).status(make_offer()) is None


def test_retry_after_crash_resends_byte_identical_bytes() -> None:
    ports = Ports(crash_after_submit=True)
    with pytest.raises(SimulatedCrash):
        make_runner(ports).run_round(make_offer())
    assert _stored(ports)["state"] == "sealed"

    # A restarted process would draw different noise; it must not retrain.
    ports.update = (0.75, 0.5)
    status = make_runner(ports).run_round(make_offer())

    assert status.state is C.SUBMITTED
    assert ports.calls.count("train") == 1
    assert len(ports.submitted) == 2
    assert ports.submitted[0] == ports.submitted[1]
    assert status.update_digest == _digest(ports.submitted[0])


def test_submission_failure_is_resumable_and_retries_same_bytes() -> None:
    ports = Ports(submit_failures=1)
    runner = make_runner(ports)
    sealed = runner.run_round(make_offer())

    assert sealed.state is C.SEALED
    assert sealed.reason_code is R.SUBMISSION_FAILED
    assert sealed.resumable is True
    assert not ports.submitted

    ports.update = (0.75, 0.5)
    submitted = runner.run_round(make_offer())
    assert submitted.state is C.SUBMITTED
    assert submitted.update_digest == sealed.update_digest
    assert ports.calls.count("train") == 1
    assert _digest(ports.submitted[0]) == sealed.update_digest


def _sealed_payload(ports: Ports) -> bytes:
    return base64.b64decode(_stored(ports)["payload"])


def test_resend_identical_bytes_submits_sealed_update_once() -> None:
    ports = Ports(submit_failures=1)
    runner = make_runner(ports)
    runner.run_round(make_offer())
    payload = _sealed_payload(ports)

    status = runner.resend_update(make_offer(), payload)
    assert status.state is C.SUBMITTED
    assert ports.submitted == [payload]

    # A further identical resend of a submitted round transmits nothing.
    assert runner.resend_update(make_offer(), payload) == status
    assert ports.submitted == [payload]


@pytest.mark.parametrize("submit_failures", [0, 1])
def test_different_bytes_for_the_same_round_fail_closed(submit_failures: int) -> None:
    ports = Ports(submit_failures=submit_failures)
    runner = make_runner(ports)
    runner.run_round(make_offer())
    submissions = list(ports.submitted)
    calls = list(ports.calls)

    with pytest.raises(FederatedClientRunnerError) as error:
        runner.resend_update(make_offer(), b'{"lora.a": [9.0, 9.0]}')

    assert error.value.reason_code is R.UPDATE_DIGEST_CONFLICT
    assert str(error.value) == "federated client runner refused: update_digest_conflict"
    assert ports.submitted == submissions
    assert ports.calls == calls


def test_resend_without_a_sealed_update_fails_closed() -> None:
    ports = Ports()
    with pytest.raises(FederatedClientRunnerError) as error:
        make_runner(ports).resend_update(make_offer(), b"synthetic")
    assert error.value.reason_code is R.UPDATE_NOT_SEALED
    assert ports.calls == []


def test_resend_after_declined_round_fails_closed() -> None:
    ports = Ports(permitted=False)
    runner = make_runner(ports)
    runner.run_round(make_offer())
    with pytest.raises(FederatedClientRunnerError) as error:
        runner.resend_update(make_offer(), b"synthetic")
    assert error.value.reason_code is R.UPDATE_NOT_SEALED


def _expire_clock(ports: Ports) -> None:
    ports.now = datetime(2026, 1, 3, 1, tzinfo=UTC)


def _hold(ports: Ports) -> None:
    ports.round_state = FederatedRoundState.HELD


def _abort(ports: Ports) -> None:
    ports.round_state = FederatedRoundState.ABORTED


def _aggregate(ports: Ports) -> None:
    ports.round_state = FederatedRoundState.AGGREGATING


@pytest.mark.parametrize(
    ("change", "state", "reason"),
    [
        (_hold, C.ABORTED, R.ROUND_HELD),
        (_abort, C.ABORTED, R.ROUND_ABORTED),
        (_expire_clock, C.EXPIRED, R.DEADLINE_PASSED),
        (_aggregate, C.EXPIRED, R.COLLECTION_CLOSED),
    ],
)
def test_no_submission_after_hold_abort_or_deadline(
    change: Any, state: C, reason: R
) -> None:
    ports = Ports(submit_failures=1)
    runner = make_runner(ports)
    sealed = runner.run_round(make_offer())
    payload = _sealed_payload(ports)
    change(ports)

    status = runner.run_round(make_offer())
    assert status.state is state
    assert status.reason_code is reason
    assert status.update_digest == sealed.update_digest
    assert status.resumable is False
    assert _stored(ports)["payload"] is None
    assert ports.calls.count("submit") == 1
    assert not ports.submitted

    with pytest.raises(FederatedClientRunnerError) as error:
        runner.resend_update(make_offer(), payload)
    assert error.value.reason_code is reason
    assert ports.calls.count("submit") == 1


@pytest.mark.parametrize(
    ("change", "state", "reason"),
    [
        (_hold, C.ABORTED, R.ROUND_HELD),
        (_abort, C.ABORTED, R.ROUND_ABORTED),
        (_expire_clock, C.EXPIRED, R.DEADLINE_PASSED),
        (_aggregate, C.EXPIRED, R.COLLECTION_CLOSED),
    ],
)
def test_resend_of_sealed_update_after_hold_abort_or_deadline_fails_closed(
    change: Any, state: C, reason: R
) -> None:
    ports = Ports(submit_failures=1)
    runner = make_runner(ports)
    runner.run_round(make_offer())
    payload = _sealed_payload(ports)
    change(ports)

    with pytest.raises(FederatedClientRunnerError) as error:
        runner.resend_update(make_offer(), payload)
    assert error.value.reason_code is reason
    assert runner.status(make_offer()).state is state
    assert ports.calls.count("submit") == 1


@pytest.mark.parametrize(
    ("change", "state", "reason"),
    [
        (_hold, C.ABORTED, R.ROUND_HELD),
        (_abort, C.ABORTED, R.ROUND_ABORTED),
        (_expire_clock, C.EXPIRED, R.DEADLINE_PASSED),
        (_aggregate, C.EXPIRED, R.COLLECTION_CLOSED),
    ],
)
def test_closed_round_is_not_trained(change: Any, state: C, reason: R) -> None:
    ports = Ports()
    change(ports)
    status = make_runner(ports).run_round(make_offer())
    assert (status.state, status.reason_code) == (state, reason)
    assert "train" not in ports.calls
    assert "verify" not in ports.calls


def test_submission_waits_for_the_submission_window() -> None:
    ports = Ports(now=datetime(2026, 1, 1, 12, tzinfo=UTC))
    runner = make_runner(ports)
    early = runner.run_round(make_offer())

    assert early.state is C.SEALED
    assert early.reason_code is R.SUBMISSION_WINDOW_NOT_OPEN
    assert early.resumable is True
    assert "submit" not in ports.calls

    ports.now = SUBMISSION_TIME
    assert runner.run_round(make_offer()).state is C.SUBMITTED
    assert ports.calls.count("train") == 1


def test_submission_waits_for_coordinator_collection() -> None:
    ports = Ports(round_state=FederatedRoundState.PREFLIGHT)
    runner = make_runner(ports)
    early = runner.run_round(make_offer())
    assert (early.state, early.reason_code) == (
        C.SEALED,
        R.SUBMISSION_WINDOW_NOT_OPEN,
    )
    ports.round_state = FederatedRoundState.COLLECTING
    assert runner.run_round(make_offer()).state is C.SUBMITTED


@pytest.mark.parametrize("value", [None, "collecting", 3])
def test_unknown_coordinator_state_fails_closed(value: object) -> None:
    ports = Ports(round_state=value)
    status = make_runner(ports).run_round(make_offer())
    assert status.state is C.RECEIVED
    assert status.reason_code is R.COORDINATOR_STATE_UNAVAILABLE
    assert status.resumable is True
    assert "train" not in ports.calls


def test_coordinator_state_port_error_before_submission_is_resumable() -> None:
    ports = Ports()
    runner = make_runner(ports)
    ports.round_state = FederatedRoundState.PREFLIGHT
    runner.run_round(make_offer())
    ports.raise_in.add("round_state")
    status = runner.run_round(make_offer())
    assert (status.state, status.reason_code) == (
        C.SEALED,
        R.COORDINATOR_STATE_UNAVAILABLE,
    )
    assert "submit" not in ports.calls


def test_tampered_sealed_payload_is_refused_before_submission() -> None:
    ports = Ports(submit_failures=1)
    runner = make_runner(ports)
    runner.run_round(make_offer())
    (key,) = ports.store
    record = json.loads(ports.store[key])
    record["payload"] = base64.b64encode(b'{"lora.a": [9.0]}').decode()
    ports.store[key] = json.dumps(record).encode()

    with pytest.raises(FederatedClientRunnerError) as error:
        runner.run_round(make_offer())
    assert error.value.reason_code is R.CHECKPOINT_INVALID
    assert ports.calls.count("submit") == 1
    assert not ports.submitted


def _drifted(_ports: Ports) -> dict[str, Any]:
    return {"manifest_digest": "sha256:" + "f" * 64}


def _set(attribute: str, value: object):
    def apply(ports: Ports) -> dict[str, Any]:
        setattr(ports, attribute, value)
        return {}

    return apply


def _memory(_ports: Ports) -> dict[str, Any]:
    return {"required_memory_class": FederatedMemoryClass.LARGE}


def _steps(_ports: Ports) -> dict[str, Any]:
    return {"local_steps": 11}


def _raises(name: str):
    def apply(ports: Ports) -> dict[str, Any]:
        ports.raise_in.add(name)
        return {}

    return apply


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        (_drifted, R.MANIFEST_DIGEST_DRIFT),
        (_set("verified", False), R.MANIFEST_UNVERIFIED),
        (_set("verified", 1), R.MANIFEST_UNVERIFIED),
        (_raises("verify"), R.MANIFEST_UNVERIFIED),
        (_set("compatible", False), R.CAPABILITY_INCOMPATIBLE),
        (_raises("capability"), R.CAPABILITY_INCOMPATIBLE),
        (_set("permitted", False), R.DATA_USE_DENIED),
        (_set("permitted", "yes"), R.DATA_USE_DENIED),
        (_raises("data_use"), R.DATA_USE_DENIED),
        (_memory, R.MEMORY_CLASS_EXCEEDED),
        (_steps, R.STEP_BUDGET_EXCEEDED),
        (_set("budget", False), R.SITE_BUDGET_EXHAUSTED),
        (_raises("budget"), R.SITE_BUDGET_EXHAUSTED),
    ],
)
def test_preflight_failures_decline_before_training(setup: Any, reason: R) -> None:
    ports = Ports()
    overrides = setup(ports)
    status = make_runner(ports).run_round(make_offer(**overrides))

    assert status.state is C.DECLINED
    assert status.reason_code is reason
    assert status.update_digest is None
    assert status.resumable is False
    for port in ("train", "clip", "privatize", "encode", "submit"):
        assert port not in ports.calls
    assert ports.saved_states[-1] == "declined"


def test_manifest_drift_after_sealing_aborts_without_submission() -> None:
    ports = Ports(submit_failures=1)
    runner = make_runner(ports)
    sealed = runner.run_round(make_offer())
    drifted = make_offer(manifest=MANIFEST + b" ")

    status = runner.run_round(drifted)
    assert (status.state, status.reason_code) == (C.ABORTED, R.MANIFEST_DIGEST_DRIFT)
    assert status.manifest_digest == sealed.manifest_digest
    assert ports.calls.count("submit") == 1
    assert _stored(ports)["payload"] is None


def test_manifest_drift_before_sealing_declines() -> None:
    ports = Ports(cancel_at_step=1, cancel_mode=FederatedCancellationMode.SUSPEND)
    runner = make_runner(ports)
    assert runner.run_round(make_offer()).state is C.VERIFIED
    ports.cancellation = None
    ports.calls.clear()

    status = runner.run_round(make_offer(manifest=MANIFEST + b" "))
    assert (status.state, status.reason_code) == (C.DECLINED, R.MANIFEST_DIGEST_DRIFT)
    assert ports.calls == []


def test_cooperative_abort_during_training_emits_no_update() -> None:
    ports = Ports(cancel_at_step=2)
    status = make_runner(ports).run_round(make_offer())

    assert (status.state, status.reason_code) == (C.ABORTED, R.CANCELLED)
    assert status.update_digest is None
    assert status.resumable is False
    for port in ("clip", "privatize", "encode", "submit"):
        assert port not in ports.calls
    stored = _stored(ports)
    assert stored["payload"] is None and stored["update_digest"] is None


def test_cancellation_before_training_never_calls_trainer() -> None:
    ports = Ports(cancellation=FederatedCancellationMode.ABORT)
    status = make_runner(ports).run_round(make_offer())
    assert (status.state, status.reason_code) == (C.ABORTED, R.CANCELLED)
    assert "train" not in ports.calls


def test_suspension_is_resumable_and_completes_later() -> None:
    ports = Ports(cancel_at_step=2, cancel_mode=FederatedCancellationMode.SUSPEND)
    runner = make_runner(ports)
    suspended = runner.run_round(make_offer())

    assert (suspended.state, suspended.reason_code) == (C.VERIFIED, R.SUSPENDED)
    assert suspended.resumable is True
    assert suspended.update_digest is None
    assert "clip" not in ports.calls

    ports.cancellation = None
    ports.cancel_at_step = None
    resumed = runner.run_round(make_offer())
    assert resumed.state is C.SUBMITTED
    assert ports.calls.count("train") == 2
    assert len(ports.submitted) == 1


def test_suspension_after_sealing_keeps_the_sealed_digest() -> None:
    ports = Ports(submit_failures=1)
    runner = make_runner(ports)
    sealed = runner.run_round(make_offer())
    ports.cancellation = FederatedCancellationMode.SUSPEND
    suspended = runner.run_round(make_offer())
    assert (suspended.state, suspended.reason_code) == (C.SEALED, R.SUSPENDED)
    assert suspended.update_digest == sealed.update_digest

    ports.cancellation = FederatedCancellationMode.ABORT
    aborted = runner.run_round(make_offer())
    assert (aborted.state, aborted.reason_code) == (C.ABORTED, R.CANCELLED)
    assert ports.calls.count("submit") == 1


@pytest.mark.parametrize("swallow_stop", [False, True])
def test_step_budget_breach_aborts_without_partial_update(
    swallow_stop: bool,
) -> None:
    ports = Ports(steps=5, swallow_stop=swallow_stop)
    status = make_runner(ports).run_round(make_offer(local_steps=3))
    assert (status.state, status.reason_code) == (C.ABORTED, R.STEP_BUDGET_EXCEEDED)
    assert status.update_digest is None
    assert "clip" not in ports.calls


def test_wall_clock_limit_breach_aborts() -> None:
    ports = Ports(clock_step=timedelta(minutes=30))
    limits = FederatedClientResourceLimits(
        max_wall_clock_seconds=1800,
        max_steps=10,
        memory_class=FederatedMemoryClass.SMALL,
    )
    status = make_runner(ports, limits).run_round(make_offer())
    assert (status.state, status.reason_code) == (
        C.ABORTED,
        R.WALL_CLOCK_LIMIT_EXCEEDED,
    )
    assert "clip" not in ports.calls


def test_round_deadline_during_training_expires() -> None:
    ports = Ports(
        now=datetime(2026, 1, 2, 23, 30, tzinfo=UTC), clock_step=timedelta(minutes=10)
    )
    limits = FederatedClientResourceLimits(
        max_wall_clock_seconds=86400,
        max_steps=10,
        memory_class=FederatedMemoryClass.SMALL,
    )
    status = make_runner(ports, limits).run_round(make_offer())
    assert (status.state, status.reason_code) == (C.EXPIRED, R.DEADLINE_PASSED)
    assert "clip" not in ports.calls


def test_memory_breach_report_aborts() -> None:
    ports = Ports(memory_breach_at_step=2, swallow_stop=True)
    status = make_runner(ports).run_round(make_offer())
    assert (status.state, status.reason_code) == (C.ABORTED, R.MEMORY_LIMIT_EXCEEDED)
    assert "clip" not in ports.calls


def test_crash_during_training_resumes_from_verified() -> None:
    ports = Ports(crash_in={"train"})
    with pytest.raises(SimulatedCrash):
        make_runner(ports).run_round(make_offer())
    assert _stored(ports)["state"] == "training"

    ports.crash_in.clear()
    status = make_runner(ports).run_round(make_offer())
    assert status.state is C.SUBMITTED
    assert ports.saved_states == [
        "received",
        "verified",
        "training",
        "verified",
        "training",
        "sealed",
        "submitted",
    ]


def _bad_digest(update: object, offer: FederatedClientRoundOffer) -> Any:
    return FederatedEncodedUpdate(
        payload=b"synthetic-payload", metadata=encoded(b"other").metadata
    )


def _unclipped(update: object, offer: FederatedClientRoundOffer) -> Any:
    policy = FederatedUpdatePolicy(
        model_digest=MODEL_DIGEST, parameters=(PARAMETER,), require_clipped=False
    )
    return encoded(b"synthetic-payload", clipped=False, policy=policy)


def _wrong_policy(update: object, offer: FederatedClientRoundOffer) -> Any:
    other = FederatedParameterMetadata(name="lora.b", shape=(2,), dtype="float32")
    policy = FederatedUpdatePolicy(model_digest=MODEL_DIGEST, parameters=(other,))
    return encoded(b"synthetic-payload", parameters=(other,), policy=policy)


def _oversized(update: object, offer: FederatedClientRoundOffer) -> Any:
    return encoded(b"x" * 65)


def _not_encoded(update: object, offer: FederatedClientRoundOffer) -> Any:
    return b"synthetic-payload"


def _empty(update: object, offer: FederatedClientRoundOffer) -> Any:
    return FederatedEncodedUpdate(payload=b"", metadata=encoded(b"").metadata)


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        (_raises("train"), R.TRAINING_FAILED),
        (_raises("clip"), R.CLIPPING_FAILED),
        (_raises("privatize"), R.PRIVACY_MECHANISM_FAILED),
        (_raises("encode"), R.ENCODING_FAILED),
        (_set("encode_override", _not_encoded), R.ENCODING_FAILED),
        (_set("encode_override", _empty), R.ENCODING_FAILED),
        (_set("encode_override", _oversized), R.ENCODING_FAILED),
        (_set("encode_override", _bad_digest), R.UPDATE_METADATA_INVALID),
        (_set("encode_override", _unclipped), R.UPDATE_METADATA_INVALID),
        (_set("encode_override", _wrong_policy), R.UPDATE_METADATA_INVALID),
    ],
)
def test_local_pipeline_failures_abort_without_sealing(setup: Any, reason: R) -> None:
    ports = Ports()
    setup(ports)
    limits = FederatedClientResourceLimits(
        max_wall_clock_seconds=3600,
        max_steps=10,
        memory_class=FederatedMemoryClass.SMALL,
        max_update_bytes=64,
    )
    status = make_runner(ports, limits).run_round(make_offer())
    assert (status.state, status.reason_code) == (C.ABORTED, reason)
    assert status.update_digest is None
    assert "submit" not in ports.calls
    assert "sealed" not in ports.saved_states


def test_store_failure_before_sealing_never_submits() -> None:
    ports = Ports(save_failure_state="sealed")
    with pytest.raises(FederatedClientRunnerError) as error:
        make_runner(ports).run_round(make_offer())
    assert error.value.reason_code is R.CHECKPOINT_STORE_FAILED
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__ is True
    assert "submit" not in ports.calls


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: {**record, "state": "unknown"},
        lambda record: {**record, "round_digest": "sha256:" + "0" * 64},
        lambda record: {**record, "schema_version": "v0"},
        lambda record: {**record, "extra": SENTINELS[0]},
        lambda record: {**record, "payload": "not base64!"},
        lambda record: {**record, "state": "submitted"},
        lambda record: {**record, "update_digest": None},
    ],
)
def test_invalid_checkpoints_fail_closed(mutate: Any) -> None:
    ports = Ports(submit_failures=1)
    runner = make_runner(ports)
    runner.run_round(make_offer())
    (key,) = ports.store
    ports.store[key] = json.dumps(mutate(json.loads(ports.store[key]))).encode()
    with pytest.raises(FederatedClientRunnerError) as error:
        runner.run_round(make_offer())
    assert error.value.reason_code is R.CHECKPOINT_INVALID
    assert SENTINELS[0] not in str(error.value)
    assert ports.calls.count("submit") == 1


@pytest.mark.parametrize("data", [b"not json", b"[]", b'{"a": 1, "a": 2}', 7])
def test_unparseable_checkpoints_fail_closed(data: object) -> None:
    ports = Ports()
    ports.store[make_runner(ports).checkpoint_key(make_offer())] = data
    with pytest.raises(FederatedClientRunnerError) as error:
        make_runner(ports).run_round(make_offer())
    assert error.value.reason_code is R.CHECKPOINT_INVALID
    assert ports.calls == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"round_ref": SENTINELS[1]},
        {"round_ref": SENTINELS[2]},
        {"round_ref": ""},
        {"manifest": b""},
        {"manifest": SENTINELS[2]},
        {"manifest_digest": SENTINELS[0]},
        {"schedule": SENTINELS[1]},
        {"update_policy": None},
        {"local_steps": 0},
        {"local_steps": True},
        {"required_memory_class": "small"},
    ],
)
def test_invalid_offers_fail_closed_without_echo(overrides: dict[str, Any]) -> None:
    with pytest.raises(FederatedClientRunnerError) as error:
        make_offer(**overrides)
    assert error.value.reason_code is R.INVALID_OFFER
    for sentinel in SENTINELS:
        assert sentinel not in str(error.value)
        assert sentinel not in repr(error.value)


def test_invalid_configuration_fails_closed() -> None:
    ports = Ports()
    with pytest.raises(FederatedClientRunnerError) as error:
        FederatedClientRunner(
            manifest_verifier=object(),
            capability=_Facade(ports, is_compatible="is_compatible"),
            data_use_policy=_Facade(ports, permits="permits"),
            trainer=_Facade(ports, train="train"),
            clipper=_Facade(ports, clip="clip"),
            privacy_mechanism=_Facade(
                ports, has_budget="has_budget", privatize="privatize"
            ),
            encoder=_Facade(ports, encode="encode"),
            submitter=_Facade(ports, round_state="get_round_state", submit="submit"),
            clock=_Facade(ports, now="current_time"),
            store=_Facade(ports, load="load", save="save"),
            limits=FederatedClientResourceLimits(
                max_wall_clock_seconds=60,
                max_steps=1,
                memory_class=FederatedMemoryClass.SMALL,
            ),
        )
    assert error.value.reason_code is R.INVALID_CONFIGURATION


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_wall_clock_seconds": 0},
        {"max_steps": -1},
        {"memory_class": "large"},
        {"max_update_bytes": 0},
    ],
)
def test_invalid_limits_fail_closed(overrides: dict[str, Any]) -> None:
    arguments = {
        "max_wall_clock_seconds": 60,
        "max_steps": 1,
        "memory_class": FederatedMemoryClass.SMALL,
    }
    arguments.update(overrides)
    with pytest.raises(FederatedClientRunnerError) as error:
        FederatedClientResourceLimits(**arguments)
    assert error.value.reason_code is R.INVALID_CONFIGURATION


@pytest.mark.parametrize(
    "moment", [datetime(2026, 1, 2, 6), "2026-01-02T06:00:00Z", None]
)
def test_invalid_clock_values_fail_closed(moment: object) -> None:
    ports = Ports(now=moment)
    with pytest.raises(FederatedClientRunnerError) as error:
        make_runner(ports).run_round(make_offer())
    assert error.value.reason_code is R.INVALID_CLOCK
    assert ports.calls == []


def test_status_records_and_errors_contain_no_private_values() -> None:
    round_ref = SENTINELS[0]
    ports = Ports(submit_failures=1)
    ports.raise_in.add("verify")
    runner = make_runner(ports)
    offer = make_offer(round_ref=round_ref)
    surfaces = [repr(offer), repr(runner)]

    declined = runner.run_round(offer)
    surfaces += [declined.to_json(), repr(declined), str(declined.to_dict())]

    ports.raise_in.clear()
    second = make_offer(round_ref=round_ref + "-2")
    sealed = runner.run_round(second)
    surfaces += [sealed.to_json(), repr(sealed)]
    for name in ("train", "clip", "privatize", "encode"):
        failing = Ports(raise_in={name})
        status = make_runner(failing).run_round(offer)
        surfaces += [status.to_json(), repr(status)]

    with pytest.raises(FederatedClientRunnerError) as error:
        runner.resend_update(second, SENTINELS[3].encode())
    surfaces += [str(error.value), repr(error.value), repr(error.value.args)]
    assert error.value.__cause__ is None

    control_ports = Ports()
    make_runner(control_ports).run_round(offer)
    surfaces.append(repr(control_ports.controls[0]))

    encoded_update = encoded(SENTINELS[3].encode())
    surfaces.append(repr(encoded_update))

    for key, data in ports.store.items():
        surfaces += [key, json.loads(data)["round_digest"]]
        assert round_ref.encode() not in data

    text = "\n".join(surfaces)
    for sentinel in SENTINELS:
        assert sentinel not in text
    assert not re.search(r"\d+\.\d{3,}", text)


def test_status_record_is_compatible_with_round_status_contracts() -> None:
    ports = Ports(cancel_at_step=1)
    aborted = make_runner(ports).run_round(make_offer())
    assert aborted.round_reason_code is FederatedRoundReasonCode.OPERATOR_CANCELLED
    round_status = build_federated_round_status(
        state=FederatedRoundState.ABORTED,
        participant_count=1,
        completed_participant_count=0,
        required_quorum=1,
        reason_code=aborted.round_reason_code,
    )
    assert round_status.reason_code is FederatedRoundReasonCode.OPERATOR_CANCELLED

    submitted = make_runner(Ports()).run_round(make_offer())
    assert submitted.round_reason_code is None
    build_federated_round_status(
        state=FederatedRoundState.COLLECTING,
        participant_count=1,
        completed_participant_count=1,
        required_quorum=1,
        aggregate_digest_refs=(submitted.update_digest,),
    )

    payload = json.loads(submitted.to_json())
    assert payload == {
        "manifest_digest": _digest(MANIFEST),
        "reason_code": None,
        "resumable": False,
        "round_digest": _digest(b"synthetic-round-1"),
        "round_reason_code": None,
        "schema_version": FEDERATED_CLIENT_RUNNER_SCHEMA_VERSION,
        "state": "submitted",
        "update_digest": submitted.update_digest,
    }
    assert submitted.to_json().endswith("}\n")


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        (_set("budget", False), FederatedRoundReasonCode.PRIVACY_GATE_FAILED),
        (_set("permitted", False), FederatedRoundReasonCode.POLICY_GATE_FAILED),
        (_raises("train"), FederatedRoundReasonCode.ROUND_ERROR),
        (_raises("privatize"), FederatedRoundReasonCode.PRIVACY_GATE_FAILED),
    ],
)
def test_round_reason_codes_map_to_abort_reasons(setup: Any, expected: Any) -> None:
    ports = Ports()
    setup(ports)
    assert make_runner(ports).run_round(make_offer()).round_reason_code is expected


def test_every_reason_code_has_a_round_reason_mapping() -> None:
    from openmed.training.federated_client_runner import round_reason_for

    for reason in R:
        mapped = round_reason_for(reason)
        assert isinstance(mapped, FederatedRoundReasonCode)


def test_transition_table_is_closed_and_terminal_states_are_final() -> None:
    assert set(FEDERATED_CLIENT_TRANSITIONS) == set(C)
    assert FEDERATED_CLIENT_TERMINAL_STATES == frozenset(
        {C.DECLINED, C.SUBMITTED, C.ABORTED, C.EXPIRED}
    )
    for state in FEDERATED_CLIENT_TERMINAL_STATES:
        assert FEDERATED_CLIENT_TRANSITIONS[state] == frozenset()
    assert C.TRAINING not in FEDERATED_CLIENT_TRANSITIONS[C.SEALED]
    assert C.SEALED in FEDERATED_CLIENT_TRANSITIONS[C.TRAINING]
    assert C.TRAINING not in FEDERATED_CLIENT_TRANSITIONS[C.RECEIVED]


def test_reason_codes_are_stable_snake_case() -> None:
    for reason in R:
        assert re.fullmatch(r"[a-z]+(?:_[a-z]+)*", reason.value)
        assert str(reason) == reason.value


def test_module_source_is_ascii_only() -> None:
    root = Path(__file__).resolve().parents[3]
    for path in (
        root / "openmed" / "training" / "federated_client_runner.py",
        root / "docs" / "training" / "federated-client-runner.md",
    ):
        path.read_bytes().decode("ascii")


@pytest.mark.integration
def test_documented_example_runs_offline() -> None:
    root = Path(__file__).resolve().parents[3]
    page = (root / "docs" / "training" / "federated-client-runner.md").read_text(
        encoding="utf-8"
    )
    blocks = re.findall(r"```python\n(.*?)```", page, flags=re.DOTALL)
    assert blocks
    namespace: dict[str, Any] = {}
    for block in blocks:
        exec(compile(block, "federated-client-runner.md", "exec"), namespace)
    assert namespace["status"].state is C.SUBMITTED


def test_sealed_retry_reverifies_signed_offer_terms_before_submission():
    ports = Ports(submit_failures=1)
    assert make_runner(ports).run_round(make_offer()).state is C.SEALED
    ports.verified = False
    status = make_runner(ports).run_round(make_offer())
    assert status.state is C.ABORTED
    assert status.reason_code is R.MANIFEST_UNVERIFIED
    assert not ports.submitted
    assert ports.calls.count("train") == 1


@pytest.mark.parametrize("stop", ("deadline", "cancel"))
def test_submission_rechecks_conditions_after_coordinator_call(stop):
    ports = Ports(submit_failures=1)
    assert make_runner(ports).run_round(make_offer()).state is C.SEALED
    original = ports.get_round_state

    def slow_state(offer):
        if stop == "deadline":
            ports.now = SCHEDULE.aggregation_starts_at
        else:
            ports.cancellation = FederatedCancellationMode.ABORT
        return original(offer)

    ports.get_round_state = slow_state
    status = make_runner(ports).run_round(make_offer())
    assert status.state is (C.EXPIRED if stop == "deadline" else C.ABORTED)
    assert not ports.submitted
