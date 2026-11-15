"""Reference v4.0 service recovery drill with value-free evidence.

The drill planned for v3.4 was folded into v4.0. It declares one failure per
category (storage, process, network and corrupted state) and walks each
through six phases: ``inject``, ``detect``, ``contain``, ``recover``,
``verify`` and ``resume``. Every phase is a pluggable step. The defaults
exercise recovery code that is on master today:

* storage: the async job metadata store rolls back a failed write and keeps
  committed records across a reopen (:mod:`openmed.service.jobs`);
* process: committed job metadata survives an abrupt worker stop;
* network: the model-backend circuit breaker opens, sheds, admits one
  recovery probe and closes (:mod:`openmed.service.resilience`); and
* corrupted state: the Journey artifact store refuses to serve or silently
  replace a corrupted blob while unaffected blobs stay readable
  (:mod:`openmed.structured.store.local`).

A phase whose capability is not on master yet (encrypted restore, leased
durable jobs, audit continuity) is recorded as ``not_available`` with its
owning issue, which keeps the report ``incomplete`` and blocks a release.
Callers inject an implementation for a ``(category, phase)`` pair when the
owning work lands.

The report holds controlled codes, counts, issue references and a canonical
self-digest only. It never records paths, record contents, exception text or
host details. The drill runs in a private temporary directory, uses synthetic
records only and makes no network call.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from openmed.core.audit import stable_hash
from openmed.eval.journey_release import (
    JourneyReleaseError,
    _git_output,
    _parse_timestamp,
    _write_release_json,
)

RECOVERY_DRILL_SCHEMA: Final = "openmed.service_release.recovery_drill.v1"
RECOVERY_DRILL_PROFILE: Final = "reference-v4.0"
RECOVERY_DRILL_RELEASE: Final = "v4.0"

STORAGE: Final = "storage"
PROCESS: Final = "process"
NETWORK: Final = "network"
CORRUPTED_STATE: Final = "corrupted_state"
DRILL_CATEGORIES: Final = (STORAGE, PROCESS, NETWORK, CORRUPTED_STATE)
DRILL_PHASES: Final = ("inject", "detect", "contain", "recover", "verify", "resume")

STEP_PASSED: Final = "passed"
STEP_FAILED: Final = "failed"
STEP_NOT_AVAILABLE: Final = "not_available"
STEP_STATES: Final = (STEP_PASSED, STEP_FAILED, STEP_NOT_AVAILABLE)

DRILL_COMPLETE: Final = "complete"
DRILL_INCOMPLETE: Final = "incomplete"
DRILL_FAILED: Final = "failed"
DRILL_STATES: Final = (DRILL_COMPLETE, DRILL_INCOMPLETE, DRILL_FAILED)

OBJECTIVE_MET: Final = "met"
OBJECTIVE_MISSED: Final = "missed"
OBJECTIVE_NOT_MEASURED: Final = "not_measured"
OBJECTIVE_STATES: Final = (OBJECTIVE_MET, OBJECTIVE_MISSED, OBJECTIVE_NOT_MEASURED)

# Data-integrity objectives: each target is a maximum count, measured by the
# listed steps. Recovery-time objectives belong to the reference deployment
# declaration and are not measured by this drill.
DRILL_OBJECTIVES: Final[Mapping[str, tuple[tuple[str, str], ...]]] = MappingProxyType(
    {
        "admissions_during_open_circuit": ((NETWORK, "contain"),),
        "corrupted_bytes_served": ((CORRUPTED_STATE, "detect"),),
        "lost_committed_records": ((STORAGE, "verify"), (PROCESS, "verify")),
        "partial_writes_visible": ((STORAGE, "contain"),),
        "unverified_restored_records": (
            (STORAGE, "recover"),
            (CORRUPTED_STATE, "recover"),
        ),
    }
)

# Codes this module can place in a report, by role. The release gate's
# failure-code inventory includes them.
DRILL_CHECK_CODES: Final = (
    "abrupt_worker_stop_simulated",
    "admission_resumed",
    "admission_resumed_after_failure",
    "admission_resumed_after_restart",
    "backend_connection_failures_injected",
    "blob_bytes_corrupted",
    "checkpoint_replay",
    "circuit_closed_after_probe",
    "circuit_opened",
    "committed_records_survive_reopen",
    "committed_records_survive_restart",
    "continuity_manifest_verification",
    "corrupted_blob_not_silently_replaced",
    "encrypted_backup_restore",
    "failed_write_rolled_back",
    "integrity_failure_detected",
    "orphaned_lease_detection",
    "persist_fault_armed",
    "poison_job_quarantine",
    "requests_shed_while_open",
    "single_recovery_probe_admitted",
    "step_raised",
    "step_result_invalid",
    "unaffected_records_served",
    "verified_restore_to_inactive_target",
    "write_failure_surfaced",
)
DRILL_REPORT_CODES: Final = (
    "drill_category_missing",
    "drill_digest_mismatch",
    "drill_report_inconsistent",
    "drill_report_invalid",
    "unknown_schema",
)
DRILL_ERROR_CODES: Final = (
    "candidate_commit_invalid",
    "drill_invalid",
    "drill_step_key_invalid",
    "drill_unavailable",
    "generated_at_invalid",
    "not_available_step_invalid",
    "output_exists",
    "output_unwritable",
    "work_dir_not_empty",
)

_CODE_RE: Final = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_ISSUE_RE: Final = re.compile(r"#[1-9][0-9]{0,6}\Z")
_COMMIT_RE: Final = re.compile(r"[0-9a-f]{40}\Z")
_DIGEST_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REPORT_FIELDS: Final = frozenset(
    {
        "candidate_commit",
        "generated_at",
        "objectives",
        "profile",
        "release",
        "report_digest",
        "scenarios",
        "schema",
        "state",
        "summary",
    }
)
_SUMMARY_FIELDS: Final = frozenset(
    {
        "objectives_met",
        "objectives_missed",
        "objectives_not_measured",
        "scenario_count",
        "steps_failed",
        "steps_not_available",
        "steps_passed",
    }
)

# Synthetic job metadata. These records carry no submitted text.
_DRILL_TIMESTAMP: Final = "2026-01-01T00:00:00Z"
_DRILL_EXPIRY: Final = "2999-01-01T00:00:00Z"
_SYNTHETIC_BLOB: Final = b"synthetic recovery drill artifact v1"
_SYNTHETIC_PEER_BLOB: Final = b"synthetic recovery drill peer artifact v1"


class RecoveryDrillError(ValueError):
    """Raised when the drill cannot produce a safe report.

    Attributes:
        code: A value-free reason code.
    """

    def __init__(self, code: str) -> None:
        self.code = code if _CODE_RE.fullmatch(code) else "drill_invalid"
        super().__init__(self.code)


@dataclass(frozen=True)
class DrillStepOutcome:
    """Value-free outcome of one drill step.

    Attributes:
        state: ``passed``, ``failed`` or ``not_available``.
        check: Controlled code naming what the step verified.
        observed: Objective counts measured by this step.
        owner: Issue reference that owns an unavailable capability.
    """

    state: str
    check: str
    observed: Mapping[str, int] = field(default_factory=dict)
    owner: str | None = None


@dataclass
class DrillContext:
    """Private, in-memory state shared by the steps of one scenario.

    Attributes:
        category: The scenario category.
        work_dir: A private directory for this scenario. It never enters the
            report.
        state: Scratch objects handed from one phase to the next.
    """

    category: str
    work_dir: Path
    state: dict[str, Any] = field(default_factory=dict)


DrillStep = Callable[[DrillContext], DrillStepOutcome]


def not_available(owner: str, check: str) -> DrillStep:
    """Return a step that records a capability owned by another issue.

    Args:
        owner: Issue reference such as ``#3577``.
        check: Controlled code naming the missing capability.

    Returns:
        A step that always reports ``not_available``.
    """

    if not _ISSUE_RE.fullmatch(owner) or not _CODE_RE.fullmatch(check):
        raise RecoveryDrillError("not_available_step_invalid")

    def step(context: DrillContext) -> DrillStepOutcome:
        del context
        return DrillStepOutcome(STEP_NOT_AVAILABLE, check, owner=owner)

    return step


def _outcome(passed: bool, check: str, **observed: int) -> DrillStepOutcome:
    return DrillStepOutcome(STEP_PASSED if passed else STEP_FAILED, check, observed)


# --- storage: job metadata store write failure ------------------------------


def _job_record(job_id: str, status: str) -> dict[str, Any]:
    return {
        "created_at": _DRILL_TIMESTAMP,
        "documents": [],
        "expires_at": _DRILL_EXPIRY,
        "id": job_id,
        "status": status,
        "updated_at": _DRILL_TIMESTAMP,
    }


def _fault_injecting_job_store(path: Path) -> Any:
    from openmed.service.jobs import LocalJobStore

    class _FaultInjectingJobStore(LocalJobStore):
        """Job store whose next persist fails like a full disk."""

        fail_next_persist = False

        def _persist_locked(self) -> None:
            if self.fail_next_persist:
                self.fail_next_persist = False
                raise OSError(28, "drill fault")
            super()._persist_locked()

    return _FaultInjectingJobStore(path)


def _open_job_store(path: Path) -> Any:
    from openmed.service.jobs import LocalJobStore

    return LocalJobStore(path)


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _storage_inject(context: DrillContext) -> DrillStepOutcome:
    path = context.work_dir / "jobs.json"
    store = _fault_injecting_job_store(path)
    committed = store.create(_job_record("job-drill-committed", "queued"))
    context.state.update(
        store=store,
        path=path,
        committed=committed,
        digest_before=_file_digest(path),
    )
    store.fail_next_persist = True
    return _outcome(True, "persist_fault_armed")


def _storage_detect(context: DrillContext) -> DrillStepOutcome:
    store = context.state["store"]
    try:
        store.create(_job_record("job-drill-pending", "queued"))
    except OSError:
        return _outcome(True, "write_failure_surfaced")
    return _outcome(False, "write_failure_surfaced")


def _storage_contain(context: DrillContext) -> DrillStepOutcome:
    store, path = context.state["store"], context.state["path"]
    leftovers = [item for item in path.parent.iterdir() if item.name != path.name]
    visible = int(store.get("job-drill-pending") is not None)
    visible += int(_file_digest(path) != context.state["digest_before"])
    visible += len(leftovers)
    intact = store.get("job-drill-committed") == context.state["committed"]
    return _outcome(
        intact and visible == 0,
        "failed_write_rolled_back",
        partial_writes_visible=visible,
    )


def _storage_verify(context: DrillContext) -> DrillStepOutcome:
    reopened = _open_job_store(context.state["path"])
    context.state["reopened"] = reopened
    lost = int(reopened.get("job-drill-committed") != context.state["committed"])
    unexpected = int(reopened.get("job-drill-pending") is not None)
    return _outcome(
        lost == 0 and unexpected == 0,
        "committed_records_survive_reopen",
        lost_committed_records=lost,
    )


def _storage_resume(context: DrillContext) -> DrillStepOutcome:
    context.state["reopened"].create(_job_record("job-drill-pending", "queued"))
    durable = _open_job_store(context.state["path"]).get("job-drill-pending")
    return _outcome(durable is not None, "admission_resumed_after_failure")


# --- process: abrupt worker stop ---------------------------------------------


def _process_inject(context: DrillContext) -> DrillStepOutcome:
    path = context.work_dir / "jobs.json"
    store = _open_job_store(path)
    running = store.create(_job_record("job-drill-running", "running"))
    context.state.update(path=path, running=running)
    # The worker stops without shutdown, cleanup or a terminal update.
    del store
    return _outcome(True, "abrupt_worker_stop_simulated")


def _process_verify(context: DrillContext) -> DrillStepOutcome:
    restarted = _open_job_store(context.state["path"])
    context.state["restarted"] = restarted
    lost = int(restarted.get("job-drill-running") != context.state["running"])
    return _outcome(
        lost == 0, "committed_records_survive_restart", lost_committed_records=lost
    )


def _process_resume(context: DrillContext) -> DrillStepOutcome:
    context.state["restarted"].create(_job_record("job-drill-after-restart", "queued"))
    durable = _open_job_store(context.state["path"]).get("job-drill-after-restart")
    return _outcome(durable is not None, "admission_resumed_after_restart")


# --- network: model-backend connection failures ------------------------------


class _DrillClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _network_inject(context: DrillContext) -> DrillStepOutcome:
    from openmed.service.resilience import (
        ResilienceManager,
        ServiceResilienceConfig,
    )

    clock = _DrillClock()
    manager = ResilienceManager(
        config=ServiceResilienceConfig(
            max_attempts=1,
            failure_threshold=2,
            recovery_timeout_seconds=30.0,
            backoff_jitter_seconds=0.0,
        ),
        clock=clock,
        sleep=lambda _seconds: None,
    )
    calls = {"count": 0, "healthy": False}

    def backend() -> str:
        calls["count"] += 1
        if not calls["healthy"]:
            raise ConnectionError
        return "ok"

    context.state.update(manager=manager, clock=clock, calls=calls, backend=backend)
    surfaced = 0
    for _attempt in range(2):
        try:
            manager.execute("drill-backend", backend)
        except ConnectionError:
            surfaced += 1
    return _outcome(surfaced == 2, "backend_connection_failures_injected")


def _circuit_state(context: DrillContext) -> str:
    snapshot = context.state["manager"].snapshots()["drill-backend"]
    return str(snapshot.state)


def _network_detect(context: DrillContext) -> DrillStepOutcome:
    return _outcome(_circuit_state(context) == "open", "circuit_opened")


def _network_contain(context: DrillContext) -> DrillStepOutcome:
    from openmed.service.resilience import CircuitBreakerOpenError

    calls = context.state["calls"]
    before = calls["count"]
    shed = False
    try:
        context.state["manager"].execute("drill-backend", context.state["backend"])
    except CircuitBreakerOpenError:
        shed = True
    admitted = calls["count"] - before
    return _outcome(
        shed and admitted == 0,
        "requests_shed_while_open",
        admissions_during_open_circuit=admitted,
    )


def _network_recover(context: DrillContext) -> DrillStepOutcome:
    from openmed.service.resilience import CircuitBreakerOpenError

    manager = context.state["manager"]
    context.state["clock"].now += 31.0
    context.state["calls"]["healthy"] = True
    manager.check_available("drill-backend")
    second_probe_rejected = False
    try:
        manager.check_available("drill-backend")
    except CircuitBreakerOpenError:
        second_probe_rejected = True
    manager.record_success("drill-backend")
    return _outcome(second_probe_rejected, "single_recovery_probe_admitted")


def _network_verify(context: DrillContext) -> DrillStepOutcome:
    snapshot = context.state["manager"].snapshots()["drill-backend"]
    return _outcome(
        snapshot.state == "closed" and snapshot.failures == 0,
        "circuit_closed_after_probe",
    )


def _network_resume(context: DrillContext) -> DrillStepOutcome:
    calls = context.state["calls"]
    before = calls["count"]
    result = context.state["manager"].execute("drill-backend", context.state["backend"])
    return _outcome(
        result == "ok" and calls["count"] == before + 1, "admission_resumed"
    )


# --- corrupted state: Journey artifact blob corruption -----------------------


def _artifact(content: bytes, suffix: str) -> Any:
    from openmed.clinical.journey_contracts import ClinicalArtifact, sha256_digest

    return ClinicalArtifact(
        artifact_id=f"artifact_drill{suffix * 12}",
        artifact_type="clinical_note",
        media_type="text/plain",
        content_hash=sha256_digest(content),
        byte_size=len(content),
        source_id=f"source_drill{suffix * 12}",
        recorded_at=_DRILL_TIMESTAMP,
    )


def _blob_location(root: Path, content_hash: str) -> Path:
    digest = content_hash.removeprefix("sha256:")
    return root / "blobs" / "sha256" / digest[:2] / digest


def _corrupted_inject(context: DrillContext) -> DrillStepOutcome:
    from openmed.structured.store.local import LocalArtifactStore

    root = context.work_dir / "artifacts"
    store = LocalArtifactStore(root)
    affected = _artifact(_SYNTHETIC_BLOB, "a")
    peer = _artifact(_SYNTHETIC_PEER_BLOB, "b")
    stored = store.put_bytes(affected, _SYNTHETIC_BLOB).ok
    stored = store.put_bytes(peer, _SYNTHETIC_PEER_BLOB).ok and stored
    blob = _blob_location(root, affected.content_hash)
    data = bytearray(blob.read_bytes())
    data[0] ^= 0x01
    blob.chmod(0o600)
    blob.write_bytes(bytes(data))
    context.state.update(store=store, affected=affected, peer=peer)
    return _outcome(stored, "blob_bytes_corrupted")


def _corrupted_detect(context: DrillContext) -> DrillStepOutcome:
    result = context.state["store"].get_bytes(context.state["affected"].content_hash)
    served = int(result.ok)
    return _outcome(
        served == 0 and result.code == "artifact_integrity_failed",
        "integrity_failure_detected",
        corrupted_bytes_served=served,
    )


def _corrupted_contain(context: DrillContext) -> DrillStepOutcome:
    store, affected = context.state["store"], context.state["affected"]
    overwrite = store.put_bytes(affected, _SYNTHETIC_BLOB)
    still_fenced = not store.get_bytes(affected.content_hash).ok
    return _outcome(
        not overwrite.ok and still_fenced, "corrupted_blob_not_silently_replaced"
    )


def _corrupted_resume(context: DrillContext) -> DrillStepOutcome:
    store, peer = context.state["store"], context.state["peer"]
    result = store.get_bytes(peer.content_hash)
    return _outcome(
        result.ok and result.value == _SYNTHETIC_PEER_BLOB, "unaffected_records_served"
    )


# Phases whose capability is owned by open work: (owning issue, check code).
UNAVAILABLE_DRILL_STEPS: Final[Mapping[tuple[str, str], tuple[str, str]]] = (
    MappingProxyType(
        {
            (STORAGE, "recover"): ("#3577", "encrypted_backup_restore"),
            (PROCESS, "detect"): ("#3574", "orphaned_lease_detection"),
            (PROCESS, "contain"): ("#3574", "poison_job_quarantine"),
            (PROCESS, "recover"): ("#3574", "checkpoint_replay"),
            (CORRUPTED_STATE, "recover"): (
                "#3577",
                "verified_restore_to_inactive_target",
            ),
            (CORRUPTED_STATE, "verify"): ("#3599", "continuity_manifest_verification"),
        }
    )
)

DEFAULT_DRILL_STEPS: Final[Mapping[tuple[str, str], DrillStep]] = MappingProxyType(
    {
        (STORAGE, "inject"): _storage_inject,
        (STORAGE, "detect"): _storage_detect,
        (STORAGE, "contain"): _storage_contain,
        (STORAGE, "verify"): _storage_verify,
        (STORAGE, "resume"): _storage_resume,
        (PROCESS, "inject"): _process_inject,
        (PROCESS, "verify"): _process_verify,
        (PROCESS, "resume"): _process_resume,
        (NETWORK, "inject"): _network_inject,
        (NETWORK, "detect"): _network_detect,
        (NETWORK, "contain"): _network_contain,
        (NETWORK, "recover"): _network_recover,
        (NETWORK, "verify"): _network_verify,
        (NETWORK, "resume"): _network_resume,
        (CORRUPTED_STATE, "inject"): _corrupted_inject,
        (CORRUPTED_STATE, "detect"): _corrupted_detect,
        (CORRUPTED_STATE, "contain"): _corrupted_contain,
        (CORRUPTED_STATE, "resume"): _corrupted_resume,
        **{
            key: not_available(owner, check)
            for key, (owner, check) in UNAVAILABLE_DRILL_STEPS.items()
        },
    }
)


# --- runner ------------------------------------------------------------------


def run_recovery_drill(
    *,
    candidate_commit: str,
    generated_at: str,
    work_dir: str | Path | None = None,
    steps: Mapping[tuple[str, str], DrillStep] | None = None,
) -> dict[str, Any]:
    """Run the reference drill and return its self-digested report.

    Args:
        candidate_commit: Full 40-character commit the drill ran against.
        generated_at: RFC 3339 UTC timestamp in whole seconds.
        work_dir: Optional empty directory for the drill's private files.
            A temporary directory is used and removed when omitted.
        steps: Optional steps by ``(category, phase)`` that replace the
            defaults in :data:`DEFAULT_DRILL_STEPS`.

    Returns:
        The report, including ``report_digest``.

    Raises:
        RecoveryDrillError: If an argument or injected step key is invalid.
    """

    if not isinstance(candidate_commit, str) or not _COMMIT_RE.fullmatch(
        candidate_commit
    ):
        raise RecoveryDrillError("candidate_commit_invalid")
    try:
        _parse_timestamp(generated_at, "generated_at")
    except JourneyReleaseError:
        raise RecoveryDrillError("generated_at_invalid") from None
    registry = dict(DEFAULT_DRILL_STEPS)
    for key, step in (steps or {}).items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or key[0] not in DRILL_CATEGORIES
            or key[1] not in DRILL_PHASES
            or not callable(step)
        ):
            raise RecoveryDrillError("drill_step_key_invalid")
        registry[key] = step

    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="openmed-drill-") as temporary:
            scenarios = _run_scenarios(Path(temporary), registry)
    else:
        root = Path(work_dir)
        if root.is_symlink() or (root.exists() and any(root.iterdir())):
            raise RecoveryDrillError("work_dir_not_empty")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        scenarios = _run_scenarios(root, registry)
    return _build_report(scenarios, candidate_commit, generated_at)


def _run_scenarios(
    root: Path,
    registry: Mapping[tuple[str, str], DrillStep],
) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for category in DRILL_CATEGORIES:
        directory = root / category
        directory.mkdir(mode=0o700)
        context = DrillContext(category=category, work_dir=directory)
        rows = [
            _run_step(registry[(category, phase)], context, phase)
            for phase in DRILL_PHASES
        ]
        scenarios.append(
            {
                "category": category,
                "id": category,
                "state": _scenario_state(rows),
                "steps": rows,
            }
        )
    return scenarios


def _scenario_state(rows: Sequence[Mapping[str, Any]]) -> str:
    states = {row["state"] for row in rows}
    if STEP_FAILED in states:
        return DRILL_FAILED
    if STEP_NOT_AVAILABLE in states:
        return DRILL_INCOMPLETE
    return DRILL_COMPLETE


def _run_step(step: DrillStep, context: DrillContext, phase: str) -> dict[str, Any]:
    try:
        outcome = step(context)
    except Exception:
        # Exception text may carry paths or record content; keep the code only.
        outcome = DrillStepOutcome(STEP_FAILED, "step_raised")
    if not _outcome_valid(outcome):
        outcome = DrillStepOutcome(STEP_FAILED, "step_result_invalid")
    return {
        "check": outcome.check,
        "observed": dict(sorted(outcome.observed.items())),
        "owner": outcome.owner,
        "phase": phase,
        "state": outcome.state,
    }


def _outcome_valid(outcome: Any) -> bool:
    if not isinstance(outcome, DrillStepOutcome):
        return False
    if outcome.state not in STEP_STATES or not _is_code(outcome.check):
        return False
    if (outcome.state == STEP_NOT_AVAILABLE) != (outcome.owner is not None):
        return False
    if outcome.owner is not None and not _ISSUE_RE.fullmatch(str(outcome.owner)):
        return False
    return all(
        name in DRILL_OBJECTIVES and type(value) is int and value >= 0
        for name, value in dict(outcome.observed).items()
    )


def _build_report(
    scenarios: Sequence[Mapping[str, Any]],
    candidate_commit: str,
    generated_at: str,
) -> dict[str, Any]:
    steps = {
        (scenario["category"], row["phase"]): row
        for scenario in scenarios
        for row in scenario["steps"]
    }
    objectives = []
    for name, sources in DRILL_OBJECTIVES.items():
        rows = [steps[key] for key in sources]
        if any(row["state"] == STEP_NOT_AVAILABLE for row in rows):
            objectives.append(
                {
                    "id": name,
                    "observed": None,
                    "state": OBJECTIVE_NOT_MEASURED,
                    "target": 0,
                }
            )
            continue
        observed = sum(int(row["observed"].get(name, 0)) for row in rows)
        measured = all(name in row["observed"] for row in rows)
        state = OBJECTIVE_MISSED if observed > 0 or not measured else OBJECTIVE_MET
        objectives.append(
            {"id": name, "observed": observed, "state": state, "target": 0}
        )
    all_steps = [row for scenario in scenarios for row in scenario["steps"]]
    summary = {
        "objectives_met": _count(objectives, OBJECTIVE_MET),
        "objectives_missed": _count(objectives, OBJECTIVE_MISSED),
        "objectives_not_measured": _count(objectives, OBJECTIVE_NOT_MEASURED),
        "scenario_count": len(scenarios),
        "steps_failed": _count(all_steps, STEP_FAILED),
        "steps_not_available": _count(all_steps, STEP_NOT_AVAILABLE),
        "steps_passed": _count(all_steps, STEP_PASSED),
    }
    if summary["steps_failed"] or summary["objectives_missed"]:
        state = DRILL_FAILED
    elif summary["steps_not_available"] or summary["objectives_not_measured"]:
        state = DRILL_INCOMPLETE
    else:
        state = DRILL_COMPLETE
    report: dict[str, Any] = {
        "candidate_commit": candidate_commit,
        "generated_at": generated_at,
        "objectives": objectives,
        "profile": RECOVERY_DRILL_PROFILE,
        "release": RECOVERY_DRILL_RELEASE,
        "scenarios": [dict(item) for item in scenarios],
        "schema": RECOVERY_DRILL_SCHEMA,
        "state": state,
        "summary": summary,
    }
    report["report_digest"] = recovery_drill_digest(report)
    return report


def _count(rows: Sequence[Mapping[str, Any]], state: str) -> int:
    return sum(1 for row in rows if row["state"] == state)


def recovery_drill_digest(report: Mapping[str, Any]) -> str:
    """Return the canonical digest of a report, excluding ``report_digest``.

    Args:
        report: A drill report.

    Returns:
        A prefixed SHA-256 digest of the canonical JSON.
    """

    return stable_hash({k: v for k, v in report.items() if k != "report_digest"})


def validate_recovery_drill_report(report: Any) -> tuple[str, ...]:
    """Check a report's structure and self-digest.

    Args:
        report: A decoded report.

    Returns:
        Sorted value-free problem codes; empty when the report is well formed.
        Codes: ``unknown_schema``, ``drill_report_invalid``,
        ``drill_digest_mismatch``, ``drill_report_inconsistent`` and
        ``drill_category_missing:<category>``.
    """

    if not isinstance(report, Mapping) or report.get("schema") != (
        RECOVERY_DRILL_SCHEMA
    ):
        return ("unknown_schema",)
    try:
        problems = set(_structure_problems(report))
    except (KeyError, TypeError, AttributeError, ValueError):
        return ("drill_report_invalid",)
    if problems:
        return tuple(sorted(problems))
    if report["report_digest"] != recovery_drill_digest(report):
        return ("drill_digest_mismatch",)
    # The digest is unkeyed, so states, objectives and counts are re-derived
    # from the step rows rather than trusted.
    rescored = [
        {**scenario, "state": _scenario_state(scenario["steps"])}
        for scenario in report["scenarios"]
    ]
    rebuilt = _build_report(
        rescored, report["candidate_commit"], report["generated_at"]
    )
    if rebuilt != dict(report):
        return ("drill_report_inconsistent",)
    return ()


def _structure_problems(report: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if set(report) != _REPORT_FIELDS:
        return ["drill_report_invalid"]
    if (
        report["profile"] != RECOVERY_DRILL_PROFILE
        or report["release"] != RECOVERY_DRILL_RELEASE
        or not isinstance(report["candidate_commit"], str)
        or not _COMMIT_RE.fullmatch(report["candidate_commit"])
        or not isinstance(report["report_digest"], str)
        or not _DIGEST_RE.fullmatch(report["report_digest"])
        or report["state"] not in DRILL_STATES
    ):
        problems.append("drill_report_invalid")
    _parse_timestamp(report["generated_at"], "generated_at")
    categories: list[str] = []
    for scenario in report["scenarios"]:
        if set(scenario) != {"category", "id", "state", "steps"}:
            problems.append("drill_report_invalid")
            continue
        categories.append(scenario["category"])
        phases = [row["phase"] for row in scenario["steps"]]
        if (
            scenario["category"] not in DRILL_CATEGORIES
            or scenario["state"] not in DRILL_STATES
            or tuple(phases) != DRILL_PHASES
            or not all(_row_valid(row) for row in scenario["steps"])
        ):
            problems.append("drill_report_invalid")
    if len(categories) != len(set(categories)):
        problems.append("drill_report_invalid")
    problems.extend(
        f"drill_category_missing:{category}"
        for category in DRILL_CATEGORIES
        if category not in categories
    )
    names = [row["id"] for row in report["objectives"]]
    if sorted(names) != sorted(DRILL_OBJECTIVES) or not all(
        set(row) == {"id", "observed", "state", "target"}
        and row["state"] in OBJECTIVE_STATES
        and row["target"] == 0
        and (row["observed"] is None or _is_count(row["observed"]))
        for row in report["objectives"]
    ):
        problems.append("drill_report_invalid")
    summary = report["summary"]
    if set(summary) != _SUMMARY_FIELDS or not all(
        _is_count(value) for value in summary.values()
    ):
        problems.append("drill_report_invalid")
    return problems


def _row_valid(row: Mapping[str, Any]) -> bool:
    return (
        set(row) == {"check", "observed", "owner", "phase", "state"}
        and row["state"] in STEP_STATES
        and _is_code(row["check"])
        and (row["state"] == STEP_NOT_AVAILABLE) == (row["owner"] is not None)
        and (row["owner"] is None or bool(_ISSUE_RE.fullmatch(str(row["owner"]))))
        and isinstance(row["observed"], Mapping)
        and all(
            name in DRILL_OBJECTIVES and _is_count(value)
            for name, value in row["observed"].items()
        )
    )


def _is_code(value: Any) -> bool:
    return isinstance(value, str) and _CODE_RE.fullmatch(value) is not None


def _is_count(value: Any) -> bool:
    return type(value) is int and value >= 0


# --- command line ------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the recovery-drill command-line parser."""

    parser = argparse.ArgumentParser(
        description="Run the reference v4.0 service recovery drill offline."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--candidate", help="Candidate commit; default HEAD.")
    parser.add_argument("--generated-at", help="RFC 3339 UTC; default now.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _write_report(report: Mapping[str, Any], path: Path, *, overwrite: bool) -> None:
    try:
        _write_release_json(report, path, overwrite=overwrite)
    except JourneyReleaseError:
        code = "output_exists" if path.exists() else "output_unwritable"
        raise RecoveryDrillError(code) from None
    except OSError:
        raise RecoveryDrillError("output_unwritable") from None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the drill and write its report.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` for a ``complete`` report, ``1`` for ``incomplete`` or
        ``failed``, ``2`` when no report could be produced.
    """

    args = build_arg_parser().parse_args(argv)
    try:
        candidate = args.candidate or _git_output(
            args.repo_root.resolve(), "rev-parse", "HEAD"
        )
        generated_at = args.generated_at or datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat().replace("+00:00", "Z")
        report = run_recovery_drill(
            candidate_commit=candidate, generated_at=generated_at
        )
        _write_report(report, args.output, overwrite=args.overwrite)
    except RecoveryDrillError as exc:
        print(f"service-recovery-drill: {exc.code}", file=sys.stderr)
        return 2
    except (JourneyReleaseError, OSError, subprocess.SubprocessError):
        print("service-recovery-drill: drill_unavailable", file=sys.stderr)
        return 2
    if args.json:
        import json

        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"service-recovery-drill: {report['state']}")
        for scenario in report["scenarios"]:
            for row in scenario["steps"]:
                owner = f" ({row['owner']})" if row["owner"] else ""
                print(
                    f"  [{row['state']}] {scenario['category']}.{row['phase']}: "
                    f"{row['check']}{owner}"
                )
    return 0 if report["state"] == DRILL_COMPLETE else 1


__all__ = [
    "CORRUPTED_STATE",
    "DEFAULT_DRILL_STEPS",
    "DRILL_CATEGORIES",
    "DRILL_CHECK_CODES",
    "DRILL_COMPLETE",
    "DRILL_ERROR_CODES",
    "DRILL_FAILED",
    "DRILL_INCOMPLETE",
    "DRILL_OBJECTIVES",
    "DRILL_PHASES",
    "DRILL_REPORT_CODES",
    "NETWORK",
    "PROCESS",
    "RECOVERY_DRILL_PROFILE",
    "RECOVERY_DRILL_RELEASE",
    "RECOVERY_DRILL_SCHEMA",
    "STEP_FAILED",
    "STEP_NOT_AVAILABLE",
    "STEP_PASSED",
    "STORAGE",
    "UNAVAILABLE_DRILL_STEPS",
    "DrillContext",
    "DrillStep",
    "DrillStepOutcome",
    "RecoveryDrillError",
    "build_arg_parser",
    "main",
    "not_available",
    "recovery_drill_digest",
    "run_recovery_drill",
    "validate_recovery_drill_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
