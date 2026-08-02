"""Resume planning and straggler classification for distributed batch runs.

This module answers two questions about an interrupted distributed
de-identification run, and nothing else:

* **What still has to run?** :func:`resume_plan` reads a
  :class:`~openmed.processing.run_manifest.BatchRunManifest`, re-verifies the
  recorded output digests of completed shards, and returns the shards that a
  resumed run must recompute, each with the reason it was selected.
* **Which in-flight shards are lagging?** :func:`classify_stragglers` compares
  the elapsed time of running shards against the observed per-document rate of
  the shards that already finished, and reports the outliers.

It **classifies only**. It never executes a shard, never kills or reassigns a
worker, and never increments :attr:`ShardRecord.attempts`; executors own all of
that. ``attempts`` is read for validation alone, so a shard that has burned its
attempt budget is reported as exhausted rather than being queued again.

Two independent mechanisms decide whether a ``RUNNING`` shard is stale:

* **Orphan detection is clock-free.** A running shard whose ``worker_id`` is
  absent from ``live_workers`` lost its worker and is queued for recomputation.
  Passing ``live_workers=None`` means "no worker survives", the correct reading
  when the whole run died and is being resumed from its manifest.
* **Straggler detection uses an injected clock.** It never fires below
  ``min_completed_baseline`` finished shards, because a median computed from
  one or two samples is noise rather than a baseline.

Timestamps are epoch seconds, matching :func:`time.time` as used by
:func:`~openmed.processing.run_manifest.build_run_manifest`; the ``clock``
argument exists so callers and tests can supply that reading deterministically.
"""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from openmed.processing.distributed import ShardPlan
from openmed.processing.run_manifest import (
    BatchRunManifest,
    ShardOutputDigestMismatchError,
    ShardRecord,
    ShardStatus,
    validate_shard_outputs,
)

RESUME_SCHEMA_VERSION = 1

# Deliberately *not* named ``HASH_NAMESPACE``: ``openmed.processing.distributed``
# exports a constant of that name with a different value, and the two are easy
# to confuse when both modules are imported together. Callers should never need
# this directly — use :func:`worker_ref` rather than rebuilding the digest.
RESUME_HASH_NAMESPACE = "openmed.processing.resume"

DEFAULT_STRAGGLER_MULTIPLIER = 1.5
DEFAULT_MIN_COMPLETED_BASELINE = 3

Clock = Callable[[], float]


class ResumeError(ValueError):
    """Base error raised when a run cannot be resumed safely."""


class PlanFingerprintMismatchError(ResumeError):
    """Raised when a manifest does not describe the expected sharding."""


class RunIdMismatchError(ResumeError):
    """Raised when a manifest belongs to a different run than requested."""


class ShardIdentityMismatchError(ResumeError):
    """Raised when two attempt records do not describe the same shard."""


def worker_ref(worker_id: str | None) -> str | None:
    """Return a stable, non-reversible reference for an operator's worker id.

    Worker identifiers are free text. The run manifest bounds their length and
    rejects control characters, but nothing stops an operator naming workers
    after the records they process, so a raw id is not safe to publish in a
    report. This maps one to an opaque token that preserves the only property a
    report needs: the same worker yields the same token, and different workers
    yield different ones, so a reader can still ask which shards a given worker
    is running.

    Every layer that publishes a worker reference must call this rather than
    rebuilding the digest, so that a worker seen on the shard path and on the
    straggler path of one report carries the same token. Two hand-rolled
    constructions that each look correct will not agree, and the mismatch is
    silent: the reader simply finds no correlation.

    ``None`` maps to ``None``, because an absent worker is not a worker whose
    identity needs protecting.
    """

    if worker_id is None:
        return None
    digest = hashlib.sha256(
        f"{RESUME_HASH_NAMESPACE}:worker:{worker_id}".encode("utf-8")
    )
    return f"w-{digest.hexdigest()[:12]}"


class ResumeReason(str, Enum):
    """Why a shard was selected for recomputation by a resumed run."""

    PENDING = "pending"
    FAILED = "failed"
    ORPHANED = "orphaned"
    OUTPUT_MISSING = "output_missing"
    OUTPUT_MISMATCH = "output_mismatch"


@dataclass(frozen=True)
class ShardResumeDecision:
    """One shard queued for recomputation, with the reason it was queued."""

    shard_id: int
    reason: ResumeReason
    attempts: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", ResumeReason(self.reason))
        if self.shard_id < 0:
            raise ResumeError("shard_id must be non-negative")
        if self.attempts < 0:
            raise ResumeError("attempts must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, PHI-free representation."""

        return {
            "shard_id": self.shard_id,
            "reason": self.reason.value,
            "attempts": self.attempts,
        }


@dataclass(frozen=True)
class StragglerCandidate:
    """A running shard whose per-document rate lags the completed baseline.

    The candidate is advisory. Executors decide whether to launch a speculative
    duplicate, and reconcile the outcome through
    :func:`reconcile_shard_attempts`.
    """

    shard_id: int
    worker_id: str | None
    attempts: int
    elapsed_seconds: float
    per_document_seconds: float
    baseline_per_document_seconds: float
    threshold_per_document_seconds: float
    elapsed_floor_seconds: float = 0.0

    @property
    def slowdown_ratio(self) -> float:
        """How many times the baseline per-document rate this shard is taking."""

        return self.per_document_seconds / self.baseline_per_document_seconds

    @property
    def worker_ref(self) -> str | None:
        """Return this candidate's publishable worker reference.

        Delegates to the module-level :func:`worker_ref` so that every layer
        emitting a worker reference agrees by construction rather than by two
        literals happening to match.
        """

        return worker_ref(self.worker_id)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation carrying no operator free text.

        ``worker_id`` is deliberately absent; :attr:`worker_ref` stands in for
        it. The raw id remains on the object for executors that must act on the
        worker, but it is never serialised.
        """

        return {
            "shard_id": self.shard_id,
            "worker_ref": self.worker_ref,
            "attempts": self.attempts,
            "elapsed_seconds": self.elapsed_seconds,
            "per_document_seconds": self.per_document_seconds,
            "baseline_per_document_seconds": self.baseline_per_document_seconds,
            "threshold_per_document_seconds": self.threshold_per_document_seconds,
            "elapsed_floor_seconds": self.elapsed_floor_seconds,
        }


@dataclass(frozen=True)
class ResumePlan:
    """Description of what a resumed run must do next.

    :meth:`to_dict` is the publishable form; see it for exactly which fields
    are machine-derived and which single field is operator-supplied.
    """

    run_id: str
    plan_fingerprint: str
    decisions: tuple[ShardResumeDecision, ...]
    completed: tuple[int, ...]
    in_flight: tuple[int, ...]
    exhausted: tuple[int, ...]
    unmeasurable: tuple[int, ...]
    stragglers: tuple[StragglerCandidate, ...]
    straggler_baseline_seconds: float | None
    fingerprint: str
    schema_version: int = RESUME_SCHEMA_VERSION

    @property
    def shard_ids(self) -> tuple[int, ...]:
        """Shard ids the resumed run has to recompute, in ascending order."""

        return tuple(decision.shard_id for decision in self.decisions)

    @property
    def is_complete(self) -> bool:
        """Whether every shard finished with a valid output.

        Exhausted shards make a run terminal, not complete: they produced no
        usable output and will never be retried, so a driver looping on
        ``while not plan.is_complete`` must exit reporting failure rather than
        success. Check :attr:`is_exhausted` to tell the two apart.
        """

        return not self.decisions and not self.in_flight and not self.exhausted

    @property
    def is_exhausted(self) -> bool:
        """Whether the run stopped with shards that spent their attempt budget."""

        return bool(self.exhausted)

    @property
    def straggler_detection_enabled(self) -> bool:
        """Whether a usable baseline existed, so an empty result means "none".

        When this is ``False`` no running shard can be reported, because there
        were too few finished shards to form a baseline or the observed rates
        were unusable. An empty :attr:`stragglers` then means "not measured"
        rather than "nothing lagging".
        """

        return self.straggler_baseline_seconds is not None

    def reasons(self) -> dict[int, ResumeReason]:
        """Map each queued shard id to the reason it was queued."""

        return {decision.shard_id: decision.reason for decision in self.decisions}

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe recovery report for this plan.

        Every value is machine-derived — a shard id, a counter, a timing, a
        digest or a hashed reference — with one exception: ``run_id``. That is
        operator-supplied and written verbatim, exactly as the run manifest
        writes it, so it must not embed a document identifier or any other
        record-derived value. It is reproduced rather than hashed because it is
        the key that ties a report to its manifest, and the manifest already
        publishes it in the clear.

        Worker identifiers get no such exemption: they are free text that
        nothing validates and they are not a join key, so they appear only as
        the hashed :attr:`StragglerCandidate.worker_ref`. No document text,
        document identifier or exception message can reach this payload.
        """

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "plan_fingerprint": self.plan_fingerprint,
            "fingerprint": self.fingerprint,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "completed": list(self.completed),
            "in_flight": list(self.in_flight),
            "exhausted": list(self.exhausted),
            "unmeasurable": list(self.unmeasurable),
            "straggler_detection_enabled": self.straggler_detection_enabled,
            "straggler_baseline_seconds": self.straggler_baseline_seconds,
            "stragglers": [candidate.to_dict() for candidate in self.stragglers],
        }


def resume_plan(
    manifest: BatchRunManifest,
    *,
    root: str | Path | None = None,
    run_id: str | None = None,
    expected_plan: ShardPlan | None = None,
    live_workers: Iterable[str] | None = None,
    max_attempts: int | None = None,
    clock: Clock | None = None,
    straggler_multiplier: float = DEFAULT_STRAGGLER_MULTIPLIER,
    min_completed_baseline: int = DEFAULT_MIN_COMPLETED_BASELINE,
    min_elapsed_seconds: float | None = None,
) -> ResumePlan:
    """Return what a resumed run must recompute, and which shards lag.

    A shard is queued when it is pending, when its last attempt failed, when it
    was running on a worker that is no longer live, or when it is marked
    completed but its recorded output has gone missing or no longer matches its
    digest. Completed shards with intact outputs are never recomputed, which is
    what makes resuming cheap.

    ``run_id`` and ``expected_plan`` are guards: resuming against a manifest
    from another run, or against a manifest whose sharding no longer matches the
    corpus, raises rather than silently recomputing the wrong partition.

    ``live_workers`` names the workers still known to be alive. ``None`` treats
    every running shard as orphaned, the correct assumption when the run is
    being resumed after the whole job died.

    ``root`` is the directory the manifest's relative output paths resolve
    against, and should almost always be supplied. Omitting it resolves them
    against the current working directory, where a finished run's outputs will
    not be found, and every completed shard is then requeued as
    ``OUTPUT_MISSING`` — a silent full-corpus recomputation.

    ``max_attempts`` bounds recomputation: a shard that already used its budget
    is reported in :attr:`ResumePlan.exhausted` instead of being queued again.
    Attempt counters are only read here; executors own incrementing them. Note
    that exhausted shards make the plan terminal but *not* complete; see
    :attr:`ResumePlan.is_complete`.

    Running shards that carry no start time, or that hold no documents, cannot
    be timed and so can never be reported as stragglers. They are listed in
    :attr:`ResumePlan.unmeasurable` so a stuck one is visible rather than
    sitting in ``in_flight`` forever.
    """

    if run_id is not None and run_id != manifest.run_id:
        raise RunIdMismatchError(
            f"manifest belongs to run {manifest.run_id!r}, not {run_id!r}"
        )
    if expected_plan is not None and expected_plan.fingerprint != (
        manifest.plan_fingerprint
    ):
        raise PlanFingerprintMismatchError(
            "manifest plan fingerprint does not match the expected shard plan; "
            "the corpus or shard count changed since the run started"
        )
    if max_attempts is not None and max_attempts < 1:
        raise ResumeError("max_attempts must be greater than zero")

    validation = validate_shard_outputs(manifest, root=root)
    missing = set(validation.missing)
    mismatched = set(validation.mismatched)
    live = None if live_workers is None else frozenset(live_workers)

    decisions: list[ShardResumeDecision] = []
    completed: list[int] = []
    in_flight: list[int] = []
    exhausted: list[int] = []

    for record in manifest.shards:
        reason = _resume_reason(
            record, missing=missing, mismatched=mismatched, live=live
        )
        if reason is None:
            if record.status is ShardStatus.COMPLETED:
                completed.append(record.shard_id)
            else:
                in_flight.append(record.shard_id)
            continue
        if max_attempts is not None and record.attempts >= max_attempts:
            exhausted.append(record.shard_id)
            continue
        decisions.append(
            ShardResumeDecision(
                shard_id=record.shard_id,
                reason=reason,
                attempts=record.attempts,
            )
        )

    unmeasurable = [
        record.shard_id
        for record in manifest.shards
        if record.shard_id in set(in_flight)
        and record.status is ShardStatus.RUNNING
        and (record.started_at is None or record.document_count < 1)
    ]
    stragglers = classify_stragglers(
        manifest,
        clock=clock,
        multiplier=straggler_multiplier,
        min_completed=min_completed_baseline,
        min_elapsed_seconds=min_elapsed_seconds,
        shard_ids=in_flight,
    )
    ordered_decisions = tuple(sorted(decisions, key=lambda item: item.shard_id))
    return ResumePlan(
        run_id=manifest.run_id,
        plan_fingerprint=manifest.plan_fingerprint,
        decisions=ordered_decisions,
        completed=tuple(completed),
        in_flight=tuple(in_flight),
        exhausted=tuple(exhausted),
        unmeasurable=tuple(unmeasurable),
        stragglers=stragglers,
        straggler_baseline_seconds=_baseline_per_document_seconds(
            manifest, min_completed=min_completed_baseline
        ),
        fingerprint=_fingerprint_resume(
            manifest.run_id,
            manifest.plan_fingerprint,
            ordered_decisions,
        ),
    )


def classify_stragglers(
    manifest: BatchRunManifest,
    *,
    clock: Clock | None = None,
    multiplier: float = DEFAULT_STRAGGLER_MULTIPLIER,
    min_completed: int = DEFAULT_MIN_COMPLETED_BASELINE,
    min_elapsed_seconds: float | None = None,
    shard_ids: Iterable[int] | None = None,
) -> tuple[StragglerCandidate, ...]:
    """Return the running shards lagging the completed per-document baseline.

    Shards hold unequal numbers of documents, so raw durations are not
    comparable. Both the baseline and each candidate are therefore expressed as
    seconds per document: the baseline is the *median* rate of the shards that
    finished, and a running shard is a straggler once its rate exceeds
    ``multiplier`` times that median. The median is deliberate. One hung shard
    that has already completed at an enormous rate would drag a mean far enough
    to hide every subsequent straggler, whereas it moves a median hardly at
    all.

    Per-document rates assume shard duration is roughly proportional to
    document count. Real shards also pay a fixed cost per attempt, such as
    loading a model, that a small shard amortises over few documents; judged on
    rate alone a healthy two-document shard can look slower than a healthy
    hundred-document one. A shard is therefore never reported before it has run
    at least ``min_elapsed_seconds``, which defaults to the median *total*
    duration of the finished shards, so no shard is speculated on until it has
    had as long as a typical shard needed end to end.

    The result is empty, never a guess, when fewer than ``min_completed`` shards
    have finished or when the observed rates give no usable baseline. Those two
    cases are indistinguishable here; :func:`resume_plan` records the baseline
    it used so callers can tell "nothing lagging" from "not measured".
    ``clock`` supplies the current epoch reading and defaults to
    :func:`time.time`.
    """

    if multiplier <= 0:
        raise ResumeError("multiplier must be greater than zero")
    if min_completed < 1:
        raise ResumeError("min_completed must be greater than zero")
    if min_elapsed_seconds is not None and (
        not math.isfinite(min_elapsed_seconds) or min_elapsed_seconds < 0
    ):
        raise ResumeError("min_elapsed_seconds must be a non-negative number")

    now = clock or time.time
    baseline = _baseline_per_document_seconds(manifest, min_completed=min_completed)
    if baseline is None:
        return ()

    floor = _elapsed_floor_seconds(manifest, override=min_elapsed_seconds)
    threshold = baseline * multiplier
    selected = None if shard_ids is None else frozenset(shard_ids)
    current = float(now())
    candidates: list[StragglerCandidate] = []

    for record in manifest.shards:
        if record.status is not ShardStatus.RUNNING or record.started_at is None:
            continue
        if record.document_count < 1:
            continue
        if selected is not None and record.shard_id not in selected:
            continue
        elapsed = current - record.started_at
        if elapsed <= 0 or not math.isfinite(elapsed):
            continue
        if elapsed < floor:
            continue
        per_document = elapsed / record.document_count
        if per_document <= threshold:
            continue
        candidates.append(
            StragglerCandidate(
                shard_id=record.shard_id,
                worker_id=record.worker_id,
                attempts=record.attempts,
                elapsed_seconds=elapsed,
                per_document_seconds=per_document,
                baseline_per_document_seconds=baseline,
                threshold_per_document_seconds=threshold,
                elapsed_floor_seconds=floor,
            )
        )

    return tuple(sorted(candidates, key=lambda item: item.shard_id))


def prepare_resume(
    manifest: BatchRunManifest,
    plan: ResumePlan,
    *,
    updated_at: float | None = None,
) -> BatchRunManifest:
    """Return ``manifest`` with every queued shard reset to a clean pending state.

    This is the distributed analogue of truncating the uncommitted tail of a
    single-process checkpoint. Resetting the status is what performs that
    truncation: a queued shard is no longer ``COMPLETED``, so nothing downstream
    may treat its output as final. The per-attempt traces that would otherwise
    describe a run that did not happen — start and completion times, the worker
    that ran it, the error it raised — are cleared with it.

    What is deliberately *not* cleared is the recorded ``output_path``,
    ``output_digest`` and ``output_bytes`` of an earlier successful attempt.
    Executors use the previously recorded digest as the expected value when the
    shard runs again, which is how a shard that silently produces different
    bytes on a rerun is caught. Clearing it here would hand the executor no
    expectation to compare against and quietly disable that check for exactly
    the shards a resume is about to recompute. Attempt counters survive too,
    because they are the audit trail of how much work a shard has consumed.

    ``plan`` must have been computed from ``manifest``. Shard ids are always
    ``0`` to ``shard_count - 1``, so a plan from another run would always
    overlap and would reset shards that run still depends on; the run id and
    plan fingerprint are checked here for that reason, exactly as
    :func:`resume_plan` checks them before deciding anything.
    """

    if plan.run_id != manifest.run_id:
        raise RunIdMismatchError(
            f"resume plan belongs to run {plan.run_id!r}, not {manifest.run_id!r}"
        )
    if plan.plan_fingerprint != manifest.plan_fingerprint:
        raise PlanFingerprintMismatchError(
            "resume plan was computed for a different shard plan than this "
            "manifest describes"
        )

    queued = set(plan.shard_ids)
    if not queued:
        return manifest

    updated = manifest
    for shard_id in sorted(queued):
        record = manifest.shard(shard_id)
        # ``replace`` rather than a fresh record: it preserves the shard's
        # identity, its attempt count and its recorded output by default, so a
        # field added to ShardRecord later is carried across a resume instead of
        # being silently dropped here.
        updated = updated.with_shard(
            replace(
                record,
                status=ShardStatus.PENDING,
                started_at=None,
                completed_at=None,
                worker_id=None,
                error_type=None,
            ),
            updated_at=updated_at,
        )
    return updated


def reconcile_shard_attempts(
    first: ShardRecord,
    second: ShardRecord,
) -> ShardRecord:
    """Return the single record that survives two attempts at the same shard.

    Speculative re-execution can finish a shard twice. Because shard writes are
    content addressed and idempotent, two completed attempts must agree on their
    output digest; disagreement means one attempt is not reproducible and is
    raised rather than papered over.

    Among agreeing attempts the winner is the one that used fewer attempts,
    then the one with a worker id, then the lexicographically smaller worker
    id, then the earlier start, the earlier completion, the smaller output path
    and the smaller output size. That ordering is total over every field the
    record carries, so the outcome does not depend on the order in which the
    two results arrive. Two attempts that tie on all of them are
    indistinguishable in the manifest, and the first argument is returned.
    """

    if first.shard_id != second.shard_id:
        raise ShardIdentityMismatchError(
            "cannot reconcile attempts from different shards"
        )
    if first.fingerprint != second.fingerprint:
        raise ShardIdentityMismatchError(
            f"shard {first.shard_id} attempts disagree on the shard fingerprint"
        )

    first_done = first.status is ShardStatus.COMPLETED
    second_done = second.status is ShardStatus.COMPLETED
    if first_done and not second_done:
        return first
    if second_done and not first_done:
        return second
    if not first_done and not second_done:
        return first if _attempt_rank(first) <= _attempt_rank(second) else second

    if first.output_digest != second.output_digest:
        raise ShardOutputDigestMismatchError(
            f"shard {first.shard_id} attempts produced different output digests"
        )
    return first if _attempt_rank(first) <= _attempt_rank(second) else second


def _resume_reason(
    record: ShardRecord,
    *,
    missing: set[int],
    mismatched: set[int],
    live: frozenset[str] | None,
) -> ResumeReason | None:
    if record.status is ShardStatus.COMPLETED:
        if record.shard_id in missing:
            return ResumeReason.OUTPUT_MISSING
        if record.shard_id in mismatched:
            return ResumeReason.OUTPUT_MISMATCH
        return None
    if record.status is ShardStatus.FAILED:
        return ResumeReason.FAILED
    if record.status is ShardStatus.RUNNING:
        if live is not None and record.worker_id is not None:
            if record.worker_id in live:
                return None
        return ResumeReason.ORPHANED
    return ResumeReason.PENDING


def _completed_durations(manifest: BatchRunManifest) -> list[tuple[float, int]]:
    """Return finite ``(duration, document_count)`` pairs for finished shards.

    Non-finite timings are dropped rather than propagated. A single ``NaN``
    would otherwise poison every comparison that follows, because comparisons
    against ``NaN`` are all false, and silently disable straggler detection for
    the whole run.
    """

    return [
        (record.duration_seconds, record.document_count)
        for record in manifest.completed_shards()
        if record.duration_seconds is not None
        and record.document_count > 0
        and math.isfinite(record.duration_seconds)
        and record.duration_seconds >= 0
    ]


def _baseline_per_document_seconds(
    manifest: BatchRunManifest,
    *,
    min_completed: int,
) -> float | None:
    # ``_completed_durations`` already guarantees a finite, non-negative
    # duration and a document count of at least one, so every rate here is
    # finite by construction.
    rates = [
        duration / document_count
        for duration, document_count in _completed_durations(manifest)
    ]
    if len(rates) < min_completed:
        return None
    baseline = statistics.median(rates)
    if not math.isfinite(baseline) or baseline <= 0:
        return None
    return baseline


def _elapsed_floor_seconds(
    manifest: BatchRunManifest,
    *,
    override: float | None,
) -> float:
    if override is not None:
        return float(override)
    durations = [duration for duration, _ in _completed_durations(manifest)]
    if not durations:
        return 0.0
    floor = statistics.median(durations)
    return floor if math.isfinite(floor) and floor > 0 else 0.0


def _attempt_rank(record: ShardRecord) -> tuple[Any, ...]:
    """Return the total order used to pick between two reconcilable attempts."""

    return (
        record.attempts,
        record.worker_id is None,
        record.worker_id or "",
        record.started_at is None,
        record.started_at if record.started_at is not None else 0.0,
        record.completed_at is None,
        record.completed_at if record.completed_at is not None else 0.0,
        record.output_path or "",
        record.output_bytes is None,
        record.output_bytes if record.output_bytes is not None else 0,
    )


def _fingerprint_resume(
    run_id: str,
    plan_fingerprint: str,
    decisions: tuple[ShardResumeDecision, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"{RESUME_HASH_NAMESPACE}:v{RESUME_SCHEMA_VERSION}".encode("utf-8"))
    for value in (run_id, plan_fingerprint):
        encoded = value.encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
        digest.update(b"\0")
    for decision in decisions:
        digest.update(str(decision.shard_id).encode("ascii"))
        digest.update(b":")
        digest.update(decision.reason.value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = [
    "DEFAULT_MIN_COMPLETED_BASELINE",
    "DEFAULT_STRAGGLER_MULTIPLIER",
    "RESUME_SCHEMA_VERSION",
    "PlanFingerprintMismatchError",
    "ResumeError",
    "ResumePlan",
    "ResumeReason",
    "RunIdMismatchError",
    "ShardIdentityMismatchError",
    "ShardResumeDecision",
    "StragglerCandidate",
    "classify_stragglers",
    "prepare_resume",
    "reconcile_shard_attempts",
    "resume_plan",
    "worker_ref",
]
