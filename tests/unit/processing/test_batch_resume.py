"""Tests for resumable distributed batch runs and straggler classification."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

import pytest

import openmed.processing
from openmed.processing import (
    PlanFingerprintMismatchError,
    ResumeError,
    ResumeReason,
    RunIdMismatchError,
    RunManifestError,
    ShardIdentityMismatchError,
    ShardOutputDigestMismatchError,
    ShardRecord,
    ShardStatus,
    StragglerCandidate,
    build_run_manifest,
    classify_stragglers,
    plan_document_shards,
    prepare_resume,
    reconcile_shard_attempts,
    resume_plan,
    shard_output_digest,
    validate_shard_outputs,
    worker_ref,
)
from openmed.processing.distributed import ShardPlan

SHARD_COUNT = 10
DOCUMENT_COUNT = 40
RUN_ID = "run-resume-0001"
# A worker id shaped like a real misconfiguration: an operator naming workers
# after the record being processed drags a document id and an MRN with it.
LEAKY_WORKER = "worker-for-doc-0007-MRN-123456"


def _documents() -> list[dict[str, str]]:
    """Build synthetic documents algorithmically; no real record is used."""

    return [
        {"id": f"doc-{index:04d}", "text": f"synthetic record {index}"}
        for index in range(DOCUMENT_COUNT)
    ]


def _plan() -> ShardPlan:
    return plan_document_shards(_documents(), shard_count=SHARD_COUNT)


def _redact(document_id: str) -> str:
    """Deterministic stand-in for de-identification: a pure function of the id."""

    return hashlib.sha256(f"redacted::{document_id}".encode("utf-8")).hexdigest()


def _run_shard(
    manifest,
    plan: ShardPlan,
    shard_id: int,
    *,
    root: Path,
    worker_id: str,
    started_at: float,
    completed_at: float,
):
    """Minimal stand-in executor: writes a shard output and records it.

    Output content depends only on the shard's document ids, so an interrupted
    run and an uninterrupted run are byte-comparable.
    """

    shard = plan.shards[shard_id]
    output_path = root / f"shard-{shard_id:05d}.txt"
    output_path.write_text(
        "".join(f"{doc_id} {_redact(doc_id)}\n" for doc_id in shard.document_ids),
        encoding="utf-8",
    )
    return manifest.with_shard(
        ShardRecord(
            shard_id=shard_id,
            fingerprint=shard.fingerprint,
            document_count=shard.document_count,
            status=ShardStatus.COMPLETED,
            attempts=manifest.shard(shard_id).attempts + 1,
            started_at=started_at,
            completed_at=completed_at,
            output_path=output_path.name,
            output_digest=shard_output_digest(output_path),
            output_bytes=output_path.stat().st_size,
            worker_id=worker_id,
        ),
        updated_at=completed_at,
    )


def _document_digests(root: Path) -> dict[str, str]:
    """Collapse every shard output into one per-document digest map."""

    digests: dict[str, str] = {}
    for output in sorted(root.glob("shard-*.txt")):
        for line in output.read_text(encoding="utf-8").splitlines():
            document_id, digest = line.split(" ")
            assert document_id not in digests
            digests[document_id] = digest
    return digests


def _completed_manifest(plan: ShardPlan, root: Path, shard_ids):
    """Run ``shard_ids`` to completion at deterministic synthetic timings."""

    manifest = build_run_manifest(plan, run_id=RUN_ID, created_at=1_000.0)
    for offset, shard_id in enumerate(shard_ids):
        manifest = _run_shard(
            manifest,
            plan,
            shard_id,
            root=root,
            worker_id=f"worker-{shard_id % 4}",
            started_at=1_000.0 + offset,
            completed_at=1_000.5 + offset,
        )
    return manifest


def _mark_running(manifest, plan: ShardPlan, shard_id: int, *, worker_id, started_at):
    shard = plan.shards[shard_id]
    return manifest.with_shard(
        ShardRecord(
            shard_id=shard_id,
            fingerprint=shard.fingerprint,
            document_count=shard.document_count,
            status=ShardStatus.RUNNING,
            attempts=1,
            started_at=started_at,
            worker_id=worker_id,
        ),
        updated_at=started_at,
    )


def test_resume_recomputes_only_incomplete_shards(tmp_path: Path) -> None:
    plan = _plan()
    manifest = _completed_manifest(plan, tmp_path, range(7))

    resumed = resume_plan(manifest, root=tmp_path)

    assert resumed.shard_ids == (7, 8, 9)
    assert resumed.completed == (0, 1, 2, 3, 4, 5, 6)
    assert set(resumed.reasons().values()) == {ResumeReason.PENDING}
    assert not resumed.is_complete


def test_resume_requeues_missing_and_mismatched_outputs(tmp_path: Path) -> None:
    plan = _plan()
    manifest = _completed_manifest(plan, tmp_path, range(SHARD_COUNT))
    assert resume_plan(manifest, root=tmp_path).is_complete

    (tmp_path / "shard-00002.txt").unlink()
    (tmp_path / "shard-00005.txt").write_text("tampered\n", encoding="utf-8")

    resumed = resume_plan(manifest, root=tmp_path)

    assert resumed.shard_ids == (2, 5)
    assert resumed.reasons() == {
        2: ResumeReason.OUTPUT_MISSING,
        5: ResumeReason.OUTPUT_MISMATCH,
    }


def test_resume_queues_failed_shards(tmp_path: Path) -> None:
    plan = _plan()
    manifest = _completed_manifest(plan, tmp_path, range(SHARD_COUNT - 1))
    manifest = manifest.with_shard(
        ShardRecord(
            shard_id=9,
            fingerprint=plan.shards[9].fingerprint,
            document_count=plan.shards[9].document_count,
            status=ShardStatus.FAILED,
            attempts=2,
            error_type="TimeoutError",
        ),
        updated_at=2_000.0,
    )

    resumed = resume_plan(manifest, root=tmp_path)

    assert resumed.reasons() == {9: ResumeReason.FAILED}
    assert resumed.decisions[0].attempts == 2


def test_running_shard_on_a_live_worker_is_not_requeued(tmp_path: Path) -> None:
    plan = _plan()
    manifest = _completed_manifest(plan, tmp_path, range(SHARD_COUNT - 1))
    manifest = _mark_running(
        manifest, plan, 9, worker_id="worker-live", started_at=1_010.0
    )

    live = resume_plan(manifest, root=tmp_path, live_workers={"worker-live"})
    assert live.shard_ids == ()
    assert live.in_flight == (9,)

    dead = resume_plan(manifest, root=tmp_path, live_workers={"worker-other"})
    assert dead.reasons() == {9: ResumeReason.ORPHANED}

    crashed = resume_plan(manifest, root=tmp_path, live_workers=None)
    assert crashed.reasons() == {9: ResumeReason.ORPHANED}


def test_interrupted_run_matches_uninterrupted_by_document_digest(
    tmp_path: Path,
) -> None:
    """Killing 30 percent of workers mid-run must not change any output."""

    plan = _plan()

    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    reference = _completed_manifest(plan, reference_root, range(SHARD_COUNT))
    expected_digests = _document_digests(reference_root)
    assert len(expected_digests) == DOCUMENT_COUNT
    assert validate_shard_outputs(reference, root=reference_root).all_valid

    interrupted_root = tmp_path / "interrupted"
    interrupted_root.mkdir()
    survivors = [0, 1, 2, 3, 4, 5, 6]
    killed = [7, 8, 9]
    manifest = _completed_manifest(plan, interrupted_root, survivors)
    for shard_id in killed:
        manifest = _mark_running(
            manifest,
            plan,
            shard_id,
            worker_id=f"doomed-{shard_id}",
            started_at=1_020.0,
        )

    resumed = resume_plan(
        manifest,
        root=interrupted_root,
        run_id=RUN_ID,
        expected_plan=plan,
        live_workers={"worker-0", "worker-1", "worker-2", "worker-3"},
    )
    assert resumed.shard_ids == tuple(killed)
    assert set(resumed.reasons().values()) == {ResumeReason.ORPHANED}

    manifest = prepare_resume(manifest, resumed, updated_at=1_030.0)
    for shard_id in resumed.shard_ids:
        assert manifest.shard(shard_id).status is ShardStatus.PENDING
        assert manifest.shard(shard_id).output_digest is None
        assert manifest.shard(shard_id).attempts == 1

    for offset, shard_id in enumerate(resumed.shard_ids):
        manifest = _run_shard(
            manifest,
            plan,
            shard_id,
            root=interrupted_root,
            worker_id="worker-recovered",
            started_at=1_040.0 + offset,
            completed_at=1_040.5 + offset,
        )

    assert _document_digests(interrupted_root) == expected_digests
    assert validate_shard_outputs(manifest, root=interrupted_root).all_valid
    assert resume_plan(manifest, root=interrupted_root).is_complete
    assert manifest.shard(9).attempts == 2


def test_manifest_validates_after_recovery(tmp_path: Path) -> None:
    plan = _plan()
    manifest = _completed_manifest(plan, tmp_path, range(SHARD_COUNT))
    (tmp_path / "shard-00003.txt").write_text("corrupted\n", encoding="utf-8")

    assert not validate_shard_outputs(manifest, root=tmp_path).all_valid
    resumed = resume_plan(manifest, root=tmp_path)
    manifest = prepare_resume(manifest, resumed, updated_at=1_100.0)
    manifest = _run_shard(
        manifest,
        plan,
        3,
        root=tmp_path,
        worker_id="worker-recovered",
        started_at=1_110.0,
        completed_at=1_110.5,
    )

    validation = validate_shard_outputs(manifest, root=tmp_path)
    assert validation.all_valid
    assert len(validation.valid) == SHARD_COUNT
    assert manifest.pending_shards() == ()


def test_resume_rejects_a_different_run_or_shard_plan(tmp_path: Path) -> None:
    plan = _plan()
    manifest = _completed_manifest(plan, tmp_path, range(3))

    with pytest.raises(RunIdMismatchError):
        resume_plan(manifest, root=tmp_path, run_id="run-other")

    other_plan = plan_document_shards(_documents(), shard_count=SHARD_COUNT - 1)
    with pytest.raises(PlanFingerprintMismatchError):
        resume_plan(manifest, root=tmp_path, expected_plan=other_plan)

    resume_plan(manifest, root=tmp_path, run_id=RUN_ID, expected_plan=plan)


def test_exhausted_attempt_budget_is_reported_not_requeued(tmp_path: Path) -> None:
    plan = _plan()
    manifest = _completed_manifest(plan, tmp_path, range(SHARD_COUNT - 2))
    for shard_id, attempts in ((8, 3), (9, 1)):
        manifest = manifest.with_shard(
            ShardRecord(
                shard_id=shard_id,
                fingerprint=plan.shards[shard_id].fingerprint,
                document_count=plan.shards[shard_id].document_count,
                status=ShardStatus.FAILED,
                attempts=attempts,
                error_type="RuntimeError",
            ),
            updated_at=2_000.0,
        )

    resumed = resume_plan(manifest, root=tmp_path, max_attempts=3)

    assert resumed.shard_ids == (9,)
    assert resumed.exhausted == (8,)
    with pytest.raises(ResumeError):
        resume_plan(manifest, root=tmp_path, max_attempts=0)


def _straggler_manifest(plan: ShardPlan, *, completed: int, running_started_at: float):
    manifest = build_run_manifest(plan, run_id=RUN_ID, created_at=0.0)
    for shard_id in range(completed):
        shard = plan.shards[shard_id]
        manifest = manifest.with_shard(
            ShardRecord(
                shard_id=shard_id,
                fingerprint=shard.fingerprint,
                document_count=shard.document_count,
                status=ShardStatus.COMPLETED,
                attempts=1,
                started_at=0.0,
                completed_at=float(shard.document_count),
                output_path=f"shard-{shard_id:05d}.txt",
                output_digest=f"sha256:{'0' * 64}",
                worker_id=f"worker-{shard_id}",
            ),
            updated_at=100.0,
        )
    return _mark_running(
        manifest,
        plan,
        SHARD_COUNT - 1,
        worker_id="worker-slow",
        started_at=running_started_at,
    )


def test_straggler_is_detected_against_the_per_document_baseline() -> None:
    """Baseline is one second per document; the runner is at three."""

    plan = _plan()
    documents = plan.shards[SHARD_COUNT - 1].document_count
    manifest = _straggler_manifest(plan, completed=5, running_started_at=0.0)

    stragglers = classify_stragglers(
        manifest, clock=lambda: 3.0 * documents, multiplier=1.5, min_completed=3
    )

    assert [candidate.shard_id for candidate in stragglers] == [SHARD_COUNT - 1]
    candidate = stragglers[0]
    assert candidate.worker_id == "worker-slow"
    assert candidate.baseline_per_document_seconds == pytest.approx(1.0)
    assert candidate.threshold_per_document_seconds == pytest.approx(1.5)
    assert candidate.per_document_seconds == pytest.approx(3.0)
    assert candidate.slowdown_ratio == pytest.approx(3.0)

    assert (
        classify_stragglers(
            manifest, clock=lambda: 1.2 * documents, multiplier=1.5, min_completed=3
        )
        == ()
    )


def test_straggler_detection_needs_a_minimum_completed_baseline() -> None:
    """Shard 2 of this synthetic plan is empty and cannot supply a rate."""

    plan = _plan()
    assert plan.shards[2].is_empty
    documents = plan.shards[SHARD_COUNT - 1].document_count
    clock = 10.0 * documents

    thin = _straggler_manifest(plan, completed=2, running_started_at=0.0)
    assert classify_stragglers(thin, clock=lambda: clock, min_completed=3) == ()

    # Shards 0, 1, 2 complete, but the empty one contributes no per-document
    # rate, so the baseline is still one sample short.
    empty_padded = _straggler_manifest(plan, completed=3, running_started_at=0.0)
    assert classify_stragglers(empty_padded, clock=lambda: clock, min_completed=3) == ()

    enough = _straggler_manifest(plan, completed=4, running_started_at=0.0)
    assert len(classify_stragglers(enough, clock=lambda: clock, min_completed=3)) == 1

    with pytest.raises(ResumeError):
        classify_stragglers(enough, clock=lambda: clock, min_completed=0)
    with pytest.raises(ResumeError):
        classify_stragglers(enough, clock=lambda: clock, multiplier=0)


def test_empty_shards_resume_and_validate_like_any_other(tmp_path: Path) -> None:
    """Empty shards are retained by the planner and must round-trip cleanly."""

    plan = _plan()
    assert plan.shards[2].document_count == 0

    manifest = build_run_manifest(plan, run_id=RUN_ID, created_at=1_000.0)
    assert 2 in resume_plan(manifest, root=tmp_path).shard_ids

    manifest = _completed_manifest(plan, tmp_path, range(SHARD_COUNT))
    assert (tmp_path / "shard-00002.txt").read_text(encoding="utf-8") == ""
    assert validate_shard_outputs(manifest, root=tmp_path).all_valid
    assert resume_plan(manifest, root=tmp_path).is_complete


def test_straggler_classification_never_touches_orphaned_shards() -> None:
    plan = _plan()
    documents = plan.shards[SHARD_COUNT - 1].document_count
    manifest = _straggler_manifest(plan, completed=5, running_started_at=0.0)

    resumed = resume_plan(
        manifest,
        root=None,
        live_workers={"worker-slow"},
        clock=lambda: 5.0 * documents,
    )
    assert resumed.in_flight == (SHARD_COUNT - 1,)
    assert [item.shard_id for item in resumed.stragglers] == [SHARD_COUNT - 1]

    orphaned = resume_plan(
        manifest, root=None, live_workers=set(), clock=lambda: 5.0 * documents
    )
    assert orphaned.reasons()[SHARD_COUNT - 1] is ResumeReason.ORPHANED
    assert orphaned.stragglers == ()


def _manifest_with_durations(
    plan: ShardPlan,
    durations,
    *,
    running: int,
    running_started_at: float = 0.0,
):
    """Build a manifest whose completed shards have explicit total durations."""

    manifest = build_run_manifest(plan, run_id=RUN_ID, created_at=0.0)
    for shard_id, duration in durations.items():
        shard = plan.shards[shard_id]
        manifest = manifest.with_shard(
            ShardRecord(
                shard_id=shard_id,
                fingerprint=shard.fingerprint,
                document_count=shard.document_count,
                status=ShardStatus.COMPLETED,
                attempts=1,
                started_at=0.0,
                completed_at=duration,
                output_path=f"shard-{shard_id:05d}.txt",
                output_digest=f"sha256:{'0' * 64}",
                worker_id=f"worker-{shard_id}",
            ),
            updated_at=100.0,
        )
    return _mark_running(
        manifest, plan, running, worker_id="worker-slow", started_at=running_started_at
    )


def _force_nonfinite_duration(manifest, shard_id: int, value: float):
    """Force a non-finite timing past the manifest's own validation.

    ``ShardRecord`` now rejects non-finite timestamps at construction, so this
    state is unreachable through the public API and no test can reach it
    honestly. The bypass exists only to pin the defence-in-depth filter in the
    baseline computation: if that validation is ever relaxed, or a record
    arrives from a future writer that does not enforce it, one bad timing must
    still not disable straggler detection for the whole run.
    """

    with pytest.raises(RunManifestError):
        ShardRecord(
            shard_id=shard_id,
            fingerprint="f",
            document_count=1,
            started_at=0.0,
            completed_at=value,
        )
    object.__setattr__(manifest.shard(shard_id), "completed_at", value)
    return manifest


def test_exhausted_run_is_terminal_but_not_complete(tmp_path: Path) -> None:
    """A run that produced no output must never report itself complete."""

    plan = _plan()
    manifest = build_run_manifest(plan, run_id=RUN_ID, created_at=1_000.0)
    for shard in plan.shards:
        manifest = manifest.with_shard(
            ShardRecord(
                shard_id=shard.shard_id,
                fingerprint=shard.fingerprint,
                document_count=shard.document_count,
                status=ShardStatus.FAILED,
                attempts=3,
                error_type="RuntimeError",
            ),
            updated_at=2_000.0,
        )

    resumed = resume_plan(manifest, root=tmp_path, max_attempts=3)

    assert resumed.decisions == ()
    assert resumed.in_flight == ()
    assert resumed.exhausted == tuple(range(SHARD_COUNT))
    assert resumed.is_exhausted
    assert not resumed.is_complete
    assert resumed.to_dict()["exhausted"] == list(range(SHARD_COUNT))


def test_prepare_resume_keeps_the_digest_the_executor_compares_against(
    tmp_path: Path,
) -> None:
    """Clearing the recorded digest would disable the rerun determinism check.

    ``run_shard_plan`` seeds each task's ``expected_digest`` from
    ``manifest.shard(id).output_digest`` and only compares when that is not
    ``None``, so a resume that blanked it would let a shard produce different
    bytes on the retry unnoticed — for exactly the shards being recomputed.
    """

    plan = _plan()
    manifest = _completed_manifest(plan, tmp_path, range(SHARD_COUNT))
    original = manifest.shard(3)
    (tmp_path / "shard-00003.txt").write_text("corrupted\n", encoding="utf-8")

    resumed = resume_plan(manifest, root=tmp_path)
    assert resumed.reasons() == {3: ResumeReason.OUTPUT_MISMATCH}
    prepared = prepare_resume(manifest, resumed, updated_at=1_100.0)
    requeued = prepared.shard(3)

    # Truncated: nothing may treat this shard's output as final any more.
    assert requeued.status is ShardStatus.PENDING
    assert requeued.started_at is None
    assert requeued.completed_at is None
    assert requeued.worker_id is None
    assert requeued.error_type is None

    # Retained: the anchor the executor needs, plus the audit trail.
    assert requeued.output_digest == original.output_digest
    assert requeued.output_path == original.output_path
    assert requeued.output_bytes == original.output_bytes
    assert requeued.attempts == original.attempts
    assert requeued.fingerprint == original.fingerprint
    assert requeued.document_count == original.document_count

    # Untouched shards are not disturbed at all.
    assert prepared.shard(4) == manifest.shard(4)


def test_prepare_resume_refuses_a_plan_from_another_run(tmp_path: Path) -> None:
    """Shard ids always overlap, so an unbound plan would clear live outputs."""

    plan = _plan()
    manifest = _completed_manifest(plan, tmp_path, range(SHARD_COUNT))
    (tmp_path / "shard-00004.txt").unlink()
    resumed = resume_plan(manifest, root=tmp_path)
    assert resumed.shard_ids == (4,)

    foreign = build_run_manifest(plan, run_id="OTHER-RUN", created_at=5.0)
    with pytest.raises(RunIdMismatchError):
        prepare_resume(foreign, resumed)

    other_plan = plan_document_shards(_documents(), shard_count=SHARD_COUNT)
    relabelled = build_run_manifest(other_plan, run_id=RUN_ID, created_at=5.0)
    object.__setattr__(relabelled, "plan_fingerprint", "different-fingerprint")
    with pytest.raises(PlanFingerprintMismatchError):
        prepare_resume(relabelled, resumed)

    prepare_resume(manifest, resumed)


def test_small_shards_are_not_speculated_on_before_they_could_finish() -> None:
    """Fixed per-attempt cost makes small shards look slow on rate alone."""

    plan = _plan()
    load_seconds = 9.0
    durations = {
        shard_id: load_seconds + plan.shards[shard_id].document_count
        for shard_id in (0, 1, 3, 4)
    }
    manifest = _manifest_with_durations(plan, durations, running=6)
    healthy = plan.shards[6].document_count
    assert healthy == 2
    # A healthy 2-document shard cannot finish before load + 2 = 11.0s.
    probe = 10.9

    unguarded = classify_stragglers(
        manifest, clock=lambda: probe, min_completed=3, min_elapsed_seconds=0.0
    )
    assert [item.shard_id for item in unguarded] == [6]

    guarded = classify_stragglers(manifest, clock=lambda: probe, min_completed=3)
    assert guarded == ()
    assert statistics.median(durations.values()) == 12.5


def test_running_shards_that_cannot_be_timed_are_surfaced(tmp_path: Path) -> None:
    """A hung shard must never sit in ``in_flight`` invisibly forever."""

    plan = _plan()
    durations = {shard_id: float(shard_id + 10) for shard_id in (0, 1, 3)}
    manifest = _manifest_with_durations(plan, durations, running=7)

    # (a) RUNNING with no start time, and (b) RUNNING holding no documents.
    for shard_id, started_at in ((6, None), (2, 0.0)):
        shard = plan.shards[shard_id]
        manifest = manifest.with_shard(
            ShardRecord(
                shard_id=shard_id,
                fingerprint=shard.fingerprint,
                document_count=shard.document_count,
                status=ShardStatus.RUNNING,
                attempts=1,
                started_at=started_at,
                worker_id="worker-live",
            ),
            updated_at=50.0,
        )
    assert plan.shards[2].document_count == 0
    assert manifest.shard(6).started_at is None

    resumed = resume_plan(
        manifest,
        root=tmp_path,
        live_workers={"worker-live", "worker-slow"},
        clock=lambda: 10_000.0,
    )

    assert set(resumed.unmeasurable) == {2, 6}
    assert set(resumed.unmeasurable) <= set(resumed.in_flight)
    assert 7 not in resumed.unmeasurable
    assert resumed.to_dict()["unmeasurable"] == list(resumed.unmeasurable)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf])
@pytest.mark.parametrize("poisoned_shard", [0, 1, 3, 4, 5])
def test_one_nonfinite_timing_does_not_disable_detection(
    poisoned_shard: int,
    bad_value: float,
) -> None:
    """A single non-finite timing must not poison the baseline for the run.

    ``statistics.median`` sorts its input and every comparison against ``NaN``
    is false, so a ``NaN`` corrupts the result only when it settles at or
    before the median position; ``inf`` instead sorts to the end and drags a
    high median. Whether one bad timing disables detection therefore depends on
    which shard carries it and on which flavour of non-finite it is, so this is
    parametrised over both rather than tested once.
    """

    plan = _plan()
    durations = {shard_id: float(shard_id + 10) for shard_id in (0, 1, 3, 4, 5)}
    manifest = _manifest_with_durations(plan, durations, running=9)
    manifest = _force_nonfinite_duration(manifest, poisoned_shard, bad_value)
    assert not math.isfinite(manifest.shard(poisoned_shard).completed_at)

    stragglers = classify_stragglers(manifest, clock=lambda: 10_000.0, min_completed=3)
    assert [item.shard_id for item in stragglers] == [9]
    assert math.isfinite(stragglers[0].baseline_per_document_seconds)


def test_an_infinite_rate_is_excluded_from_the_median_not_merely_survived() -> None:
    """An ``inf`` left in the sample shifts the median and hides a straggler."""

    plan = _plan()
    # Rates 1.0, 2.0, 3.0 and one infinite.
    durations = {0: 6.0, 1: 4.0, 3: 9.0, 4: 8.0}
    manifest = _manifest_with_durations(plan, durations, running=9)
    manifest = _force_nonfinite_duration(manifest, 4, math.inf)
    running_documents = plan.shards[9].document_count
    elapsed = 3.4 * running_documents

    # Excluding the infinite rate: median 2.0, threshold 3.0 -> 3.4 is a
    # straggler. Retaining it: median 2.5, threshold 3.75 -> 3.4 is not.
    assert statistics.median([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert statistics.median([1.0, 2.0, 3.0, math.inf]) == pytest.approx(2.5)

    stragglers = classify_stragglers(manifest, clock=lambda: elapsed, min_completed=3)

    assert [item.shard_id for item in stragglers] == [9]
    assert stragglers[0].baseline_per_document_seconds == pytest.approx(2.0)
    assert stragglers[0].threshold_per_document_seconds == pytest.approx(3.0)


def test_baseline_availability_is_reported_distinctly(tmp_path: Path) -> None:
    """An empty straggler tuple must be distinguishable from "not measured"."""

    plan = _plan()
    thin = _manifest_with_durations(plan, {0: 10.0}, running=9)
    thin_plan = resume_plan(
        thin, root=tmp_path, live_workers={"worker-slow"}, clock=lambda: 10_000.0
    )
    assert thin_plan.straggler_baseline_seconds is None
    assert not thin_plan.straggler_detection_enabled
    assert thin_plan.stragglers == ()
    # Serialised explicitly: an empty ``stragglers`` list is ambiguous on its
    # own, and a consumer must not have to infer this from a null baseline.
    assert thin_plan.to_dict()["straggler_detection_enabled"] is False

    durations = {shard_id: float(shard_id + 10) for shard_id in (0, 1, 3, 4)}
    measured = _manifest_with_durations(plan, durations, running=9)
    measured_plan = resume_plan(
        measured, root=tmp_path, live_workers={"worker-slow"}, clock=lambda: 10_000.0
    )
    assert measured_plan.straggler_detection_enabled
    assert measured_plan.straggler_baseline_seconds is not None
    assert measured_plan.to_dict()["straggler_baseline_seconds"] is not None
    assert measured_plan.to_dict()["straggler_detection_enabled"] is True


def test_median_baseline_resists_a_single_extreme_completed_shard() -> None:
    """A mean baseline would be dragged far enough to hide real stragglers."""

    plan = _plan()
    rates = {0: 1.0, 1: 1.0, 3: 1.0, 4: 1.0, 5: 500.0}
    durations = {
        shard_id: rate * plan.shards[shard_id].document_count
        for shard_id, rate in rates.items()
    }
    manifest = _manifest_with_durations(plan, durations, running=9)
    running_documents = plan.shards[9].document_count
    elapsed = 2.0 * running_documents

    observed = [
        duration / plan.shards[shard_id].document_count
        for shard_id, duration in durations.items()
    ]
    assert statistics.median(observed) == pytest.approx(1.0)
    assert statistics.fmean(observed) == pytest.approx(100.8)

    stragglers = classify_stragglers(manifest, clock=lambda: elapsed, min_completed=3)

    # median threshold 1.5 flags a 2.0 s/doc shard; a mean threshold of 151.2
    # would miss it entirely.
    assert [item.shard_id for item in stragglers] == [9]
    assert stragglers[0].baseline_per_document_seconds == pytest.approx(1.0)
    assert stragglers[0].per_document_seconds == pytest.approx(2.0)


def _record(shard_id: int, **overrides) -> ShardRecord:
    payload = {
        "shard_id": shard_id,
        "fingerprint": hashlib.sha256(f"shard-{shard_id}".encode()).hexdigest(),
        "document_count": 4,
        "status": ShardStatus.COMPLETED,
        "attempts": 1,
        "output_path": f"shard-{shard_id:05d}.txt",
        "output_digest": f"sha256:{'a' * 64}",
        "worker_id": "worker-a",
    }
    payload.update(overrides)
    return ShardRecord(**payload)


def test_reconcile_prefers_the_cheaper_reproducible_attempt() -> None:
    first = _record(3, attempts=2, worker_id="worker-b")
    second = _record(3, attempts=1, worker_id="worker-a")

    assert reconcile_shard_attempts(first, second) is second
    assert reconcile_shard_attempts(second, first) is second

    tie_a = _record(3, attempts=1, worker_id="worker-a")
    tie_b = _record(3, attempts=1, worker_id="worker-b")
    assert reconcile_shard_attempts(tie_a, tie_b) is tie_a
    assert reconcile_shard_attempts(tie_b, tie_a) is tie_a


def test_reconcile_is_independent_of_argument_order() -> None:
    """Attempts sharing attempts and worker id must still order deterministically."""

    early = _record(3, worker_id=None, started_at=10.0, completed_at=20.0)
    late = _record(3, worker_id=None, started_at=30.0, completed_at=40.0)

    assert reconcile_shard_attempts(early, late).started_at == 10.0
    assert reconcile_shard_attempts(late, early).started_at == 10.0

    retried_a = _record(3, worker_id="worker-a", started_at=5.0, completed_at=9.0)
    retried_b = _record(3, worker_id="worker-a", started_at=7.0, completed_at=11.0)
    assert reconcile_shard_attempts(retried_a, retried_b).started_at == 5.0
    assert reconcile_shard_attempts(retried_b, retried_a).started_at == 5.0


def test_reconcile_returns_the_first_of_two_indistinguishable_attempts() -> None:
    """Records equal on every ordered field resolve to the first argument."""

    first = _record(3, started_at=1.0, completed_at=2.0, output_bytes=64)
    second = _record(3, started_at=1.0, completed_at=2.0, output_bytes=64)
    assert first is not second

    assert reconcile_shard_attempts(first, second) is first
    assert reconcile_shard_attempts(second, first) is second


def test_reconcile_prefers_a_completed_attempt_over_an_unfinished_one() -> None:
    done = _record(3)
    running = ShardRecord(
        shard_id=3,
        fingerprint=hashlib.sha256(b"shard-3").hexdigest(),
        document_count=4,
        status=ShardStatus.RUNNING,
        attempts=1,
        started_at=5.0,
        worker_id="worker-z",
    )

    assert reconcile_shard_attempts(done, running) is done
    assert reconcile_shard_attempts(running, done) is done


def test_reconcile_rejects_digest_divergence_and_shard_mismatch() -> None:
    first = _record(3)
    diverged = _record(3, output_digest=f"sha256:{'b' * 64}", worker_id="worker-b")

    with pytest.raises(ShardOutputDigestMismatchError):
        reconcile_shard_attempts(first, diverged)

    with pytest.raises(ShardIdentityMismatchError):
        reconcile_shard_attempts(first, _record(4))
    with pytest.raises(ShardIdentityMismatchError):
        reconcile_shard_attempts(first, _record(3, fingerprint="b" * 64))
    with pytest.raises(ShardIdentityMismatchError, match="document count"):
        reconcile_shard_attempts(first, _record(3, document_count=5))


@pytest.mark.parametrize("multiplier", [float("nan"), float("inf"), float("-inf")])
def test_straggler_multiplier_must_be_finite(multiplier: float) -> None:
    with pytest.raises(ResumeError, match="finite positive"):
        classify_stragglers(
            build_run_manifest(_plan(), run_id=RUN_ID, created_at=0.0),
            multiplier=multiplier,
        )


def test_resume_report_is_phi_free(tmp_path: Path) -> None:
    """No document text or identifier may ride through the report, key or value."""

    plan = _plan()
    manifest = _completed_manifest(plan, tmp_path, [0, 1, 2, 3, 4, 7, 8, 9])
    manifest = _mark_running(manifest, plan, 5, worker_id=LEAKY_WORKER, started_at=0.0)
    manifest = manifest.with_shard(
        ShardRecord(
            shard_id=6,
            fingerprint=plan.shards[6].fingerprint,
            document_count=plan.shards[6].document_count,
            status=ShardStatus.FAILED,
            attempts=1,
            error_type="ValueError",
        ),
        updated_at=2_000.0,
    )

    report = resume_plan(
        manifest, root=tmp_path, live_workers={LEAKY_WORKER}, clock=lambda: 5_000.0
    ).to_dict()
    payload = json.dumps(report, sort_keys=True)
    json.loads(payload)

    forbidden = {document["id"] for document in _documents()}
    forbidden |= {document["text"] for document in _documents()}
    forbidden |= {word for text in forbidden for word in text.split()}

    for token in _walk_strings(report):
        assert token not in forbidden
        assert "synthetic" not in token
        assert "doc-" not in token

    # The operator-supplied worker id, and everything riding on it, is absent.
    assert LEAKY_WORKER not in payload
    assert "MRN" not in payload
    assert "123456" not in payload
    assert "worker_id" not in payload

    assert report["decisions"] == [
        {"shard_id": 6, "reason": "failed", "attempts": 1},
    ]
    assert report["schema_version"] == 1
    assert "error_type" not in payload
    assert "text" not in payload


def test_straggler_reports_hash_the_worker_id_but_keep_it_in_process() -> None:
    """The raw worker id must reach executors but never a published report."""

    plan = _plan()
    durations = {shard_id: float(shard_id + 10) for shard_id in (0, 1, 3, 4)}
    manifest = _manifest_with_durations(plan, durations, running=9)
    manifest = _mark_running(manifest, plan, 9, worker_id=LEAKY_WORKER, started_at=0.0)

    stragglers = classify_stragglers(manifest, clock=lambda: 10_000.0, min_completed=3)
    assert [item.shard_id for item in stragglers] == [9]
    candidate = stragglers[0]

    # Executors still need the real id to act on the worker.
    assert candidate.worker_id == LEAKY_WORKER

    # The serialised form does not carry it, or anything derived from it.
    payload = json.dumps(candidate.to_dict(), sort_keys=True)
    assert LEAKY_WORKER not in payload
    assert "MRN" not in payload
    assert "doc-0007" not in payload
    assert "worker_id" not in payload
    assert candidate.to_dict()["worker_ref"] == candidate.worker_ref

    # The reference is stable, opaque, and distinguishes workers.
    assert candidate.worker_ref is not None
    assert candidate.worker_ref.startswith("w-")
    assert len(candidate.worker_ref) == 14
    assert _candidate(worker_id=LEAKY_WORKER).worker_ref == candidate.worker_ref
    assert _candidate(worker_id="worker-b").worker_ref != candidate.worker_ref
    assert _candidate(worker_id=None).worker_ref is None


def test_worker_ref_is_a_shared_helper_not_a_duplicated_construction() -> None:
    """Every layer publishing a worker reference must agree by construction.

    A report can carry the same worker on more than one path. If each layer
    rebuilds the digest from its own literal, two individually-correct
    implementations disagree and the reader silently finds no correlation, so
    the construction lives in one exported function.
    """

    assert worker_ref is openmed.processing.worker_ref

    # The property is the function, not a parallel implementation.
    for candidate_worker in (LEAKY_WORKER, "worker-b", "worker-0", None):
        assert _candidate(worker_id=candidate_worker).worker_ref == worker_ref(
            candidate_worker
        )

    # Contract other layers rely on: deterministic, opaque, injective enough to
    # correlate, and never a passthrough of the raw id.
    assert worker_ref(None) is None
    assert worker_ref(LEAKY_WORKER) == worker_ref(LEAKY_WORKER)
    assert worker_ref(LEAKY_WORKER) != worker_ref("worker-b")
    assert LEAKY_WORKER not in str(worker_ref(LEAKY_WORKER))
    ref = worker_ref(LEAKY_WORKER)
    assert ref is not None and ref.startswith("w-") and len(ref) == 14

    # Namespaced, so it cannot collide with a bare sha256 of the same id.
    bare = hashlib.sha256(LEAKY_WORKER.encode("utf-8")).hexdigest()[:12]
    assert ref != bare
    assert ref != f"w-{bare}"


def _candidate(*, worker_id: str | None) -> StragglerCandidate:
    return StragglerCandidate(
        shard_id=0,
        worker_id=worker_id,
        attempts=1,
        elapsed_seconds=10.0,
        per_document_seconds=2.0,
        baseline_per_document_seconds=1.0,
        threshold_per_document_seconds=1.5,
    )


def _walk_strings(value):
    """Yield every string in a JSON-safe payload, keys included."""

    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def test_resume_plan_fingerprint_is_stable_and_reason_sensitive(
    tmp_path: Path,
) -> None:
    plan = _plan()
    manifest = _completed_manifest(plan, tmp_path, range(7))

    first = resume_plan(manifest, root=tmp_path)
    assert first.fingerprint == resume_plan(manifest, root=tmp_path).fingerprint

    (tmp_path / "shard-00000.txt").unlink()
    assert resume_plan(manifest, root=tmp_path).fingerprint != first.fingerprint
