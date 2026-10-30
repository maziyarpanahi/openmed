"""Queued job, worker restart and NDJSON stream under continuation authority.

Everything is offline and synthetic: a JSON-file job store stands in for the
durable queue, JSON-file stores stand in for the durable generation high water
and closed-epoch ledger, and the authority service is an in-process provider.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from openmed.agent.permissions.revocation import AuthorityRuntime, AuthorityStatus
from openmed.service.continuation_auth import (
    ContinuationDisposition,
    ContinuationGuard,
    ContinuationReason,
    ContinuationRecord,
    GuardedStream,
    ReconciliationState,
    continuation_error_event,
    run_continuation,
)
from tests.fixtures.service.continuation_authority import (
    SENTINEL,
    ContinuationAuthority,
    SyntheticSink,
    pending_effect,
)

pytestmark = pytest.mark.integration

DOCUMENTS = ("doc-a", "doc-b", "doc-c")


class FileGenerations:
    """Test-only durable implementation of the generation-store contract."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def observe(self, status: AuthorityStatus) -> bool:
        state = json.loads(self.path.read_text()) if self.path.exists() else {}
        key = f"{status.kind.value}:{status.digest}"
        previous = state.get(key)
        if previous is not None:
            generation, revoked = previous
            if (
                status.generation < generation
                or (status.generation == generation and status.revoked != revoked)
                or (revoked and not status.revoked)
            ):
                return False
        state[key] = [status.generation, status.revoked]
        self.path.write_text(json.dumps(state, sort_keys=True))
        return True


class FileLedger:
    """Test-only durable closed-epoch ledger shared by worker processes."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _load(self) -> list[str]:
        return json.loads(self.path.read_text()) if self.path.exists() else []

    def is_closed(self, epoch_digest: str) -> bool:
        return epoch_digest in self._load()

    def close(self, epoch_digest: str) -> bool:
        closed = set(self._load()) | {epoch_digest}
        self.path.write_text(json.dumps(sorted(closed)))
        return True


class JobStore:
    """JSON-file job queue holding only continuation metadata and outcomes."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, job_id: str, record: dict) -> None:
        jobs = self.load()
        jobs[job_id] = record
        self.path.write_text(json.dumps(jobs, sort_keys=True))

    def load(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}


def _worker_process(fixture: ContinuationAuthority, tmp_path: Path):
    """Build one worker process: fresh runtime over durable stores."""

    runtime = AuthorityRuntime(
        fixture.provider,
        FileGenerations(tmp_path / "generations.json"),
        clock=fixture.clock,
    )
    return ContinuationGuard(runtime, ledger=FileLedger(tmp_path / "ledger"))


def _document_worker(sink: SyntheticSink, on_document=None):
    def worker(ctx):
        for ordinal, document in enumerate(DOCUMENTS):
            ctx.checkpoint("document")
            effect = pending_effect(ordinal)
            ctx.commit_effect(effect, lambda: sink.commit(effect.idempotency_key))
            ctx.emit({"document": document, "text": SENTINEL})
            if on_document is not None:
                on_document(ordinal)

    return worker


def test_revoking_consent_mid_job_stops_work_and_survives_restart(tmp_path):
    authority = ContinuationAuthority()
    jobs = JobStore(tmp_path / "jobs.json")
    sink = SyntheticSink()

    # Admission in the API process.
    api = _worker_process(authority, tmp_path)
    record = api.admit("job-1", authority.contracts, checkpoints=("document",))
    jobs.save("job-1", {"status": "queued", "continuation": record.to_dict()})

    # Worker process picks the job up; consent is revoked after document 0.
    worker_guard = _worker_process(authority, tmp_path)
    restored = ContinuationRecord.from_dict(jobs.load()["job-1"]["continuation"])

    def revoke_after_first(ordinal):
        if ordinal == 0:
            authority.provider.revoke(authority.consent)

    result = run_continuation(
        worker_guard, restored, "job-1", _document_worker(sink, revoke_after_first)
    )
    outcome = result.outcome
    assert outcome.disposition is ContinuationDisposition.STOPPED
    assert outcome.reason is ContinuationReason.AUTHORITY_REVOKED
    assert outcome.checkpoint == "document"
    assert len(sink.commits) == 1
    assert outcome.committed_effect_count == 1
    assert outcome.withheld_output_count == 1
    assert outcome.reconciliation is ReconciliationState.COMMITTED_EFFECTS_RETAINED
    assert result.outputs == ()
    jobs.save(
        "job-1",
        {
            "status": outcome.disposition.value,
            "continuation": restored.to_dict(),
            "result": result.to_dict(),
        },
    )
    stored = (tmp_path / "jobs.json").read_text()
    assert SENTINEL not in stored and "job-1" in stored
    assert json.loads(stored)["job-1"]["result"]["outcome"]["compensation"] == (
        "not_attempted"
    )

    # A restarted worker that trusts only the persisted metadata, or an operator
    # who resets the status field to "queued", cannot revive the job.
    jobs.save("job-1", {**jobs.load()["job-1"], "status": "queued"})
    restarted = _worker_process(authority, tmp_path)
    again = run_continuation(
        restarted,
        ContinuationRecord.from_dict(jobs.load()["job-1"]["continuation"]),
        "job-1",
        _document_worker(sink),
        resumed=True,
    )
    assert again.outcome.reason is ContinuationReason.TERMINAL
    assert len(sink.commits) == 1

    # Even with a lost ledger, the authority service still reports revocation.
    (tmp_path / "ledger").unlink()
    fresh = _worker_process(authority, tmp_path)
    again = run_continuation(
        fresh,
        ContinuationRecord.from_dict(jobs.load()["job-1"]["continuation"]),
        "job-1",
        _document_worker(sink),
        resumed=True,
    )
    assert again.outcome.reason is ContinuationReason.AUTHORITY_REVOKED
    assert len(sink.commits) == 1


def test_queued_job_revoked_before_pickup_never_runs(tmp_path):
    authority = ContinuationAuthority()
    sink = SyntheticSink()
    api = _worker_process(authority, tmp_path)
    record = api.admit("job-2", authority.contracts, checkpoints=("document",))
    authority.provider.revoke(authority.scope)

    worker_guard = _worker_process(authority, tmp_path)
    result = run_continuation(worker_guard, record, "job-2", _document_worker(sink))
    assert sink.commits == []
    assert result.outcome.reason is ContinuationReason.AUTHORITY_REVOKED
    assert result.outcome.reconciliation is ReconciliationState.NO_EFFECTS

    # An unrevoked job on the same worker completes and releases its outputs.
    other = ContinuationAuthority()
    record = other.guard.admit("job-3", other.contracts, checkpoints=("document",))
    result = run_continuation(other.guard, record, "job-3", _document_worker(sink))
    assert result.outcome.disposition is ContinuationDisposition.COMPLETED
    assert len(result.outputs) == len(DOCUMENTS)
    assert len(sink.commits) == len(DOCUMENTS)


def test_ndjson_stream_stops_releasing_chunks_after_revocation(tmp_path):
    authority = ContinuationAuthority()
    guard = _worker_process(authority, tmp_path)
    record = guard.admit("stream-1", authority.contracts)

    class RedactedChunks:
        def __init__(self) -> None:
            self.closed = False
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= 4:
                raise StopAsyncIteration
            self.index += 1
            text = "[REDACTED] chunk" if self.index == 1 else SENTINEL
            return json.dumps({"type": "chunk", "redacted_text": text}) + "\n"

        async def aclose(self):
            self.closed = True

    async def serve():
        source = RedactedChunks()
        stream = GuardedStream(
            guard,
            record,
            "stream-1",
            source,
            terminal=lambda outcome: (
                json.dumps(continuation_error_event(outcome)) + "\n"
            ),
        )
        body = []
        async for line in stream:
            body.append(line)
            authority.provider.revoke(authority.consent)
        return body, source, stream.outcome

    body, source, outcome = asyncio.run(serve())
    assert len(body) == 2
    assert json.loads(body[0])["redacted_text"] == "[REDACTED] chunk"
    error = json.loads(body[1])["error"]
    assert error["code"] == ContinuationReason.AUTHORITY_REVOKED.value
    assert error["details"]["delivered_output_count"] == 1
    assert SENTINEL not in "".join(body)
    assert source.closed
    assert outcome.withheld_output_count == 1
    assert json.dumps(outcome.to_dict()).count("sha256:") >= 2
