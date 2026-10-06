"""Fail-closed, local composition of sealed evaluation governance primitives."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping, Sequence

from openmed.eval.governance import (
    FEEDBACK_DETAILED,
    BlindedAdjudicationPacket,
    ComparisonCase,
    ConflictOfInterestMetadata,
    FeedbackBudgetDecision,
    FeedbackBudgetLedger,
    HoldoutCommitment,
    RubricCriterion,
    SourceEvidence,
    commit_holdout_manifests,
    render_blinded_adjudication_packets,
    scan_overlap_forensics,
    verify_holdout_commitment,
    verify_sealed_identity_mapping,
)
from openmed.eval.workflows.sealed_manifest import (
    SealedWorkflowManifest,
    verify_at_evaluation_start,
)

SESSION_SCHEMA_VERSION = "openmed.eval.workflow_session.v1"
_PHASES = ("verified", "cases", "forensics", "adjudication", "release_claimed")
_TERMINAL = ("released", "budget_denied")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_ERROR_CODES = frozenset(
    {
        "invalid_digest",
        "record_invalid",
        "store_failed",
        "record_conflict",
        "budget_state_unverified",
        "invalid_configuration",
        "checkpoint_mismatch",
        "feedback_already_claimed",
        "invalid_clock",
        "holdout_commitment_mismatch",
        "invalid_order",
        "verification_failed",
        "verification_required",
        "invalid_output",
        "invalid_score",
        "execution_failed",
        "execution_required",
        "forensics_failed",
        "forensics_required",
        "adjudication_failed",
        "adjudication_required",
        "feedback_failed",
        "rerun_budget_exhausted",
        "incomplete_manifest",
        "invalid_manifest",
        "mutable_manifest",
        "incomplete_evaluation_snapshot",
        "invalid_evaluation_snapshot",
        "component_digest_mismatch",
        "component_failed",
    }
)


class EvaluationSessionError(ValueError):
    """A controlled, content-free session refusal with a stable ``code``."""

    def __init__(self, code: str) -> None:
        self.code = (
            code if type(code) is str and code in _ERROR_CODES else "component_failed"
        )
        super().__init__(self.code)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _require_digest(value: Any) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise EvaluationSessionError("invalid_digest")


def evaluation_case_digest(case: ComparisonCase) -> str:
    """Hash the canonical private case for the holdout's ordered case manifest.

    Source evidence, candidate identities and reference outputs remain in memory.
    The runner receives only source evidence, never reference outputs or labels.
    """
    return _digest(
        {
            "case_ref": case.case_ref,
            "source_evidence": [item.to_dict() for item in case.source_evidence],
            "candidate_outputs": dict(case.candidate_outputs),
        }
    )


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """Digest-bound checkpoints containing only codes, counts and coarse bands."""

    session_key: str
    binding_digest: str
    steps: tuple[tuple[str, str, int, int, int | None], ...] = ()

    def __post_init__(self) -> None:
        _require_digest(self.session_key)
        _require_digest(self.binding_digest)
        if type(self.steps) is not tuple or len(self.steps) > 6:
            raise EvaluationSessionError("record_invalid")
        for index, step in enumerate(self.steps):
            if type(step) is not tuple or len(step) != 5:
                raise EvaluationSessionError("record_invalid")
            phase, digest, count, tick, band = step
            allowed = (_PHASES[index],) if index < 5 else _TERMINAL
            if type(phase) is not str or phase not in allowed:
                raise EvaluationSessionError("record_invalid")
            _require_digest(digest)
            if any(type(value) is not int or value < 0 for value in (count, tick)):
                raise EvaluationSessionError("record_invalid")
            if band is not None and (index != 5 or type(band) is not int or band < 0):
                raise EvaluationSessionError("record_invalid")

    def to_dict(self) -> dict[str, Any]:
        """Return a closed content-free record with a downstream session digest."""
        payload = {
            "schema_version": SESSION_SCHEMA_VERSION,
            "session_key": self.session_key,
            "binding_digest": self.binding_digest,
            "steps": [list(step) for step in self.steps],
        }
        return {**payload, "session_digest": _digest(payload)}

    def to_json(self) -> str:
        """Serialize the record deterministically."""
        return _json(self.to_dict())

    @classmethod
    def from_json(cls, value: str) -> SessionRecord:
        """Validate persisted schema and digest before trusting any checkpoint."""
        try:
            document = json.loads(value)
            if (
                set(document)
                != {
                    "schema_version",
                    "session_key",
                    "binding_digest",
                    "steps",
                    "session_digest",
                }
                or document["schema_version"] != SESSION_SCHEMA_VERSION
            ):
                raise ValueError
            record = cls(
                document["session_key"],
                document["binding_digest"],
                tuple(tuple(step) for step in document["steps"]),
            )
            if record.to_dict() != document:
                raise ValueError
            return record
        except Exception:
            raise EvaluationSessionError("record_invalid") from None


class SQLiteSessionStore:
    """Durable local checkpoints and atomic, at-most-once feedback claims.

    Use one evaluator-controlled database for all sessions sharing a budget
    ledger. The database stores no case text, outputs, scores, paths or keys.
    Ledger watermarks refuse a reset ledger after a restart. An uncertain claim
    stays locked: recovery sacrifices availability rather than repeating release.
    """

    def __init__(self, path: str | Path) -> None:
        """Open a local database; never use a submission-controlled location."""
        try:
            if not str(path) or str(path) == ":memory:":
                raise ValueError
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._lock = RLock()
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS sessions "
                "(session_key TEXT PRIMARY KEY, record TEXT NOT NULL)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS budgets "
                "(budget_key TEXT PRIMARY KEY, watermark TEXT NOT NULL, claim TEXT)"
            )
            self._db.commit()
        except Exception:
            raise EvaluationSessionError("store_failed") from None

    def close(self) -> None:
        """Close the local connection."""
        self._db.close()

    def load(self, session_key: str) -> SessionRecord | None:
        """Read and validate a durable session checkpoint."""
        _require_digest(session_key)
        try:
            with self._lock:
                row = self._db.execute(
                    "SELECT record FROM sessions WHERE session_key = ?", (session_key,)
                ).fetchone()
            return SessionRecord.from_json(row[0]) if row else None
        except EvaluationSessionError:
            raise
        except Exception:
            raise EvaluationSessionError("store_failed") from None

    def _check(self, previous: SessionRecord) -> None:
        row = self._db.execute(
            "SELECT record FROM sessions WHERE session_key = ?", (previous.session_key,)
        ).fetchone()
        if row is None or row[0] != previous.to_json():
            raise EvaluationSessionError("record_conflict")

    @staticmethod
    def _transition(previous: SessionRecord, updated: SessionRecord) -> None:
        if (
            type(previous) is not SessionRecord
            or type(updated) is not SessionRecord
            or previous.session_key != updated.session_key
            or previous.binding_digest != updated.binding_digest
            or len(updated.steps) != len(previous.steps) + 1
            or updated.steps[:-1] != previous.steps
        ):
            raise EvaluationSessionError("record_invalid")

    def save(self, record: SessionRecord, previous: SessionRecord | None) -> None:
        """Commit a checkpoint with compare-and-swap conflict detection."""
        if type(record) is not SessionRecord:
            raise EvaluationSessionError("record_invalid")
        if previous is None:
            if record.steps:
                raise EvaluationSessionError("record_invalid")
        else:
            self._transition(previous, record)
            if len(record.steps) > 4:
                raise EvaluationSessionError("record_invalid")
        try:
            with self._lock, self._db:
                self._db.execute("BEGIN IMMEDIATE")
                if previous is None:
                    self._db.execute(
                        "INSERT INTO sessions VALUES (?, ?)",
                        (record.session_key, record.to_json()),
                    )
                else:
                    self._check(previous)
                    self._db.execute(
                        "UPDATE sessions SET record = ? WHERE session_key = ?",
                        (record.to_json(), record.session_key),
                    )
        except EvaluationSessionError:
            raise
        except sqlite3.IntegrityError:
            raise EvaluationSessionError("record_conflict") from None
        except Exception:
            raise EvaluationSessionError("store_failed") from None

    def claim_release(
        self,
        previous: SessionRecord,
        claimed: SessionRecord,
        budget_key: str,
        watermark: str,
    ) -> None:
        """Persist release intent before touching the authoritative budget."""
        _require_digest(budget_key)
        _require_digest(watermark)
        self._transition(previous, claimed)
        if len(claimed.steps) != 5:
            raise EvaluationSessionError("record_invalid")
        try:
            with self._lock, self._db:
                self._db.execute("BEGIN IMMEDIATE")
                self._check(previous)
                row = self._db.execute(
                    "SELECT watermark, claim FROM budgets WHERE budget_key = ?",
                    (budget_key,),
                ).fetchone()
                if row and (row[0] != watermark or row[1] is not None):
                    raise EvaluationSessionError("budget_state_unverified")
                self._db.execute(
                    "INSERT OR REPLACE INTO budgets VALUES (?, ?, ?)",
                    (budget_key, watermark, claimed.session_key),
                )
                self._db.execute(
                    "UPDATE sessions SET record = ? WHERE session_key = ?",
                    (claimed.to_json(), claimed.session_key),
                )
        except EvaluationSessionError:
            raise
        except Exception:
            raise EvaluationSessionError("store_failed") from None

    def finish_release(
        self,
        previous: SessionRecord,
        finished: SessionRecord,
        budget_key: str,
        watermark: str,
    ) -> None:
        """Commit a terminal receipt and budget watermark before returning feedback."""
        _require_digest(budget_key)
        _require_digest(watermark)
        self._transition(previous, finished)
        if len(finished.steps) != 6:
            raise EvaluationSessionError("record_invalid")
        try:
            with self._lock, self._db:
                self._db.execute("BEGIN IMMEDIATE")
                self._check(previous)
                row = self._db.execute(
                    "SELECT claim FROM budgets WHERE budget_key = ?", (budget_key,)
                ).fetchone()
                if row != (previous.session_key,):
                    raise EvaluationSessionError("budget_state_unverified")
                self._db.execute(
                    "UPDATE sessions SET record = ? WHERE session_key = ?",
                    (finished.to_json(), finished.session_key),
                )
                self._db.execute(
                    "UPDATE budgets SET watermark = ?, claim = NULL WHERE budget_key = ?",
                    (watermark, budget_key),
                )
        except EvaluationSessionError:
            raise
        except Exception:
            raise EvaluationSessionError("store_failed") from None


@dataclass(frozen=True, slots=True)
class SessionFeedback:
    """Policy-gated scores, one aggregate budget decision and a session digest."""

    session_digest: str
    decision: FeedbackBudgetDecision
    per_case_scores: tuple[float, ...] | None


class EvaluationSession:
    """Enforce verification, execution, forensics, blinding and budgeted release.

    Args:
        session_key: Evaluator-issued SHA-256 identifier for this attempt.
        manifest: Sealed executable submission manifest.
        component_snapshot: Fresh local component digests; called before execution.
        holdout: Pre-published commitment to all four private manifests.
        holdout_manifests: Ordered digest manifests; case entries must use
            :func:`evaluation_case_digest` in the supplied case order.
        cases: Private comparison cases, including a placeholder for the runner's
            candidate and at least one reference candidate.
        candidate_identity: Candidate whose placeholder the runner replaces.
        candidate_manifests: Sealed manifest digest per blinded candidate.
        runner: Injected runner receiving only a case's source evidence.
        scorer: Injected score function returning a finite value in [0, 1].
        review: Injected local clinician-review callback receiving blinded packets.
        provider_digests: Exactly runner/scorer/review digests, pinning providers.
        rubric: Existing blinded-packet rubric contract.
        conflict_of_interest: Existing reviewer-screening contract.
        randomization_key: Evaluator-held packet randomization key.
        public_items: Public comparison corpus for overlap checks.
        shadow_items: Private comparison corpus for overlap checks.
        canaries: Private canary strings for overlap checks.
        ledger: Authoritative feedback ledger; epoch is the holdout commitment,
            submission is the sealed manifest. Persist its usage externally.
        store: Shared durable session store for this budget authority.
        clock: Injected nonnegative integer clock, used for checkpoint timestamps.

    Completed steps are replayed privately on restart and checked against stored
    digests before continuing. Providers must be deterministic and safe to replay.
    No content is persisted, so outputs cannot be recovered from this database.
    """

    def __init__(
        self,
        *,
        session_key: str,
        manifest: SealedWorkflowManifest,
        component_snapshot: Callable[[], Mapping[str, str]],
        holdout: HoldoutCommitment,
        holdout_manifests: Mapping[str, Sequence[str]],
        cases: Sequence[ComparisonCase],
        candidate_identity: str,
        candidate_manifests: Mapping[str, str],
        runner: Callable[[tuple[SourceEvidence, ...]], str],
        scorer: Callable[[ComparisonCase, str], float],
        review: Callable[[tuple[BlindedAdjudicationPacket, ...]], None],
        provider_digests: Mapping[str, str],
        rubric: Sequence[RubricCriterion],
        conflict_of_interest: ConflictOfInterestMetadata,
        randomization_key: bytes,
        public_items: Sequence[str],
        shadow_items: Sequence[str],
        canaries: Sequence[str],
        ledger: FeedbackBudgetLedger,
        store: SQLiteSessionStore,
        clock: Callable[[], int],
    ) -> None:
        try:
            _require_digest(session_key)
            if set(provider_digests) != {"runner", "scorer", "review"}:
                raise EvaluationSessionError("invalid_configuration")
            for digest in (*provider_digests.values(), *candidate_manifests.values()):
                _require_digest(digest)
            self._manifest = manifest
            self._snapshot = component_snapshot
            self._holdout = holdout
            self._manifests = {k: tuple(v) for k, v in holdout_manifests.items()}
            self._cases = tuple(cases)
            self._identity = candidate_identity
            self._candidate_manifests = dict(candidate_manifests)
            if (
                not self._cases
                or candidate_manifests.get(candidate_identity)
                != manifest.manifest_digest
                or any(
                    candidate_identity not in c.candidate_outputs for c in self._cases
                )
                or len({c.case_ref for c in self._cases}) != len(self._cases)
            ):
                raise EvaluationSessionError("invalid_configuration")
            self._runner, self._scorer, self._review = runner, scorer, review
            self._rubric = tuple(rubric)
            self._conflict = conflict_of_interest
            self._key = bytes(randomization_key)
            self._public, self._shadow, self._canaries = (
                tuple(public_items),
                tuple(shadow_items),
                tuple(canaries),
            )
            self._ledger, self._store, self._clock = ledger, store, clock
            self._policy = ledger.policy(holdout.commitment_digest)
            binding = _digest(
                {
                    "manifest": manifest.to_dict(),
                    "holdout": holdout.to_dict(),
                    "holdout_manifests": self._manifests,
                    "cases": [evaluation_case_digest(c) for c in self._cases],
                    "candidate_identity": candidate_identity,
                    "candidate_manifests": self._candidate_manifests,
                    "providers": dict(provider_digests),
                    "feedback_policy": self._policy.to_dict(),
                    "rubric": [r.to_dict() for r in self._rubric],
                    "conflict": conflict_of_interest.to_dict(),
                    "key_digest": "sha256:" + hashlib.sha256(self._key).hexdigest(),
                    "public": self._public,
                    "shadow": self._shadow,
                    "canaries": self._canaries,
                }
            )
            record = store.load(session_key)
            if record is None:
                record = SessionRecord(session_key, binding)
                store.save(record, None)
            elif record.binding_digest != binding:
                raise EvaluationSessionError("checkpoint_mismatch")
            if len(record.steps) >= 5:
                raise EvaluationSessionError("feedback_already_claimed")
            self._record: SessionRecord = record
            self._cursor = 0
            self._lock = RLock()
            self._outputs: tuple[str, ...] = ()
            self._scores: tuple[float, ...] = ()
        except EvaluationSessionError:
            raise
        except Exception:
            raise EvaluationSessionError("invalid_configuration") from None

    @property
    def record(self) -> SessionRecord:
        """Return the immutable content-free record for downstream reporting."""
        return self._record

    def _tick(self) -> int:
        tick = self._clock()
        if type(tick) is not int or tick < 0:
            raise EvaluationSessionError("invalid_clock")
        return tick

    def _step(self, phase: str, digest: str, count: int) -> None:
        if self._cursor < len(self._record.steps):
            saved = self._record.steps[self._cursor]
            if saved[:3] != (phase, digest, count):
                raise EvaluationSessionError("checkpoint_mismatch")
        else:
            updated = replace(
                self._record,
                steps=(*self._record.steps, (phase, digest, count, self._tick(), None)),
            )
            self._store.save(updated, self._record)
            self._record = updated
        self._cursor += 1

    def _verify_inputs(self) -> None:
        verification = verify_at_evaluation_start(self._manifest, self._snapshot())
        if not verification.eligible_for_sealed_results:
            raise EvaluationSessionError(verification.reason_codes[0])
        if (
            not verify_holdout_commitment(self._holdout).valid
            or self._manifests.get("case")
            != tuple(evaluation_case_digest(case) for case in self._cases)
            or commit_holdout_manifests(
                self._holdout.holdout_version, self._manifests
            ).commitment_digest
            != self._holdout.commitment_digest
        ):
            raise EvaluationSessionError("holdout_commitment_mismatch")

    def verify(self) -> None:
        """Verify both seals before exposing any case to the injected runner."""
        with self._lock:
            if self._cursor != 0:
                raise EvaluationSessionError("invalid_order")
            try:
                self._verify_inputs()
                self._step("verified", self._record.binding_digest, len(self._cases))
            except EvaluationSessionError:
                raise
            except Exception:
                raise EvaluationSessionError("verification_failed") from None

    def run_cases(self) -> None:
        """Execute and score verified cases privately, returning no output or score."""
        with self._lock:
            if self._cursor != 1:
                raise EvaluationSessionError("verification_required")
            try:
                self._verify_inputs()
                outputs, scores = [], []
                for case in self._cases:
                    output = self._runner(tuple(case.source_evidence))
                    if type(output) is not str or not output.strip():
                        raise EvaluationSessionError("invalid_output")
                    score = self._scorer(case, output)
                    if (
                        type(score) not in (int, float)
                        or not math.isfinite(score)
                        or not 0 <= score <= 1
                    ):
                        raise EvaluationSessionError("invalid_score")
                    outputs.append(output)
                    scores.append(float(score))
                self._step("cases", _digest([outputs, scores]), len(outputs))
                self._outputs, self._scores = tuple(outputs), tuple(scores)
            except EvaluationSessionError:
                raise
            except Exception:
                raise EvaluationSessionError("execution_failed") from None

    def run_forensics(self) -> None:
        """Run all existing overlap checks and bind their content-free report."""
        with self._lock:
            if self._cursor != 2:
                raise EvaluationSessionError("execution_required")
            try:
                report = scan_overlap_forensics(
                    self._outputs,
                    public_items=self._public,
                    shadow_items=self._shadow,
                    canaries=self._canaries,
                    submission_manifest=self._manifest,
                    holdout_commitment=self._holdout,
                )
                self._step("forensics", _digest(report.to_dict()), len(report.signals))
            except EvaluationSessionError:
                raise
            except Exception:
                raise EvaluationSessionError("forensics_failed") from None

    def build_adjudication(self) -> None:
        """Render and audit blinded packets, then invoke the injected local review."""
        with self._lock:
            if self._cursor != 3:
                raise EvaluationSessionError("forensics_required")
            try:
                cases = tuple(
                    replace(
                        case,
                        candidate_outputs={
                            **case.candidate_outputs,
                            self._identity: output,
                        },
                    )
                    for case, output in zip(self._cases, self._outputs)
                )
                packets, mapping = render_blinded_adjudication_packets(
                    packet_set_ref=self._record.session_key,
                    cases=cases,
                    rubric=self._rubric,
                    conflict_of_interest=self._conflict,
                    holdout_commitment_digest=self._holdout.commitment_digest,
                    submission_manifest_digests=self._candidate_manifests,
                    randomization_key=self._key,
                )
                if not verify_sealed_identity_mapping(
                    packets, mapping, self._key
                ).valid:
                    raise EvaluationSessionError("adjudication_failed")
                digest = _digest(
                    {
                        "packets": [p.to_dict() for p in packets],
                        "mapping_digest": mapping.mapping_commitment,
                    }
                )
                # Verify a replayed packet checkpoint before repeating reviewer I/O.
                if self._cursor < len(self._record.steps):
                    if self._record.steps[self._cursor][:3] != (
                        "adjudication",
                        digest,
                        len(packets),
                    ):
                        raise EvaluationSessionError("checkpoint_mismatch")
                self._review(packets)
                self._step("adjudication", digest, len(packets))
            except EvaluationSessionError:
                raise
            except Exception:
                raise EvaluationSessionError("adjudication_failed") from None

    def release_feedback(self) -> SessionFeedback:
        """Claim release durably, consume one session attempt, then return feedback.

        Exact per-case scores require the ledger's detailed-feedback decision.
        Otherwise only its aggregate coarse band is returned. Session records
        contain decision digests and coarse bands, never exact scores.
        """
        with self._lock:
            if len(self._record.steps) >= 5:
                raise EvaluationSessionError("feedback_already_claimed")
            if self._cursor < 3:
                raise EvaluationSessionError("forensics_required")
            if self._cursor != 4:
                raise EvaluationSessionError("adjudication_required")
            epoch, submission = (
                self._holdout.commitment_digest,
                self._manifest.manifest_digest,
            )
            budget_key = _digest([epoch, submission])
            try:
                snapshot = self._ledger.snapshot(epoch, submission)
                watermark = _digest(
                    {
                        "policy": self._policy.to_dict(),
                        "usage": snapshot.to_dict(),
                    }
                )
                claimed = replace(
                    self._record,
                    steps=(
                        *self._record.steps,
                        (
                            "release_claimed",
                            _digest(snapshot.to_dict()),
                            1,
                            self._tick(),
                            None,
                        ),
                    ),
                )
                self._store.claim_release(self._record, claimed, budget_key, watermark)
                self._record = claimed
                decision = self._ledger.record_official_attempt(
                    epoch, submission, score=sum(self._scores) / len(self._scores)
                )
                finished = replace(
                    claimed,
                    steps=(
                        *claimed.steps,
                        (
                            "released" if decision.accepted else "budget_denied",
                            _digest(decision.to_dict()),
                            decision.official_attempts,
                            self._tick(),
                            decision.coarse_band,
                        ),
                    ),
                )
                self._store.finish_release(
                    claimed,
                    finished,
                    budget_key,
                    _digest(
                        {
                            "policy": self._policy.to_dict(),
                            "usage": self._ledger.snapshot(epoch, submission).to_dict(),
                        }
                    ),
                )
                self._record = finished
                if not decision.accepted:
                    raise EvaluationSessionError(decision.reason_code)
                return SessionFeedback(
                    finished.to_dict()["session_digest"],
                    decision,
                    self._scores
                    if decision.feedback_level == FEEDBACK_DETAILED
                    else None,
                )
            except EvaluationSessionError:
                raise
            except Exception:
                raise EvaluationSessionError("feedback_failed") from None

    def run(self) -> SessionFeedback:
        """Execute the ordered pipeline, verifying replayed checkpoints on restart."""
        with self._lock:
            self.verify()
            self.run_cases()
            self.run_forensics()
            self.build_adjudication()
            return self.release_feedback()
