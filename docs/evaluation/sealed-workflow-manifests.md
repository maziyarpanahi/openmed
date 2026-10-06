# Sealed workflow manifests

Workflow benchmark results are comparable only when the executable submission
surface cannot change between submission and evaluation. OpenMed represents
that surface as a deterministic, digest-only manifest and verifies it again at
evaluation start. The process is local and performs no network requests.

## Governed components

Every manifest contains exactly one lowercase `sha256:` digest for each of
these components:

- `model`
- `tokenizer`
- `tool_inventory`
- `prompt`
- `policy`
- `container`
- `threshold`
- `post_processing`

The tool-inventory digest must come from a canonical, PHI-safe inventory. Do
not place tool arguments, endpoints, credentials, descriptions, clinical
values, or paths in the manifest. The other entries likewise identify bytes or
canonical configuration documents by digest; they do not embed their content.

## Seal a submission

Compute the component digests locally, then seal the complete mapping:

```python
from openmed.eval.workflows import seal_workflow_manifest

component_digests = {
    "container": "sha256:" + "1" * 64,
    "model": "sha256:" + "2" * 64,
    "policy": "sha256:" + "3" * 64,
    "post_processing": "sha256:" + "4" * 64,
    "prompt": "sha256:" + "5" * 64,
    "threshold": "sha256:" + "6" * 64,
    "tokenizer": "sha256:" + "7" * 64,
    "tool_inventory": "sha256:" + "8" * 64,
}
manifest = seal_workflow_manifest(component_digests)
serialized = manifest.to_json()
```

`to_json()` uses sorted, compact canonical JSON. `manifest_digest` seals the
schema version and the complete component mapping. The returned Python object
is frozen, and its component mapping is read-only.

## Verify at evaluation start

Immediately before evaluation, recompute all eight digests from the local
artifacts and configuration, then compare them with the submitted manifest:

```python
from openmed.eval.workflows import verify_at_evaluation_start

verification = verify_at_evaluation_start(manifest, component_digests)
if not verification.eligible_for_sealed_results:
    # Keep the run out of sealed-result publication.
    print(verification.to_dict())
```

Only a complete, canonical, unmodified manifest whose component digests all
match the evaluation-start snapshot is eligible for sealed results. Missing or
malformed data, extra fields, a broken manifest seal, and changed components
all fail closed.

Verification reports contain only closed reason codes and governed component
names. They never echo submitted values. Store or log the report rather than
the supplied manifest when producing evaluation diagnostics.

This mechanism establishes submission immutability for benchmark comparison.
It is not a compliance certification or an autonomous clinical decision
guarantee.

## Fail-closed evaluation sessions

`openmed.eval.workflows.session.EvaluationSession` composes the Python sealed
manifest, holdout commitment, overlap, blinded-packet and feedback-budget
contracts. It performs no network requests. Runners, scoring functions and
clinician review are injected; the session does not implement clinical scoring,
validate clinician credentials or make clinical decisions. Independent-site
execution remains separate.

A session follows this fixed order:

1. `verify()` verifies the executable manifest against a fresh injected component
   snapshot, recomputes all four holdout manifests and binds the actual cases.
2. `run_cases()` repeats verification immediately before execution, then calls
   the injected runner once per case. The runner receives only source evidence,
   never reference candidates or labels. Outputs and scores stay in memory.
3. `run_forensics()` runs all existing overlap checks over the generated outputs
   against evaluator-supplied public, shadow and canary corpora.
4. `build_adjudication()` substitutes the generated candidate into the existing
   comparison cases, renders and audits the blinded packets, and passes them to
   the injected local review callback. Packets and private identity mappings are
   never saved in the session database.
5. `release_feedback()` durably claims release, records **one official attempt
   for the whole session** in the ledger, and commits a terminal receipt before
   returning anything. The session's mean score is the aggregate ledger input.

`run()` executes those steps in order. Intermediate methods return `None`; only
`release_feedback()` returns scores. A detailed ledger decision permits the
ordered `SessionFeedback.per_case_scores`; a coarse decision returns `None` for
that field and only the ledger's aggregate band. Exhausted reruns return no
feedback. Forensic signals remain evidence, not automatic contamination verdicts
or claims that an unflagged model is clean.

### Inputs and budget authority

Use the existing `ComparisonCase`, `SourceEvidence`, `RubricCriterion` and
`ConflictOfInterestMetadata` contracts. Each comparison case needs a placeholder
for `candidate_identity` and at least one reference candidate. Build the ordered
holdout `case` manifest with `evaluation_case_digest(case)`; that helper hashes
the case reference, ordered evidence and identity-keyed reference/placeholder
outputs without writing their content. Pass all four private digest manifests
alongside the pre-published `HoldoutCommitment`. Their recomputed commitment must
match before any case reaches the runner. Label/template/randomization manifests
retain their existing evaluator-defined representation.

The feedback epoch is the holdout's `commitment_digest`; the submission identity
is the executable `manifest_digest`. Supply an authoritative `FeedbackBudgetLedger`
with a policy for that epoch. Its read-only `policy(epoch_digest)` accessor lets
the session bind the exact immutable policy. No new dependency is needed.

Use a new evaluator-issued SHA-256 `session_key` for each official attempt, and
one durable `SQLiteSessionStore(path)` for all attempts sharing a budget authority.
The injected component snapshot must recompute local artifact digests, rather
than return an old cached snapshot. `provider_digests` must contain exactly
`runner`, `scorer` and `review` digests supplied by the evaluator. The binding
also covers cases, corpora, rubric, reviewer metadata, candidate manifests,
randomization-key digest and feedback policy. Provider identity declarations are
the evaluator's responsibility; the session does not inspect executable code.

The runner accepts `tuple[SourceEvidence, ...]` and returns text. The scorer
accepts `(ComparisonCase, output_text)` and returns a finite score in `[0, 1]`.
The review callback accepts `tuple[BlindedAdjudicationPacket, ...]`; it returns
no session feedback. Review transport, authentication and clinical judgments
remain under evaluator control. Use deterministic providers that can safely be
replayed, and keep every injected provider local for PHI workflows.

### Recovery and content-free records

`session.record.to_json()` contains a closed schema, session and input-binding
digests, ordered step digests, counts, nonnegative integer clock ticks and an
optional aggregate coarse band. Its `session_digest` binds the entire record
for downstream reporting. Exact scores, evidence, outputs, identities, paths,
credentials and randomization keys are absent, including from SQLite journal
writes. Use an injected `clock: Callable[[], int]` for deterministic tests.

Reopening a session validates the record schema and digest. Because content is
not persisted, it privately replays completed steps from verification and
compares their digests before advancing. A changed output or scoring result
refuses recovery before forensics or feedback. Original checkpoint timestamps
are retained. This recovery model requires safe, deterministic runner/scorer/
review callbacks; it is not an exactly-once execution guarantee for their side
effects. Keep any separately retained case data or provider state in the
existing evaluator-controlled private systems, outside the session database.

SQLite compare-and-swap writes prevent competing sessions from claiming the
same release. A durable `release_claimed` checkpoint is committed **before** the
ledger is touched. Claimed, released and denied sessions cannot release again,
including after a restart. If a crash occurs around budget consumption or
terminal persistence, the claim stays locked and no feedback is returned;
recovery does not guess whether consumption occurred. This deliberately permits
lost feedback rather than duplicate feedback.

The existing ledger is in memory: the evaluator must preserve its authoritative
usage across process restarts and serialize its operations. The session store
persists a policy/usage watermark and refuses a reset or changed ledger for an
already used budget pair. An uncertain release also locks that budget pair.
Keep the database durable, access-controlled and shared by all sessions using
that ledger. Digest checks detect accidental corruption, not malicious database
rollback or replacement; neither a fresh database nor a fresh ledger is a valid
recovery procedure. In-memory or temporary SQLite database names are rejected.

### Stable refusals and synthetic example

`EvaluationSessionError.code` exposes controlled codes without echoing provider
exceptions or input values. Key refusals include:

| Trigger | Code |
| --- | --- |
| Cases requested before verification | `verification_required` |
| A sealed executable component changes | `component_digest_mismatch` |
| Actual cases or private manifests differ from the holdout | `holdout_commitment_mismatch` |
| Feedback or packets requested before overlap scanning | `forensics_required` |
| Feedback requested before blinded review | `adjudication_required` |
| Replay/configuration differs from a stored checkpoint | `checkpoint_mismatch` |
| Budget exhausted | `rerun_budget_exhausted` |
| Release previously claimed, completed or denied | `feedback_already_claimed` |
| Ledger watermark differs or budget consumption is uncertain | `budget_state_unverified` |
| Concurrent checkpoint/release conflict | `record_conflict` |

For a fully configured synthetic session with scores `0.75` and `0.25`, a
policy granting one detailed attempt returns `per_case_scores=(0.75, 0.25)`
and aggregate `detailed_score=0.5`. A later allowed coarse attempt with boundaries
`(0.5, 0.8)` returns `per_case_scores=None`, `detailed_score=None` and
`coarse_band=1`. Only digests, counts and clock ticks are persisted for the first
attempt; the later coarse receipt may also persist band `1`. These are synthetic
contract examples, not clinical validation or release evidence.
