# Guarded clinical brief

Start with the [local summarizer](summarization.md), the
[clinical NLI gate](nli-verification.md), and the
[synthetic walkthrough and recording script](../demo/clinical-brief.md).

`openmed.clinical.build_clinical_brief()` composes the existing local clinical
guards into an immutable `ClinicalBrief`. It never treats a generated summary as
a diagnosis or as approval to act. Successful results still need human review.

## Input and review boundary

Pass a note or a `DeidentificationResult`, a local summarizer alias, a built-in
profile (default `bhc`), and an explicit `BriefContext`. A raw note is first
de-identified locally. Missing runtime/model assets fail closed. Without a
reviewed context the result is refused; the composer never manufactures review
history or assumes that de-identification is evidence approval.

`BriefContext` contains the existing `EvidencePacket`, its content digest, one
`BriefFact` per reference, a calibrated local three-class NLI callback with its
`NLIThresholds`, and a local privacy detector. The callback must identify the
same calibration as its thresholds. A trained default NLI artifact is a separate
release prerequisite; this API does not invent one or fall back to a heuristic.

Before reviewing references, calculate `brief_policy_fingerprint(text, facts,
profile)`. This binds the review history to the exact de-identified text,
clinical axes and profile version. Create review transitions through the existing
review-state API only for work actually reviewed. Changed content or annotations
require a new review. The current evidence-packet contract is **synthetic-only**;
this implementation does not widen that boundary to real patient data.

The initial alignment policy is deliberately strict: each atomic generated claim
must exactly equal one unique reviewed source span. Paraphrases, ambiguous spans,
missing profile fields, demographic evidence, contradictions and NLI abstentions
are refused. This is not a general paraphrase-grounding system. Clinical axes come
from reviewed evidence, never inferred from a generated answer.

## Fixed execution and failure behavior

The stage order is not configurable: de-identification, sections, evidence,
summary-input contract, section plan, length budget, profile, generation, safety
envelope, demographic minimization, atomic claims, assertion/temporal/experiencer
pairs, calibrated NLI, citation consistency/boundaries/minimality, temporal order,
unsupported claims, coverage, citation support, empty-evidence guard, privacy,
review packet and provenance. Empty evidence is also rejected before generation.

Only reviewed spans are sent to the generator. Temporal uncertainty is retained
for human review; document order is not a claim that events occurred in that
order. Deterministic citation-span checks do not imply clinician adjudication.
The summary and the assembled protected response both pass the configured privacy
detector. Python outbound sockets are blocked; trusted callbacks are not isolated
plugins and must not invoke external processes/services themselves.

`refusal_reason` is a `BriefRefusal` enum. Refusals contain no partial summary.
Exceptions from backends are not copied into reports or chained into public
errors. The `stages` trace records entered stages; on refusal the last stage
identifies the failure boundary, not a claim that the stage passed.

## Output handling

- `brief.summary` and `brief.to_response()` are protected application output.
  Do not log or persist them without an explicit application privacy boundary.
- `brief.to_dict()` omits all source and summary text and returns aggregate
  metrics, offsets, digests, fixed reasons, review packet and provenance.
- The deterministic `digest` binds the safe record, including the summary digest.
- The result never marks clinician review approved, even when all checks pass.

The synthetic unit fixtures use explicitly labelled test-only NLI scores and
privacy detectors. They verify orchestration and refusal behavior, not model
quality, clinical validation or a release benchmark.

## CLI and service interfaces

`openmed brief note.txt --model extractive --profile bhc --review-id <digest>
--context-factory my_application.review:provider --summary-output summary.txt
--review-output review.json` writes separate, newly created mode-0600 files. It
never overwrites existing destinations. Exit 1 means refusal; stdout contains
only counts, status and a digest. The installed factory is trusted local Python
configuration, not an upload or a server request parameter. It returns a callable
`(original_text, review_id) -> (DeidentificationResult, BriefContext)` backed by
the application's existing review and access-control store. Do not use this
mechanism to invent review histories or accept untrusted executable modules.

`POST /brief`, Python `OpenMedClient.brief(text, model="extractive",
review_id=...)`, TypeScript `client.brief({text, model: "extractive",
review_id})`, and the read-only MCP `openmed_brief` tool share the protected
response. Models are restricted to local aliases; remote URLs, approval records
and NLI scores are not accepted in requests. The opaque review reference is 64
lowercase hexadecimal characters. Configure REST with
`app.state.brief_context_provider` or MCP with the local runtime's
`brief_context_provider`; that provider must authorize access for the caller.
Missing review configuration returns a typed refusal, not an unguarded summary.
REST access logs add only bounded outcome vocabulary and counts to the normal
request duration. Never log the response: `summary` is protected content.

## Cancellation and deadlines

`build_clinical_brief(..., cancellation=...)` accepts an optional caller-owned
`BriefCancellation`, exported from `openmed.clinical`. Existing calls without a
context preserve their result and digest. For a whole-request deadline, start the
existing extraction budget clock before review lookup:

```python
from openmed.clinical import BriefCancellation, build_clinical_brief
from openmed.core.budget import RequestBudget

cancellation = BriefCancellation(RequestBudget(max_wall_time=10).start())
# An application thread may call cancellation.cancel() at any time.
brief = build_clinical_brief(
    deidentified, model="extractive", context=reviewed_context,
    cancellation=cancellation,
)
```

All fixed pipeline stages check the context before entering. Generation, each NLI
call, both privacy scans and the final publication boundary also check after work.
A trusted local summarizer, NLI callback, privacy detector or review lookup can
explicitly accept the keyword `cancellation` and call `cancellation.check()` inside
its own work. Legacy signatures receive no extra keyword. No stage can be removed
or reordered by the context. The clock uses the existing `BudgetClock` wall-time
semantics (elapsed time greater than the allowance); no second extraction budget
or wire-supplied evidence policy is introduced.

Caller cancellation and `KeyboardInterrupt`/`asyncio.CancelledError` produce
`refusal_reason="cancelled"`. Expiry produces `"deadline_exceeded"`; ordinary
backend failures retain `"stage_failed"`. Explicit cancellation wins when both
are observed at the same checkpoint. Observed expiry stays terminal. Interrupted
results are refused with an empty summary, citations, verdicts, metrics and
provenance. Only entered stage names and fixed diagnostics survive; no partial
protected output is returned as success.

`brief_response(..., cancellation=...)` covers application-owned review lookup
as well as composition. The REST route cancels its context when its awaiting task
is cancelled. A disconnected client may be unable to receive the refusal. This
is not a general job scheduler or HTTP disconnect monitor. The CLI accepts
`--timeout-seconds`, maps interruption to the same empty response in its JSON
`data` envelope and exits 1. It creates no output files for an interrupted request
and removes files it reserved if interrupted during output writing.

### Provider cooperation and resource ownership

A synchronous provider, including the current Python MLX `generate` call, may not
cooperate while blocked or generating. Cancellation cannot kill that call or
promise immediate memory reclamation. Its returned output or exception is checked
before any next stage or publication; late output is discarded. The REST worker
may finish later but cannot publish its result through the cancelled request.
The Python MLX adapter drops the model and tokenizer references it owns on every
terminal path after loading. Caller-owned providers, native loaded model actors
and runtime-wide caches remain application-owned; no global cache flush is used.
Providers must release their own transient resources with `finally`/`defer`. They
must not persist, log or externally publish intermediate output themselves.

OpenMedKit uses `ClinicalBriefCancellation`: native task cancellation maps to
`ClinicalBriefError.cancelled`; an application-supplied, thread-safe
`deadlineExpired` closure (using its existing local clock) maps to
`.deadlineExceeded`. `ClinicalBrief.compose` and the Maple `brief` adapter guard
generation, local evaluation, rendered-packet verification and publication.
Evaluators may capture the context for checks inside their own evidence/NLI
stages. No mobile lifecycle policy or Apple Foundation Models cloud fallback is
added. All successful briefs still require qualified human review.
