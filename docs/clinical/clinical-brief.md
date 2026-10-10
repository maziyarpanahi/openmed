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
same calibration as its thresholds. Existing, independently qualified local
checkpoints or caller-supplied calibrated providers can satisfy this contract;
training or publishing a new OpenMed model is not a feature or SDK-release
prerequisite. This API does not invent a default artifact, manufacture calibrated
scores, or fall back to a heuristic. Missing evidence or providers still produce
typed refusal rather than unsafe output.

Before reviewing references, calculate `brief_policy_fingerprint(text, facts,
profile)`. This binds the review history to the exact de-identified text,
clinical axes and profile version. Create review transitions through the existing
review-state API only for work actually reviewed. Changed content or annotations
require a new review. The existing `EvidencePacket` and `BriefContext` remain
**synthetic-only**.
Reviewed-local evidence uses the separate opt-in admission contract below.

Plain-string generation retains the strict extraction path: each atomic claim
must exactly equal one unique reviewed source span. An opt-in structured path
also accepts paraphrases through explicit bindings (below). Ambiguous spans,
missing profile fields, demographic evidence, contradictions and NLI abstentions
are refused. Clinical axes come from reviewed evidence, never from generation.

## Explicit paraphrase bindings (v1)

A trusted local backend may implement `generate_brief(evidence, *, mode)` instead
of `summarize(text, *, mode)`. It receives a tuple of `BriefGenerationEvidence`
records containing only reviewed de-identified spans, opaque `reference_id`s and
original half-open Unicode-scalar source offsets. It returns
`BriefGenerationResult(claims, schema_version=1)` from
`openmed.clinical.summarize_backends`:

For example, when the supplied synthetic evidence contains only the reviewed
span `Symptoms improved after fluids.`, a deterministic test generator can
propose the following paraphrase. A complete brief must also satisfy its profile.

```python
from openmed.clinical.summarize_backends import (
    BriefGeneratedClaim,
    BriefGenerationResult,
)

class SyntheticGenerator:
    def generate_brief(self, evidence, *, mode):
        # Contract illustration only; not a model or a clinical verifier.
        return BriefGenerationResult((
            BriefGeneratedClaim(
                "Symptoms improved following fluid treatment.",
                (evidence[0].reference_id,),
            ),
        ))
```

The result admits 1–64 atomic claims and at most 8,192 UTF-8 output bytes.
Each claim requires exactly one reviewed reference. Empty, invented, repeated or
multiple references, overlapping reviewed spans, compound claims, unknown schema
versions and free-form structured results refuse. There is no similarity search
or confidence-based evidence inference. Multi-reference synthesis needs a future
version that defines how reviewed clinical axes combine; v1 does not guess.
Direct generation-contract validation also refuses malformed Unicode with a
value-free error that retains no chained decoder exception.

Claim text is joined with one space; the composer computes output offsets and
maps the selected reference to original source offsets. Explicit bindings grant
only alignment: every claim still passes the assertion, temporal, experiencer,
calibrated NLI, citation and privacy gates. No threshold changes. Successful
audits add `generation_contract` and `claim_bindings` with reference digests;
legacy exact-match packets remain unchanged. Original identifiers are checked
against generated text, and the complete response is privacy-scanned.

The bundled audit and response schemas and service response model admit this
optional metadata. Both fields must occur together, and the binding sequence
must match the citation count and order. Unknown metadata is refused. Legacy
HTTP responses retain their original fields without added null placeholders.

OpenMedKit's `ClinicalBriefGeneration` and `ClinicalBriefGeneratedClaim` use the
same v1 wire fields (`schema_version`, `claims`, `text`, `reference_ids`). Supply
`boundGeneration` and caller-authorized `reviewedEvidence` to
`ClinicalBrief.validate` alongside the packet from the trusted on-device
evidence/NLI evaluator. Native checks independently bind reference digests and
source/output offsets and require entailment verdicts. The caller must validate
atomicity and all reviewed clinical axes through that evaluator; model output
cannot authorize evidence. With no explicit bindings, native validation keeps
exact extraction. Built-in model adapters continue their existing contracts;
this change adds no trained artifact, cloud fallback or clinical-validation claim.

## Reviewed-local admission (v1)

`ReviewedLocalEvidence` is a separate contract (`kind=reviewed_local_evidence`,
`schema_version=1`, `provenance_class=reviewed_local`). Its source is the exact
**de-identified** artifact whose spans will enter generation, not the original
patient record. `reviewed_source_digest(text)` hashes those UTF-8 bytes. Offsets
are half-open Unicode-scalar coordinates (`unicode_scalar_half_open`); Python
code-point and Swift Unicode-scalar counts match, including supplementary
characters. The packet contains one source digest and length, a policy digest,
1–64 offset-only references, and a nullable `LocalReviewReceipt`. Metadata
integers are non-negative signed 64-bit values on both platforms. Malformed
Unicode source strings produce a controlled invalid-evidence refusal without
retaining their encoding error or source content.

Source/reference/receipt/authority IDs have the form `source:`, `ref:`,
`receipt:`, or `authority:` followed by 64 lowercase hexadecimal characters.
Applications must mint opaque IDs, never encode names, paths or credentials.
Digests use `sha256:` plus 64 lowercase hexadecimal characters. Parsers reject
unknown keys, text payloads, duplicate reference IDs and invalid span boundaries.
`to_dict()`, `to_json()` and `from_json()` round-trip controlled metadata only.
Decoding or changing a provenance marker never constitutes admission.

A receipt contains opaque receipt and authority IDs, the packet's
`evidence_digest`, and UTC Unix `issued_at`/`expires_at` seconds. The digest binds
all version, provenance, source, policy, reference and offset fields, excluding
the receipt itself. The receipt is a **locator**, not a bearer credential or a
self-authenticating signature. The application must supply these narrow local
protocols through `ReviewedLocalBriefContext`:

- `CurrentLocalSource.current_digest(source_id)` authorizes source access and
  returns the current de-identified source digest, or `None`.
- `ReviewAuthorityVerifier.verify(receipt, evidence_digest=..., now=...)`
  resolves an independently held review record, verifies the **entire** receipt
  and caller/reviewer authority, and checks current revocation. It returns
  `ReviewAuthorityStatus.CURRENT`, `REVOKED`, or `MISMATCHED`. Booleans and
  caller-controlled approval markers are refused. No default authority exists.

The context also supplies the ordinary content digest, reviewed `BriefFact`s,
calibrated NLI and privacy callbacks, plus an optional trusted clock. Recompute
`brief_policy_fingerprint(text, facts, profile)` and review the new packet digest
in the application's review store. Never manufacture an approval merely to make
an example pass. A minimal application composition, after actual local review,
is:

```python
from openmed.clinical import ReviewedLocalBriefContext, build_clinical_brief

context = ReviewedLocalBriefContext(
    packet=reviewed_packet,             # loaded from authorized local custody
    content_digest=reviewed_content_digest,
    facts=reviewed_facts,
    nli_predict=local_calibrated_nli,
    thresholds=local_thresholds,
    privacy_detector=local_privacy_detector,
    source=authorized_source_store,
    authority=local_review_registry,
)
brief = build_clinical_brief(deidentified_result, model="extractive", context=context)
# Protected response: brief.to_response(); value-free audit: brief.to_dict().
```

`admit_reviewed_local_evidence()` checks source/policy bindings, receipt existence,
full digest binding, issuance/expiry, current source custody and review authority.
After source and review callbacks return, the trusted clock is read again:
expired receipts, invalid clocks and time rollback fail before generation.
The brief calls it at the evidence stage and again immediately before generation;
an admitted record is not a reusable authorization token. Local stores must
provide consistent reads and enforce their own concurrent-update/access policy.
Only admitted spans enter the generator; all existing NLI, citation and privacy
guards still run. Successful audit `metrics.reviewed_evidence` contains only the
value-free contract, and the generated output still requires human review.

Refusals are typed codes: `invalid_reviewed_evidence`, `review_receipt_missing`,
`review_receipt_expired`, `review_receipt_mismatched`, `review_receipt_revoked`,
`review_authority_unavailable`, `review_source_unavailable`,
`review_source_changed`, and `review_policy_changed`. A receipt with a future
issuance time is mismatched; expiry is inclusive (`now >= expires_at`). Store and
verifier exceptions are discarded without copying their messages or chains.
Changed axes/profile bindings require fresh review. Unavailable stores fail
closed before generation. Ordinary synthetic behavior and defaults are retained.

OpenMedKit exposes the same v1 metadata, digest and offset convention through
`ReviewedLocalEvidence.fromJSON`, `toJSON`, and `admit`. The
`ClinicalBrief.reviewedLocal` adapter admits and rechecks before calling a supplied
**on-device** generator over reviewed spans, then validates the supplied complete
local evidence/NLI evaluation through the existing native guards. Its current
policy digest must be recomputed by the trusted local evaluator; the SDK does
not mint that policy or review. This adds no inference backend, reviewer UI,
training prerequisite, cloud fallback, or autonomous clinical action. Both
platforms' fixtures are wholly synthetic even though they exercise the
`reviewed_local` provenance class; they do not qualify real clinical use.

## Reviewed-local admission (v1)

`ReviewedLocalEvidence` is a separate contract (`kind=reviewed_local_evidence`,
`schema_version=1`, `provenance_class=reviewed_local`). Its source is the exact
**de-identified** artifact whose spans will enter generation, not the original
patient record. `reviewed_source_digest(text)` hashes those UTF-8 bytes. Offsets
are half-open Unicode-scalar coordinates (`unicode_scalar_half_open`); Python
code-point and Swift Unicode-scalar counts match, including supplementary
characters. The packet contains one source digest and length, a policy digest,
1–64 offset-only references, and a nullable `LocalReviewReceipt`. Metadata
integers are non-negative signed 64-bit values on both platforms. Malformed
Unicode source strings produce a controlled invalid-evidence refusal without
retaining their encoding error or source content.

Source/reference/receipt/authority IDs have the form `source:`, `ref:`,
`receipt:`, or `authority:` followed by 64 lowercase hexadecimal characters.
Applications must mint opaque IDs, never encode names, paths or credentials.
Digests use `sha256:` plus 64 lowercase hexadecimal characters. Parsers reject
unknown keys, text payloads, duplicate reference IDs and invalid span boundaries.
`to_dict()`, `to_json()` and `from_json()` round-trip controlled metadata only.
Decoding or changing a provenance marker never constitutes admission.

A receipt contains opaque receipt and authority IDs, the packet's
`evidence_digest`, and UTC Unix `issued_at`/`expires_at` seconds. The digest binds
all version, provenance, source, policy, reference and offset fields, excluding
the receipt itself. The receipt is a **locator**, not a bearer credential or a
self-authenticating signature. The application must supply these narrow local
protocols through `ReviewedLocalBriefContext`:

- `CurrentLocalSource.current_digest(source_id)` authorizes source access and
  returns the current de-identified source digest, or `None`.
- `ReviewAuthorityVerifier.verify(receipt, evidence_digest=..., now=...)`
  resolves an independently held review record, verifies the **entire** receipt
  and caller/reviewer authority, and checks current revocation. It returns
  `ReviewAuthorityStatus.CURRENT`, `REVOKED`, or `MISMATCHED`. Booleans and
  caller-controlled approval markers are refused. No default authority exists.

The context also supplies the ordinary content digest, reviewed `BriefFact`s,
calibrated NLI and privacy callbacks, plus an optional trusted clock. Recompute
`brief_policy_fingerprint(text, facts, profile)` and review the new packet digest
in the application's review store. Never manufacture an approval merely to make
an example pass. A minimal application composition, after actual local review,
is:

```python
from openmed.clinical import ReviewedLocalBriefContext, build_clinical_brief

context = ReviewedLocalBriefContext(
    packet=reviewed_packet,             # loaded from authorized local custody
    content_digest=reviewed_content_digest,
    facts=reviewed_facts,
    nli_predict=local_calibrated_nli,
    thresholds=local_thresholds,
    privacy_detector=local_privacy_detector,
    source=authorized_source_store,
    authority=local_review_registry,
)
brief = build_clinical_brief(deidentified_result, model="extractive", context=context)
# Protected response: brief.to_response(); value-free audit: brief.to_dict().
```

`admit_reviewed_local_evidence()` checks source/policy bindings, receipt existence,
full digest binding, issuance/expiry, current source custody and review authority.
After source and review callbacks return, the trusted clock is read again:
expired receipts, invalid clocks and time rollback fail before generation.
The brief calls it at the evidence stage and again immediately before generation;
an admitted record is not a reusable authorization token. Local stores must
provide consistent reads and enforce their own concurrent-update/access policy.
Only admitted spans enter the generator; all existing NLI, citation and privacy
guards still run. Successful audit `metrics.reviewed_evidence` contains only the
value-free contract, and the generated output still requires human review.

Refusals are typed codes: `invalid_reviewed_evidence`, `review_receipt_missing`,
`review_receipt_expired`, `review_receipt_mismatched`, `review_receipt_revoked`,
`review_authority_unavailable`, `review_source_unavailable`,
`review_source_changed`, and `review_policy_changed`. A receipt with a future
issuance time is mismatched; expiry is inclusive (`now >= expires_at`). Store and
verifier exceptions are discarded without copying their messages or chains.
Changed axes/profile bindings require fresh review. Unavailable stores fail
closed before generation. Ordinary synthetic behavior and defaults are retained.

OpenMedKit exposes the same v1 metadata, digest and offset convention through
`ReviewedLocalEvidence.fromJSON`, `toJSON`, and `admit`. The
`ClinicalBrief.reviewedLocal` adapter admits and rechecks before calling a supplied
**on-device** generator over reviewed spans, then validates the supplied complete
local evidence/NLI evaluation through the existing native guards. Its current
policy digest must be recomputed by the trusted local evaluator; the SDK does
not mint that policy or review. This adds no inference backend, reviewer UI,
training prerequisite, cloud fallback, or autonomous clinical action. Both
platforms' fixtures are wholly synthetic even though they exercise the
`reviewed_local` provenance class; they do not qualify real clinical use.

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

## JSON Schema exports

The bundled brief audit and response schemas admit the optional, closed
`metrics.reviewed_evidence` v1 metadata. Runtime record validation also checks its
bounded offsets, unique references and receipt time ordering. This describes an
audit record; it does not replace current source custody or independent review
authority. The original synthetic evidence-packet schema remains unchanged.

The bundled Draft 2020-12 files in `openmed/core/schemas/json/` describe the
brief audit and protected response, evidence packet, NLI verification list and
SDOH evidence report. `openmed.clinical.record_schemas` exports and fingerprints
these records without loading a runtime or contacting a service. Optional
`validate_clinical_record(name, record)` validation requires `jsonschema` and
reports controlled schema names and field locations without submitted values.

The audit schemas reject nested source and summary text; the response schema
retains its explicitly protected `summary` field. Schema validation does not
prove that a supplied record came from the producer, that its digests are
correct, or that clinical review has occurred. Record versions and Python
producer behavior remain unchanged.

## CLI and service interfaces

`openmed brief note.txt --model extractive --profile bhc --review-id <digest>
--context-factory my_application.review:provider --summary-output summary.txt
--review-output review.json` writes separate, newly created mode-0600 files. It
never overwrites existing destinations. Exit 1 means refusal; stdout contains
only counts, status and a digest. The installed factory is trusted local Python
configuration, not an upload or a server request parameter. It returns a callable
`(original_text, review_id) -> (DeidentificationResult, context)` where `context`
is a `BriefContext` or opt-in `ReviewedLocalBriefContext` backed by
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

Structured `generate_brief` backends receive the same request context when their
signature explicitly declares `cancellation`. Late structured results are
discarded too. Native `ClinicalBrief.reviewedLocal` accepts the same deadline
context and checks admission callbacks, generation, evaluation and publication;
cancelled generation cannot reach the evaluator. Both bundled brief record
schemas admit the controlled `cancelled` and `deadline_exceeded` refusal codes.
## Extracted facts and explicit review receipts

`LocalBriefContextProvider` in `openmed.clinical.brief_context` adapts existing
Python Journey `ClinicalFact` and `EvidenceLocator` records into `BriefContext`.
It implements the callable used by all four Python/CLI/REST/MCP seams above.
This issue's adapter targets those Python extraction and transport contracts;
the existing native brief packet contract is unchanged.

Two narrow application-owned protocols keep extraction, access control and review
authority separate from context construction:

- `BriefExtractionProvider.resolve(text, review_id)` authorizes the caller and
  resolves the current protected `BriefExtraction`, or returns `None`. Include
  the complete selected facts, locators, mappings and known conflicts.
- `BriefReviewVerifier.verify(review_id, plan)` checks an **existing** explicit
  receipt, including current authorization, validity and revocation. Return
  `(stored_binding_digest, approved_evidence_packet)` or `None`. Never construct
  approval transitions merely because the provider requested verification.

`BriefExtraction` contains a `DeidentificationResult`, artifact/source identities,
facts and locators, and one `BriefFactMapping(fact_id, profile_field, temporality)`
per selected fact. Set `synthetic=True` only for synthetic test evidence. The
adapter preserves the existing synthetic-only packet boundary; reviewed-local
admission is separate work in #3651. It does not certify that caller-labelled
data is synthetic and is not a production evidence/review store.

Assertion, certainty and experiencer are read from the existing normalized fact
attributes (`assertion`, `certainty`, `experiencer`). Temporality is explicitly
declared in the mapping and reviewed; lifecycle status and dates do not imply a
temporal axis. Missing/unknown axes or field states refuse instead of becoming
negative facts. Unsupported axes, profile fields and demographic facts refuse.
There is no inferred fact type-to-profile table or heuristic axis fallback.

Each fact must have exactly one untransformed `text_span` locator in the selected
de-identified artifact. Offsets are half-open Python character positions. The
span must fit within one detected section. Missing locators, mixed subjects or
encounters, duplicate/shared/overlapping spans, duplicate source claims and
non-text/transformed locators fail closed. Original-source offsets require an
upstream mapping and renewed review; the adapter never guesses a redaction shift.
An open conflict involving selected facts blocks construction.

Before review, use `plan_brief_context(extraction, profile="bhc",
thresholds=thresholds)` to get the unapproved plan. Its `binding.digest` binds
the original and de-identified sources, artifact/source identities,
de-identification method/mapping/entities, complete fact and locator records,
explicit field/temporal mappings, conflicts, versioned profile and full calibrated
threshold policy. Source/reference identifiers in the plan are synthetic-prefixed
digests rather than copied record identifiers. The existing packet policy binds
content and mapped axes; the separate receipt binding also covers full fact and
calibration identities. Changed evidence requires a new explicit receipt. The
adapter compares the stored binding and exact packet spans, validates approval
history and rechecks the plan after verification. It creates no approval itself.

```python
from openmed.clinical.brief_context import LocalBriefContextProvider

# All dependencies below belong to the trusted local application.
provider = LocalBriefContextProvider(
    extraction_provider=authorized_extraction_store,
    review_verifier=independent_receipt_store,
    thresholds=reviewed_calibration,
    nli_predict=calibrated_local_predictor,
    privacy_detector=local_privacy_detector,
    profile="bhc",
)
app.state.brief_context_provider = provider  # REST
# Also return provider from a CLI context factory, or inject it into MCP runtime.
outcome = provider.build(original_text, opaque_review_id)
safe_diagnostics = outcome.to_dict()  # controlled code and binding digest only
```

`build` returns `BriefContextOutcome` with a `BriefContextCode`: `ready`,
`missing_facts`, `missing_mapping`, `conflicted_facts`, `unsupported_mapping`,
`unknown_fact`, `missing_source`, `ambiguous_source`, `invalid_offsets`,
`review_required`, `evidence_changed`, `invalid_evidence` or
`provider_unavailable`. Its artifact/context are protected application values;
only `to_dict()` is a diagnostic serialization. The callable raises a value-free
`BriefContextError` on refusal; existing transports convert failed lookups into
their existing `invalid_evidence` refusal, with no summary or generation.

No model is downloaded or initialized to construct a context. Python outbound
sockets are blocked during provider calls, as during composition. Trusted
callbacks must not launch external network clients or log source text. A context
with `code="ready"` admits evidence to the guarded composer; a successful brief
still has `status="needs_review"`. Synthetic callbacks and fixtures establish
contract behavior only, not clinical validation or model qualification.

## Passive FHIR R4 documents

`openmed.clinical.exporters.fhir.export_brief_document()` projects a successful
brief into the closed `openmed.clinical.brief.fhir-r4.v1` subset. It runs offline
without a transport, SMART credentials, models or EHR writes. Refused results,
missing or changed provenance, and any requested finalization are rejected.
The current `ClinicalBrief` result always requires review: approved **source
evidence** does not approve the generated document.

```python
from datetime import datetime, timezone
from openmed.clinical.exporters.fhir import (
    export_brief_document,
    import_brief_document,
)

# brief is an existing successful ClinicalBrief. local_detector must be a
# configured local leakage detector; an empty test detector is not a PHI guard.
document = export_brief_document(
    brief,
    recorded_at=datetime.now(timezone.utc),
    privacy_detector=local_detector,
    original_identifiers=original_identifier_tokens,
)
audit = document.to_dict()         # codes, counts, offsets/digests; no narrative
protected = document.to_response() # protected narrative; do not log
restored = import_brief_document(
    protected,
    privacy_detector=local_detector,
    original_identifiers=original_identifier_tokens,
)
assert restored.to_response() == protected
```

The response has `bundle`, `document_reference` and `document_url`:

- The document Bundle includes a **preliminary** Composition first, its local
  Device author, and one evidence DocumentReference per cited claim. Every
  Composition reference resolves inside the Bundle. The Bundle has a deterministic
  identifier bound to the projected metadata and explicit UTC export time.
- Composition narrative preserves the summary. Claim sections follow citation
  order and retain Unicode-scalar output/source offsets in the controlled metadata
  extension. A final Limitations section preserves the human-review disclaimer.
- Evidence attachments contain only `urn:sha256` commitments, fixed titles and
  media types. They contain no source payloads, private paths, original identifiers
  or remote locations. Commitments cannot retrieve source text; applications must
  resolve evidence through their own authorized local evidence store.
- The separate document DocumentReference is `current` with `docStatus=preliminary`.
  Its attachment URL equals `document_url`, an opaque local identifier for the
  Bundle in this response. No bytes are uploaded or external URLs dereferenced.
- The controlled extension preserves the brief, summary, source, provenance and
  policy digests, evidence identifiers/hashes, citations, queued review status and
  conversion-loss codes. It cannot attest approval or reconstruct a ClinicalBrief.

Conversion losses explicitly report unavailable profile section names and omitted
source payloads, evaluation metrics, detailed review packets and model details.
The brief contract does not expose profile section boundaries, so the exporter
preserves claim order without inventing profile headings or clinical context.
Dates are explicit **export times**, never inferred patient dates. No patient
identity or clinical attester is synthesized.

This subset follows the R4 [Composition](https://hl7.org/fhir/R4/composition.html),
[DocumentReference](https://hl7.org/fhir/R4/documentreference.html) and
[document Bundle](https://hl7.org/fhir/R4/documents.html) structures; it does not
claim full R4, IPS, US Core or national implementation-guide conformance.
`import_brief_document()` is the subset validator: it rejects unknown fields,
altered narrative, unsafe status, attachment changes, external/dangling references,
write requests and changed section order. Imported commitments are structural
evidence links, not authentication of an external author or validation against
an unavailable source store.

Both directions require the configured detector on decoded string surfaces and
the assembled document, including extensions and attachment metadata. All
findings block output. Original identifier tokens are also prohibited across
those surfaces. `BriefDocumentError` contains only a controlled code and never
chains callback exceptions. `repr(document)` and `to_dict()` omit narrative;
only explicit `to_response()` returns it. These gates supplement upstream
de-identification and do not establish model quality or authorize clinical use.

OpenMedKit mirrors this subset with `ClinicalBriefFHIRDocument.export(brief,
recordedAt:date, originalIdentifiers:tokens, privacyCheck:localCheck)` and
`ClinicalBriefFHIRDocument.validate(documentJSON:data,
originalIdentifiers:tokens, privacyCheck:localCheck)`. `responseJSON()` is
protected output; `auditJSON()` and the description are value-free. A shared
synthetic fixture checks Python/Swift byte-identical export and round trips.
Native callbacks are trusted on-device application code; no remote fallback or
EHR transport is provided.
