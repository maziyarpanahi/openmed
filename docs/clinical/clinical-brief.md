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

## JSON Schema exports

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
