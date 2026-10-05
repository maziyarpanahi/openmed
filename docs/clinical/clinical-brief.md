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
require a new review. The existing `EvidencePacket` and `BriefContext` remain
**synthetic-only**.
Reviewed-local evidence uses the separate opt-in admission contract below.

The initial alignment policy is deliberately strict: each atomic generated claim
must exactly equal one unique reviewed source span. Paraphrases, ambiguous spans,
missing profile fields, demographic evidence, contradictions and NLI abstentions
are refused. This is not a general paraphrase-grounding system. Clinical axes come
from reviewed evidence, never inferred from a generated answer.

## Reviewed-local admission (v1)

`ReviewedLocalEvidence` is a separate contract (`kind=reviewed_local_evidence`,
`schema_version=1`, `provenance_class=reviewed_local`). Its source is the exact
**de-identified** artifact whose spans will enter generation, not the original
patient record. `reviewed_source_digest(text)` hashes those UTF-8 bytes. Offsets
are half-open Unicode-scalar coordinates (`unicode_scalar_half_open`); Python
code-point and Swift Unicode-scalar counts match, including supplementary
characters. The packet contains one source digest and length, a policy digest,
1–64 offset-only references, and a nullable `LocalReviewReceipt`.

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
