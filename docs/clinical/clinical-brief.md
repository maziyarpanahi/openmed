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
