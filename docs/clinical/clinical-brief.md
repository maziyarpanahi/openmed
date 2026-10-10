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

Claim text is joined with one space; the composer computes output offsets and
maps the selected reference to original source offsets. Explicit bindings grant
only alignment: every claim still passes the assertion, temporal, experiencer,
calibrated NLI, citation and privacy gates. No threshold changes. Successful
audits add `generation_contract` and `claim_bindings` with reference digests;
legacy exact-match packets remain unchanged. Original identifiers are checked
against generated text, and the complete response is privacy-scanned.

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
