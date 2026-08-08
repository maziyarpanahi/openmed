# Post-de-identification summary safety

OpenMed summaries are downstream of de-identification. The summary stage does
not accept raw text or a plain mapping as its input boundary: callers first
create a `VerifiedDeidentifiedArtifact` from a structured de-identification
result. The artifact retains only the de-identified payload, a content hash,
source hash, safe identifiers, and verification metadata. Original text,
detected entities, and re-identification mappings are not copied into summary
provenance.

The stage also requires an explicit human-review mode. If either gate is
missing, it returns a refusal envelope and does not invoke the summary
producer. Every ready or refused envelope carries the same non-diagnostic
disclaimer:

> This summary is for human review only. It is not a diagnosis, medical advice,
> or a substitute for qualified clinical judgment.

## Local usage

```python
from openmed.clinical.summary_envelope import (
    run_summary_stage,
    verify_deidentified_artifact,
)

deidentified = verify_deidentified_artifact(
    deidentification_result,
    artifact_id="synthetic-note-001",
    provenance={"policy": "offline-synthetic"},
)

envelope = run_summary_stage(
    deidentified,
    lambda text: {"section_count": text.count("\n\n") + 1},
    human_review_mode=True,
    provenance={"producer": "local-summary-v1"},
)

if envelope.status == "refused":
    print(envelope.refusal_reasons)
else:
    print(envelope.summary)
```

`run_summary_stage` passes only the de-identified payload to the producer. A
producer exception is converted into the fixed refusal code
`summary_generation_failed`; the original exception is not chained through the
summary boundary. The module itself performs no network call and has no model
or service dependency. Supply a local deterministic producer and keep its
summary output aggregate or otherwise de-identified before serialization.

The envelope's `provenance` is intentionally metadata-only. It contains the
artifact identifier, SHA-256 hashes, de-identification method, verification
method, and caller-supplied safe metadata. Do not put source text, names,
identifiers, dates, addresses, entity surfaces, or re-identification mappings
in provenance, logs, exceptions, reports, or committed fixtures.

This guardrail is an assistive workflow control. It is not a compliance
certification, a clinical decision guarantee, or a substitute for deployment-
specific privacy and clinical review.
