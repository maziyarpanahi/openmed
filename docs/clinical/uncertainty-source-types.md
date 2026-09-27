# Typed clinical uncertainty sources

Confidence is not a sufficient uncertainty contract for a guarded clinical
output. OpenMed keeps the causes of uncertainty separate so a reviewer can see
whether the active concern is evidence quality, model ambiguity, policy
insufficiency, temporal resolution, or an unresolved conflict.

## Compose sources

The public source types are `evidence`, `model`, `policy`, `temporal`, and
`conflict`. Each source has a controlled reason code and may carry opaque
provenance references. Sources are deduplicated and sorted in a stable order;
they are never reduced to a second aggregate score.

```python
from openmed.clinical import (
    SourceSpan,
    UncertaintySources,
    build_guarded_suggestion,
    conflict_uncertainty,
    evidence_uncertainty,
    model_uncertainty,
    policy_uncertainty,
    temporal_uncertainty,
)

sources = UncertaintySources.compose(
    evidence_uncertainty(
        "insufficient",
        references=("synthetic-evidence-reference",),
    ),
    model_uncertainty("ambiguous"),
    policy_uncertainty("review_required"),
    temporal_uncertainty("unresolved"),
    conflict_uncertainty("unresolved"),
)

guarded = build_guarded_suggestion(
    "Review the finding",
    [SourceSpan(start=12, end=24, label="synthetic")],
    0.72,
    uncertainty_sources=sources,
)

payload = guarded.to_dict()
assert payload["uncertainty_sources"]["active_source_count"] == 5
```

`GuardedSuggestion.to_dict()` includes the `uncertainty_sources` disclosure
when a source collection is supplied. Its `active_sources` list contains every
active source independently, including its type and reason code. An inactive
source is retained in the in-memory composition for deterministic reuse but is
not presented as an active uncertainty.

## Privacy and local-first behavior

Reason codes use a closed vocabulary. References that are not already
`sha256:<64 lowercase hex>` identifiers are hashed before storage, so raw note
text, identifiers, and other caller values do not enter the disclosure,
`repr`, exceptions, or JSON. Use offsets and opaque references for review
provenance; do not put source text in a reason code.

The implementation is deterministic, uses only the Python standard library,
and performs no mandatory network or filesystem call. The disclosure is an
assistive review artifact, not a diagnosis, treatment instruction, compliance
certification, or autonomous clinical decision. Guarded outputs still require
independent qualified-clinician review.
