# Atomic summary claim segmentation

OpenMed can split a **post-de-identification** summary into deterministic claim
spans before citation or natural-language-inference checks. Every emitted claim
retains an exact half-open character offset into the supplied summary.

```python
from openmed.clinical.summary_claim_segments import segment_summary_claims

summary = "Signal alpha improved, but marker beta persisted."
result = segment_summary_claims(summary)

for claim in result.segments:
    assert summary[claim.start : claim.end] == claim.text
    print(claim.offset, claim.review_required)
```

The implementation is local and deterministic. It uses the package's pinned
sentence segmenter, makes no network request, loads no model, and adds no
telemetry. Semicolons, newlines, and sentence endings are stable boundaries.
For English summaries, commas, `and`, `but`, and `yet` split only when both
sides have an explicit subject and finite predicate.

## Review gate

Conservative segmentation is deliberate. A clause that still contains a
coordination, alternative, causal, temporal, or relative-clause marker is kept
intact and receives:

```text
review_required = true
review_reason = "unsegmentable_compound"
```

Callers must stop before claim-level citation and NLI checks until a reviewer
resolves that span. Non-English summaries retain deterministic sentence-level
offsets but receive `unsupported_language`, because the English clause rules
cannot establish that those spans are atomic.

## Privacy boundary

Pass only summary text produced after de-identification. `SummaryClaimSegment`
contains its exact summary slice because downstream verification needs it, but
`repr()`, `to_dict()`, and `to_json()` omit claim text and expose only offsets,
counts, language, schema version, and controlled review metadata. Exceptions
use fixed messages and do not echo submitted values. This is a verification
aid, not a compliance certification or autonomous clinical decision system.
