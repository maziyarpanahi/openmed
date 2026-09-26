# Citation minimality checks

Citation minimality checks make overbroad evidence visible in guarded clinical
outputs. The caller declares the source span required for each atomic claim;
the local checker compares that span with each citation, counts deterministic
Unicode word and punctuation tokens, and returns offsets and controlled review
statuses.

The checker is a review aid. It does not infer whether a source semantically
entails a claim, select evidence, certify compliance, or make a clinical
decision. Required spans must be supplied by a human-reviewed annotation or an
upstream evidence component with its own guardrails.

## Check a citation

Identifiers in the report must be opaque `sha256:<64 lowercase hex>` references.
Source text is used only during the local check and is not retained in the
returned report.

```python
import hashlib

from openmed.clinical.citation_minimality import (
    AtomicClaim,
    ClaimCitation,
    CitationSpan,
    check_citation_minimality,
)


def reference(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


source = "alpha beta gamma delta"
claim = AtomicClaim(
    claim_id=reference("claim-1"),
    required_span=CitationSpan(start=6, end=10),  # beta
)
citation = ClaimCitation(
    claim_id=claim.claim_id,
    citation_id=reference("citation-1"),
    source_span=CitationSpan(start=6, end=16),  # beta gamma
)

report = check_citation_minimality(source, [claim], [citation])
print(report.to_json())
```

The example is flagged as `excess_context`: the citation contains two tokens,
while the required span contains one. The record includes both offsets, the
two token counts, the excess token count, and a review flag. The cited words
are never copied into JSON, Markdown, exceptions, or the immutable report.

## Review statuses and context budgets

Each citation receives one of these statuses:

- `minimal`: the citation contains the required span and does not exceed the
  configured `max_excess_tokens` allowance (zero by default).
- `excess_context`: the citation contains the required span but has more
  context tokens than the allowance. The report exposes the left and right
  excess offsets for reviewer navigation.
- `missing_required_span`: the citation does not fully contain the required
  span and is always flagged for review.

`max_excess_tokens` is an explicit policy choice, not a semantic quality
threshold. A non-zero allowance can accommodate a small amount of context,
but the report still exposes its token count and offsets.

The token counter uses only the Python standard library and a fixed Unicode
word/punctuation rule. It performs no model lookup, terminology lookup, or
network call. Invalid offsets, identifiers, and policy values fail closed with
fixed-category errors that do not echo submitted values.

## Privacy and review boundary

Reports contain only opaque claim/citation references, half-open character
offsets, token counts, controlled statuses, and aggregate counts. They do not
contain source text, claim text, prompts, model output, filesystem paths, or
credentials. The offsets let an authorized reviewer navigate the original
source in the caller's controlled system; they are not a substitute for that
review. Citation minimality is not a compliance certification or a guarantee
of clinical correctness.
