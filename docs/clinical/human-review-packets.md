# Human-review packets

`openmed.clinical.review_packet` renders a compact, deterministic evidence
artifact for a human reviewer. It combines typed findings, citations, and
policy or quality-gate results without making a diagnosis, treatment choice,
release approval, or compliance determination.

The renderer is local and mechanical. It does not load a model or make a
network call. Record order is normalized before the packet identifier is
computed, so equivalent inputs produce byte-stable JSON.

## Safe-by-default rendering

Source values belong in the protected fields (`protected_text`, or the
`text`/`value` compatibility aliases on a finding). The default JSON and
Markdown output contains only a `source_hash` and a
`protected_text_available` flag. Citation excerpts and quotes follow the same
policy. Gate details are limited to structured values and omit free-form source
fields.

```python
from openmed.clinical import (
    ReviewCitation,
    ReviewFinding,
    ReviewGateResult,
    build_review_packet,
    render_review_packet,
)

packet = build_review_packet(
    findings=(
        ReviewFinding(
            finding_id="finding-1",
            label="renal_function_measure",
            confidence=0.78,
            uncertainty="uncertain",
            source_start=12,
            source_end=24,
            protected_text="<protected source value>",
            citation_ids=("citation-guideline",),
        ),
    ),
    citations=(
        ReviewCitation(
            citation_id="citation-guideline",
            source="local-guideline",
            locator="section-4",
            title="Synthetic local review guidance",
        ),
    ),
    gate_results=(
        ReviewGateResult(
            gate_id="uncertainty-policy",
            passed=False,
            reason="requires_review",
            severity="warning",
            blocking=True,
        ),
    ),
)

safe_json = render_review_packet(packet)
safe_markdown = render_review_packet(packet, format="markdown")
```

Both default renderings are safe to save in an audit or review queue. The
packet remains an assistive artifact and should be reviewed by a qualified
human before any downstream action.

## Explicit local opt-in

If a local reviewer has a justified need to inspect the protected source, the
caller must opt in on that render operation:

```python
local_json = render_review_packet(
    packet,
    include_protected_text=True,
)
```

The opt-in is intentionally per-render and is never persisted in the packet
object. Callers should keep this output local and avoid placing it in logs,
shared reports, fixtures, or telemetry. The packet object and its default
serializers never expose protected values accidentally.

## Gate interpretation

Gate results describe upstream policy or quality checks. A failed blocking gate
sets `review_status` to `blocked`; a failed non-blocking gate sets it to
`review_required`; passing gates set it to `ready_for_review`. These statuses do
not constitute clinical decisions. When no gate results are supplied, the
status is `not_evaluated`.
