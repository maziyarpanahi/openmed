# Multimodal Processing Summary Diffs

Compare two [processing summaries](processing-summaries.md) without reopening
assets or reading OCR, transcripts, or source content. Pass the earlier summary
first; every numeric delta is **after minus before**.

```python
from openmed.multimodal.processing_diff import (
    diff_processing_summaries,
    render_processing_diff_markdown,
)

# Both inputs are ProcessingSummary values from summarize_processing_run().
difference = diff_processing_summaries(before_summary, after_summary)
print(difference.to_json())
print(render_processing_diff_markdown(difference))
```

`ProcessingDiff` includes signed changes in asset count, bytes, duration,
assets with output digests, per-media asset/byte/page/frame totals, terminal
outcomes, and abstention stage/reason buckets. JSON and Markdown use fixed
field and row order. Media types are alphabetized; outcomes retain their
closed-code order; abstentions are sorted by stage and reason.

Digest changes compare the **multiset of `(input_sha256, output_sha256)` pairs**.
An unchanged pair is omitted. A new or changed pair appears under added
digests, and a missing or replaced pair under removed digests. Repeated pairs
carry a positive `count`. The artifact does not include asset identifiers, file
paths, media content, model output, or free text. A changed digest is evidence
of a changed digest pair, not a measure of clinical quality.

The current diff schema version is `1`. Inputs must be `ProcessingSummary`
values using the current processing summary schema; other types or versions
raise `ProcessingDiffError`. This comparison makes no release decision and
never reads files.
