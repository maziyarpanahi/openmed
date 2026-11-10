# Structured review override reasons

`openmed.clinical.review_override_reasons` records why a human reviewer changed
or confirmed an automated result. Schema version 1 defines five stable codes:
`accept`, `correct`, `reject`, `defer`, and `insufficient_evidence`.

```python
from openmed.clinical.review_override_reasons import create_review_override

override = create_review_override(
    "insufficient_evidence",
    local_note="Available evidence did not support a final decision.",
)

safe_event = override.to_telemetry_dict()
local_record = override.to_local_dict()
```

The optional note is intended for a protected local review store. It is hidden
from `repr()` and omitted from `to_dict()`, `to_telemetry_dict()`,
`to_aggregate_dict()`, and `aggregate_review_overrides()`. Only the explicit
`to_local_dict()` method includes note content. Applications must keep that
local record under their own access, retention, and encryption controls.

Aggregation reports only the schema version, total records, and counts for all
five reason codes. The implementation is deterministic and performs no network
calls. These records support human review; they do not certify compliance or
authorize autonomous clinical decisions.
