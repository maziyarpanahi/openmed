# SDOH Longitudinal Status Resolution

`resolve_sdoh_longitudinal_status()` groups value-free observations into dated
episodes. Observations recorded for the same category and effective time remain
together, including contradictory statuses and every source reference.

```python
from datetime import datetime, timezone

from openmed.clinical.sdoh_deduplicate import SDOHSourceReference
from openmed.clinical.sdoh_longitudinal import (
    SDOHStatusObservation,
    resolve_sdoh_longitudinal_status,
)

observation = SDOHStatusObservation(
    observation_id="observation-1",
    category="food_access",
    status="secure",
    effective_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
    source=SDOHSourceReference("document-local-1", "v1", 8, 20),
    supports_current=True,
)

result = resolve_sdoh_longitudinal_status([observation])[0]
assert result.current_status == "secure"
```

An episode with more than one status emits `same_time_status_conflict` and has
no resolved status. If that episode is latest, the result also emits
`latest_episode_unresolved`. An unambiguous latest episode still does not become
the current status unless at least one source explicitly sets
`supports_current=True`; otherwise `no_current_support` is emitted.

This prevents stale historical notes from being promoted to a current finding.
It does not infer that the latest document is clinically correct. The output is
deterministic, contains no source text, and is intended for human review rather
than eligibility, diagnosis, or treatment decisions.
