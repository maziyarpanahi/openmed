# Versioned public clinical-trial records

OpenMed can explicitly synchronize public study metadata from the official
clinical-trial registry API into an integrity-checked local cache. Network
access occurs only when `ClinicalTrialSource.fetch_page()` is called. Reading,
filtering, and inspecting history after synchronization are fully offline.

The synchronization request exposes only public pagination controls. It has no
patient, subject, free-text profile, or clinical-journey input. Do not use this
metadata synchronization boundary for matching a person to a study.

## Fetch and cache a page

```python
from openmed.clinical.trials import ClinicalTrialSource, LocalTrialStore

source = ClinicalTrialSource()
page = source.fetch_page(
    retrieved_at="2026-09-21T08:00:00Z",
    page_size=100,
)
store = LocalTrialStore("./trial-cache")
report = store.apply_page(page)
```

Each cached study version preserves its registry identifier, retrieval time,
source digest, status, last-update date, titles, conditions, interventions,
locations, and raw public eligibility text. Reapplying identical source
content is a no-op. Changed content, including a withdrawn status, appends a
new immutable version.

`TrialSyncReport.next_page_token` can be supplied to the next explicit fetch.
The opaque cursor is never interpreted by OpenMed.

## Offline queries and history

```python
from openmed.clinical.trials import TrialQuery

withdrawn = store.query(TrialQuery(statuses=("WITHDRAWN",)))
history = store.history("NCT00000001")
latest = store.latest("NCT00000001")
```

Queries operate on the latest verified version and can filter by public
status, condition, or location country. Results are sorted by study identifier
for deterministic use. History remains ordered by retrieval time.

## Integrity and compatibility

The cache is written atomically with owner-only permissions. Its envelope,
each source response, and every study version are content-addressed. A modified
cache raises `TrialCacheCorruptionError`; a required upstream field changing
shape raises `TrialSchemaDriftError`; an unsupported stored major version
raises `TrialUnsupportedError`.

The bundled `clinical_trial_study.schema.json` describes version `1.0.0` of the
public study artifact. The cache contains public registry metadata only. It is
not a recommendation, eligibility decision, enrollment action, or substitute
for qualified clinical review.
