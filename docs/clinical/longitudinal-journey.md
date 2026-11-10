# Point-in-Time Longitudinal Journey

OpenMed materializes a longitudinal patient journey directly from immutable
artifacts, evidence locators, clinical facts, conflicts, resolution events,
canonical pointers, and review packets. The journey is a derived query view;
it does not create a second event database or copy source bytes.

The API is deterministic and local after the caller has supplied any required
assets. It is assistive infrastructure for review and application workflows,
not an autonomous clinical decision system.

## Query the current journey

```python
from openmed.clinical.journey import JourneyQuery, query_journey

result = query_journey(
    store,
    JourneyQuery(
        subject_id="subject_aaaaaaaaaaaaaaaa",
        limit=100,
    ),
)

if result.ok:
    page = result.value
    assert page is not None
    for event in page.events:
        print(event.event_type, event.journey_state)
```

`store` implements the backend-neutral `JourneyQueryStore` protocol. The
bundled SQLite and PostgreSQL stores expose the same point-in-time read
surface.

Non-success outcomes remain typed:

- `unknown` means the subject or requested snapshot is not present;
- `partial` means a requested time filter cannot be proven because a fact has
  no usable time or its immutable provenance is incomplete;
- `conflict` means a cursor belongs to different filters or the materialized
  ordering graph is contradictory;
- `unsupported` means a stored fact cannot be represented by the v1 event
  type contract;
- `denied` preserves a storage-policy denial;
- `failure` identifies invalid or corrupt persisted state.

Callers must not reinterpret any of these states as an empty successful
journey.

## Stable snapshots and pagination

The first successful query binds the response to the store's latest committed
revision and returns a `JourneySnapshot`. The snapshot contains an opaque,
subject-bound `snapshot_id` plus the exact revision used for every read.

Pass that snapshot and `next_cursor` into the next query:

```python
from openmed.clinical.journey import JourneyQuery, query_journey

next_result = query_journey(
    store,
    JourneyQuery(
        subject_id=page.snapshot.subject_id,
        snapshot=page.snapshot,
        limit=100,
        cursor=page.next_cursor,
    ),
)
```

The cursor is opaque and bound to the snapshot, filters, and page size. A
cursor cannot be reused with changed filters. Later ingestion or corrections
do not alter results read through an older snapshot.

## Filters

`JourneyQuery` supports bounded filters for:

- encounter identifiers;
- inclusive time overlap;
- event type;
- clinical fact status;
- source identifier;
- review state;
- materialized journey state.

The v1 typed event set is `condition`, `medication`, `procedure`,
`laboratory`, `observation`, `encounter`, and `social_determinant`. The aliases
`lab`, `labs`, `sdoh`, and `social` normalize to their canonical types.

Time filters accept ISO dates, partial year or month values, and timezone-aware
datetimes. Partial dates are compared through conservative bounds while their
original precision remains unchanged on the fact. Missing time is not guessed:
a time-constrained query returns `partial/journey_time_unknown`.

## Deterministic ordering without invented precision

Events are ordered by their conservative time bounds, declared precision, and
opaque fact identifier. Equal timestamps use the fact identifier as a stable
tie-breaker. That tie-breaker changes only presentation order; it does not emit
a temporal edge or claim that one equal-time event occurred first.

Unknown-time events remain in a deterministic trailing bucket when no time
filter is applied.

## Current, historical, conflicted, and corrected facts

`journey_state` distinguishes:

- `current`: selected by the canonical pointer, or an unreconciled fact with
  no conflict or superseding correction;
- `historical`: rejected by the latest resolution or superseded by a later
  correction;
- `conflicted`: included in a conflict whose latest resolution still defers,
  reopens, branches, or is absent.

Correction lineage is separate from current-state selection:

- `none` has no correction relationship;
- `amends` is a correction of an earlier fact;
- `superseded` has a later correction;
- `amends_and_superseded` is an intermediate fact in a correction chain.

This separation keeps a corrected current fact distinguishable from the
historical fact it supersedes without mutating either record.

## Provenance drill-down

Every `JourneyEvent` carries:

- the complete immutable `ClinicalFact`, including its derivation hash and
  schema version;
- every `EvidenceLocator` referenced by that fact;
- the corresponding `ClinicalArtifact` metadata and source identifier;
- every conflict involving the fact;
- append-only resolution history with policy versions and supersession links;
- the visible canonical pointer version and store revision;
- current review states from migrated review-packet versions.

The artifact path contains content hashes and coordinates, not source bytes.
Applications can use the content-addressed artifact store for an explicitly
authorized source lookup.

## Evidence-linked timeline graph

Each page includes one value-free `EvidenceLinkedTimelineGraph`. Graph nodes
reference the same event and fact identifiers returned on that page; they do
not duplicate clinical values.

Edges record:

- `chronological_precedes` only when conservative time bounds prove strict
  ordering;
- `corrects` for immutable parent-to-correction lineage;
- `relation.<role>` for normalized fact relation participants.

Every edge retains evidence identifiers, a derivation hash, and an ordering
basis. Chronological and correction cycles are rejected as a typed conflict.
Equal or overlapping time bounds do not create a chronological edge.

## Privacy-safe operational reporting

`JourneyPage.to_dict()` is the authorized clinical payload and therefore can
contain clinical fact values. Do not write it to ordinary logs, traces, or
metrics.

Use `JourneyPage.to_safe_dict()` for operational reporting. It contains only
schema versions, event counts, journey-state counts, and pagination presence;
it excludes subject, event, fact, source, evidence, and clinical values.

Committed tests and fixtures are synthetic. Credentials, restricted
vocabularies, data-use-agreement corpora, raw patient records, and source bytes
are never bundled by this feature.

## Compatibility

The page, snapshot, event, and graph contracts declare schema version `1.0.0`
and compatibility policy `same_major`. The bundled
`journey_view.schema.json` validates serialized pages. Persisted source
records retain their own explicit contract versions; the journey does not
rewrite them during materialization.
