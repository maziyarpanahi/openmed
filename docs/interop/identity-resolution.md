# Patient and encounter identity resolution

OpenMed provides a conservative local resolver for linking opaque source-local
patient and encounter keys to opaque canonical keys. Exact links are evaluated
deterministically. Optional candidate plugins may propose possible links, but
their output is always `ambiguous` and always requires a recorded review.

The resolver does not inspect names, addresses, dates of birth, medical record
numbers, or free text. It does not automatically merge uncertain identities.
It is an identity-resolution building block, not a clinical identity service or
a substitute for deployment-specific stewardship and validation.

## Outcome contract

Every successful resolution has one of four explicit states:

| State | Canonical key | Candidate keys | Review required |
| --- | --- | --- | --- |
| `matched` | exactly one | exactly the selected key | no |
| `unmatched` | none | none | no |
| `ambiguous` | none | one or more plugin candidates | yes |
| `conflict` | none | two or more exact-link candidates | yes |

`ambiguous` and `conflict` never expose a selected canonical key. Callers must
not treat either state as a match. The Python contracts and bundled JSON Schema
both enforce these invariants.

Operational failures use the shared `StoreResult` states: `partial`, `unknown`,
`conflict`, `unsupported`, `denied`, and `failure` remain distinct from success.
For example, a policy review is `partial`, an absent resolution is `unknown`, a
newer storage schema is `unsupported`, and a candidate-plugin failure remains a
`failure`; none silently becomes `unmatched`.

## Use opaque source-local keys

`SourceIdentityKey` contains an entity type, source identifier, and local key.
All three identifiers are controlled or opaque tokens:

```python
from openmed.interop.identity import SourceIdentityKey

source_key = SourceIdentityKey(
    entity_type="patient",
    source_id="source_0123456789abcdef",
    local_key="local_fedcba9876543210",
)
```

Do not put a raw medical record number, account number, name, email address, or
other patient value in these fields. Derive opaque keys before calling the
resolver, preferably with a deployment-owned keyed pseudonymization function.
Keep the secret outside OpenMed records, logs, fixtures, and configuration
checked into source control. A plain unsalted hash of a small identifier space
is not sufficient pseudonymization.

Patient and encounter keys occupy separate entity domains. An encounter link
cannot satisfy a patient request, even if its opaque source values happen to be
the same.

## Create an exact-link registry

The built-in store is local SQLite with owner-only filesystem permissions,
ordered migrations, foreign keys, full synchronization, and write-ahead
logging:

```python
from openmed.clinical.journey_contracts import canonical_digest
from openmed.interop.identity import (
    ExactIdentityResolver,
    IdentityLink,
    IdentityResolutionRequest,
    IdentityResolutionStore,
)

store = IdentityResolutionStore("./identity/registry.sqlite3")
link = IdentityLink(
    link_id="link_0123456789abcdef",
    source_key=source_key,
    canonical_key="patient_0123456789abcdef",
    evidence_digest=canonical_digest({"receipt": "synthetic-v1"}),
    policy_id="openmed.identity.default",
    policy_version="1.0.0",
    recorded_at="2026-01-02T03:04:05Z",
)
assert store.add_link(link).ok

request = IdentityResolutionRequest(
    request_id="request_0123456789abcdef",
    entity_type="patient",
    source_keys=(source_key,),
    purpose="care",
    role="clinician",
    attributes=("identified_access",),
    policy_id="openmed.identity.default",
    policy_version="1.0.0",
    requested_at="2026-01-02T04:04:05Z",
)
result = ExactIdentityResolver(store).resolve(request)
assert result.ok
assert result.value is not None
assert result.value.state == "matched"
```

The default access policy routes resolution through the identified-projection
boundary. It requires an allowed role and the `identified_access` attribute.
Denied or review-required policy decisions occur before link lookup and before
a resolution is persisted.

Resolution identity is deterministic for the request content, policy version,
resolver version, candidates, and evidence. Reprocessing the same state returns
the same resolution and an idempotent `created=False` result.

## Optional candidate plugins

An `IdentityCandidatePlugin` can integrate a caller-owned local matcher. The
plugin receives the value-safe request and returns candidate keys, basis-point
scores, evidence digests, and versioned plugin provenance. It must not place
patient values in those fields.

```python
from openmed.interop.identity import (
    CompositeIdentityResolver,
    ProbabilisticIdentityCandidate,
)
from openmed.structured.store import StoreResult


class LocalCandidates:
    def candidates(self, request):
        candidate = ProbabilisticIdentityCandidate(
            canonical_key="patient_0123456789abcdef",
            score_basis_points=9750,
            evidence_digest=canonical_digest({"features": "local-v1"}),
            plugin_id="local.identity.matcher",
            plugin_version="1.0.0",
        )
        return StoreResult.success((candidate,))


resolver = CompositeIdentityResolver(
    ExactIdentityResolver(store),
    LocalCandidates(),
)
result = resolver.resolve(request)
assert result.value is not None
assert result.value.state == "ambiguous"
assert result.value.review_required
assert result.value.canonical_key is None
```

Exact links always run first. A plugin runs only when exact resolution is
`unmatched`. One candidate with a perfect score is still ambiguous: thresholds
can rank review work, but cannot authorize a merge. Duplicate candidates are
canonicalized deterministically. Plugin denial, unsupported behavior, or
failure is returned unchanged and does not create a resolution.

## Record a review decision

An uncertain resolution remains isolated until `apply_review()` commits one
explicit decision:

- `confirm_match` selects an existing candidate;
- `merge` selects an existing candidate and replaces conflicting active links;
- `split` creates a new canonical key that is not among the candidates;
- `confirm_unmatched` deactivates the involved exact links without selecting a
  replacement.

```python
from openmed.interop.identity import IdentityReviewDecision

decision = IdentityReviewDecision(
    decision_id="decision_0123456789abcdef",
    resolution_id=result.value.resolution_id,
    action="confirm_match",
    selected_canonical_key="patient_0123456789abcdef",
    reviewer_digest=canonical_digest({"reviewer": "local-steward-role"}),
    evidence_digest=canonical_digest({"review": "synthetic-receipt-v1"}),
    policy_id=result.value.policy_id,
    policy_version=result.value.policy_version,
    decided_at="2026-01-02T05:04:05Z",
)
assert store.apply_review(decision).ok
```

The store writes the review, deactivates prior links, and adds replacement links
in one immediate transaction. A candidate mismatch, policy-version mismatch,
repeated different review, or invalid split rolls back without changing active
links. Review records are append-only. Active status is a separate storage
projection so historical link payloads and their integrity hashes do not need
to be rewritten.

Reviewer and review evidence fields accept SHA-256 digests only. They are
provenance anchors, not authorization by themselves. The surrounding
application must authenticate reviewers, authorize the action, retain the
protected source receipt separately, and apply its retention policy.

## Persistence and recovery

Use `IdentityResolutionStore.open()` when startup must return typed failure
states instead of raising:

```python
opened = IdentityResolutionStore.open("./identity/registry.sqlite3")
if not opened.ok:
    raise RuntimeError(opened.code)
store = opened.value
```

The store refuses symlinked database paths and newer or checksum-drifted
migrations. `integrity_check()` reparses every record, recomputes canonical
hashes, and runs SQLite foreign-key checks. Run it after restore or unexpected
shutdown. It detects storage corruption; it does not prove that a source link
was clinically or administratively correct.

Back up the database and its `-wal` file using a SQLite-aware snapshot process.
Protect the directory as identified data even though public payloads contain
only opaque identifiers and digests. Keyed pseudonyms may still be linkable and
must follow the deployment's access, deletion, and retention controls.

## Schemas and local verification

The six public records use schema version `1.0.0`: source key, request,
evidence, resolution, link, and review decision. Their JSON Schemas ship in the
package and load without network access:

```python
from openmed.interop.identity import load_all_identity_schemas

schemas = load_all_identity_schemas()
assert schemas["resolution"]["schema_version"] == 1
```

The synthetic tests cover duplicates, source collisions, amendments, splits,
merges, deterministic reprocessing, transaction rollback, policy denial,
plugin failure, restart recovery, migration refusal, schema validation, and
absence of raw-value canaries in persisted payloads. A committed golden fixture
also exercises the public request, exact-link, policy, store, resolver, evidence,
and outcome contracts together:

```bash
pytest tests/unit/interop/identity/test_identity_resolution.py -q
```

All fixtures are synthetic. A deployment should add representative evaluation
outside the repository, including source-system collisions, identifier reuse,
late amendments, household similarity, twins, cross-facility transfer, and
reviewer disagreement. Do not use a match for autonomous clinical action
without deployment-specific validation and human oversight.
