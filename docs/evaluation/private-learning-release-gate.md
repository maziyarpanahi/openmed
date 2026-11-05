# Private-learning release gate

`openmed.eval.suites.private_learning_release` turns the private-learning
release definition of done into one signed, offline-verifiable evidence
packet. It follows the conventions of the
[v3.1 agent release gates](v3.1-agent-gates.md) and the exact-source candidate
packet: canonical JSON, SHA-256 digests, closed reason codes and an
HMAC-SHA256 signature from a caller-controlled key.

The evaluator is metadata-only. It never imports the evidence producers
(protocol profile, site lifecycle, chaos suite, DP utility study, language
cohort, synthetic controls, compatibility matrix and release-assurance
artifacts). Each producer is referenced only by its evidence digest and schema
version, so the gate does not wait on unmerged work. It performs no file,
process or network I/O.

A `READY` packet is not release authority. Every packet carries
`release_authority: "not_granted"`. Tagging, publishing and creating a GitHub
release still require separate explicit authorization after an exact-SHA
release-candidate audit.

## Definition-of-done catalog

The catalog is closed and versioned per release. Release `v3.3` contains the
following items, in this order. The private-learning workstream was later
regrouped into a separate release plan. A new catalog version would record
that scope. The gate does not change silently.

| Item | Priority | Privacy invariant |
| --- | --- | --- |
| `mvp_scope_closed` | P0 | no |
| `protocol_profile` | P0 | yes |
| `site_lifecycle` | P0 | yes |
| `multi_node_chaos` | P0 | no |
| `dp_utility_study` | P0 | yes |
| `language_cohort` | P0 | no |
| `synthetic_release_controls` | P0 | yes |
| `compatibility_matrix` | P1 | no |
| `promotion_privacy_gate` | P0 | yes |
| `promotion_leakage_gate` | P0 | yes |
| `promotion_utility_gate` | P0 | no |
| `promotion_language_gate` | P0 | no |
| `promotion_drift_gate` | P0 | no |
| `promotion_lineage_gate` | P0 | no |
| `promotion_evidence_gate` | P0 | no |
| `release_notes` | P0 | no |
| `migration_guidance` | P0 | no |
| `api_protocol_diff` | P0 | no |
| `security_privacy_evidence` | P0 | yes |
| `release_candidate_manifest` | P0 | no |

The promotion gates are separate items on purpose. The release plan says no
private-learning model is promoted unless privacy, leakage, utility, language,
drift, lineage and evidence gates each pass separately. The packet never
collapses items into a score, a pass rate or a weighted total.

## Manifest

The manifest uses schema
`openmed.eval.private_learning_release_manifest.v1`. Every object has exactly
the documented fields. Unknown fields, free text and malformed values raise
`PrivateLearningReleaseError("manifest_invalid")`, or
`release_unsupported` for a release without a catalog. The messages are fixed
codes and never echo input.

```json
{
  "schema_version": "openmed.eval.private_learning_release_manifest.v1",
  "release": {"release_id": "v3.3", "git_commit": "<40 or 64 lowercase hex>"},
  "policy": {"max_evidence_age_seconds": 86400},
  "items": [
    {
      "item_id": "protocol_profile",
      "priority": "P0",
      "privacy_invariant": true,
      "non_compensable": true,
      "required_schema_version": "openmed.federated.protocol_profile.v1",
      "required_evidence_digest": "sha256:<64 lowercase hex>"
    }
  ],
  "moved_work": [
    {
      "work_ref": "compatibility_matrix",
      "priority": "P1",
      "privacy_invariant": false,
      "target_milestone": "v4.2",
      "rationale_code": "blocked_by_external_dependency",
      "rationale_digest": "sha256:<digest of the recorded rationale>"
    }
  ]
}
```

The values above are placeholders, not release evidence. List `items` in
catalog order, with every catalog item that has not been moved. Each item must
repeat the catalog priority and privacy flag, and declare
`non_compensable: true`.

`moved_work` records work that leaves this release. `work_ref` is a catalog
item ID or a public work reference such as `gh-3877`. `rationale_digest` binds
the rationale recorded in the release plan. `rationale_code` is one of
`blocked_by_external_dependency`, `deferred_by_scope_policy`,
`split_into_later_milestone` or `superseded_by_later_work`.

## Evidence entries

Each producer supplies one entry per item, signed with
`sign_private_learning_evidence(entry, evidence_key)`:

```json
{
  "schema_version": "openmed.eval.private_learning_release_evidence.v1",
  "item_id": "protocol_profile",
  "evidence_schema_version": "openmed.federated.protocol_profile.v1",
  "evidence_digest": "sha256:<64 lowercase hex>",
  "source_commit": "<40 or 64 lowercase hex>",
  "generated_at": "2026-11-01T10:00:00Z",
  "status": "passed",
  "signature": "sha256:<HMAC-SHA256 over the other fields>"
}
```

Entries may arrive in any order. An entry with an unknown field, such as a
site ID, path, endpoint, example, tensor or free-form message, is refused as
`evidence_invalid`. The packet never echoes the refused value.

## Evaluation

```python
from openmed.eval.suites.private_learning_release import (
    PrivateLearningReleasePacket,
    evaluate_private_learning_release,
)

packet = evaluate_private_learning_release(
    manifest,
    evidence,
    source_commit=checkout_commit,
    evidence_key=evidence_key_bytes,
    signing_key=packet_key_bytes,
    clock=utc_clock,
)
stored = packet.to_json()

restored = PrivateLearningReleasePacket.from_json(stored)
assert restored.verify(packet_key_bytes)
```

`source_commit` is the full commit of the evaluated checkout, for example
from `git rev-parse HEAD`. Keys are caller-controlled bytes of at least 32
bytes. The evidence key and the packet key may differ. `clock` must return a
timezone-aware datetime, and the packet records it in UTC seconds.

Every item is evaluated independently. One failure never masks another, and
all applicable codes are listed. An item passes only when it has exactly one
entry and all of these conditions hold:

- the entry is signed with the evidence key;
- it matches the manifest digest and schema version;
- it was produced for the manifest commit;
- it is no older than `max_evidence_age_seconds`;
- it is not from the future; and
- its status is `passed`.

The decision is `READY` only when every required item passes, every move is
accepted, and no release-level blocker exists.

### Item reason codes

| Code | Meaning |
| --- | --- |
| `evidence_missing` | No entry for the item |
| `evidence_duplicate` | More than one entry; all are excluded |
| `evidence_invalid` | Entry violates the closed evidence schema |
| `evidence_unsigned` | Entry carries no signature |
| `evidence_signature_invalid` | Signature does not verify (tampered or wrong key) |
| `evidence_digest_mismatch` | Evidence digest differs from the manifest |
| `evidence_schema_mismatch` | Evidence schema version differs from the manifest |
| `evidence_commit_mismatch` | Entry was produced for another commit |
| `evidence_stale` | Entry is older than the manifest policy allows |
| `evidence_from_future` | Entry is newer than the evaluation clock |
| `evidence_failed` | Producer reported a failed result |
| `item_undeclared` | Catalog item is missing from the manifest and not validly moved |
| `item_declared_twice` | Item appears more than once in the manifest |
| `item_contract_mismatch` | Manifest priority or privacy flag differs from the catalog |
| `item_compensable` | Manifest does not declare the item non-compensable |
| `item_move_refused` | A refused move tried to remove this item |

### Release reason codes

| Code | Meaning |
| --- | --- |
| `release_commit_mismatch` | Manifest commit differs from the evaluated checkout |
| `manifest_item_order_invalid` | Manifest items are not in catalog order |
| `manifest_item_duplicate` | A manifest item is repeated |
| `manifest_item_unknown` | A manifest item is not in the catalog |
| `evidence_invalid` | An entry cannot be attributed to an item |
| `evidence_unexpected_item` | An entry names an unknown or moved item |
| `moved_work_refused` | At least one moved-work row was refused |

### Moved-work reason codes

Moved P1 and P2 work is accepted only with a recorded rationale and a later
milestone. A move can never remove a P0 or privacy-invariant item. The
catalog priority and privacy flag win over the manifest's declaration, so
relabeling a P0 item as P1 is also refused. A refused move keeps the item
required, and the item fails as `item_undeclared` plus `item_move_refused`.

| Code | Meaning |
| --- | --- |
| `moved_rationale_missing` | Rationale code or digest is absent |
| `moved_rationale_invalid` | Rationale code is not in the closed set (not echoed) |
| `moved_target_not_later` | Target milestone is not after the release |
| `moved_p0_refused` | Moved work is P0 |
| `moved_privacy_invariant_refused` | Moved work protects a privacy invariant |
| `moved_contract_mismatch` | Declared priority or privacy flag differs from the catalog |
| `moved_item_still_declared` | Moved catalog item is still listed in `items` |
| `moved_duplicate` | The same work is moved more than once |

## Packet

The packet schema is `openmed.eval.private_learning_release_packet.v1`. It
contains:

- the release ID, manifest commit, observed source commit and UTC evaluation
  time;
- the manifest, catalog and evidence-set digests;
- one row per catalog item, in catalog order, with its priority, privacy
  flag, status (`passed`, `failed` or `moved`), reason codes, required digest
  and schema version, and the digest of the accepted entry; and
- the moved-work rows with acceptance and reason codes.

`packet_digest` hashes the canonical unsigned packet. `signature` is
HMAC-SHA256 over every field including `packet_digest`, formatted as
`sha256:<64 lowercase hex>`.

`from_dict` and `from_json` enforce the closed schema. They reject duplicate
JSON keys, NaN, reordered items, unknown reason codes, unsigned packets and
decisions that disagree with the item rows. `verify` recomputes the digest
and HMAC with constant-time comparisons. Verification needs only the packet
and the key: no Git, network or model provider. HMAC needs the shared secret,
so it is not a public-key attestation.

## Privacy limits

Packets contain only closed codes, digests, commit SHAs, schema identifiers,
milestone names and UTC timestamps. They contain no raw PHI, site identities,
key material, private paths, rationale text or producer payloads. Schema
identifiers must match `openmed.<segments>.v<N>`. Use public schema names for
them, never site or patient labels. The bundled tests run a sentinel scan over
packets built from inputs seeded with synthetic identifying values. Digests
bind content, not its origin or clinical validity. Producers and their
evidence stores remain the caller's responsibility.
