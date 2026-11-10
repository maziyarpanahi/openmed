# 21 CFR Part 11 audit trail

`openmed.compliance.part11` emits local, PHI-safe technical evidence for
audit-trail review in workflows that may be subject to 21 CFR Part 11. It
records the actor, action, UTC time, before/after state references, and reason
code. Each state is represented by a bounded label and a `sha256:` digest; raw
source values, replacements, and clinical text are not written to the record.

This feature is an implementation aid, not a certification or legal opinion.
The adopting organization remains responsible for system validation, access
control, identity verification, retention, backup, training, written policies,
and qualified regulatory review. Electronic-signature workflow and identity
provisioning are deliberately out of scope.

## Emit a record offline

Use a safe state reference directly when the digest is already available:

```python
from openmed.compliance import Part11AuditEmitter, hash_state

trail = Part11AuditEmitter()
trail.emit(
    actor_id="reviewer-001",
    action="record.update",
    before_state=hash_state("synthetic-before-state", label="pending"),
    after_state=hash_state("synthetic-after-state", label="approved"),
    reason_code="synthetic-review",
    timestamp="2026-08-04T10:00:00Z",
)

assert trail.verify()
trail.write("part11-audit-trail.json")
```

`hash_state` hashes caller-held material and retains only its digest and label.
Applications that already have a digest can pass a mapping such as
`{"label": "approved", "hash": "sha256:<64 lowercase hex digits>"}`.
State mappings containing `value`, `text`, `raw`, or other uncommitted fields
are rejected.

## Record and chain fields

Each exported record contains:

| Field | Meaning |
| --- | --- |
| `record_id` | Stable non-PHI identifier for the audit event. |
| `actor_id` | Caller-supplied actor or pseudonymous identity reference. |
| `action` | The controlled operation that was performed. |
| `timestamp_utc` | Normalized, timezone-aware UTC timestamp. |
| `before_state` / `after_state` | State `label` plus `sha256:` `hash`; no raw value. |
| `reason_code` | Controlled reason for the operation. |
| `chain_sequence` | Position of the corresponding append-only chain entry. |
| `chain_previous_hash` | Hash of the preceding chain entry. |
| `chain_entry_hash` | Hash of the chain entry containing the record payload. |
| `record_hash` | Independent hash binding the record to its chain reference. |

The JSON artifact also includes the complete `chain`, the exported `records`,
`head_hash`, `trail_hash`, and the readiness crosswalk. `Part11AuditEmitter.verify`
checks the chain links, every record hash, every record-to-chain payload match,
the retained head, and the trail crosswalk. Verification is fully offline.

## CLI export

Create a local JSON input containing safe event objects. The input may be a
list or an object with an `events` array:

```json
[
  {
    "actor_id": "reviewer-001",
    "action": "record.update",
    "timestamp_utc": "2026-08-04T10:00:00Z",
    "before_state": {
      "label": "pending",
      "hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    },
    "after_state": {
      "label": "approved",
      "hash": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
    },
    "reason_code": "synthetic-review"
  }
]
```

Export it locally:

```bash
openmed compliance part11-export events.json \
  --output part11-audit-trail.json
```

The equivalent noun/verb spelling is
`openmed compliance part11 export events.json --output part11-audit-trail.json`.
The command verifies the generated artifact before publication and writes the
output atomically. Add `--json` for a machine-readable command result and
`--overwrite` when replacing an existing file is intentional.

## Part 11 readiness crosswalk

The machine-readable source is
`openmed.compliance.PART11_READINESS_CHECKLIST`; the same crosswalk is embedded
in every exported artifact. `partial` means the emitter supplies technical
evidence but the deployment must complete the control. `external` means the
control is primarily organizational or explicitly outside this feature.

| Clause | Emitter field mapping | Boundary |
| --- | --- | --- |
| 11.10(a) | `record_hash`, `trail_hash` | Validation evidence is external. |
| 11.10(b) | `chain`, `records`, `record_hash` | Deterministic electronic copy is supplied. |
| 11.10(c) | `chain_entry_hash`, `chain_previous_hash`, `trail_hash` | Retention and storage controls are external. |
| 11.10(d) | `actor_id` | Authorization and provisioning are external. |
| 11.10(e) | `actor_id`, `action`, `timestamp_utc`, `before_state`, `after_state`, `reason_code`, `chain_entry_hash` | Primary emitter coverage. |
| 11.10(f) | `action`, `reason_code`, `chain_sequence` | Workflow sequencing is caller-owned. |
| 11.10(g) | `actor_id`, `record_id` | Authority checks are external. |
| 11.10(h) | `actor_id` | Device checks are external. |
| 11.10(i) | `actor_id` | Training evidence is external. |
| 11.10(j) | `actor_id`, `action`, `reason_code` | Written policies are external. |
| 11.10(k) | `record_hash`, `trail_hash` | Documentation control is external. |
| 11.30 | `record_hash`, `chain_entry_hash`, `trail_hash` | Confidentiality controls are deployment-owned. |
| 11.50 | `actor_id`, `timestamp_utc`, `action` | Electronic signatures are out of scope. |
| 11.70 | `record_id`, `chain_entry_hash` | Signature linking is out of scope. |
| 11.100 | `actor_id` | Identity provisioning and signature uniqueness are out of scope. |
| 11.200 | `actor_id`, `record_id` | Signature components are out of scope. |
| 11.300 | `actor_id` | Credential lifecycle is deployment-owned. |

The crosswalk intentionally makes gaps visible. A valid hash chain proves
integrity of this artifact; it does not prove that an organization has met
every Part 11 procedural or operational requirement.
