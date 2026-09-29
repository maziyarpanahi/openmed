# Federated update batch preflight

Coordinators can check a bounded batch of anonymous dense adapter updates
against one independently supplied policy without opening tensor values.
`check_federated_update_batch()` reuses the per-envelope update contract for
every submission, groups the surviving schemas, and returns a deterministic
report of counts, reason codes, and per-envelope verdicts.

```python
from openmed.training.federated_update_batch import (
    FederatedUpdateBatchPolicy,
    check_federated_update_batch,
)

# update_policy was built independently of the submitted envelopes.
policy = FederatedUpdateBatchPolicy(update_policy, minimum_group_size=5)
report = check_federated_update_batch(submitted, policy=policy)
print(report.received, report.accepted, report.rejected, report.ok)
```

Every envelope is validated on its own. A malformed envelope is rejected with
`update_schema_invalid` and never stops the remaining submissions. A repeated
declared update digest is rejected with `update_duplicate` after its first
occurrence is accepted. When `expected_fingerprint` is supplied, an envelope
whose schema fingerprint differs is rejected with
`update_fingerprint_mismatch` and its fingerprint is omitted from the report.

## Verdicts and reason codes

| Reason code | Status | Meaning |
| --- | --- | --- |
| `batch_limit_exceeded` | skipped | The submission held more envelopes than `max_updates`. |
| `batch_empty` | skipped | The submission held no envelopes. |
| `batch_group_suppressed` | skipped | One schema group fell below `minimum_group_size`. |
| `batch_digest_withheld` | skipped | At least one accepted digest was not disclosed. |
| `update_accepted` | accepted | The envelope matched the policy and the expected schema. |
| `update_duplicate` | rejected | The declared update digest already appeared in the batch. |
| `update_fingerprint_mismatch` | rejected | The schema fingerprint differed from `expected_fingerprint`. |
| `update_schema_invalid` | rejected | The envelope failed the per-envelope update contract. |

Findings are aggregated per reason code in ascending reason-code order, so a
report shows how many envelopes each reason covers rather than which envelope
produced it.

## Batch limits

`max_updates` bounds one submission and defaults to 64; the hard ceiling is 512.
A submission above the bound returns immediately with `limit_exceeded` set, a
single `batch_limit_exceeded` finding, and no per-envelope outcomes: nothing is
validated and no digest is read. `report.ok` is true only when the batch stayed
within the bound and no envelope was rejected.

## Suppression and digest disclosure

Accepted envelopes are grouped by schema fingerprint. A group smaller than
`minimum_group_size`, which defaults to the shared federated floor, is
suppressed and reported as `batch_group_suppressed`; suppression counts, not
identities, appear in the report. Declared update digests stay out of the report
unless the policy sets `disclose_digests=True`, and even then a suppressed group
withholds its digests: `batch_digest_withheld` reports how many accepted digests
were omitted. Disclosure remains a coordinator decision, because a declared
digest is disclosure-sensitive metadata.

## Privacy

Reports retain no client or site identifier, patient count, path, endpoint,
message, parameter name, tensor, gradient, example, or local metric. An invalid
envelope contributes only its reason code, and the validator never echoes
submitted values, keys, or parse fragments in its error messages. Schema
fingerprints are derived only from the fields the update contract already
validates.

## Determinism

Findings are ordered by reason code and groups by fingerprint, and the
per-envelope outcomes keep submitted order. `to_json()` renders sorted-key
indented JSON with a trailing newline, so the same input bytes produce the same
report bytes; a golden digest in the focused tests pins that output.

## Not in scope

Reading tensor values, identifying submitters, training, secure aggregation, and
automatic client registration are out of scope. A clean batch report says
nothing about tensor contents, clipping truthfulness, privacy guarantees, or
whether a submission should be aggregated.

Run the offline checks with:

```sh
uv run --frozen --extra dev python -m pytest tests/unit/training/test_federated_update_batch.py -q
```
