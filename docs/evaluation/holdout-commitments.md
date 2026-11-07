# Sealed holdout commitments

Evaluators can prove that benchmark cases, labels, templates, and
randomization inputs were fixed before submissions were inspected without
publishing the live holdout. OpenMed represents each private manifest as an
ordered Merkle tree of item digests and publishes only the four roots, item
counts, a public holdout version, and a commitment digest. All operations are
local and perform no network requests.

## Prepare private manifests

Create one ordered digest sequence for each governed manifest kind:

- `case`
- `label`
- `template`
- `randomization`

Each item must be a lowercase `sha256:` digest. Canonicalize items before
hashing them. For labels, short templates, or other low-entropy values, hash a
canonical envelope containing an evaluator-held random nonce. A digest is a
commitment, not encryption; unsalted low-entropy inputs may be guessable. Keep
the source values and nonces at the evaluator site.

The four sequences are ordered. The item index is domain-separated into every
leaf, so changing item order changes the root and duplicate item digests remain
individually provable.

## Publish a pre-submission commitment

Use a public, non-sensitive version identifier and commit the digest-only
manifests:

```python
from openmed.eval.governance import commit_holdout_manifests

manifests = {
    "case": ("sha256:" + "1" * 64,),
    "label": ("sha256:" + "2" * 64,),
    "template": ("sha256:" + "3" * 64,),
    "randomization": ("sha256:" + "4" * 64,),
}
commitment = commit_holdout_manifests("benchmark-2026.09", manifests)
published_json = commitment.to_json()
```

Publish or timestamp `published_json` before accepting submissions. It contains
no item digests or raw holdout values. Store the private manifests separately;
they are required to generate later audit proofs.

The schema fixes SHA-256, leaf and internal-node domain separation, canonical
JSON, and duplicate-last handling for odd Merkle levels. Any schema change must
use a new schema version rather than silently changing those rules.

## Selectively prove an included item

During an authorized audit, use the original private manifest to disclose a
proof for only one indexed item:

```python
from openmed.eval.governance import (
    create_inclusion_proof,
    verify_inclusion_proof,
)

proof = create_inclusion_proof(commitment, "case", 0, manifests["case"])
verification = verify_inclusion_proof(commitment, proof)
assert verification.valid
```

Proof generation first recomputes the selected manifest root and refuses to
produce a proof if the private sequence does not match the published
commitment. The proof reveals the selected item digest, its index, and the
minimum sibling-digest path. It does not reveal other item digests or contents.

Auditors can verify parsed JSON mappings as well as the Python objects.
Verification fails closed and returns only stable reason codes; it does not
echo malformed caller input. Log the verification report, not private
manifests or raw proof source values.

Merkle commitments establish pre-publication integrity and selective
membership. They do not establish dataset quality, regulatory compliance, or
an autonomous clinical decision guarantee.

## Federated patient-level site holdouts

Federated sites reuse the same commitment primitives to fix a patient-level
train/holdout split before the first training round. A site assigns each
opaque patient key with a keyed HMAC-SHA256 under a site-held secret, so every
note of one patient lands in the same partition and the assignment is
reproducible only with that secret. Holdout membership is committed as keyed
membership tags; the split policy, federation binding, and a keyed secret
fingerprint fill the remaining manifests.

```python
from openmed.training.federated_holdout_split import (
    FederatedHoldoutLedger,
    FederatedHoldoutSplitError,
    HoldoutRotationReason,
    build_site_holdout_split,
)

notes = [
    (f"note-{index}-{part}", f"patient-{index:03d}")
    for index in range(40)
    for part in range(3)
]
split = build_site_holdout_split(
    [patient for _, patient in notes],
    site_secret=b"site-held-secret-of-32-bytes-min",
    federation_id="fed-2026-q4",
    holdout_fraction=0.25,
)
train, holdout = split.partition_examples(notes, patient_key_of=lambda n: n[1])
evidence = split.export_evidence()
assert set(evidence.to_dict()) == {
    "commitment_digest",
    "holdout_patient_count_band",
    "schema_version",
    "train_patient_count_band",
}
```

`export_evidence()` is the only artifact that may leave the site. It contains
the commitment digest and coarse patient-count bands (`suppressed` below the
minimum group size, then `under_100`, `100_to_999`, `1000_to_9999`, and
`10000_or_more`). Patient keys, note identifiers, membership tags, exact
counts, and the site secret stay local; the split object excludes them from
`repr` and refuses pickling. Callers canonicalize patient keys before building
the split; keys are not normalized.

The training port admits a batch only when every example belongs to the
committed train partition. A batch containing a committed holdout member is
refused in full with the stable code `holdout_member_in_training`, and patients
that were not part of the committed split are refused with
`unassigned_patient`:

```python
split.admit_training_examples(train, patient_key_of=lambda n: n[1])
try:
    split.admit_training_examples(
        [*train, holdout[0]], patient_key_of=lambda n: n[1]
    )
except FederatedHoldoutSplitError as error:
    assert error.code.value == "holdout_member_in_training"
```

The coordinator keeps a digest-only `FederatedHoldoutLedger`. Training for a
site may start only after `require_active_commitment()` returns its digest.
Re-registering the same digest is idempotent; any different split, including
one built with a new secret, fraction, cohort, or epoch, fails closed with
`recommitment_without_rotation` until `record_rotation()` records a controlled
reason. A retired digest can never become active again
(`retired_commitment_reused`). Evaluation evidence must cite the active digest
in `holdout_commitment_digest`; citing a different commitment fails with
`evaluation_commitment_mismatch`, and results that cite a rotated-out digest
fail with `holdout_commitment_rotated`:

```python
ledger = FederatedHoldoutLedger("fed-2026-q4")
ledger.register("site-a", evidence.to_dict())
result = {"holdout_commitment_digest": ledger.require_active_commitment("site-a")}
assert ledger.verify_evaluation_evidence("site-a", result).valid

ledger.record_rotation("site-a", HoldoutRotationReason.COHORT_REFRESH)
rotated = build_site_holdout_split(
    [patient for _, patient in notes],
    site_secret=b"site-held-secret-of-32-bytes-new",
    federation_id="fed-2026-q4",
    holdout_fraction=0.25,
    rotation_epoch=1,
)
ledger.register("site-a", rotated.export_evidence())
stale = ledger.verify_evaluation_evidence("site-a", result)
assert stale.reason_codes == ("holdout_commitment_rotated",)
```

`to_json()` and `from_dict()` persist the ledger as digests, site references,
and rotation reasons only; restoring tampered state fails with
`invalid_ledger_state`. The split governs evaluation integrity only. It does
not grant training authorization, establish cohort representativeness, or
trigger any clinical action.
