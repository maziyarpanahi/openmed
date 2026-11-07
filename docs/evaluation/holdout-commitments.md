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
