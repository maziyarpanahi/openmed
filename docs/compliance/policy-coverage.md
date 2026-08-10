# Privacy-policy coverage matrix

OpenMed can produce a deterministic crosswalk between bundled policy actions,
the canonical label taxonomy, structured-column semantics, synthetic fixture
identifiers, and focused tests. The artifact is a release-review aid; it is not
a compliance certification or a clinical decision guarantee.

## Generate the matrix

Generation uses bundled package resources and identifier-only coverage
bindings. It does not load fixture payloads, inspect user data, or make a
network call.

```python
from pathlib import Path

from openmed.compliance import generate_policy_coverage

result = generate_policy_coverage(Path("audit-evidence/policy-coverage"))
print(result.manifest_path)
print(result.markdown_path)
print(result.matrix.coverage_percent)
```

The destination contains:

- `policy-coverage.json`: a machine-readable matrix with policy resource
  hashes, row identifiers, action names, counts, statuses, and binding hashes.
- `policy-coverage.md`: the same matrix rendered for review.

Repeated runs against the same local policy resources produce identical bytes.
No timestamp, host path, fixture text, span, or source value is included.

## Matrix semantics

Each row is one canonical label action in one bundled policy profile. The
resource path points to the action key in the versioned policy JSON. Rows with
an action other than `keep` are required rules. A required row is covered only
when it has both a synthetic fixture identifier and a focused-test reference;
missing required evidence raises `UncoveredPolicyRuleError` by default.

The matrix also records the structured semantic fields that map to each
canonical label. The structured-field map is hashed independently so adding,
removing, or relabeling a field changes the evidence fingerprint. The bundled
fixture catalog uses category identifiers such as
`synthetic-policy-direct-identifiers`; it deliberately contains no fixture
payloads or sensitive values.

## Review boundary

The matrix proves that policy definitions have declared local evidence links.
It does not measure model recall, certify a jurisdictional control, or replace
focused behavioral tests. Reviewers should inspect the referenced tests and
run the narrow policy-coverage test before relying on the artifact.
