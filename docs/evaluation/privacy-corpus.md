# Synthetic privacy regression corpus

`openmed.eval.privacy_corpus` defines a small, versioned contract for privacy
regression cases. It keeps synthetic fixture coverage, policy expectations, and
severity in one deterministic manifest without committing fixture source text.

The manifest is an evaluation artifact only. It is not a compliance
certification, a clinical decision rule, or a guarantee of de-identification
quality.

## Manifest contents

Each manifest contains:

- `cases`: a case identifier, coverage category, policy profile, severity,
  fixture length, SHA-256 fixture hash, and expected finding counts;
- `policy_profiles`: required categories, required severities, and the
  expected aggregate critical-leakage count for each policy;
- `required_categories`: the categories that the complete corpus must cover;
- `manifest_hash`: a SHA-256 hash of the canonical manifest contents; and
- `synthetic_only: true`, which is required by validation.

Cases contain no `text` or arbitrary metadata fields. Use
`make_privacy_case()` when a local synthetic fixture needs to be registered:

```python
from openmed.eval.privacy_corpus import (
    PrivacyFindingExpectation,
    PrivacyPolicyProfile,
    build_privacy_corpus_manifest,
    make_privacy_case,
)

case = make_privacy_case(
    "direct_identifier_case",
    {"text": "synthetic fixture text", "spans": []},
    category="direct_identifier",
    policy_profile_id="strict_redaction",
    severity="critical",
    expected_findings=(
        PrivacyFindingExpectation(
            finding_id="critical_leakage",
            severity="critical",
            expected_count=0,
            critical_leakage=True,
        ),
    ),
)
profile = PrivacyPolicyProfile(
    profile_id="strict_redaction",
    required_categories=("direct_identifier",),
    required_severities=("critical",),
)
manifest = build_privacy_corpus_manifest([case], [profile])
```

The fixture is used only while `make_privacy_case()` computes its content hash.
The returned case and manifest retain the hash and length, never the source
text. Hashing is canonical: mapping key order does not affect the result, and
changing fixture content changes the hash.

## Validation and offline loading

`default_privacy_corpus_manifest()` returns the built-in metadata-only
synthetic registry. `load_privacy_corpus_manifest()` returns the same registry
without a path, or loads a local JSON manifest when given a path. Neither path
performs a network request.

Use `validate_privacy_corpus_manifest()` before consuming a manifest. It
checks schema and synthetic-only declarations, unique identifiers, SHA-256
integrity, profile references, category and severity coverage, and aggregate
critical-leakage expectations. The returned `PrivacyCoverageReport` contains
only identifiers, categories, severities, and counts, so it is suitable for
logs and reports.

```python
from openmed.eval.privacy_corpus import (
    default_privacy_corpus_manifest,
    validate_privacy_corpus_manifest,
)

report = validate_privacy_corpus_manifest(default_privacy_corpus_manifest())
assert report.valid
```

Do not add real patient text, credentials, restricted datasets, or raw
identifiers to a fixture or to a manifest field. Keep committed fixtures
synthetic and local-first; the hash is the provenance handle for any
out-of-band fixture content.
