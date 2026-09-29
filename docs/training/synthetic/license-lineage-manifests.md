# Synthetic dataset license and lineage manifests

Synthetic releases need one reviewable record of where every byte came from.
`validate_license_lineage` checks that record against committed policy before
any artifact is redistributed: which source class produced the dataset, which
license expression covers it, which generator digest and model provenance are
attached, which upstream dependencies contributed, and whether redistribution
is allowed. Every gap fails closed.

```python
from openmed.training.synthetic import (
    LicenseLineageSourceClass,
    RedistributionDecision,
    SyntheticDatasetDependency,
    SyntheticDatasetLicenseManifest,
    SyntheticGeneratorProvenance,
    validate_license_lineage,
    assert_redistribution_allowed,
)

manifest = SyntheticDatasetLicenseManifest(
    dataset_id="synthetic-discharge-summaries",
    source_class=LicenseLineageSourceClass.SYNTHETIC_GENERATED,
    license_expression="Apache-2.0",
    generator=SyntheticGeneratorProvenance(
        generator_id="openmed.synthetic.generator",
        digest="sha256:" + "a" * 64,
        version="1.0.0",
    ),
    redistribution=RedistributionDecision.ALLOWED,
    dependencies=(
        SyntheticDatasetDependency(
            dependency_id="corpus-alpha",
            source_class=LicenseLineageSourceClass.PUBLIC_DOMAIN_CORPUS,
            license_expression="MIT",
            redistribution=RedistributionDecision.ALLOWED,
        ),
    ),
)

report = validate_license_lineage(manifest)
print(report.verdict.value, report.reason_codes)
assert_redistribution_allowed(report)
```

`to_dict()`/`to_json()` render byte-stable evidence: findings are sorted by
field, dependency, and reason code, dependencies are ordered by identifier, and
two runs over the same manifest produce identical JSON.

## Manifest contents

- `dataset_id`: a lowercase identifier naming the release.
- `source_class`: one of `synthetic_generated`, `synthetic_augmented`,
  `model_generated`, `public_domain_corpus`, `licensed_corpus`,
  `manual_curation`.
- `license_expression`: one SPDX short identifier, assessed by
  `assess_license_expression`. Expressions such as `A OR B` are rejected
  instead of evaluated.
- `generator`: generator identifier plus a required `sha256:` content digest
  and an optional version.
- `model_provenance`: optional model identifier, license expression and
  digest; can be made mandatory with
  `SyntheticLicenseLineagePolicy(require_model_provenance=True)`.
- `dependencies`: bounded, ordered records with their own source class,
  license expression, redistribution decision and optional digest.
- `redistribution`: `allowed`, `restricted`, `prohibited`, or `unknown`.

`SyntheticDatasetLicenseManifest.from_dict()` and `from_json()` reject unknown
keys, missing required sections, and non-list dependency payloads. An
unrecognized enum value stays absent on purpose, so validation reports it
instead of raising.

## Failure modes

Every reason code blocks the manifest; there is no "warning" tier.

| Reason code | Trigger |
| --- | --- |
| `source_class_unknown` | source class absent or unrecognized |
| `license_expression_missing` | license expression absent or blank |
| `license_expression_malformed` | expression is not a single SPDX short identifier |
| `license_expression_restricted` | committed restricted identifier (`GPL-3.0-only`, `AGPL-3.0-only`, `CC-BY-NC-4.0`, `SSPL-1.0`, …) |
| `license_expression_unknown` | unlisted identifier or `LicenseRef-` value |
| `generator_digest_missing` | generator absent or without a digest |
| `generator_digest_malformed` | digest is not `sha256:` plus 64 lowercase hex characters |
| `model_provenance_missing` | model provenance required by policy but absent |
| `model_digest_malformed` | model digest present but not a `sha256:` digest |
| `dependency_lineage_missing` | dependency lacks source class, license, or redistribution decision |
| `dependency_digest_missing` | policy requires dependency digests |
| `dependency_digest_malformed` | dependency digest present but not a `sha256:` digest |
| `dependency_limit_exceeded` | dependency count above the policy bound |
| `redistribution_unknown` | redistribution decision absent or unknown |
| `redistribution_restricted` | redistribution explicitly restricted |
| `redistribution_prohibited` | redistribution explicitly prohibited |

Unknown identifiers may be allowed only through the explicit
`SyntheticLicenseLineagePolicy(allow_unknown_licenses=True)` review switch;
restricted and malformed identifiers always block.

## Determinism, bounds, and privacy

- `manifest_digest()` returns the stable content digest of the canonical
  manifest, so reordering dependencies cannot change the recorded digest.
- Bounds: at most `MAX_DEPENDENCIES` (256) dependencies per manifest, 64 by
  default, and license expressions shorter than 128 characters.
- Reports never contain raw manifest text. Unknown, malformed, and `LicenseRef`
  values are reported by closed reason code only, so identifiers, prompts, and
  credentials cannot leak into evidence artifacts.
- `assert_redistribution_allowed()` raises an error whose message contains only
  finding and dependency counts.

Validation is fully offline: it reads no license files, performs no network
lookup, holds no registry, and adds no free-form text to its results.

## Run the offline checks with:

```sh
uv run --frozen --extra dev python -m pytest tests/unit/training/synthetic/test_license_lineage.py -q
```
