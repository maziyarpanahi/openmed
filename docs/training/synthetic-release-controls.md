# Synthetic release controls

`openmed.training.synthetic.governance` decides whether a synthetic artifact,
such as a generated dataset or a split derived from one, may be released. It
checks disclosure, lineage, licenses, withdrawal, benchmark contamination,
downstream regression, and subgroup quality in one local call, and it fails
closed: a missing control is a failure, not a pass.

Every control consumes plain data. Lineage comes from the existing
[`openmed.compliance.artifact_lineage`](../compliance/artifact-lineage.md)
manifests, and the
other inputs are license records, generator and model digests, benchmark
canary and overlap counts, and per-slice success counts. Decisions and reports
carry digests, controlled identifiers, closed codes, counts, and rates. They
never include record text, real records, license values, canary strings, or the
names of suppressed groups, and validation makes no network calls.

These controls do not show that synthetic data is anonymous or that it is
clinically equivalent to any real population. They are not legal advice.

## Release blockers

`validate_synthetic_release()` returns a `SyntheticReleaseDecision` whose
`passed` flag is true only when no control produced a finding. Each finding is
a closed code with a count:

| Control | Blocking codes |
| --- | --- |
| `label` | `label_missing`, `intended_use_missing`, `intended_use_prohibited`, `generator_disclosure_missing`, `model_digest_missing` |
| `lineage` | `lineage_missing`, `lineage_integrity_failed`, `lineage_artifact_absent`, `lineage_ancestor_missing`, `generator_not_in_lineage` |
| `license` | `license_missing`, `license_conflict`, `license_not_permitted`, `license_redistribution_disallowed` |
| `withdrawal` | `artifact_withdrawn` |
| `contamination` | `benchmark_evidence_missing`, `benchmark_evidence_mismatch`, `benchmark_evidence_empty`, `required_benchmark_missing`, `benchmark_canary_detected`, `benchmark_overlap_exceeded` |
| `regression` | `regression_evidence_missing`, `regression_evidence_insufficient`, `critical_regression` |
| `subgroup` | `subgroup_report_missing`, `subgroup_evidence_insufficient`, `subgroup_disparity_exceeded` |

`assert_synthetic_release()` raises `SyntheticReleaseBlockedError` with the
codes only and attaches the decision. Malformed inputs raise
`SyntheticGovernanceError`; error messages name fields or positions, never
values.

## Synthetic labels

`SyntheticArtifactLabel` is the machine-readable label. It binds the
artifact's lineage node hash and content digest to:

- `content_kind`: `fully_synthetic` or `partially_synthetic`.
- `intended_uses` and `prohibited_uses` from the closed `SyntheticUse`
  vocabulary. `clinical_decision_making`, `population_inference`,
  `real_data_substitution`, and `reidentification` are always prohibited and
  cannot be removed. An intended use that is also prohibited blocks release.
- `generators`: one `GeneratorDisclosure` per generator with its identifier,
  version, `method`, code and configuration digest, lineage node, and model
  weight digests. `statistical_model`, `neural_model`, and `hybrid` generators
  must disclose at least one model digest.

`to_dict()` always adds `"synthetic": true` and three fixed disclosures:
`anonymity_not_guaranteed`, `not_clinically_equivalent_to_real_population`,
and `not_real_patient_data`. `from_mapping()` and `from_json()` reject labels
that are not marked synthetic, alter the disclosures, or carry unknown fields.
`label_digest` commits to the canonical label.

## Lineage and licenses

The artifact's node must be present in an `ArtifactLineageManifest` whose
integrity check passes, every ancestor must be present, and each generator's
`lineage_node_hash` must be an ancestor of the artifact.

Every node in the artifact's lineage closure, including the artifact itself,
needs a `LicenseRecord`. A license passes when the permissive SPDX allowlist
used by `normalize_spdx_identifier()` accepts it, or when it exactly matches
(ignoring case) one of `SyntheticReleasePolicy.permitted_data_licenses`, which
defaults to `CC-BY-4.0` and `CC0-1.0`. Unknown identifiers, `LicenseRef-`
values, expressions, and copyleft or share-alike licenses fail. Every record
must also allow redistribution, and two different records for one node fail as
a conflict. `evidence_digest` can point at a license and lineage manifest.

## Withdrawal

`propagate_withdrawal()` takes a lineage manifest and `WithdrawalNotice`
values. A notice carries a node hash, a closed `WithdrawalReason`, and an
optional ISO effective date; free text is rejected. Propagation follows every
parent reference, so a malformed edge can only widen a withdrawal. The
returned `WithdrawalEvidence` lists each invalidated node with its type, its
distance from the withdrawn node, the withdrawn hashes that reach it, and the
reason codes. Notices for nodes outside the graph are counted as unresolved
and still invalidate their own hash. Evidence is deterministic regardless of
notice order and has an `evidence_digest`.

During validation, an artifact that is withdrawn directly or descends from a
withdrawn node fails with `artifact_withdrawn`. Withdrawing a descendant, such
as a model trained on the artifact, does not block the artifact itself.

## Benchmark contamination

`BenchmarkContaminationEvidence` records, for one benchmark version, how many
artifact records were checked, how many overlap the benchmark, how many
canaries were scanned for, and how many were found. It names the artifact
content digest it was measured on, and evidence for other content fails.

`measure_benchmark_contamination()` builds that evidence locally. Benchmark
owners publish `fingerprint_record()` digests instead of records: text is
NFKC-normalized, case-folded, and whitespace-collapsed before hashing, so the
overlap is exact after normalization. Canary strings must have at least eight
characters. Any canary hit blocks release, and overlap blocks release when its
rate exceeds `max_benchmark_overlap_rate` (default `0.0`).
`required_benchmarks` lists benchmarks that must have evidence. Near-duplicate
and memorization audits are separate.

## Slices, suppression, and uncertainty

Downstream regression and subgroup quality use proportion metrics given as
success and trial counts per slice. The whole set is the `overall` axis with
the `all` group.

Slices with fewer trials than the declared `minimum_cell_size` (default `11`,
matching the common rule of not reporting cells of 1 to 10) are suppressed.
When exactly one group on an axis is suppressed, the next smallest group on
that axis is suppressed too, so the hidden group cannot be recovered by
subtraction. Suppressed slices are counted, never named. Released slices
report a coarse count band instead of exact counts:

| Band | Trials |
| --- | --- |
| `minimum_to_under_double` | At least the minimum and less than twice it |
| `double_to_under_fourfold` | At least twice and less than four times the minimum |
| `fourfold_or_more` | At least four times the minimum |

`build_subgroup_quality_report()` reports each released slice with a Wilson
score interval and, per axis, the gap between the highest and lowest released
groups. `gap_supported` is true only when the two intervals do not overlap.
Disparities are reported only, unless `max_subgroup_gap` is set; then a
supported gap above it blocks release. A missing report, or one where every
slice is suppressed, blocks release unless `require_subgroup_report` is false.

`build_downstream_regression_report()` compares the candidate trained with the
artifact against a baseline for each task, metric, and slice, using Newcombe's
hybrid score interval for the difference. A slice is critical when the
candidate's point estimate drops by at least `critical_regression_drop`
(default `0.05`). Slices hidden only by complementary suppression still count
toward critical regressions; slices below the minimum cell size cannot be
assessed. Each task and metric needs an assessable `overall` slice, otherwise
the evidence is insufficient.

## Example

```python
from openmed.compliance.artifact_lineage import (
    ArtifactLineageNode,
    build_artifact_lineage_manifest,
)
from openmed.training.synthetic.governance import (
    DownstreamComparison,
    GeneratorDisclosure,
    LicenseRecord,
    SubgroupMetric,
    SyntheticArtifactLabel,
    fingerprint_record,
    measure_benchmark_contamination,
    validate_synthetic_release,
)

generator = ArtifactLineageNode.create(
    artifact_type="synthetic_generator", transformation="configure"
)
dataset = ArtifactLineageNode.create(
    artifact_type="synthetic_dataset",
    parents=[{"type": generator.artifact_type, "hash": generator.artifact_hash}],
    transformation="generate",
)
lineage = build_artifact_lineage_manifest([generator, dataset])
content_digest = "sha256:" + "a" * 64  # digest of the released bytes

label = SyntheticArtifactLabel(
    artifact_hash=dataset.artifact_hash,
    content_digest=content_digest,
    content_kind="fully_synthetic",
    generators=[
        GeneratorDisclosure(
            generator_id="openmed.locale_phi",
            generator_version="1.0.0",
            method="template",
            generator_digest="sha256:" + "b" * 64,
            lineage_node_hash=generator.artifact_hash,
        )
    ],
    intended_uses=["model_training"],
)
evidence = measure_benchmark_contamination(
    ["Synthetic note: follow-up in two weeks."],
    benchmark_id="openmed.privacy_corpus",
    benchmark_digest="sha256:" + "c" * 64,
    artifact_content_digest=content_digest,
    benchmark_fingerprints=[fingerprint_record("Benchmark placeholder row.")],
    canaries=["openmed-canary-7f3c9e21"],
)
decision = validate_synthetic_release(
    label,
    lineage,
    licenses=[
        LicenseRecord(node.artifact_hash, "Apache-2.0", redistribution_allowed=True)
        for node in (generator, dataset)
    ],
    benchmark_evidence=[evidence],
    regression=[DownstreamComparison("pii_ner", "recall", 90, 100, 89, 100)],
    subgroup_metrics=[SubgroupMetric("label_validity", "overall", "all", 95, 100)],
)
print(decision.passed, decision.codes)
```

## Binding points

- `SyntheticReleaseDecision.to_capsule_check()` returns the
  `status`/`checked_count`/`evidence_digest` shape of a release-capsule check,
  with `pass` or `fail` and a bare hexadecimal digest, so a capsule can commit
  to the decision. The label's `content_digest` is the artifact commitment a
  capsule records.
- `SyntheticReleaseDecision.to_gate_check()` returns an
  `openmed.eval.release_gates.GateCheck` named `synthetic_release_controls`
  carrying only finding codes and the evidence digest.
- `LicenseRecord.evidence_digest` can hold the digest of a license and lineage
  manifest; `GeneratorDisclosure.generator_digest` and `model_digests` are the
  same SHA-256 commitments a capsule pins for its generators.
- Fidelity scores and split plans can feed `SubgroupMetric` and
  `DownstreamComparison` counts, and a memorization finding can be recorded as
  a `memorization_detected` withdrawal.
