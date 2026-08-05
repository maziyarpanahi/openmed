# Structured Release Risk and Expert-Review Evidence

Removing names, medical-record numbers, and other direct identifiers is
necessary, but it does not by itself make a structured dataset safe to release.
Combinations such as age, geography, dates, facility, and rare diagnoses can
still distinguish a person. OpenMed provides a local workflow to discover those
quasi-identifiers, measure disclosure risk over explicit privacy units,
generalize or suppress records, validate the materialized result, and prepare
aggregate evidence for a qualified expert.

!!! warning "Decision support, not a determination"

    OpenMed's output supports qualified expert review. It is **not** an Expert
    Determination, compliance certification, legal conclusion, or guarantee of
    zero re-identification risk. The qualified expert remains responsible for
    the population, release context, recipients, auxiliary data, composition
    with other releases, methods, thresholds, and documented conclusion.

The workflow is offline and uses CSV, TSV, JSONL/NDJSON, or Parquet files. It
does not send records to a service.

## The release workflow

Use the stages in this order:

1. **Inventory direct identifiers and de-identify free text.** Declare
   structured direct identifiers so the release workflow removes them, and run
   narrative or image content through its separate de-identification path.
2. **Discover candidate roles.** Run a bounded, advisory scan to profile column
   roles and candidate quasi-identifier sets.
3. **Review roles.** A reviewer confirms direct identifiers,
   quasi-identifiers, sensitive attributes, free-text columns, and the privacy
   unit for the intended release.
4. **Assess the complete dataset.** Measure k-anonymity, l-diversity, and
   t-closeness over all declared privacy units.
5. **Assess an explicit reference population when one is available.** Measure
   exact k-map, exact-linkage risk, and delta-presence against a
   caller-supplied table whose attack-population relevance and containment
   assumptions have been reviewed.
6. **Anonymize.** Search reviewed generalization hierarchies and, within an
   explicit budget, suppress complete privacy units.
7. **Reread and validate.** Reopen the materialized release and verify row
   counts, identifier removal, prior residual-policy validation, and a typed
   content digest where the format preserves scalar types.
8. **Prepare and gate expert-review evidence.** Export deterministic aggregate
   JSON and Markdown, verify evidence integrity, and use the technical CI gate
   before review.
9. **Let the expert author any attestation.** An expert can sign their own
   identity, qualifications, methodology, conclusion, reassessment time, and
   evidence bindings. OpenMed never supplies or infers those statements.

Do not skip role review because discovery found no candidate. A no-candidate
result is reported as `insufficient-discovery`; it is not evidence that the
dataset is safe.

## What the roles mean

Discovery preserves overlapping candidate roles so a reviewer can see
ambiguity instead of losing it. The release policy is stricter: every source
column must receive an explicit disposition, and one physical column cannot be
both a quasi-identifier and a sensitive attribute in the current enforcement
model. Resolve that ambiguity before assessment, or create separately reviewed
derived columns whose semantics support the two analyses.

| Role | Meaning in this workflow |
| --- | --- |
| `direct-id` | Directly identifies a person and must not enter the release. |
| `internal-linkage` | Links rows inside the trusted environment, such as a patient or encounter key. |
| `quasi-id` | Can distinguish or link people when combined with other data. |
| `sensitive` | Attribute whose within-class disclosure is measured with l-diversity and t-closeness. |
| `free-text` | Narrative content requiring a separate text de-identification path. |
| `safe` | Reviewed as outside the other roles for this release context. |

The original single `role` field remains in discovery output for compatibility.
The `roles` and `column_role_sets` fields preserve overlaps. Explicit `--role`,
`--qi`, `--sensitive`, and `--privacy-unit` choices take precedence over
heuristics. For assessment and anonymization, use `--non-sensitive` for
reviewed columns that remain in the release and `--exclude` for reviewed
columns that must be removed. Direct identifiers and the privacy-unit key are
also removed. An unclassified column stops the workflow.

Comma-separated flags remain convenient for simple schemas. Repeat the literal
forms `--qi-column`, `--sensitive-column`, `--direct-id-column`,
`--non-sensitive-column`, and `--exclude-column` when a reviewed column name
contains a comma. These forms also preserve spaces and Unicode exactly. They
can be combined with their comma-separated counterparts, and duplicate names
are removed before the policy canonicalizes them deterministically.

## Privacy models and exact semantics

OpenMed does not choose a universal regulatory threshold. The caller must set
`k` and should set `l` and `t` explicitly after expert review.

### k-anonymity

`k` is the smallest equivalence-class size for the declared
quasi-identifiers. When `--privacy-unit patient_id` is supplied, class size is
the number of distinct patients, not the number of rows. Repeated encounters
therefore do not inflate k.

For keyed longitudinal data, each privacy unit is represented by a
deterministic sorted multiset of its **joint row-level QI tuples**. This keeps
the relationship between QI values that occurred in the same row and retains
repeated-row multiplicity. Two patients with crossed age/facility pairings, or
with different numbers of otherwise identical encounters, therefore do not
collapse into the same class. Hierarchy transformations are applied to each
profile value and the complete joint multiset is canonicalized again before
post-transform assessment.

Every row must contain a non-empty privacy-unit value for full assessment.
Privacy-unit identifiers and quasi-identifier cells use their exact typed
published representation for grouping. OpenMed does not silently merge
Unicode normalization variants, case variants, alternate timezone-offset
spellings, or signed-zero spellings. Normalize such values before assessment
only under a reviewed data-cleaning rule, because representation differences
can themselves be linkable.

Suppression removes all rows belonging to a suppressed privacy unit, and the
privacy-unit column is removed before release output is written.

### l-diversity

OpenMed supports two variants:

- `--l-metric distinct --l 2` requires at least two distinct values for each
  declared sensitive attribute in every equivalence class.
- `--l-metric entropy --l 2` requires Shannon entropy of at least
  `log2(2) = 1` bit for each declared sensitive attribute in every class.
  More generally, the integer `L` supplied to `--l` becomes a threshold of
  `log2(L)` bits.

Missing, null, or empty sensitive values are rejected. Apply and document an
explicit preprocessing policy before assessment; OpenMed will not turn
missingness into an extra diversity category.

For l-diversity, semantic aliases such as Unicode canonical equivalents and
equal timezone-aware instants collapse conservatively so aliases cannot inflate
the distinct-value or entropy claim. For t-closeness, OpenMed also measures the
exact published sensitive representations; representation variants correlated
with an equivalence class therefore remain visible as distribution shift.

When one privacy unit has multiple values for a sensitive attribute, OpenMed
does not treat the whole value set as one diversity category. A nontrivial
l-diversity or t-closeness claim fails closed until the data is normalized to
one reviewed value per privacy unit or evaluated with a dedicated
multi-valued disclosure model.

### t-closeness

OpenMed currently uses variational distance, also called total-variation
distance. `--t 0.2` requires every equivalence class's sensitive-attribute
distribution to be no more than `0.2` from the complete release distribution.
Smaller thresholds are stricter; `1.0` is the least restrictive accepted
threshold.

The release assessment also reports exact-match sample identity-risk
aggregates. It does not by itself estimate population uniqueness or risk
against external auxiliary datasets.

### Exact reference-population assessment

When a reviewed reference table is available, `assess_population_risk()`
performs an exact, offline comparison between the intended sample and that
table. It does not download population data and does not extrapolate beyond
the supplied rows.

For a sample profile with frequency \(f\) and reference-population frequency
\(F\):

- **k-map** is the smallest \(F\) among sample profiles. The configured
  `target_k_map` requires every sample profile to occur at least that many
  times in the reference population.
- **Exact-linkage risk** is \(1/F\), reported as maximum and mean risk over
  sample units.
- **Delta-presence** is \(f/F\), also reported as maximum and mean. The
  configured maximum must be between `0` and `1`.

Both thresholds are mandatory in the Python API and CLI. OpenMed does not
silently treat `k-map = 1` or `delta-presence = 1` as an acceptable policy.

With privacy-unit keys, the sample and reference tables use the same joint
longitudinal multiset semantics described above; the two key columns may have
different names. Without keys, each row is one analysis unit.

The model has two mandatory assumptions: the supplied table represents the
anticipated attack population for the declared QIs, and the sample units are
contained in that population under compatible row-level or keyed semantics.
The assessment fails closed when a sample profile is absent or when \(f > F\).
An absent profile receives `achieved_k_map = 0` and conservative linkage and
delta-presence risk of `1`; a frequency inconsistency remains visible even
when its computed \(f/F\) exceeds `1`. Scalar types and published
representations must be compatible across both tables.

The JSON result is aggregate-only: it includes counts, rates, policy verdicts,
and separate sample, reference, schema, policy, and integrity digests. It does
not serialize profile keys, cell values, privacy-unit identifiers, or source
paths. The integrity digest detects mutation; use an external signature when
authenticity or provenance matters.

### Longitudinal cross-document linkage

`longitudinal_risk_report()` measures whether stable evidence can cluster two
or more de-identified notes back to the same privacy unit. Run it only after
free text and structured fields have passed direct-identifier de-identification.
Each input note needs a stable source-side patient key and may provide age,
rare-condition, and surrogate evidence in its fields or audit spans. The key is
converted to an HMAC-SHA256 patient pseudonym before it enters the report; note
identifiers and evidence values are also HMAC digests. Supply an installation-
specific HMAC key through a secret-management channel and do not publish it.

The current estimator deliberately uses a conservative patient-level bound. A
patient with fewer than two notes, or without a stable attack fingerprint, has
a bound of `0.0`. A patient with reusable surrogate evidence, an age trajectory,
or rare-attribute evidence has a bound of `1.0`. The report's
`linkage_success_upper_bound` is the maximum patient bound, so adding documents
cannot lower it. This is a safety upper bound, not a calibrated probability or
the observed success rate of a particular attacker.

```python
from openmed.risk import longitudinal_risk_report

linkage_report = longitudinal_risk_report(
    deidentified_notes,
    hmac_key=longitudinal_hmac_key,
    patient_key_fields=("patient_id",),
)
```

The report schema is versioned independently of the dashboard:

| Field | Meaning |
| --- | --- |
| `schema_version` | Currently `1`. Consumers must reject unknown versions. |
| `patient_count`, `document_count` | Number of pseudonymous patients and de-identified notes assessed. |
| `linkage_success_upper_bound` | Maximum conservative patient-level bound. |
| `mean_patient_linkage_upper_bound` | Arithmetic mean of the patient bounds. |
| `linkable_patient_count` | Patients whose bound is greater than zero. |
| `residual_direct_identifier_leakage` | Fraction of notes with detected residual direct identifiers. |
| `residual_direct_identifier_leakage_count` | Total residual direct-identifier findings. |
| `patient_risks` | Per-patient hashed breakdown used to validate the aggregate values. |
| `high_risk_patients` | Rows tied at the nonzero maximum bound. Empty when the maximum is zero. |

Each `patient_risks` row contains `patient_pseudonym`, document, evidence, and
direct-identifier counts, its linkage bound, aggregate stable-surrogate, age,
rare-attribute, and category counts, a hashed attack fingerprint, and hashed
evidence references. Evidence references contain a note index, note and value
digests, and optional start/end offsets and section metadata. They never contain
the evidence value. The dashboard and signed gate apply a narrower allowlist:
only digests, offsets, counts, and scores are retained; category, source,
section, and any unknown fields are discarded.

Add the panel to either existing HTML dashboard with the optional
`longitudinal` argument. The aggregate release dashboard is the appropriate
choice for expert-review handoff:

```python
from pathlib import Path

from openmed.risk import render_release_assessment_dashboard

dashboard_html = render_release_assessment_dashboard(
    release_assessment,
    longitudinal=linkage_report,
)
Path("longitudinal-release-dashboard.html").write_text(
    dashboard_html,
    encoding="utf-8",
)
```

The panel shows corpus counts and upper bounds, then the highest-risk cohort and
its hashed evidence. It validates every displayed digest, integer, offset, and
probability and ignores all other input fields. Omitting `longitudinal` preserves
the previous dashboard output.

For release, first run the focused check, then attach the same report to the
complete benchmark evidence under `metrics.longitudinal_linkage_risk` and use
the normal signed release gate. The default linkage ceiling is zero; a caller
may choose another ceiling only under a documented release policy.

```python
from openmed.eval import ReleaseGate, evaluate_cross_document_linkage_gate

linkage_check = evaluate_cross_document_linkage_gate(
    linkage_report,
    ceiling=approved_linkage_ceiling,
)
if not linkage_check.passed:
    raise RuntimeError(linkage_check.reason)

benchmark_payload = benchmark_report.to_dict()
benchmark_payload["metrics"]["longitudinal_linkage_risk"] = linkage_report
signed_verdict = ReleaseGate(
    cross_document_linkage_ceiling=approved_linkage_ceiling,
    signing_key=release_signing_key,
).evaluate(benchmark_payload, baseline_evidence)
if not signed_verdict.verify(release_signing_key):
    raise RuntimeError("Release-gate signature verification failed")
```

A passing bound means only that no modeled fingerprint exceeded the configured
ceiling in the supplied corpus. It does not model cross-institution linkage,
unknown auxiliary data, errors in the patient key, or attacks using signals the
extractor did not capture. An empty highest-risk cohort is not proof of zero
re-identification risk. Keep the report and dashboard under the same release-
governance controls as other hashed evidence, because stable digests can still
support correlation.

## Run the synthetic example

The repository includes a fully offline walkthrough with fabricated patients:

```bash
DEMO_ROOT="$(mktemp -d)"
python examples/structured_release_risk.py \
  --output-dir "$DEMO_ROOT/python"
```

The script writes a synthetic source table, advisory discovery manifest,
pre-release assessment, validated release, output-validation report, and
expert-review evidence. Its final summary should show:

```json
{
  "discovery_advisory": true,
  "expert_review_evidence_verified": true,
  "materialized_release_valid": true,
  "post_release_meets_policy": true,
  "pre_release_meets_policy": false,
  "qualified_expert_review_required": true
}
```

See
[`examples/structured_release_risk.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/structured_release_risk.py)
for the complete Python implementation.

The companion population-risk example uses a fabricated sample and fabricated
reference population:

```bash
python examples/structured_population_risk.py
```

It prints aggregate exact k-map, linkage-risk, and delta-presence evidence plus
the assessment digest. See
[`examples/structured_population_risk.py`](https://github.com/maziyarpanahi/openmed/blob/master/examples/structured_population_risk.py)
for the complete input and policy.

## CLI walkthrough

The following commands reuse the example's synthetic cohort. Use a new output
directory because release commands do not overwrite artifacts by default.

```bash
SOURCE="$DEMO_ROOT/python/synthetic-cohort.jsonl"
CLI_OUT="$DEMO_ROOT/cli"
mkdir "$CLI_OUT"
```

### 1. Advisory discovery

```bash
openmed risk discover "$SOURCE" \
  --output "$CLI_OUT/qi-discovery.json" \
  --sample-rows 10000 \
  --privacy-unit patient_id \
  --qi age,zip,visit_date \
  --sensitive disease \
  --include-safe-candidates \
  --role full_name=direct-id \
  --role encounter_id=direct-id,internal-linkage
```

The output contains schema names and aggregate counts, not the source path,
cell values, record identifiers, equivalence-class keys, or low-entropy value
hashes. Sampled discovery is always marked advisory. `--full-scan` reads all
rows and marks dataset coverage complete, but it still does not replace role
review. By default, bounded combination search uses columns already identified
as QI or sensitive candidates. `--include-safe-candidates` also searches
reviewed scalar columns currently classified as safe, which can find a risky
combination whose individual columns look innocuous. Direct identifiers,
internal-linkage columns, and free text remain excluded from that broader
candidate scope.

For a schema containing commas, spaces, or Unicode, use repeatable literal
flags instead of placing those names in a comma-separated value:

```bash
openmed risk discover "$SOURCE" \
  --output "$CLI_OUT/literal-qi-discovery.json" \
  --privacy-unit patient_id \
  --qi-column "Age, grouped" \
  --qi-column "Région de visite" \
  --sensitive-column "Diagnostic principal"
```

Review at least:

- the selected privacy unit and whether repeated rows represent one person;
- externally linkable demographics, dates, geography, facilities, and rare
  clinical attributes;
- columns discovered as both `quasi-id` and `sensitive`, which must be resolved
  before constructing the enforcement policy;
- direct identifiers, internal linkage keys, and free text;
- truncated candidate columns, exhausted search budgets, and other
  completeness metadata.

### 2. Complete pre-release assessment

After the reviewer confirms roles and chooses thresholds:

```bash
openmed risk assess "$SOURCE" \
  --output "$CLI_OUT/pre-release-assessment.json" \
  --dashboard "$CLI_OUT/pre-release-dashboard.html" \
  --qi age,zip,visit_date \
  --sensitive disease \
  --direct-id full_name,encounter_id \
  --privacy-unit patient_id \
  --k 2 \
  --l 2 \
  --l-metric distinct \
  --t 0
```

`risk assess` reads the complete table. Its JSON contains aggregate class-size,
identity-risk, l-diversity, and t-closeness evidence. It omits raw
equivalence-class keys, members, privacy-unit values, and sensitive values. It
writes that aggregate assessment but exits `1` when the declared policy is not
met; the synthetic cohort is intentionally in that state before anonymization.
Inspect that expected result before continuing. Other runtime failures also use
a nonzero exit and must not be ignored.

The optional `--dashboard` is a self-contained HTML view of the same
aggregate-only assessment. It contains no raw rows, QI keys, privacy-unit
identifiers, or sensitive values. Treat schema names and all derived evidence
according to the release-governance policy even when the artifact is designed
for safe expert handoff.

### 3. Compare with an explicit reference population

Run an exact population comparison only after reviewing the reference table
and its relationship to the intended release:

```bash
SAMPLE="/trusted/intended-sample.parquet"
REFERENCE="/trusted/reference-population.parquet"

openmed risk population-assess "$SAMPLE" "$REFERENCE" \
  --output "$CLI_OUT/population-risk.json" \
  --qi age_band,region \
  --sample-privacy-unit sample_patient_key \
  --population-privacy-unit population_person_key \
  --k-map 5 \
  --max-delta-presence 0.2
```

Use repeated `--qi-column` flags when a literal QI name contains a comma.
Both privacy-unit flags must be supplied together, or both omitted for
row-level analysis. The command writes aggregate evidence even when a
configured threshold is missed, then exits `1`. Missing profiles, incompatible
scalar types, inconsistent schemas, and sample profile frequencies greater
than reference frequencies fail closed rather than being extrapolated.

### 4. Anonymize, reread, validate, and produce evidence

Create a local reviewed-assumptions note. Its contents are hashed into the
evidence binding but are never copied into the shareable evidence:

```bash
cat > "$CLI_OUT/reviewed-assumptions.md" <<'EOF'
Population, recipient, release, and reasonably available auxiliary-data
assumptions were reviewed for this synthetic walkthrough.
EOF

openmed risk anonymize "$SOURCE" \
  --output "$CLI_OUT/validated-release.jsonl" \
  --evidence "$CLI_OUT/expert-review-evidence.json" \
  --evidence-markdown "$CLI_OUT/expert-review-evidence.md" \
  --dashboard "$CLI_OUT/post-release-dashboard.html" \
  --qi age,zip,visit_date \
  --sensitive disease \
  --direct-id full_name,encounter_id \
  --privacy-unit patient_id \
  --k 2 \
  --l 2 \
  --l-metric distinct \
  --t 0 \
  --max-suppression-rate 0 \
  --max-lattice-nodes 100000 \
  --max-suppression-subsets 100000 \
  --privacy-unit-kind patient \
  --population-scope release_cohort \
  --release-model restricted \
  --recipient-model named_researchers \
  --auxiliary-data-model reasonably_available \
  --assumptions-notes "$CLI_OUT/reviewed-assumptions.md"
```

The command performs full residual assessment before identifier removal, writes
the release to a private staging file, rereads it, and validates the staged
artifact. It then stages the expert-review JSON and Markdown plus any requested
dashboard and publishes every requested output as one rollback-safe
transaction, with the sensitive release published last. It fails closed if the
hierarchy search exceeds
`--max-lattice-nodes` or `--max-suppression-subsets`, the suppression budget is
insufficient, the transformed data misses a configured privacy target, the
release is empty, output paths alias an input or one another, or output
validation fails. Within those explicit bounds, search is exhaustive unless
the first exact candidate has zero information loss and therefore reaches the
mathematical lower bound. That exact proof prunes irrelevant remaining nodes
and subsets while recording `optimality_proven = true`, `complete = false`,
and an unknown total suppression-subset count. The evidence records candidate
totals and limits; a budget overrun fails without claiming an optimum.

Any `other` or `other_documented` release-context choice requires
`--assumptions-notes`; the notes file remains local and only its binding digest
appears in the expert-review bundle.

Built-in hierarchies cover age bands, date coarsening, and geography
coarsening. Arbitrary facilities, categories, and clinical codes use only
exact-or-suppressed levels unless the reviewer supplies a semantic hierarchy
with `--hierarchies reviewed-hierarchies.json`. OpenMed does not infer clinical
parents from string prefixes. Every supplied hierarchy must start with a
canonical identity level with loss `0`, no mappings, and no default. Coarsening
levels must have positive loss, and hierarchy outputs cannot use OpenMed's
reserved internal namespace.

Verify the aggregate evidence:

```bash
openmed compliance expert-review-verify \
  "$CLI_OUT/expert-review-evidence.json"
```

Verification proves that the deterministic evidence structure and integrity
hash agree. It does not add an expert conclusion or turn the artifact into an
Expert Determination.

Run the same strict parser and technical release gate in CI:

```bash
openmed risk gate "$CLI_OUT/expert-review-evidence.json"
```

The gate exits `0` only when evidence integrity, exhaustive-search completion
or an exact zero-loss lower-bound proof, the configured k/l/t policy,
post-transform results, and required composition metadata pass. Malformed or
mutated evidence and failed technical policy return a nonzero exit. A passing
gate remains technical evidence, not release authorization or an Expert
Determination. Evidence schema version 3 adds the explicit
`search.optimality_proven` field. Version 2 evidence remains parseable, but an
incomplete pruned proof requires version 3 to pass the gate.

## Python API

The same workflow is available without the CLI:

```python
from pathlib import Path

from openmed.compliance import (
    ReleaseAssumptions,
    build_release_expert_review_evidence,
)
from openmed.core.audit import stable_hash
from openmed.risk import (
    AnonymityPolicy,
    anonymize_release,
    assess_population_risk,
    assess_release,
    validate_released_output,
)
from openmed.structured import read_table, scan_table, write_table

source = Path("synthetic-cohort.jsonl")

discovery = scan_table(
    source,
    privacy_unit="patient_id",
    quasi_identifier_columns=("age", "zip", "visit_date"),
    sensitive_columns=("disease",),
    include_safe_candidates=True,
)
# A human reviews discovery before constructing the release policy.
policy = AnonymityPolicy(
    quasi_identifiers=("age", "zip", "visit_date"),
    sensitive_attributes=("disease",),
    direct_identifiers=("full_name", "encounter_id"),
    privacy_unit="patient_id",
    target_k=2,
    target_l=2,
    l_metric="distinct",
    target_t=0.0,
    max_lattice_nodes=100_000,
    max_suppression_subsets=100_000,
)

rows = read_table(source)  # complete, in-memory assessment
before = assess_release(rows, policy)
result = anonymize_release(rows, policy)

release_path = Path("validated-release.jsonl")
write_table(release_path, result.records)
reread = read_table(release_path)
validation = validate_released_output(reread, result)
if not validation.passed:
    raise RuntimeError("Materialized release failed validation")

assumptions = ReleaseAssumptions(
    privacy_unit="patient",
    population_scope="release_cohort",
    release_model="restricted",
    recipient_model="named_researchers",
    auxiliary_data_model="reasonably_available",
    notes_digest=stable_hash({"kind": "reviewed-release-assumptions"}),
)
evidence = build_release_expert_review_evidence(
    result,
    validation=validation,
    assumptions=assumptions,
)

Path("pre-release-assessment.json").write_text(
    before.to_json(), encoding="utf-8"
)
Path("expert-review-evidence.json").write_text(
    evidence.to_json(), encoding="utf-8"
)
Path("expert-review-evidence.md").write_text(
    evidence.to_markdown(), encoding="utf-8"
)
assert evidence.verify()
```

Assess an intended sample against an explicit, locally supplied reference
population with the same API:

```python
population_assessment = assess_population_risk(
    read_table(Path("intended-sample.parquet")),
    read_table(Path("reference-population.parquet")),
    ("age_band", "region"),
    sample_privacy_unit="sample_patient_key",
    population_privacy_unit="population_person_key",
    target_k_map=5,
    max_delta_presence=0.2,
)
Path("population-risk.json").write_text(
    population_assessment.to_json(),
    encoding="utf-8",
)
if not population_assessment.meets_policy:
    raise RuntimeError("Reference-population policy was not met")
```

The QI sequence is canonicalized for evidence, so its input order does not
change the assessment digest. Changing the data, schema, privacy-unit model, or
thresholds does change the relevant binding digest.

Use `preserve_scalar_types=False` when calling
`validate_released_output()` on a CSV or TSV that was written and reread.
Those formats cannot provide typed-digest equality, so validation canonicalizes
the expected scalars to their delimited string representation and requires
lexical dataset-digest equality. It also rejects policy-column encodings that
collapse distinct typed values to the same delimited text, and checks row
counts, the exact materialized schema, forbidden identifier columns, and the
completed in-memory residual policy validation. JSONL and Parquet retain typed
digest comparison.

### Pandas, Polars, and release gates

The optional DataFrame adapters use the same policy and report types:

```python
from openmed.interop import get_adapter

pandas_frame = ...  # authorized local DataFrame
polars_frame = ...  # authorized local DataFrame
get_adapter("pandas")  # registers the .openmed accessor
pandas_report = pandas_frame.openmed.assess_release(policy)
pandas_result = pandas_frame.openmed.anonymize_release(policy)

polars_adapter = get_adapter("polars")
polars_report = polars_adapter.assess_release(polars_frame, policy)
polars_result = polars_adapter.anonymize_release(polars_frame, policy)
```

The standalone release gate checks evidence integrity, exhaustive completion or
an exact zero-loss optimality proof, configured k/l/t thresholds,
post-transform violations, and multi-release composition metadata. For more
than one release, both longitudinal-linkage and prior-release-overlap
assessments must be recorded and the composition status must be
`no_material_increase_observed`:

```python
from openmed.eval import evaluate_release_risk_evidence

check = evaluate_release_risk_evidence(evidence.to_dict())
if not check.passed:
    raise RuntimeError(check.reason)
```

A passing technical gate does not authorize release and does not constitute an
Expert Determination.

### Expert-authored signed handoff

An expert can create a provider-neutral Ed25519 envelope after the technical
evidence verifies. The expert, not OpenMed, supplies the identity,
qualifications, scope and methodology, conclusion, reassessment time, signing
key, and trusted key identifier. The optional supporting-evidence mapping can
bind the population assessment without copying its contents. Install
`openmed[integrity]` for signing and verification:

```python
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from openmed.compliance import (
    ExpertAttestationEnvelope,
    create_expert_attestation,
)

# Synthetic demonstration only. In production, the expert controls key
# creation, storage, rotation, and public-key distribution.
expert_private_key = Ed25519PrivateKey.generate()
expert_public_key = expert_private_key.public_key()
issued_at = datetime.now(timezone.utc)
supporting_digests = {
    "population_risk": population_assessment.digest,
}

attestation = create_expert_attestation(
    evidence,
    expert_identity="Dr. Taylor Example",
    qualifications="Independent statistical disclosure-control expert",
    scope_and_methodology=(
        "Reviewed the population, recipients, release context, "
        "transformations, residual risk, utility, and supporting evidence."
    ),
    conclusion="very_small_risk",
    issued_at=issued_at,
    reassessment_at=issued_at + timedelta(days=365),
    private_key=expert_private_key,
    key_id="expert-key-2026",
    supporting_evidence_digests=supporting_digests,
)
Path("expert-attestation.json").write_text(
    attestation.to_json(),
    encoding="utf-8",
)

parsed = ExpertAttestationEnvelope.from_json(
    Path("expert-attestation.json").read_text(encoding="utf-8")
)
verification = parsed.verify(
    evidence=evidence,
    public_key=expert_public_key,
    expected_key_id="expert-key-2026",
    expected_supporting_evidence_digests=supporting_digests,
    as_of=datetime.now(timezone.utc),
)
if not all(
    (
        verification.cryptographically_valid,
        verification.key_id_matches,
        verification.evidence_integrity_valid,
        verification.bindings_match,
        verification.fresh,
    )
):
    raise RuntimeError("Expert-attestation verification failed")
if verification.conclusion != "very_small_risk":
    raise RuntimeError(f"Expert conclusion: {verification.conclusion}")
```

The same verification is available from the CLI once the expert's public key
has been distributed through a trusted channel. Set `POPULATION_DIGEST` to the
`integrity_digest` in the verified population-risk artifact:

```bash
openmed compliance expert-attestation-verify \
  expert-attestation.json \
  --evidence expert-review-evidence.json \
  --public-key trusted-expert-public.pem \
  --key-id expert-key-2026 \
  --supporting-evidence "population_risk=$POPULATION_DIGEST"
```

The command returns a nonzero exit for a signature, key-ID, evidence-integrity,
or binding mismatch. It reports the expert-stated conclusion and freshness
separately; neither is converted into automatic release approval.

The allowed conclusions are `very_small_risk`, `requires_changes`, and
`not_approved`. Verification deliberately returns cryptographic validity, key
match, evidence integrity, binding match, conclusion, and freshness as
independent facts; converting the result to `bool` raises an error. A valid
signature says that the holder of the supplied key signed the envelope. The
reviewer must establish trust in that public key and the named expert through a
separate governance process.

## Artifact handling

| Artifact | Contents | Handling |
| --- | --- | --- |
| Source table | Original structured records | Sensitive; keep inside the trusted boundary. |
| Discovery manifest | Column names, roles, aggregate profiles, search metadata | Designed to avoid row values; still review schema-name sensitivity and governance before sharing. |
| Release assessment | Aggregate class sizes, risk metrics, warnings, and digests | Aggregate-safe technical evidence; not a determination. |
| Reference-population assessment | Aggregate k-map, exact-linkage, delta-presence, model-consistency findings, and binding digests | Aggregate-safe technical evidence; the reference model and its assumptions still require expert review. |
| Aggregate HTML dashboard | Self-contained visualization of a release assessment | Contains no raw rows or profile keys; govern schema names and derived evidence. |
| Materialized release | Generalized and possibly suppressed records | Still a data release; apply the approved recipient and access controls. |
| Validation report | Counts, identifier-column findings, policy status, and digests | Aggregate-safe technical evidence. |
| Expert-review JSON/Markdown | Reviewed roles, assumptions, model results, transformations, utility, search, composition status, and integrity hash | Handoff to a qualified expert; not a completed determination. |
| Expert attestation | Expert-authored conclusion, qualifications, methodology, reassessment time, evidence bindings, and signature | Verify the trusted key, bindings, conclusion, and freshness independently; not automated authorization. |

The expert-review evidence builder uses an allow-listed schema. It does not
accept records, samples, record IDs, source paths, raw equivalence-class keys,
or transformed rows. Evidence construction requires a passing materialized
output validation object and binds the actual output dataset and schema,
the separate source-dataset digest, reviewed policy roles including any keyed
privacy-unit attribute, hierarchy, configuration, and relevant software module
content.

Transformation evidence distinguishes whole privacy units and their rows
removed by record suppression from QI cells replaced by a suppression
hierarchy. It also records per-field affected privacy-unit and cell counts.
Search evidence reports the evaluated and possible lattice nodes separately
from the evaluated and possible equivalence-class suppression subsets. When
the exact release already has zero information loss and meets policy, zero is a
global lower bound: the remaining positive-loss candidates are pruned,
`optimality_proven` is true, exhaustive `complete` is false, and the
unenumerated suppression-subset total is recorded as `null` rather than
invented or computed through an unbounded combinatorial count.

## Current limitations

- Complete assessment and anonymization load the table into memory. Use bounded
  discovery for very large inputs and size the trusted execution environment
  before a full run. High-dimensional hierarchy lattices and suppression
  subsets can exhaust the explicit search budgets; there is no out-of-core or
  distributed anonymization engine.
- Population-aware metrics use only the explicit reference table supplied by
  the caller. OpenMed does not extrapolate to an unobserved population,
  estimate sampling uncertainty, attach confidence intervals, or decide
  whether the table matches the real anticipated attacker and auxiliary data.
  Without that table, the reported exact-match identity-risk aggregates remain
  sample-only.
- Longitudinal rows within one release use joint multiset profiles, but
  cross-release composition is not assessed automatically. The default
  evidence records composition as `not_assessed`; qualified review must cover
  linkage and overlap with prior or concurrent releases.
- Generalization quality depends on reviewed hierarchies. Unknown categories
  fall back to exact-or-suppressed behavior.
- Multi-valued sensitive attributes require a dedicated disclosure model or
  normalization to one reviewed value per privacy unit before making a
  nontrivial l-diversity or t-closeness claim.
- CSV and TSV cannot preserve scalar types. OpenMed validates their lexical
  representation and rejects type-colliding policy values, but Parquet is the
  better choice when date, decimal, binary, or other typed columns matter.
- Free text, images, genomic data, and other unsupported modalities require
  separate analysis before release.
- Automated discovery cannot know every release-context fact. A qualified
  expert must review roles, assumptions, transformations, utility, residual
  risk, and limitations before reaching any conclusion.

For the broader legal and operational boundary, see
[Compliance Posture](compliance.md).
