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
5. **Anonymize.** Search reviewed generalization hierarchies and, within an
   explicit budget, suppress complete privacy units.
6. **Reread and validate.** Reopen the materialized release and verify row
   counts, identifier removal, prior residual-policy validation, and a typed
   content digest where the format preserves scalar types.
7. **Prepare expert-review evidence.** Export deterministic aggregate JSON and
   Markdown, then verify the evidence integrity before review.

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

## Privacy models and exact semantics

OpenMed does not choose a universal regulatory threshold. The caller must set
`k` and should set `l` and `t` explicitly after expert review.

### k-anonymity

`k` is the smallest equivalence-class size for the declared
quasi-identifiers. When `--privacy-unit patient_id` is supplied, class size is
the number of distinct patients, not the number of rows. Repeated encounters
therefore do not inflate k.

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

The assessment also reports exact-match sample identity-risk aggregates. It
does not estimate population uniqueness or risk against external auxiliary
datasets.

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
  --role full_name=direct-id \
  --role encounter_id=direct-id,internal-linkage
```

The output contains schema names and aggregate counts, not the source path,
cell values, record identifiers, equivalence-class keys, or low-entropy value
hashes. Sampled discovery is always marked advisory. `--full-scan` reads all
rows and marks dataset coverage complete, but it still does not replace role
review.

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

### 3. Anonymize, reread, validate, and produce evidence

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
artifact. It then stages both expert-review evidence files and publishes all
three outputs as one rollback-safe transaction, with the sensitive release
published last. It fails closed if the hierarchy search exceeds
`--max-lattice-nodes` or `--max-suppression-subsets`, the suppression budget is
insufficient, the transformed data misses a configured privacy target, the
release is empty, output paths alias an input or one another, or output
validation fails. Within those explicit bounds, every eligible
equivalence-class suppression subset is evaluated for every lattice node. The
evidence records both candidate totals and limits; a budget overrun fails
without claiming an optimum.

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

The standalone release gate checks evidence integrity, search completeness,
configured k/l/t thresholds, post-transform violations, and multi-release
composition metadata. For more than one release, both longitudinal-linkage and
prior-release-overlap assessments must be recorded and the composition status
must be `no_material_increase_observed`:

```python
from openmed.eval import evaluate_release_risk_evidence

check = evaluate_release_risk_evidence(evidence.to_dict())
if not check.passed:
    raise RuntimeError(check.reason)
```

A passing technical gate does not authorize release and does not constitute an
Expert Determination.

## Artifact handling

| Artifact | Contents | Handling |
| --- | --- | --- |
| Source table | Original structured records | Sensitive; keep inside the trusted boundary. |
| Discovery manifest | Column names, roles, aggregate profiles, search metadata | Designed to avoid row values; still review schema-name sensitivity and governance before sharing. |
| Release assessment | Aggregate class sizes, risk metrics, warnings, and digests | Aggregate-safe technical evidence; not a determination. |
| Materialized release | Generalized and possibly suppressed records | Still a data release; apply the approved recipient and access controls. |
| Validation report | Counts, identifier-column findings, policy status, and digests | Aggregate-safe technical evidence. |
| Expert-review JSON/Markdown | Reviewed roles, assumptions, model results, transformations, utility, search, composition status, and integrity hash | Handoff to a qualified expert; not a completed determination. |

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
from the evaluated and possible equivalence-class suppression subsets.

## Current limitations

- Complete assessment and anonymization load the table into memory. Use bounded
  discovery for very large inputs and size the trusted execution environment
  before a full run.
- The reported exact-match risk is sample risk. OpenMed does not estimate
  population uniqueness or linkability against public, recipient-supplied, or
  otherwise reasonably available auxiliary data.
- Cross-release and longitudinal composition are not assessed automatically.
  The default evidence records composition as `not_assessed`.
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
