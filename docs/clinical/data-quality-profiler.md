# Clinical Data-quality Profiler

`openmed.structured.quality` provides a deterministic, local-only profile for a
batch of extracted and grounded results before the batch is used for OMOP loads,
cohort resolution, or analytics. It reports completeness, Athena-backed
grounding coverage, conformance, and plausibility checks in one structured
report.

The profiler accepts mappings containing `entities`, `spans`,
`grounded_spans`, or a `fields` mapping. Required fields can be supplied per
batch or per record. A note's completeness score is the product of its required
field coverage and its grounded-span coverage; the report also retains the
underlying counts and rates so downstream consumers can choose a stricter
policy.

## Python

```python
from openmed.structured.quality import profile_jsonl

report = profile_jsonl(
    "grounded-results.jsonl",
    athena_index="/path/to/caller-supplied/athena-export",
    required_fields=("condition", "drug", "measurement"),
    completeness_floor=0.90,
)

if not report.passed:
    report.raise_for_gate()

print(report.to_json())
print(report.human_summary)
```

Athena exports are caller-supplied. OpenMed does not bundle SNOMED CT, RxNorm,
LOINC, UMLS, or other restricted terminology content. A standard concept is
counted only when the span's code or concept identifier matches a standard
concept in the supplied index and, when present, its domain matches.

## CLI and REST

```bash
openmed profile quality \
  --input grounded-results.jsonl \
  --athena /path/to/athena-export \
  --required-field condition \
  --required-field measurement \
  --completeness-floor 0.90 \
  --json
```

The command emits the report and returns exit code `1` when the quality report
fails. `POST /profile` accepts the same JSONL content, a completeness floor,
required fields, and an optional caller-supplied Athena index. It returns the
report's `pass` or `fail` status so an orchestrator can make the same gate
decision without parsing human text.

The OMOP loader and the in-memory note-to-CDM ETL accept
`completeness_floor`, `quality_floor`, and `required_fields`. A failed gate
raises `QualityGateError` before any downstream rows are built. The REST
`/omop/load` and `/cohort/resolve` routes return a PHI-free `409` rejection
envelope when their configured floor is not met.

## Privacy and review boundary

Reports contain aggregate counts, category names, date/offset locations, and
normalizer provenance. They do not contain note text, extracted values,
identifiers, or vocabulary concept names. Invalid dates and measurements are
flagged; the profiler never repairs or silently drops them. Plausibility and
grounding results remain evidence for human review and are not clinical
decisions.
