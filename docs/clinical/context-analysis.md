# Clinical context analysis

`analyze_clinical_context` composes existing SDK section detection, entity span
validation, and ConText/experiencer analysis. It consumes entities from an upstream
extractor and returns source offsets and cue evidence without copying source
surfaces into the result. It does not load an NLP model or call a service.

```python
from openmed.clinical import analyze_clinical_context

text = "Familienanamnese: Mutter mit Diabetes."
start = text.index("Diabetes")
result = analyze_clinical_context(
    text,
    [{"start": start, "end": start + len("Diabetes"), "label": "Disease"}],
    language="de",
    entity_coverage_complete=True,
)
assertion = result["tasks"]["assertions"]["records"][0]
assert assertion["experiencer"] == "family"
```

The upstream caller must verify complete input-token coverage before setting
`entity_coverage_complete=True`. With incomplete coverage, only the independent
section task can succeed. Entity/assertion records are withheld. Unknown labels,
inconsistent surfaces/offsets, nonfinite scores, more than 2,000 entities, and
sources larger than 100,000 characters fail explicitly. Missing confidence stays
unknown. Local entity IDs are deterministic after offset ordering/deduplication.

Each requested task has its own `complete`, `status`, and `records` fields. A
fully processed result remains `needs_review`: no language has an independent
clinical qualification receipt. English and German assertion rules are available
as preview language packs. Unsupported, mixed, or uncertain automatic language
cannot silently use English assertion rules; the assertion task reports
`unsupported`, while section/entity results remain separately inspectable. Supply
an explicit supported language for context analysis after reviewing that choice.

The result includes clinical context axes, their `local`/`section`/`default`
provenance, and source ranges for scoped context cues, local subject cues, and
section headers. Evidence records contain offsets and categories, not the cue
text. These remain heuristic context annotations, not confirmed patient facts.

The cooperative 30-second postprocessing deadline and cancellation checks run
between SDK stages and during evidence assembly. They do not preempt an active
Python helper. Counts and source limits bound admitted work. Services must run
this function on their controlled worker, never the HTTP event loop.

## Structured tasks

Contract `clinical-context-v2` adds opt-in `medications`, `labs`, `vitals`, and
`relations` tasks. The default remains sections/entities/assertions. The upstream
extractor must supply compatible attribute spans: Drug/Chemical, Dose, Route,
Frequency, Duration, Form, Strength, Lab Test, Lab Value, Reference Range,
Abnormal Flag, Vital Sign, Anatomy and Severity. This API does not find missing
attributes itself. A complete task means all supplied spans were processed, not
that every clinical concept in the source was extracted.

Structured tasks also require complete upstream coverage and a supported EN/DE
preview language. They compute context even when `assertions` is not explicitly
requested. Records retain negation, family/other experiencer, temporality and
uncertainty; `coding_eligible` is false. They are unconfirmed candidates.

- **Medications:** the SDK medication-candidate policy requires an upstream score
  of at least 0.75, rejects measurement abbreviations, and performs no grounding.
  Missing confidence cannot become a perfect score. The language is forwarded
  to the observation filter, so German `K 4,2 mmol/L` is not a drug candidate.
  Linked dose/frequency/duration spans use localized normalization; other
  attributes retain evidence without invented normalization. Pattern and link
  scores are heuristic, not calibrated clinical probabilities.
- **Labs and measurements:** supplied analyte/value/range/flag spans link only
  within one source scope. Canonical magnitudes and range bounds include their
  units; for example, `55 %` becomes `0.55` in unit `1`, with the original value
  range preserved as evidence. Missing range units use the explicit measurement
  unit. Unknown units, unparsed values, unlinked analytes and orphan value spans
  remain inspectable without fabricated values. Explicit lab flags take
  precedence over reference comparisons. No critical threshold is invented.
- **Vitals:** each located source span must describe one measurement. A span
  containing both blood pressure and heart rate is unparsed, rather than silently
  retaining the first measurement. Blood-pressure components retain their source
  context and captured units. No missing unit is inferred.
- **Relations:** offset-only drug-dose/route, problem-anatomy and finding-severity
  candidates retain both endpoints' context. Source scopes prevent links across
  lines, sentence/semicolon boundaries, contrastive clauses and sections.

Linking scopes admit at most 64 entities; relation output is capped at 4,096
records per task. Limit failures and ambiguous same-offset labels return no
structured findings for that task. Other independent tasks can remain complete.
Unexpected helper failures use a finite error code without exception/source text.
Cancellation or timeout still aborts the entire composition. These synthetic
regressions and bounds do not provide independent language qualification.
