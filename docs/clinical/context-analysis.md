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

Contract `clinical-context-v3` includes opt-in `medications`, `labs`, `vitals`, and
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
  Same-label adjacent fragments of an explicitly localized decimal quantity can
  be rejoined only when the original text parses as one valid amount/unit. The
  structured reference records both source parts and the repair rule; original
  entity records stay unchanged. Strength remains strength, including a
  `DRUG_STRENGTH` relation, and cannot become a prescribed dose. Its amount/unit
  use the quantity normalizer. Partial units, different quantities and source
  scope/context changes cannot be joined.
  Original-source quantity boundaries are checked before normalization. A
  decimal tail cannot become a smaller dose, including when model fragments
  have conflicting Dose/Strength labels. Unrecognized or incomplete quantities
  expose only a finite reason and `recognized=false`, without a guessed value,
  unit or source-bearing parser message. Unsafe quantity tails cannot form
  dosage relations. Section-header predictions remain in the entity result but
  cannot become structured medication/lab findings.
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

## Events and timelines

`events` and `timeline` are additional opt-in EN/DE preview tasks. The SDK composes
its medication-change/lab-trend frame builders and timeline assembler over the
validated entities. Drug heads retain the 0.75 candidate threshold. Original
head, trigger, attribute and date offsets accompany each candidate; source
surfaces and raw helper messages are excluded. Numeric fragment guards also apply
to old/new event doses. Strength is not reinterpreted as dose.

German trigger rules distinguish start, restart, stop, increase, decrease and hold,
and rising/falling/stable laboratory trends. A continued regimen is not inferred
to have restarted. Competing heads in one clause remain clinical mentions when
the action cannot be assigned unambiguously. An elevated lab beside a medication
must not become a dose-increase event. Events cannot link across sentence, line,
semicolon, contrastive or section boundaries. Source dates are protected from
being split at internal punctuation when those scopes are constructed.

Event head context and trigger context are both retained. A `stopped` trigger
with negated trigger context does not assert that the medication was stopped.
Family, historical, uncertain and negated findings remain reviewable candidates;
every event has `coding_eligible=false`.

```python
result = analyze_clinical_context(
    text,
    entities,
    language="de",
    tasks=["events", "timeline"],
    reference_date="2026-09-08",  # Only when this is the document's known date.
)
```

The optional ISO `reference_date` is accepted only with the timeline task.
Relative expressions remain unanchored without it. The implementation does not
substitute the current date or the document date for an event that lacks usable
date evidence. A clinical event can use one unambiguous date from its own scope;
competing dates, invalid dates and birth-date contexts remain unanchored. German
numeric dates follow explicit DMY rules, full German month names and relative
day/week/month/year phrases are supported, and two-digit years remain ambiguous.
English ambiguous slash dates remain unresolved. No text translation occurs.

The timeline presents anchored events in timestamp order and unanchored events
in a separate source-order group. Presentation order is not a claimed temporal
or causal relation. Interval anchors retain both endpoints; an unanchored event
is not asserted to occur after the anchored group. The original four context axes
are retained rather than narrowed to the assembler's historical context enums.

At most 2,048 temporal spans, 64 entities per source scope, 128 event triggers per
scope/engine and 4,096 output event records are admitted. Dependent task failures
return empty results with finite errors; independent sections/entities remain
available. These are deterministic preview rules, with no independent event or
timeline accuracy qualification.
