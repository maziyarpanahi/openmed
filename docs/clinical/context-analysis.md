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

Contract `clinical-context-v5` includes opt-in `medications`, `labs`, `vitals`, and
`relations` tasks. The default remains sections/entities/assertions. The upstream
extractor must supply compatible attribute spans: Drug/Chemical, Dose, Route,
Frequency, Duration, Form, Strength, Lab Test, Lab Value, Reference Range,
Abnormal Flag, Vital Sign, Anatomy and Severity. A bounded source pattern can
also recover a written amount immediately after a detected drug; other missing
attributes still require extraction. A complete task means all supplied spans
were processed, not that every clinical concept in the source was extracted.

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
  entity records stay unchanged for this repair. Strength remains strength,
  including a `DRUG_STRENGTH` relation, and cannot become a prescribed dose. Its amount/unit
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
  A single combined Lab Test span containing a single-token name and one complete, parseable
  written quantity may supply both parts. A combined Vital Sign span may do so
  only for explicit LVEF (or the English/German full name) and a written percent.
  This treats LVEF as a named measurement in `labs`; it does not add a vital-sign
  class. The original entity and any unparsed vital record remain inspectable.
  Derived name/value references have deterministic pattern provenance and no
  assigned confidence; `source_parts` retains the original model offsets,
  label and score. Context is inherited from that model span. Overlapping model
  evidence, incomplete quantities, comparisons, ranges, multiple values and
  scope crossings cannot supply such a derived measurement. It borrows no
  reference range or abnormal flag from neighboring observations, supplies no
  missing unit or code, and remains an unconfirmed review candidate.
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

German `Therapie`/`Behandlung` and `Verlauf`/`Klinischer Verlauf` headings map to
neutral `treatment` and `clinical_course` sections. They end inherited family
history scope without assigning an unverified terminology code or a future/past
temporal prior.

For a Drug prediction ending *inside* a decimal amount followed by a recognized
unit, the SDK can trim the drug boundary to the preceding word. It retains the
original prediction in `source_parts`, the complete quantity offsets, and
`score_kind=model_score_before_boundary_repair`: that confidence describes the
original span, not a new calibrated boundary. Integer suffixes in drug names,
unknown units and partial compound units cannot trigger this repair.

Medication records also carry `written_amounts`. These are normalized complete
amounts immediately following the drug on the same line, with source offsets,
supporting Dose/Strength predictions and `semantic_type=unspecified`. They recover
the written `47,5 mg` when conflicting model labels prevent an attribute merge,
without converting it into a confirmed prescribed dose or product strength.
They do not infer a missing amount or attach a distant measurement. Existing
partial-dose rejection remains in force. These rules remain unqualified and
require review alongside the source.
Medication quantity scanning admits at most 2,000 source quantity candidates;
an excess fails explicitly with `clinical_quantity_limit`. Documents without a
drug prediction do not invoke this scan or inherit its limit.

## Events and timelines

`events` and `timeline` are additional opt-in EN/DE preview tasks. The SDK composes
its medication-change/lab-trend frame builders and timeline assembler over the
validated entities. Drug heads retain the 0.75 candidate threshold. Original
head, trigger, attribute and date offsets accompany each candidate; source
surfaces and raw helper messages are excluded. Numeric fragment guards also apply
to old/new event doses. Strength retains its original label. An explicit
`from`/`von` quantity cannot fill the new-dose role, and a `to`/`auf` quantity
cannot fill the old-dose role when the counterpart is missing. Explicit change
grammar can assign a complete Strength span to an old/new event amount, with
`role_source=explicit_change_grammar` and the original label/offsets retained.
An isolated strength without such wording cannot become an event dose.

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
