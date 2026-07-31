# Clinical Context and Extraction Depth

OpenMed's clinical context layer turns entity spans into reviewable assertion
metadata. It composes deterministic ConText-style axes, section priors, scoped
modifier hits, and lightweight normalization helpers so downstream exporters can
keep clinical text extraction transparent.

!!! warning "Advisory annotations only"
    Context and extraction-depth outputs are advisory annotations for review,
    quality checks, and downstream processing. They must not automatically
    trigger diagnosis, triage, treatment, escalation, medication changes, or any
    other clinical decision. Validate the full workflow locally before using it
    in a clinical or regulated setting.

The layer is intentionally local and mechanical:

1. A clinical NER or structured extractor proposes a target span.
2. A scope scanner finds modifier cues that actually reach that target.
3. Section metadata can add a prior, such as historical temporality in Past
   Medical History, when no stronger scoped cue is present.
4. The axis resolvers produce temporality, certainty, and negation values.
5. `ClinicalAssertion` carries the composed assertion axes for downstream
   grounding, tabular export, or FHIR resource construction.

For FHIR-specific resource shaping, see the
[FHIR interop helpers](../fhir-interop.md). For privacy-preserving examples and
surrogate text handling, start with the
[de-identification cookbook](../anonymization.md) and the
[copy/paste recipes](../examples.md).

## Context Axes

`resolve_temporality()` classifies a span as:

| Temporality | Meaning | Example cue |
| --- | --- | --- |
| `recent` | Current or active by default. | `acute` |
| `historical` | Belongs to the patient's past history. | `history of`, `s/p` |
| `hypothetical` | Conditional or not asserted as present. | `if`, `in case of` |

`resolve_uncertainty()` classifies a span as:

| Certainty | Meaning | Example cue |
| --- | --- | --- |
| `certain` | Asserted without a hedging cue. | `confirmed` |
| `uncertain` | Hedged, conditional, or possible. | `possible`, `rule out` |

`resolve_negation()` classifies polarity as:

| Negation | Meaning | Example cue |
| --- | --- | --- |
| `affirmed` | The span is not refuted. | `pneumonia confirmed` |
| `negated` | The span is explicitly refuted. | `no evidence of` |

Pseudo-negation cues, such as `not ruled out` and `no increase`, are masked
before true negation cues are counted. That keeps "pneumonia not ruled out"
affirmed but uncertain, instead of incorrectly treating it as refuted.

```python
from openmed.clinical import assert_context_axes, resolve_span_context

examples = {
    "recent": ("acute pneumonia", []),
    "historical": ("MI", ["history of"]),
    "hypothetical": ("wheezing", ["if"]),
    "uncertain": ("pneumonia", ["possible"]),
    "negated": ("pneumonia", ["no evidence of"]),
}

for name, (span, modifiers) in examples.items():
    context = resolve_span_context(span, modifiers)
    assertion = assert_context_axes(span, modifiers)
    print(name, context.temporality, context.certainty, context.negation)
    print(assertion.to_dict())
```

## Scope And Section Priors

Modifier hits should be scoped before they reach the axis resolvers. A cue only
modifies a target when no sentence boundary or coordinating terminator blocks
the path between cue and target. For example, `history of` should affect
"asthma" in "history of asthma" but not "pneumonia" in "history of asthma but
pneumonia is present".

Section priors are weaker than scoped cues. A Past Medical History section can
seed a historical modifier for otherwise unmodified spans, while a direct
hypothetical cue still wins over that prior.

```python
from openmed.clinical import resolve_span_context

section_prior_hits = {
    "historical": "history of",
}

target = "asthma"
modifier_hits = []
section_prior = "historical"

effective_hits = list(modifier_hits)
temporal_hits = {"history of", "if", "in case of"}
if section_prior and not any(hit in temporal_hits for hit in effective_hits):
    effective_hits.append(section_prior_hits[section_prior])

context = resolve_span_context(target, effective_hits)
print(context.temporality, context.certainty, context.negation)
```

Family-history sections should also remain distinguishable from patient
assertions. If an upstream extractor marks a span as family history, preserve
that section or experiencer metadata and avoid materializing it as an active
patient condition. The `ClinicalAssertion.experiencer` field is available for
callers that already have an experiencer layer.

## Assertion Records

`assert_context_axes()` returns a compact `ClinicalAssertion` for downstream
grounding. It deliberately does not build FHIR, OMOP, or other clinical records
by itself.

```python
from openmed.clinical import assert_context_axes

assertion = assert_context_axes({"text": "possible pneumonia"})
print(assertion.to_dict())
```

Optional axes such as negation and experiencer can be carried on
`ClinicalAssertion` when a caller has already resolved them:

```python
from openmed.clinical import AFFIRMED, CERTAIN, ClinicalAssertion, RECENT

assertion = ClinicalAssertion(
    temporality=RECENT,
    certainty=CERTAIN,
    negation=AFFIRMED,
    experiencer="patient",
)

print(assertion.to_dict())
```

## Axis To FHIR Mapping

The context layer emits axis values. A FHIR exporter decides whether and how to
materialize a `Condition`, `Observation`, `MedicationStatement`, or related
resource. Use this table as the documented default mapping for Condition-like
assertions:

| Axis signal | FHIR field | Default mapping | Notes |
| --- | --- | --- | --- |
| `temporality=recent` | `clinicalStatus` | `active` | Use when the span is asserted as a current patient condition. |
| `temporality=historical` | `clinicalStatus` | `inactive` or `resolved` | Preserve onset, abatement, or provenance dates when available. |
| `temporality=hypothetical` | `clinicalStatus` | no active condition | Keep as advisory metadata or a provisional planning note if retained. |
| `certainty=certain` | `verificationStatus` | `confirmed` | Apply only when not negated. |
| `certainty=uncertain` | `verificationStatus` | `provisional` | Do not drop the span; carry the uncertainty. |
| `negation=negated` | `verificationStatus` | `refuted` | Refuted findings should not become active conditions. |
| `experiencer=family` | resource choice | family-history representation | Do not turn family history into a patient active condition. |

```python
from openmed.clinical import resolve_span_context


def condition_status_for_context(text: str, modifiers: list[str]) -> dict[str, str]:
    context = resolve_span_context(text, modifiers)
    status = {
        "clinicalStatus": "active",
        "verificationStatus": "confirmed",
    }
    if context.temporality == "historical":
        status["clinicalStatus"] = "inactive"
    if context.certainty == "uncertain":
        status["verificationStatus"] = "provisional"
    if context.negation == "negated":
        status["verificationStatus"] = "refuted"
        status["clinicalStatus"] = "not-materialized-as-active"
    if context.temporality == "hypothetical":
        status["clinicalStatus"] = "not-materialized-as-active"
    return status


print(condition_status_for_context("pneumonia", ["possible"]))
print(condition_status_for_context("pneumonia", ["no evidence of"]))
```

## Timeline, Relation, And Normalization Helpers

Timeline and relation helpers sit beside the assertion axes. A timeline layer
should normalize dates and relative ordering while keeping offsets, provenance,
and section metadata. A relation layer should connect already-extracted spans,
such as medication-to-dose or finding-to-anatomy, without copying raw PHI into
logs or diagnostics.

The flat-table exporter keeps these annotations easy to inspect. It copies only
whitelisted fields into stable rows, including `normalized_text`, coding fields,
context axes, offsets, and section labels.

```python
from openmed.clinical import resolve_span_context
from openmed.clinical.exporters import flatten_clinical_entities

context = resolve_span_context("pneumonia", ["possible"])
rows = flatten_clinical_entities(
    [
        {
            "label": "condition",
            "text": "pneumonia",
            "context": context,
            "start": 24,
            "end": 33,
            "metadata": {"section": "Assessment"},
        }
    ]
)

print(rows[0])
```

For laboratory values, the shipped helpers parse simple numeric reference ranges
and derive advisory abnormal flags. They do not convert units and do not replace
the originating laboratory's own formal flags.

```python
from openmed.clinical import derive_abnormal_flag, parse_reference_range

reference_range = parse_reference_range("135-145")
print(reference_range)
print(derive_abnormal_flag(130, reference_range))
print(derive_abnormal_flag(140, "135-145", explicit_flag="N"))
```

Medication sig, problem status, family-history, and relation outputs should feed
the same record shape: normalized text or coding, assertion axes, section or
experiencer metadata, offsets, and provenance. Keep raw clinical text out of
audit artifacts unless the caller explicitly owns that PHI boundary.
