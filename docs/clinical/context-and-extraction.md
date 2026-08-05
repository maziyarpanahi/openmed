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

## Multilingual Cue Lexicons

The public resolvers and `scan_context_cues()` accept a `language` argument.
The registry normalizes a BCP-47-like value to its primary language subtag, so
`es-MX` selects the `es` pack. An unknown code falls back to English for
backward compatibility. A registered non-English pack is otherwise isolated:
missing cues keep the default recent, certain, and affirmed axes rather than
borrowing English cues.

The shipped `en`, `es`, `fr`, `de`, `zh`, `hi`, and `pt` packs live in
`openmed/clinical/lexicons/context_cues.py`. They are compact,
OpenMed-authored surface-form tables distributed with the repository under
Apache-2.0. They are not verbatim exports of a publication's supplementary
lexicon, a terminology system, or a clinical corpus.

### Research Lineage And Pack Provenance

The publications below establish the method and the review practices that
informed the OpenMed implementation. A citation is not permission to copy a
paper's corpus or lexical asset. Treat a source as methodological background
unless its terms have a clear, permissive redistribution license that is
compatible with Apache-2.0.

| Source | Influence on OpenMed | Data boundary |
| --- | --- | --- |
| [Chapman et al., *A Simple Algorithm for Identifying Negated Findings and Diseases in Discharge Summaries*](https://doi.org/10.1006/jbin.2001.1029) | NegEx's compact true-negation, pseudo-negation, and bounded-scope design. | No discharge-summary sentence or source trigger file is bundled. |
| [Harkema et al., *ConText: An Algorithm for Determining Negation, Experiencer, and Temporal Status from Clinical Reports*](https://pmc.ncbi.nlm.nih.gov/articles/PMC2757457/) | ConText-style contextual axes, directional cues, and termination boundaries. | No evaluation report or annotated condition is bundled. |
| [Chapman et al., *Extending the NegEx Lexicon for Multiple Languages*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3923890/) | The multilingual translation workflow and the need for clinician or linguist review, especially for French and German variation. | The published OWL/RDF lexicon and its source corpora were not imported. |
| [Velupillai et al., *Cue-based assertion classification for Swedish clinical text—developing a lexicon for pyConTextSwe*](https://pmc.ncbi.nlm.nih.gov/articles/PMC4104142/) | The practice of testing language-specific cues, inflections, uncertainty, and error-driven refinements independently. | Swedish is not a shipped pack, and no pyConTextSwe cue or clinical sentence was copied. |

The English pack migrated the pre-registry OpenMed tuples to preserve existing
behavior. The Spanish, French, German, Chinese, Hindi, and Portuguese packs are
OpenMed-maintained baselines authored as small lists of common surface forms.
The multilingual publications above informed their structure and review
criteria; they are not a claim that every shipped phrase occurs in those
resources. Chinese and Hindi entries in particular are OpenMed-authored
applications of the published cue-and-scope method, not translations imported
from those publications. Their committed evaluation evidence is the synthetic
fixture described below, not a restricted clinical corpus.

The Portuguese pack was authored from no external lexical source: no URL, DOI,
or third-party cue list was consulted or copied, so there is no upstream
license to honor beyond this repository's Apache-2.0 terms. Its entries are
common Brazilian and European Portuguese clinical surface forms, and every one
of them carries a behavioral regression case in
`tests/unit/clinical/test_context_multilingual.py`; an exact-set test fails if
a cue is added without one. Two review notes are recorded rather than assumed.
Bare `se` is excluded from the conditional cues because it is also the
reflexive clitic, and bare `previo`/`previa` are excluded because
`placenta previa` is a diagnosis rather than a temporal marker. The pack has
not yet had a fluent-clinician sign-off; the shipped evidence is the cue-level
regression table and the synthetic fixture, not a native-speaker review.

### Lexicon Fields

`ClinicalCueLexicon` has the following contribution contract:

| Field | Meaning and review requirement |
| --- | --- |
| `language` | Primary language code used as the registry key. Region or script subtags passed by callers normalize to this primary code. |
| `negation` | True-negation cues that can change `affirmed` to `negated`. Keep phrases that only look negative in `pseudo_negation`. |
| `pseudo_negation` | Longer phrases masked before true-negation matching, such as an inability to exclude a finding. Test each one as affirmed and, when appropriate, uncertain. |
| `historical` | Cues that move temporality from its default `recent` value to `historical`. |
| `hypothetical` | Conditional or future-contingent cues. A hypothetical cue takes precedence over a historical cue. |
| `recent` | Explicit current or acute cues. The resolver also returns `recent` when no temporal cue matches. |
| `uncertainty` | Hedging, speculation, or conditional cues that change `certain` to `uncertain`. If a phrase affects multiple axes, include and test it in every relevant tuple. |
| `backward` | Exact normalized cue strings that scope to a target on their left. Every entry must also appear in an axis tuple so the scanner can discover and categorize it. All other discovered cues scope forward. |
| `scope_terminators` | Language-specific boundaries used to trim the direct resolver's context window and validate explicit modifier hits. Sentence punctuation is also a boundary. |
| `conjunction_terminators` | Same-sentence blockers used by `scan_context_cues()` between a cue and target, such as the local equivalents of “but” or “however.” |
| `token_boundaries` | When `true`, cue matching requires Unicode word boundaries. Set to `false` only when the script or orthography needs substring matching, then add negative tests for accidental matches inside longer text. |

### Scope Direction

`scan_context_cues()` emits offset-bearing modifier hits for historical,
hypothetical, uncertainty, and negation cues. A cue before a target scopes
forward; a cue listed in `backward` must occur after the target and scopes to
the left. Cue and target must remain in the same sentence, and a localized
`conjunction_terminators` match between them stops the hit.

The resolver path also accepts a span carrying full context and offsets. It
trims that context at punctuation and `scope_terminators`, then evaluates each
axis. Pseudo-negation is masked before true negation cues are counted; an even
number of true negation cues is treated as affirmed so double negation is
deterministic. These mechanical rules need language-specific minimal pairs and
must remain advisory annotations rather than clinical decisions.

### Adding A Language Pack

1. Establish provenance before editing. Record the source URL or DOI, license,
   authoring or translation method, and the language/domain review performed.
   If a lexical source does not state permissive redistribution terms, use it
   only to understand the method and author new entries independently.
2. Add the `ClinicalCueLexicon` constant and its registry entry in
   `openmed/clinical/lexicons/context_cues.py`. Adding a pack must not require
   edits to resolver or scanner logic.
3. Extend `openmed/eval/golden/fixtures/context_multilingual.jsonl`. Update the
   metadata language list and add uniquely named synthetic minimal pairs for
   affirmed, negated, historical, hypothetical/uncertain, pseudo-negation, and
   double-negation behavior. Both the metadata row and every case must retain
   `"synthetic": true`, and each target string must occur unambiguously in its
   sentence.
4. Extend the focused tests in
   `tests/unit/clinical/test_context_multilingual.py` for forward and backward
   scope, a localized conjunction terminator, token-boundary behavior, and the
   per-axis fixture results. The tests and eval must run fully offline.

## Language-Pack Review Checklist

- [ ] The PR identifies each lexical or methodological source and its license;
      copied material is Apache-2.0-compatible and attribution requirements are
      preserved.
- [ ] No raw PHI, real clinical sentence, DUA-gated corpus, proprietary or
      source-available list, or gated terminology asset appears in code,
      fixtures, logs, or test artifacts.
- [ ] A fluent reviewer has checked meaning, spelling, diacritics, inflection,
      common abbreviations, and whether cues are plausible in the intended
      clinical register.
- [ ] True negation, pseudo-negation, uncertainty, and intentional cross-axis
      overlaps are separated correctly; every `backward` cue also belongs to an
      axis tuple.
- [ ] Forward and backward direction, sentence/clause termination, and the
      `token_boundaries` choice have positive and negative regression cases.
- [ ] The JSONL cases are fabricated minimal pairs, carry `"synthetic": true`,
      contain no identifiers, and cover affirmed, negated, historical,
      hypothetical, uncertain, pseudo-negation, and double-negation outcomes.
- [ ] The existing English cases and every registered language still meet the
      focused multilingual context gates without network access.
- [ ] The change is confined to the lexicon, fixture, focused tests, and this
      guidance; resolver logic is unchanged.

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

`assert_context(text, spans)` is the document-level API. It returns copied span
mappings with `negation`, `uncertainty`, `experiencer`, and `temporality`, and
also places those fields under `metadata["clinical_context"]` for compatibility
with formatted NER results. Input spans are not mutated.

Pass detected or upstream `SectionSpan` metadata through `sections=` to apply
section-scoped priors. The span must fall inside the section's half-open
`start`/`end` range. Canonical labels and supported LOINC section codes are
accepted. With this opt-in path, each result also contains `context_sources`
and `metadata["clinical_context_sources"]`, recording the winning source for
each axis.

Precedence is deterministic for every axis:

1. `local` — an explicit in-clause modifier wins.
2. `section` — a containing-section prior is used when no local modifier wins.
3. `default` — the existing global default is used otherwise.

Family History supplies `experiencer=family`; Past Medical History supplies
`temporality=historical`. Section header text is not treated as a local cue, so
provenance distinguishes a section prior from an explicit statement in the
section body. Omitting `sections` preserves the prior output shape and values.

```python
from openmed.clinical import assert_context
from openmed.clinical.sections import detect_sections

text = "Past Medical History:\nPneumonia."
start = text.index("Pneumonia")
[span] = assert_context(
    text,
    [{"text": "Pneumonia", "start": start, "end": start + 9}],
    sections=detect_sections(text),
)
print(span["temporality"])
print(span["context_sources"]["temporality"])
```

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
