# SDOH extraction

`openmed.clinical.sdoh.extract_sdoh(text, spans, sections=...)` dispatches
registered determinant extractors over caller-owned text. The packaged
extractors use local English cue tables and context rules; they do not load a
model or download a corpus. Findings are candidate annotations for qualified
human review, with no clinical-validity or autonomous-decision claim.

## Run a synthetic example

This example needs no NER model. The built-in social extractors scan text cues
directly, while the dispatcher filters candidates and complete finding spans
to the supplied Social History sections.

```python
from openmed.clinical.sdoh import extract_sdoh
from openmed.clinical.sections import detect_sections

text = (
    "Assessment: retired teacher.\n"
    "Social History: unemployed; lives alone; food insecurity.\n"
    "Plan: routine follow-up."
)
findings = extract_sdoh(text, spans=[], sections=detect_sections(text))
summary = sorted((item.category, item.status, item.span) for item in findings)
assert summary == [
    ("employment", "unemployed", (45, 55)),
    ("food_insecurity", "current", (70, 85)),
    ("living_status", "lives_alone", (57, 68)),
]
assert len(findings) == 3
```

Offsets are half-open Python character positions in the original `text`.
The assessment's employment cue is excluded. With `sections=[]`, no approved
Social History range exists and the dispatcher returns no findings. With
`sections=None`, the caller takes responsibility for selecting a suitable
window: cue extractors may scan the whole supplied text even when `spans=[]`.
An empty candidate list therefore does not make an unscoped document safe.

Upstream spans may be mappings or objects with integer `start`/`end` offsets.
Passing detected sections filters both those candidate spans and findings;
the source text itself is still passed to the trusted extractor. This is an
output-scoping contract, not isolation from custom Python code.
Use the [strict section-scope guard](sdoh-section-scope.md) when a workflow
needs explicit fallback policy and a counts-only exclusion report.

## Built-in determinants and statuses

These are exactly the startup keys returned by
`available_determinant_extractors()`. Registration can extend or replace them
within the current process. Food insecurity is an OpenMed extension beyond
the five core SHAC categories.

| Determinant | Packaged finding status vocabulary | Notes |
|---|---|---|
| `alcohol` | `current`, `past`, `none`, `unknown` | Explicit substance cue and its local assertion/temporal context |
| `drug` | `current`, `past`, `none`, `unknown` | Drug-use cues and local context; finite cue coverage |
| `employment` | `employed`, `unemployed`, `retired`, `disabled`, `student`, `former`, `never`, `unknown` | May also preserve a configured occupation type |
| `food_insecurity` | `current` | Packaged extension uses the controlled value `food_insecure`; absence is not a negative assertion |
| `living_status` | `housed`, `homeless`, `lives_alone`, `lives_with_family`, `assisted_living`, `former`, `never`, `unknown` | Living/housing cues and local context |
| `tobacco` | `current`, `past`, `none`, `unknown` | Tobacco-use cues; may retain a source-derived extent |

The status normalizer uses `former` and `never` for substance cues; the finding
extractor maps these to `past` and `none`. These finding labels differ from the
downstream evidence contract's `present`, `absent`, `uncertain`, `unknown` and
`refused` assertions. A custom `SDOHFinding` accepts optional status strings;
the table describes the packaged extractors, not a universal enum for plugins.

`SDOHFinding` contains category, value, optional status/extent/temporality,
source span and score. Values and extents can preserve source content.
Do not use `to_dict()` as a general audit-safe serializer. The example projects
only controlled labels and offsets. Cue-table scores are deterministic rule
weights, not calibrated probabilities or measured model performance.

## Register a custom extractor

An extractor receives `(text, candidate_spans)` and yields `SDOHFinding`
objects. Registration is process-wide; keep your own callable when replacing
an existing key with `replace=True`, and restore registrations after temporary
use. Registered callables are trusted local application code.

This synthetic extension demonstrates section scoping and cleanup. It does not
qualify a new determinant or implement every assertion/experiencer guard.

```python
from openmed.clinical.sdoh import (
    SDOHFinding,
    available_determinant_extractors,
    extract_sdoh,
    register_determinant_extractor,
    unregister_determinant_extractor,
)
from openmed.clinical.sections import detect_sections

cue = "synthetic transit barrier"
note = f"Assessment: {cue}.\nSocial History: {cue}."
candidates = [
    {"start": index, "end": index + len(cue)}
    for index in (note.index(cue), note.rindex(cue))
]
before = available_determinant_extractors()


def synthetic_transport(source, spans):
    for span in spans:
        start, end = span["start"], span["end"]
        if source[start:end] == cue:
            yield SDOHFinding(
                category="transportation",
                value="synthetic_barrier",
                status="unknown",
                extent=None,
                temporality=None,
                span=(start, end),
                score=1.0,
            )


register_determinant_extractor("synthetic_transport", synthetic_transport)
try:
    custom_findings = extract_sdoh(note, candidates, sections=detect_sections(note))
    custom_summary = [
        (item.category, item.status, item.span) for item in custom_findings
    ]
    assert custom_summary == [
        ("transportation", "unknown", (note.rindex(cue), note.rindex(cue) + len(cue)))
    ]
finally:
    unregister_determinant_extractor("synthetic_transport")
assert available_determinant_extractors() == before
```

## Cue tables and replacement data

The packaged tables are
`openmed/clinical/data/sdoh_social_cues.yaml` for employment, living status and
food insecurity, and `openmed/clinical/data/sdoh_substance_cues.yaml` for
alcohol, drug and tobacco triggers. Status normalization also uses
`openmed/clinical/data/status_vocab.yaml`. These repository-authored assets
contain synthetic/public phrases and unrestricted provenance, with no SHAC
source text.

`load_sdoh_social_cues()` returns a detached copy of the validated social table.
`load_sdoh_social_cues(path)` validates a caller-supplied replacement YAML:
schema version, determinant identities, scores, status priorities, occupation
types, food-extension metadata and unrestricted provenance must be valid.
Loading or mutating that dictionary **does not install it into the built-in
extractors**. Downstream code must consume the returned table in its own
extractor and register that callable explicitly. This guide changes no packaged
table or runtime behavior.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from openmed.clinical.sdoh import load_sdoh_social_cues

default_table = load_sdoh_social_cues()
replacement = load_sdoh_social_cues()
replacement["determinants"]["food_insecurity"]["cues"] = ["synthetic pantry cue"]
with TemporaryDirectory(prefix="openmed-sdoh-cues-") as directory:
    path = Path(directory) / "social-cues.yaml"
    path.write_text(yaml.safe_dump(replacement), encoding="utf-8")
    loaded = load_sdoh_social_cues(path)
    assert loaded["determinants"]["food_insecurity"]["cues"] == ["synthetic pantry cue"]
assert load_sdoh_social_cues() == default_table
```

## Interpretation, coverage and data policy

The packaged cue coverage is **English-only**. No multilingual recall or
automatic language detection is claimed by these cue extractors. Finite cue
lists can miss synonyms, spelling variation, unusual section layouts and
implicit context. Section detection and custom extractors need their own
qualification. Screening questions, education, unanswered templates, negation,
historical context and third-party mentions need explicit downstream handling;
an emitted candidate or high rule score is not a reviewed fact.

**Zero findings is not a negative finding.** It can mean absent cues,
unsupported wording/language, excluded sections, missing documentation or
insufficient detector recall. It does not establish that a social need is
absent, that the patient declined to answer, or that every determinant was
assessed. Do not fill missing categories with `none` or treat them as eligibility
or care decisions.

Under OpenMed's data policy, real **SHAC is DUA-gated, user-supplied and
evaluation-only**. It is never bundled with this runtime, its cue assets,
documentation examples or golden tests. The separate credentialed eval loader
is an optional user-data path; cue extraction does not load it. Food insecurity
remains the explicitly labelled OpenMed extension beyond core SHAC.

## Feed the existing evidence and review guards

Keep source text and values in the caller's protected context. Use
`evidence_from_sdoh_finding()` from the [evidence contract](sdoh-evidence.md)
to preserve controlled determinant/assertion labels and offsets, with explicit
evidence provenance, section and review state. The adapter is not permission to
invent provenance or approved review.

Apply the relevant guards independently:

- [Section scope](sdoh-section-scope.md) and
  [temporal qualifiers](sdoh-temporal-qualifiers.md) retain source boundaries.
- [Experiencer filtering](sdoh-experiencer-filtering.md) and
  [negated needs](sdoh-negated-needs.md) distinguish whose statement is present
  and what was actually denied.
- [Evidence deduplication](sdoh-evidence-deduplication.md) and
  [longitudinal resolution](sdoh-longitudinal-resolution.md) preserve lineage
  and uncertainty across observations.
- [Category completeness](sdoh-category-completeness.md) records unobserved
  coverage without inferring a negative.
- [Review routing](sdoh-review-routing.md) and
  [sensitive-use labels](sdoh-sensitive-use-labels.md) retain human review and
  prohibit automatic high-impact decisions.

The [synthetic counterfactual checks](../evaluation/sdoh-counterfactuals.md)
and [false-positive stress checks](../evaluation/sdoh-false-positive-stress.md)
provide engineering regression evidence. They do not establish clinical
validity, measured real-world recall, fairness certification or authority to
act on an individual.
