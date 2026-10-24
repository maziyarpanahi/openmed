# Low-resource language cohort and graduation contract

`openmed.clinical.languages.graduation` names the first cohort of low-resource
clinical locales and the contract a locale must pass before OpenMed claims
support for a task in it. The cohort is a small, explicit list. It is not a
promise to support every language in one release.

Every locale records:

- a **license/lineage** record for its weights, fixtures and surrogates;
- a **review role** code (`native_clinical_reviewer`), never a person's name,
  and whether that role is filled;
- a **synthetic fixture plan** with the committed golden records, the critical
  labels they already cover, a target slice size and the script or morphology
  stressors to author;
- an explicit **`experimental` or `supported` status**.

No cohort locale has native-review or evaluation evidence in the repository
yet, so all of them start `experimental` with the review role `unfilled`.

## Initial cohort

| Code | Script | Region | Route today | Golden records | Limitations |
| ---- | ------ | ------ | ----------- | -------------- | ----------- |
| `am` | Ethiopic | Horn of Africa | multilingual fallback | 2 | surrogate locale approximation |
| `sw` | Latin | East Africa | multilingual fallback | 3 | none recorded |
| `zu` | Latin | Southern Africa | multilingual fallback | 1 | none recorded |
| `or` | Odia | South Asia | multilingual fallback | 2 | none recorded |
| `pa` | Gurmukhi | South Asia | default-model placeholder | 2 | placeholder route, surrogate approximation, tokenizer script unclaimed |
| `ur` | Arabic | South Asia | default-model placeholder | 3 | placeholder route, surrogate approximation |
| `ne` | Devanagari | South Asia | user-supplied model | 0 | no bundled model, no golden fixture, no locale PHI generator |
| `yo` | Latin | West Africa | national-ID-only | 3 (no `PERSON`) | identifier-only patterns, no bundled model, no locale PHI generator |

All bundled weights and committed fixtures in the cohort are Apache-2.0. The
`ne` and `yo` routes ship no weights, so their model license is empty rather
than assumed. Restricted or DUA-gated corpora are never bundled: a lineage
record that declares them is rejected.

### Why these locales

Each criterion maps to evidence that already exists in the repository:

| Criterion | Repository evidence |
| --------- | ------------------- |
| Clinical need | Regional deployment guides ([Africa onboarding](../africa-onboarding.md), [CHW form export](../chw-form-deid.md), [India health-ID de-identification](../india-health-id-deidentification.md)) and a validator-backed national-ID pattern for every locale except `ne`. |
| Script and morphology coverage | Six scripts across eight locales: Ethiopic numerals, Bantu noun-class prefixes, Gurmukhi and Odia native digits, Odia conjuncts, right-to-left Arabic script with bidi isolates, Devanagari affixes, and Yoruba tone marks in NFC and NFD. |
| Permissive data | Apache-2.0 multilingual weights in `models.jsonl`, synthetic Apache-2.0 golden fixtures, and native or documented-approximation Faker locales. |
| On-device feasibility | `am`, `or`, `sw` and `zu` are declared by MLX variants of the multilingual privacy filter. |
| Native reviewer commitment | None recorded. Every locale keeps the role code but marks it `unfilled`, which blocks graduation until the role is active. |

The cohort deliberately spans every route a low-resource language can take
today: the declared multilingual fallback, the compatibility placeholder, a
user-supplied model and the deterministic national-ID-only path. Other
low-resource packs (for example `xh`, `ig`, `ha`, `ml`) wait for a later cohort.

`repository_language_inventory()` rebuilds this evidence from the language-pack
catalog, the model manifest, the surrogate-locale report, the synthetic locale
generators and the golden fixture metadata. It reads codes, counts and labels
only, never fixture text. `cohort_inventory_findings()` compares it with the
declared cohort, and the unit tests require zero findings. A route, locale,
script or license change, or a record that claims more evidence than the
repository holds, fails the suite. New evidence, such as an added fixture,
leaves the record conservative until it is refreshed.

## Status and fail-closed tasks

Only `deidentification` has a graduation contract. Every other task in the
closed task vocabulary is unsupported for every cohort locale and always
fails closed, with or without the experimental opt-in.

```python
from openmed.clinical.languages.graduation import (
    INITIAL_LOW_RESOURCE_COHORT as cohort,
    LanguageGraduationError,
)

assert cohort.task_status("sw", "deidentification") == "experimental"
assert cohort.require_task("sw", "deidentification", allow_experimental=True) == (
    "experimental"
)
for task in ("deidentification", "summarization"):
    try:
        cohort.require_task("sw", task)
    except LanguageGraduationError as error:
        print(task, error.code)  # task_not_supported
```

Refusals carry fixed codes: `language_not_in_cohort`, `locale_not_in_cohort`
(a region that differs from the recorded locale), `invalid_language`,
`unknown_task` and `task_not_supported`. Messages never echo the request.

This gate governs support claims and callers that opt in to it. It does not
change how `deidentify()` routes a language today. Experimental output stays
assistive and needs human review.

## Graduation contract

`evaluate_graduation()` checks aggregate evidence for one locale and task.
Every gate is independent; a single failure keeps the locale experimental.
A contract may tighten any threshold but never loosen it past its policy floor,
and it may add critical labels but never remove one.

| Gate | Default | Policy floor or cap |
| ---- | ------- | ------------------- |
| Slice size | 200 documents | at least 200 |
| Critical labels | `DATE`, `ID_NUM`, `PERSON`, `PHONE`, `STREET_ADDRESS`, `ZIPCODE` | cannot shrink |
| Gold items per critical label | 100 | at least 100 |
| Per-label point recall | 0.99 | at least 0.99 |
| Per-label Wilson 95% lower bound | 0.95 | at least 0.95 |
| Clinical terminology slice | 200 reviewed terms | at least 100 |
| Terminology coverage | 0.90 | at least 0.80 |
| Abstention probes | 50, zero failures | at least 20 |
| Cross-lingual regression per cell | 0.00 | at most 0.01 |
| Trials per cross-lingual cell | 100 | at least 50 |
| Offline package | verified offline, at least one device tier, 1 GiB | at most 2 GiB |
| Maintenance review interval | 180 days | at most 365 days |

Critical-identifier recall is **non-compensable**. Each critical label must
pass both its point recall and its Wilson lower bound on its own, and a
missing label fails rather than being skipped. The report includes the
micro-averaged recall for context, but a high average never offsets one
failing label. A small slice with perfect point recall still fails the lower
bound, which is how uncertainty enters the decision.

Cross-lingual regression is **bounded per cell**. The caller names the
already-supported reference languages (never the candidate itself), and every
`(reference language, critical label)` cell must be present, large enough and
within the regression bound. An improvement in one cell cannot offset a drop
in another. Unexpected cells also fail.

The remaining gates also block graduation:

- the review role must be `active` in the cohort record and the evidence must
  carry an `approved` review from the same role;
- evaluation fixtures must be native-authored synthetic or native-reviewed
  translations, so machine translation alone never graduates a locale;
- routes without bundled weights (`user_supplied_model`, `national_id_only`)
  cannot graduate a model-backed task.

`graduate()` re-runs the evaluation and returns the promoted record with the
report digest bound to the task. A record cannot be declared `supported`
without such a digest, an active reviewer and bundled weights.

## Evidence and readiness

Evidence is content-free: counts, controlled codes and 64-character SHA-256
references to lexicon, fixture, evaluation, native-review and package
artifacts held elsewhere. `GraduationEvidence.to_readiness_evidence()`
projects those references onto `kind`, `task`, `digest` and `model_tier`
fields, so the same evidence can feed the clinical language readiness
inventory without either module importing the other.

Reports and cohort JSON are deterministic and contain only codes, counts,
ratios and digests. They contain no reviewer names, note text, spans, paths or
model outputs.

!!! warning "Scope"
    A `supported` status is a release claim for the listed task only. It is
    not clinical validation, certification or cultural validation, and it does
    not make autonomous clinical or disclosure decisions. Digests reference
    evidence; they do not prove that the referenced study was sound.
