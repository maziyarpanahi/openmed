# Radiology Finding Profile

`openmed.clinical.radiology_profile` is a local, deterministic profile for
turning synthetic or caller-owned radiology prose into reviewable finding
records. It composes the existing finding lexicon and report section parser;
it does not download a model, contact a terminology service, or infer a
diagnosis.

!!! warning "Review aid only"

    The profile is not a medical device or a diagnostic decision engine. A
    qualified reviewer must validate the source report and any downstream use.

## Extract findings

```python
from openmed.clinical import extract_radiology_profile

text = (
    "FINDINGS: No focal consolidation in the left lower lobe. "
    "Possible 8 mm nodule in the right upper lobe."
)
records = extract_radiology_profile(text)
```

The profile reads only `FINDINGS` and `IMPRESSION` sections. A report without
headings is treated as findings prose. Recommendation text is deliberately
not interpreted as a finding, so a follow-up suggestion cannot create a new
clinical fact.

## Record contract

Each returned mapping contains:

| Field | Meaning |
| --- | --- |
| `finding` | Small normalized surface label from the existing transparent lexicon. |
| `laterality` | `left`, `right`, `bilateral`, or explicit `unknown`. |
| `size_value`, `size_unit` | Written measurement, or `None` when absent. |
| `location` | Written anatomical location, or `None` when absent. |
| `assertion` | `affirmed`, `negated`, or `unknown` for uncertain/hypothetical wording. |
| `section` | `findings` or `impression`. |
| `unknown_fields` | Stable list naming missing or indeterminate fields, such as `size` or `assertion`. |
| `evidence` | Offset-only links into the caller's source text. |

For example, a negated synthetic finding has an explicit state rather than
being silently discarded:

```python
record = records[0]
record["finding"]       # "consolidation"
record["assertion"]     # "negated"
record["laterality"]    # "left"
record["unknown_fields"]  # ["size"]
```

`evidence` contains half-open `start`/`end` offsets for the finding,
laterality, measurement value and unit, location, assertion cue, and section.
It does not copy the surrounding report or emit raw source snippets. Callers
should retain and protect the original text according to their own privacy
policy.

## Determinism and boundaries

The profile uses sentence- and clause-scoped rules. Negation cues such as
`no` bind only when they reach the finding; `possible` and similar uncertainty
cues produce the explicit `unknown` assertion state. Existing proximity
limits prevent a distant measurement or laterality mention from crossing into
another finding. Supplying `radlex_mapping` is optional and caller-owned; no
ontology is bundled or fetched.

`RADIOLOGY_FINDING_PROFILE` is the reusable callable profile object, while
`extract_radiology_profile()` is the concise function form. Both produce the
same records for the same input and options.
