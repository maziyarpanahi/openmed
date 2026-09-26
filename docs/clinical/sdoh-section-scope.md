# Strict SDOH section scope

`openmed.clinical.sdoh_section_scope` is a deterministic, local-first guard for
SDOH candidate evidence. It keeps a candidate only when its complete source
span is contained by a configured clinical section. The default allow-list is
`social_history`; assessment, plan, templated, and unsectioned content is
excluded when a Social History section is available.

```python
from openmed.clinical.sdoh_section_scope import scope_sdoh_candidates
from openmed.clinical.sections import detect_sections

note = (
    "Assessment: synthetic plan discussion.\n"
    "Social History: synthetic housing cue.\n"
    "Plan: synthetic follow-up instruction."
)
candidates = [
    {"start": note.index("synthetic housing cue"),
     "end": note.index("synthetic housing cue") + len("synthetic housing cue")},
]

scoped = scope_sdoh_candidates(note, candidates, detect_sections(note))
usable_candidates = scoped.candidates
review_report = scoped.report.to_dict()
```

The scope is configurable for workflows that have an approved section map:

```python
scoped = scope_sdoh_candidates(
    note,
    candidates,
    allowed_sections=("social_history", "living_situation"),
    fallback="reject",
)
```

Fallback behavior is explicit. The default `fallback="reject"` excludes every
candidate when no configured section is detected. `fallback="unsectioned"`
allows only candidates inside an explicitly unsectioned range. The opt-in
`fallback="document"` accepts valid document-local candidates when no approved
section exists; callers should use it only when their upstream workflow has a
separate, reviewed boundary guarantee.

`SDOHSectionScopeReport` contains only policy metadata, counts grouped by
section, controlled exclusion reasons, and the fixed review disclaimer. It
does not copy candidate text, source values, or discarded candidate objects.
Its JSON and Markdown representations are therefore suitable for local audit
metadata. The retained candidates remain available in memory for the caller's
next extraction step.

Section detection defaults to the rules-first local detector. No network call
is required, and learned section refinement is not invoked by this helper.
Scope is a review guard, not a diagnosis, autonomous clinical decision, or
compliance certification; downstream SDOH findings still require qualified
human review.
