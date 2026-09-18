# Section-preserving summary planning

`build_summary_section_plan` creates the bounded hand-off between reviewed
clinical evidence and a local summary generator. It groups approved evidence
by an explicit stable `section_id`, keeps each evidence reference attached to
that same section, and applies deterministic source ordering. The planner does
not generate a summary or make a clinical decision.

```python
from openmed.clinical import build_summary_section_plan

plan = build_summary_section_plan(
    [
        {
            "evidence_id": "e-02",
            "section_id": "s-assessment",
            "source_offset": {"start": 40, "end": 52},
            "approved": True,
        },
        {
            "evidence_id": "e-01",
            "section_id": "s-history",
            "source_offset": {"start": 10, "end": 22},
            "approved": True,
        },
    ]
)

assert plan.ready
assert [group.section_id for group in plan.groups] == [
    "s-history",
    "s-assessment",
]
```

The returned plan is metadata-only. Each reference contains its evidence ID,
section ID, approval marker, and optional source offsets. A local generator can
resolve those offsets against the operator-controlled source document while
the plan report remains free of source text, extracted values, labels, and
model output. The planner has no model or network dependency.

## Approval and ordering

The input is an approved-evidence boundary. Missing approval metadata means
the caller has already supplied approved evidence. An explicit `approved=False`
or a rejected/pending review status excludes that item. Selected references are
sorted by source start/end offsets and then stable evidence ID. Section groups
are sorted by detected section offsets when optional `sections` metadata is
provided; otherwise the first evidence offset, then section ID, determines
their order. Reordering the input cannot change `plan.to_json()`.

```python
plan = build_summary_section_plan(
    evidence,
    sections=[
        {"id": "s-history", "start": 0, "end": 30},
        {"id": "s-assessment", "start": 30, "end": 70},
    ],
)
```

## Refusal is fail-closed

Every selected item must carry an explicit stable section identifier. The
planner does not infer one from a section label, source text, or offsets. If an
approved item is unscoped, the whole plan is refused so a generator cannot
silently mix evidence from different sections:

```python
from openmed.clinical import SummaryPlanRefusalReason

refused = build_summary_section_plan(
    [{"evidence_id": "e-unscoped", "source_offset": {"start": 1, "end": 4}}]
)
assert refused.refusal_reason is SummaryPlanRefusalReason.MISSING_SECTION_ID
assert refused.groups == ()
```

`SummaryPlanRefusal` exposes a finite typed reason and rejected count. Its JSON
and `SummarySectionPlanError` contain only fixed reason codes and counts; they
never echo submitted IDs, source surfaces, or upstream exception messages.
Call `require_summary_section_plan` when the local generation boundary should
raise instead of returning a refusal.

All plans carry `requires_clinician_review=True` and
`autonomous_decision=False`. They are assistive organization artifacts and do
not certify extraction quality, establish patient facts, or authorize clinical
action.
