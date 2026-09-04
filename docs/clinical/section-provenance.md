# Clinical section-boundary provenance

`openmed.clinical.sections.provenance` validates the relationship between
normalized clinical section ranges and their source offsets. It is a
structural audit aid, not a medical device, compliance certification, or
clinical decision guarantee.

## Validate section ranges

Ranges use half-open character offsets. Top-level ranges are expected to be
ordered, non-overlapping, and to cover the source document:

```python
from openmed.clinical.sections.provenance import validate_section_provenance

source = "HPI: synthetic cough.\nPLAN: synthetic follow-up."
plan_start = source.index("PLAN")
report = validate_section_provenance(
    source,
    (
        {"id": "hpi", "start": 0, "end": plan_start},
        {"id": "plan", "start": plan_start, "end": len(source)},
    ),
)

assert report.valid
print(report.to_json())
```

The validator also accepts `SectionSpan` values and objects with `start` and
`end` attributes. Set `require_coverage=False` when validating a partial view
of a document.

## Validate source-map references

Source-map entries can be supplied directly on a range or through a local
mapping keyed by section id or input index:

```python
report = validate_section_provenance(
    source,
    (
        {"id": "hpi", "start": 0, "end": plan_start},
        {"id": "plan", "start": plan_start, "end": len(source)},
    ),
    {
        "hpi": {
            "source_start": 0,
            "source_end": plan_start,
            "source_ref": "src-a",
        },
        "plan": {
            "source_start": plan_start,
            "source_end": len(source),
            "source_ref": "src-b",
        },
    },
    require_source_map=True,
)
```

The source map is local caller-supplied data; the validator never downloads or
resolves one. A repeated reference must resolve to the same source range.
Source ranges are checked for ordering, overlap, bounds, and optional supplied
content hashes. When no explicit map is supplied, ordinary source-indexed
ranges use deterministic identity references.

## Parent containment and privacy

A child range can name a parent with `parent_id`. The child must be contained
by that parent's normalized range. Parent definitions may also be supplied via
the `parent_sections=` argument.

Reports contain only structural offsets, category/code values, counts, and
SHA-256 hashes. They do not copy labels, identifiers, source-map references, or
section text. `report.to_json()` and `report.write_json(...)` are deterministic
for identical inputs. Findings such as `gap`, `overlap`, `outside_parent`, and
`source_map` conflicts can be reviewed without exposing the source document.
