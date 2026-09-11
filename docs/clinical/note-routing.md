# Section-aware note routing

`openmed.clinical.note_router` turns the existing section detector output into
a small, profile-aware extraction plan. It is intended for local clinical
pipelines that want radiology or pathology extractors to see only the sections
they understand while retaining absolute source offsets.

The default profiles are:

| Profile | Sections |
| --- | --- |
| `radiology` | `technique`, `findings`, `impression` |
| `pathology` | `specimen`, `diagnosis`, `synoptic`, `staging`, `grading` |

Every note is first represented as a complete, non-overlapping partition. If
the caller supplies section spans, the router validates that they cover the
whole note from offset `0` through `len(text)`. The router then emits one
`SectionRoute` per section. A section that does not match a registered profile
uses the `unknown` profile and is never sent to a specialized extractor by
accident.

## Route a note

```python
from openmed.clinical.note_router import NoteRouter


def extract_radiology(source_text, route):
    # `route.start` and `route.end` are absolute half-open offsets.
    section_text = source_text[route.start : route.end]
    return {"section": route.label, "offset": route.offset, "text": section_text}


router = NoteRouter(extractors={"radiology": extract_radiology})
plan = router.route(
    "RADIOLOGY REPORT\nFINDINGS: Synthetic observation."
)

for route in plan.routes:
    print(route.profile, route.label, route.offset)
```

`route.to_dict()` contains labels, profile names, offsets, section-detector
metadata, and fallback reasons. It intentionally does not include source text.
Extractor values are kept in memory by `NoteExtractionResult.extractions`; its
`to_dict()` method serializes routing metadata without serializing those
values. Callers remain responsible for applying their own PHI-safe output
contract to extractor results.

## Run local extractors

Use `NoteRouter.extract()` when the configured extractors should be invoked.
The extractor receives `(source_text, route)`, so an extractor can preserve
absolute offsets in its own structured output. A one-argument extractor is
also accepted and receives a `SectionInput` with `.text`, `.content_text`,
`.offset`, and `.route` properties.

An optional `unknown_extractor` can process conservative fallback sections:

```python
router = NoteRouter(
    extractors={"radiology": extract_radiology},
    unknown_extractor=lambda section: {
        "section": section.route.label,
        "offset": section.offset,
    },
)
result = router.extract("HPI: Synthetic narrative.")
```

The router itself is deterministic, rules-first, and offline. It does not load
a model, fetch terminology, read credentials, or make a mandatory network
call. It is assistive extraction plumbing and does not make clinical
decisions.
