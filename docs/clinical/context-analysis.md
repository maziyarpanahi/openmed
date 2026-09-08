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
