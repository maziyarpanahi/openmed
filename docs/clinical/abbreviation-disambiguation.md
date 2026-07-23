# Clinical Abbreviation Disambiguation

Clinical short forms are often ambiguous. `MS`, for example, can refer to
multiple sclerosis, mitral stenosis, or morphine sulfate. OpenMed provides a
small offline sense inventory and a deterministic resolver that ranks meanings
from three kinds of local evidence:

- the clinical section containing the abbreviation;
- semantic types of nearby entities; and
- literal cue words or phrases in the surrounding text.

This feature is assistive support. Review ambiguous or low-evidence resolutions
before clinical use.

## Resolve one abbreviation

```python
from openmed.clinical import disambiguate_abbreviation

sense = disambiguate_abbreviation(
    "PT",
    "PT was prolonged at 18 seconds with an INR of 1.8.",
    section="coagulation",
    entity_types=("laboratory_test", "lab_value"),
)

assert sense is not None
assert sense.long_form == "prothrombin time"
print(sense.score)
print([alternative.long_form for alternative in sense.alternatives])
```

The winning `AbbreviationSense` includes its long form, semantic type, source,
normalized score, matched features, and every non-winning candidate in ranked
`alternatives`. The winner and alternative scores sum to approximately `1.0`.
An unknown short form returns `None` without making a network request or raising
an error.

## Annotate detected spans

`expand_abbreviations()` accepts mappings or objects with `start` and `end`
offsets. Span-level `section` metadata and nearby labelled spans contribute
context:

```python
from openmed.clinical import expand_abbreviations

text = "Laboratory: PT was prolonged at 18 seconds with INR 1.8."
pt_start = text.index("PT")
value_start = text.index("18")

annotations = expand_abbreviations(
    text,
    [
        {
            "start": pt_start,
            "end": pt_start + 2,
            "label": "ACRONYM",
            "section": "laboratory",
        },
        {
            "start": value_start,
            "end": value_start + 2,
            "label": "LAB_VALUE",
        },
    ],
)

assert annotations[0].sense is not None
assert annotations[0].sense.long_form == "prothrombin time"
```

Known inventory entries are recognized even when their spans have no label.
Unknown spans explicitly labelled `ABBREVIATION` or `ACRONYM` are retained with
`sense=None`. Invalid or out-of-range offsets raise `ValueError` rather than
being silently moved.

## Supply a local inventory

The bundled file is a manually authored synthetic starter set under `CC0-1.0`.
It is not derived from UMLS LRABR or another restricted vocabulary. No network
or external terminology service is used.

A local JSON inventory uses this schema:

```json
{
  "schema_version": 1,
  "provenance": {
    "source": "my local synthetic inventory"
  },
  "senses": {
    "HCM": [
      {
        "long_form": "hypertrophic cardiomyopathy",
        "semantic_type": "condition",
        "source": "local-clinical-team",
        "sections": ["cardiology"],
        "entity_types": ["condition"],
        "cue_words": ["septal hypertrophy"],
        "prior": 0.2
      }
    ]
  }
}
```

Load it and pass the result to either API:

```python
from openmed.clinical import disambiguate_abbreviation, load_sense_inventory

inventory = load_sense_inventory("clinic-abbreviations.json")
sense = disambiguate_abbreviation(
    "HCM",
    "Septal hypertrophy supports HCM.",
    section="cardiology",
    inventory=inventory,
)
```

Local candidates with the same short form and long form replace their starter
definition. New long forms extend an existing short form, and new short forms
extend the registry. Use `include_starter=False` to load only the local file.

Inventory authors are responsible for the licensing and permitted use of any
user-supplied terminology data. Restricted resources must remain outside the
OpenMed package and repository.

## Scoring behavior

Scoring is deterministic and uses fixed evidence weights. A matching section
adds `3.0`, each matching nearby entity type adds `2.0`, and each matching cue
adds `1.5`, on top of the candidate's `prior`. Raw candidate scores are divided
by their sum to expose review-friendly normalized scores. Inventory order is
the stable tie-breaker when evidence is equal.
