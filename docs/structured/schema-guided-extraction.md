# Schema-Guided JSON Extraction

`openmed.structured.extract_to_schema` projects a clinical note onto a
caller-supplied **target schema**: you declare the fields you want and their
types, and the API returns a typed JSON object plus per-field source offsets.
It binds material that has *already* been detected -- named entities, inline
`key: value` lines, and reconstructed table cells -- to the schema slots,
coercing each raw value to the declared type. It performs no model inference and
no free-form extraction; everything is deterministic and offline.

## The target schema

The schema is a small, standard subset of JSON Schema: a top-level object with
`properties` and an optional `required` list. Each property declares one scalar
`type` -- `string`, `integer`, `number`, or `boolean`.

Extraction hints ride along as extension keywords that a standard JSON Schema
validator ignores:

| Keyword | Meaning |
| --- | --- |
| `aliases` | Alternative slot labels to match in `key: value` lines and table rows, in addition to the humanized field name. |
| `entity` | An entity label (or list of labels) to bind from detected entities. |
| `enum` | The permitted values; matching is case-insensitive and the canonical form is returned. |
| `pattern` | A regular expression the raw value must fully match. |

```python
schema = {
    "type": "object",
    "required": ["patient_age", "sex"],
    "properties": {
        "patient_age": {"type": "integer", "aliases": ["age"]},
        "sex": {"type": "string", "enum": ["Male", "Female"]},
        "temperature": {"type": "number", "aliases": ["temp"]},
        "smoker": {"type": "boolean", "aliases": ["current smoker"]},
        "facility": {"type": "string", "entity": "HOSPITAL"},
    },
}
```

## Extracting

```python
from openmed.structured import extract_to_schema

note = (
    "Patient Age: 54 years\n"
    "Sex: Female\n"
    "Temperature: 37.8 C\n"
    "Current Smoker: no\n"
)

result = extract_to_schema(note, schema)

result["data"]
# {'patient_age': 54, 'sex': 'Female', 'temperature': 37.8, 'smoker': False}

result["missing_required"]
# []  (both required slots filled)
```

Detected entities and reconstructed tables are passed alongside the text and
take priority over inline text:

```python
entities = [{"label": "HOSPITAL", "text": "Mercy General", "start": 8, "end": 21}]
result = extract_to_schema(note, schema, entities=entities, tables=tables)
```

## What comes back

`extract_to_schema` returns a `SchemaExtraction` mapping:

- **`data`** -- the validated object. Only slots that filled *and* passed
  coercion and constraint checks appear here, so its values always conform to
  the declared types.
- **`bindings`** -- per-field provenance: the coerced `value`, the `raw`
  substring, its `start`/`end` offsets into the source text, and the `source`
  (`entity`, `table`, or `key_value`).
- **`missing_required`** -- required slots that no source filled (or whose only
  candidate failed validation). These are reported, never silently dropped.
- **`errors`** -- candidate values that were found but rejected, each with the
  reason, the raw text, and its offsets.

## Determinism and precedence

For every slot the sources are consulted in a fixed priority order --
**entities, then table cells, then inline `key: value` lines** -- and the first
source to yield a candidate wins. Within a single source, the earliest offset in
the document wins. The same inputs therefore always produce the same object.

A malformed *schema* raises `SchemaDefinitionError`. A malformed or empty
*document* never raises: partial extraction always returns a result, with the
gaps recorded in `missing_required` and `errors`.
