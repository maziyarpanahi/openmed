# Community Health Worker Form De-identification

OpenMed can de-identify local submission exports from ODK Central, CommCare HQ,
and KoBoToolbox before a form leaves the collection device or facility. The
handler accepts nested JSON and CSV/TSV, including long-format repeat exports,
and does not call any platform API.

## Quick start

Use `redact_chw_form` when you need the structured result and PHI-free manifest:

```python
from pathlib import Path

from openmed.multimodal import redact_chw_form

result = redact_chw_form(
    Path("submissions.csv"),
    platform="odk",  # optional when platform metadata is present
)

Path("submissions.deidentified.csv").write_text(result.text, encoding="utf-8")
print(result.platform, result.row_count)
print(result.manifest)  # policies and counts only; never raw form values
```

The generic multimodal dispatcher also detects XForm exports before falling
back to the normal tabular CSV handler:

```python
from openmed.multimodal import redact_document

document = redact_document("submissions.json")
print(document.metadata["platform"])
print(document.metadata["redaction_manifest"])
```

The narrative pass uses OpenMed's regular text de-identification pipeline. Keep
the selected model cached on the facility machine when the workflow must remain
fully offline.

## XForm field policy

Slash- and dot-delimited paths are interpreted as group/repeat paths. For
example, `household/members/national_id` and
`form.household.members.national_id` both classify the leaf as `ID_NUM` while
retaining the original key or column header.

| Field shape | Default action |
| --- | --- |
| `_uuid`, `_submission_time`, `deviceid`, `instanceID`, case IDs, `KEY`, `PARENT_KEY` | Deterministic hash |
| Person, phone, date of birth, and street-address paths | Canonical-label mask |
| National, patient, household, and record IDs | Deterministic hash |
| Visit/encounter dates | Deterministic per-record date shift that preserves intervals |
| `geopoint`, `geotrace`, `geoshape`, GPS, and coordinate paths | Generalize latitude/longitude; remove altitude and accuracy |
| Narrative descriptions, comments, counselling notes, and visit notes | Text de-identification pipeline |
| Unclassified text, including select-one/select-multiple answers | Run the text pipeline; coded values without detected PII remain unchanged |

The default geopoint output retains only rounded latitude and longitude. For a
district-only release, prefer dropping the coordinate and releasing a separate
district code obtained from the program's local administrative hierarchy. This
mirrors the DHIS2 rule of releasing the district ancestor rather than a
facility or household location.

## Policy overrides

Platform metadata and coordinates can be removed instead of transformed:

```python
result = redact_chw_form(
    "submissions.json",
    policy={
        "metadata_action": "drop",
        "geopoint_action": "drop",
    },
)
```

Metadata supports `hash` or `drop`. Geopoints support `generalize_geo` or
`drop`. `geo_precision` controls rounding from 0 through 4 decimal places. Use
`header_heuristics` to map a program-specific field to a canonical label and
`action_overrides` to set a specific path, leaf, canonical label, or policy
class to `keep`, `mask`, `hash`, `drop`, `date_shift`,
`free_text_redact`, or `generalize_geo`.

```python
result = redact_chw_form(
    "submissions.csv",
    policy={
        "header_heuristics": {"beneficiary_alias": "PERSON"},
        "action_overrides": {
            "visit/other_answer": "free_text_redact",
            "household/geopoint": "drop",
        },
    },
)
```

Dropping a CSV field removes that column. Dropping a JSON field removes its key;
repeat lists and the remaining nested structure are preserved.

## Synthetic platform fixtures

The repository includes paired JSON and CSV fixtures for each supported export
shape under `tests/unit/multimodal/fixtures/chw_forms/`:

- `odk.json` and `odk.csv`: ODK Central/OData-style metadata, slash paths, and
  repeat rows
- `commcare.json` and `commcare.csv`: nested `form`, `meta`, `case`, and dotted
  paths
- `kobo.json` and `kobo.csv`: KoBo metadata, slash-delimited fields, and nested
  repeats

Run the file-based example against any fixture:

```bash
uv run python examples/chw_form_deid.py \
  tests/unit/multimodal/fixtures/chw_forms/odk.json \
  --platform odk \
  --output /tmp/odk.deidentified.json \
  --manifest /tmp/odk.manifest.json
```

The output is deterministic for the same input, policy, and deterministic text
redactor. JSON is emitted with stable key ordering, and CSV uses stable headers,
row order, and line endings. The manifest includes field paths, canonical
labels, actions, row counts, value counts, and affected counts, but never cell
or answer values.

For safety, an unrecognized text field is not assumed to be coded: it passes
through the text de-identification pipeline. Numeric and boolean JSON values
retain their original types, and coded text with no detected PII is emitted
unchanged. CSV rows with a different width from the header are rejected rather
than truncated.

## Privacy boundary

Only file exports are in scope. The handler does not fetch submissions from ODK
Central, CommCare HQ, or KoBoToolbox and does not upload the result. Media
attachments require the existing image/OCR redaction paths. Use only synthetic
fixtures in tests and keep raw production exports inside the facility's
authorized storage boundary.
