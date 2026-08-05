# De-identified DHIS2 export

OpenMed can prepare aggregate data-value sets and tracker import objects before
they leave a facility. The exporter is deliberately transport-free: it reads
payloads and an organisation-unit hierarchy from local JSON, applies privacy
policy in memory, and returns deterministic JSON. Upload credentials, HTTP
requests, and live DHIS2 metadata changes remain outside OpenMed.

The output follows the DHIS2 structures used by the
[`/api/dataValueSets`](https://docs.dhis2.org/en/develop/using-the-api/dhis-core-version-243/data.html)
and [`/api/tracker`](https://docs.dhis2.org/en/develop/using-the-api/dhis-core-version-243/tracker.html)
imports. `aggregate.json` is a local collection whose `dataValueSets` members
are individual data-value-set request bodies; `tracker.json` contains the
`trackedEntities` and `events` arrays accepted by the tracker import.

## Facility-to-district flow

Run the steps in this order:

1. Export the facility's aggregate and tracker JSON to a protected local
   workspace.
2. Export a local `/api/organisationUnits` snapshot containing every referenced
   unit and its ancestors, with `id`, `level`, and `parent.id` fields.
3. Run `export_dhis2`. Free text is de-identified first, precise geometry is
   removed, fine-grained `orgUnit` references are replaced with the configured
   ancestor, dates are shifted or coarsened, and small aggregate cells are
   suppressed.
4. Inspect the endpoint payloads and the value-free manifest locally.
5. Only then hand the endpoint bodies to the facility's existing authenticated
   uploader.

The default target is level 3, conventionally district in a
country/province/district/facility tree. DHIS2 hierarchies are configurable, so
confirm the local level numbering before export. Any unknown unit, broken
parent link, cycle, or missing target-level ancestor fails closed. The exporter
does not query a server to fill gaps. DHIS2 describes organisation units as a
single hierarchy that supplies geographic context to data values, tracked
entities, and events in its
[organisation-unit guidance](https://docs.dhis2.org/en/implement/maintenance-and-use/organisation-unit-maintenance.html).

## Python API

```python
import json
from pathlib import Path

from openmed.clinical.exporters import DHIS2ExportConfig, export_dhis2

aggregate = json.loads(Path("facility-aggregate.json").read_text())
tracker = json.loads(Path("facility-tracker.json").read_text())

result = export_dhis2(
    aggregate,
    tracker,
    "organisation-units.json",
    config=DHIS2ExportConfig(
        generalization_level=3,
        small_cell_threshold=5,
        date_mode="shift",
        # Omit this for a deterministic per-record non-zero shift.
        date_shift_days=30,
    ),
)

Path("aggregate.json").write_text(result.aggregate_json() + "\n")
Path("tracker.json").write_text(result.tracker_json() + "\n")
Path("manifest.json").write_text(result.manifest_json() + "\n")
```

By default, `comment`, `storedBy`, `completedBy`, tracker attribute and data-value
`value` strings, and note `value` strings pass through OpenMed's
`hipaa_safe_harbor` de-identification pipeline. Non-string values in these
sensitive string fields fail closed, except for the `null` values DHIS2 uses to
delete attributes and data values. A caller may provide a local `text_redactor`
callable when it already owns a warmed model or a stricter pipeline. The
callable must return a string or an object with a `deidentified_text` string.

## Geography, dates, and small cells

- `generalization_level=3` replaces every level-4 facility UID with its level-3
  ancestor. References already at level 3 or above stay unchanged.
- Exact `geometry`, `latitude`, and `longitude` fields are removed rather than
  carried into the de-identified payload.
- `date_mode="shift"` uses the same deterministic, per-record offset derivation
  as tabular quasi-identifier redaction. Set `date_shift_days` to reuse one
  explicit non-zero offset.
- `date_mode="coarsen"` replaces event dates with the first day of the month or
  year and coarsens compatible DHIS2 reporting periods. Set
  `period_granularity` to `"month"` or `"year"`.
- `small_cell_threshold=5` removes numeric aggregate data values from 0 through
  4. Use `0`, or construct a config with `None`, only when an approved policy
  explicitly disables suppression.

Array members and JSON keys are serialized canonically, so identical input and
configuration produce byte-identical output. Suppression and transformation
paths are recorded without source values.

## Manifest and review gate

The manifest contains only policy settings, counts, and structural paths. It
does not contain comments, attribute values, usernames, organisation-unit UIDs,
or suppressed cell values. Before upload, verify at least:

- `counts.org_units_generalized` matches the expected facility references;
- `counts.precise_locations_removed` matches the expected coordinate fields;
- `counts.suppressed_aggregate_values` matches the low-volume review;
- `transformed_paths` covers all expected free-text and date fields;
- the serialized endpoint payloads pass the facility's no-PHI gate and DHIS2
  metadata validation.

For a command-line reference, run:

```bash
python examples/dhis2_district_export.py \
  --aggregate facility-aggregate.json \
  --tracker facility-tracker.json \
  --org-units organisation-units.json \
  --output-dir deidentified-dhis2
```

This command writes local files only. It never uploads them.
