# DICOM-SR provenance mapping

`openmed.multimodal.dicom_sr_provenance` links an opaque finding identifier to
the deterministic content-tree path emitted by the DICOM-SR extractor. The
result carries only structural evidence references:

- `finding_id`: a caller-supplied stable identifier;
- `item_path`: the 1-based dotted path used by `walk_sr_content_tree`;
- `template_id`: the item's template, or the nearest declared ancestor
  template; and
- `source_start` / `source_end`: half-open offsets into the extracted text when
  a matching `SourceSpan` or explicit finding offsets are available.

Concept names, rendered values, units, report text, and arbitrary finding
metadata are not copied into provenance records. Input paths and source spans
are validated strictly. Duplicate paths, conflicting path aliases, duplicate
source spans, and offsets matching multiple paths raise
`AmbiguousDicomSrItemPathError` instead of choosing a possibly incorrect
evidence link.

## Example

```python
from openmed.multimodal import extract_dicom_sr
from openmed.multimodal.dicom_sr_provenance import (
    build_dicom_sr_provenance,
    serialize_dicom_sr_provenance,
)

document = extract_dicom_sr("report.dcm")
findings = [
    {"finding_id": "finding-001", "item_path": "1.3.1.3"},
]

records = build_dicom_sr_provenance(findings, document=document)
print(serialize_dicom_sr_provenance(records))
```

The mapper is deterministic and local-only. It is an evidence-linking aid, not
a clinical interpretation or a substitute for qualified review.

Explicit offsets must fit the declared item span, and source spans must fit a
supplied document. Conflicting identifier, template, path and offset aliases
are rejected. Collections are limited to 4,096 entries and item paths to 64
levels; reference strings are limited to 4,096 characters. Rendering revalidates
typed records, and public errors discard raw exception context. Finding and
template identifiers are caller-owned opaque references, not anonymized values;
never place report text or patient identifiers in those fields.
