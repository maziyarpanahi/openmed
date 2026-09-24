# SDOH Sensitive-Use Labels

SDOH fields can affect high-impact decisions. `label_sdoh_output()` wraps every
exported field in an envelope with machine-readable allowed purposes,
prohibited automated uses, and a mandatory human-review flag.

```python
from openmed.clinical.sdoh_sensitive_use import (
    label_sdoh_output,
    serialize_labeled_sdoh_output,
)

export = label_sdoh_output(
    {"category": "synthetic_housing", "status": "unknown"}
)
payload = serialize_labeled_sdoh_output(export)
```

The default label permits clinical review, care coordination, and a
patient-requested summary. It prohibits automated eligibility, insurance
underwriting, employment, care-denial, and autonomous-diagnosis decisions.

Consumers may supply custom `SDOHSensitiveUseLabel` values, but the field names
must exactly match the exported fields. Partial labels, duplicate labels, and
unlabeled serialization are rejected. `serialize_labeled_sdoh_output()` accepts
only a validated `LabeledSDOHExport`, so labels cannot be accidentally dropped
by that serializer.

The envelope does not authorize a use merely because it is listed. Callers must
still apply consent, law, policy, and human-review requirements. Field values
are excluded from object representations and validation errors; intentional
serialization includes them only inside the labeled envelope. The module is
deterministic and performs no network calls.
