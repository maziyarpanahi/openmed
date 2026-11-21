# Build and verify a capability evidence pack

A pack binds the SDK version, a specific intended-use declaration, required
evidence lanes and aggregate measurements to a digest. A reviewer can check
that the measurements survived transport, the signer used the expected key,
the required lanes are present and the pack falls within the site's age limit.

Every pack carries these mandatory limitation codes verbatim:

- `not_clinically_validated`
- `not_a_medical_device`

This is technical evidence. Clinical validation and certification are separate
processes. Local validation remains necessary for the site's population,
workflow, equipment and human review arrangements.

## Select the capability and tier

Read the capability's declaration in `openmed/core/data/intended_use.json`.
Its tier determines additional evidence obligations. Informational use adds
notice coverage. Assistive review adds acknowledgement coverage. Restricted
use also requires human-factors and shadow-comparison evidence. A deployment
may strengthen the tier. The verifier checks the registered obligations and
the declaration digest packaged with its installed SDK.

Each lane records its state, required metrics, measured values and source
report digest. A missing report produces an unknown lane with missing metrics.
An observed zero is a measurement and must come from the report itself.

## Run the offline example

Use an empty local working directory with the OpenMed checkout's development
environment activated. The following commands create five synthetic aggregate
reports and a disposable signing key. The numbers exercise the wire format
only. No model, patient records or network connection is needed.

```sh
python - <<'PY'
import hashlib
import json
from pathlib import Path

inputs = Path("reports")
inputs.mkdir()
measurements = {
    "g1a": {"direct_identifier_recall": 1.0},
    "g2": {"recall": 1.0},
    "g3": {"critical_leakage_count": 0},
    "g4": {"quantized_recall_delta": 0.0},
    "span_integrity": {"invalid_span_count": 0},
}
index = []
for lane, metrics in measurements.items():
    report = {"lane": lane, "state": "success", "metrics": metrics}
    raw = json.dumps(report, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    (inputs / f"{digest}.json").write_bytes(raw)
    index.append({"lane": lane, "type": "review_metrics", "sha256": digest})
(inputs / "reports.json").write_text(json.dumps(index), encoding="utf-8")
PY
openmed evidence build --capability deidentify --inputs reports --output pack.json
```

A complete build returns `UNSIGNED`. Inspect the lanes, version, limitations
and source digests before signing. This example creates a new private key
file with exclusive creation and binds the resulting HMAC to the key ID
`demo`. Production reviewers need a trusted key delivered separately from
the pack. Each holder of a shared HMAC key can also sign.

```sh
python - <<'PY'
import os
import secrets
from pathlib import Path
from openmed.eval.evidence_pack import ValidationEvidencePack
from openmed.eval.evidence_pack.verify import sign_evidence_pack

pack = ValidationEvidencePack.from_json(Path("pack.json").read_bytes())
assert not pack.lane_blockers()
key = secrets.token_bytes(32)
with os.fdopen(os.open("demo.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as handle:
    handle.write(key)
signed = sign_evidence_pack(pack, key_id="demo", key=key)
with Path("signed.json").open("x", encoding="utf-8") as handle:
    handle.write(signed.to_json())
PY

as_of=$(python -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat())')
openmed evidence verify signed.json --key-id demo --key-file demo.key \
    --as-of "$as_of" --max-age-days 7 --json
```

The verification returns exit 0 and `data.ready` is true for this synthetic
example. Keep the exact evaluation timestamp and age limit with the result so
another reviewer can replay it. At the age limit the pack remains current.
A later evaluation beyond the limit yields `expired_evidence`. A timestamp
before pack generation yields `future_timestamp`.

## Interpreting failures

| Outcome | What to inspect |
| --- | --- |
| Build exits 1 with `NOT_READY` | Inspect the `missing:` blockers and the report index. An unsigned pack can still be inspected. Rebuild with `--overwrite` after restoring the missing input. |
| Verify exits 1 | Check the typed reasons. `digest_mismatch` can indicate changed metrics or lane order, `signature_mismatch` can indicate a wrong key, and `intended_use_mismatch` means the local declaration differs. Review the evidence before signing a replacement. |
| Verify exits 2 | Inspect the JSON format, key configuration and timezone-aware evaluation timestamp locally. Diagnostics omit report content. |

Use the [independent review checklist](../compliance/assurance/index.md) to
record unresolved items and the [evidence freshness guide](../compliance/evidence-freshness.md)
to select the site's age limit.
