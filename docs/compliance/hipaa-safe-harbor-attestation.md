# HIPAA Safe Harbor Attestation

OpenMed can turn a de-identification audit report into a PHI-safe evidence
artifact covering all 18 HIPAA Safe Harbor identifier categories. The
attestation records aggregate detection counts, observed actions, configured
policy actions, and hashes. It does not copy source text, span offsets,
surrogates, context, or detector evidence from the audit report.

The artifact is evidence from one run. It is not a certification, legal
sign-off, or substitute for the Safe Harbor “actual knowledge” assessment.

## Generate an attestation offline

First create an audit report during local de-identification:

```bash
openmed deid \
  --input note.txt \
  --output run-audit.json \
  --policy hipaa_safe_harbor \
  --audit
```

Then generate the attestation. Both commands run locally; the attestation
command makes no network requests:

```bash
openmed compliance safe-harbor run-audit.json \
  --output safe-harbor-attestation.json
```

Omit `--output` to print the JSON artifact to standard output.
Add `--json` to wrap stdout in OpenMed's uniform machine-readable command
envelope; the attestation is returned in its `data` field.
The command rejects an audit report whose reproducibility hash no longer
matches its contents.

## Report contents

The report contains:

- `source_report_hash`, which ties the evidence to the source audit run;
- `attestation_hash`, which makes the generated artifact independently
  reproducible;
- exactly 18 ordered category records;
- the canonical OpenMed labels mapped to each category;
- the configured policy action for every mapped label;
- aggregate detection and applied-action counts from the run; and
- explicit residual-risk flags and reasons requiring qualified expert review.

The report never includes raw span text, offsets, context, replacement values,
surrogates, or per-detector evidence. Store and handle the source audit report
under the controls appropriate for your deployment even though OpenMed audit
reports are designed to be PHI-safe.

## Coverage mapping

The mapping is derived from `openmed.core.labels.LABEL_TO_HIPAA`, so the report
stays aligned with the canonical label policy spine.

| Safe Harbor category | Canonical label coverage |
|---|---|
| Names | `PERSON`, `FIRST_NAME`, `LAST_NAME`, `MIDDLE_NAME`, `PREFIX` |
| Geographic subdivisions | `LOCATION`, `STREET_ADDRESS`, `BUILDING_NUMBER`, `ZIPCODE`, `GPS_COORDINATES`, `ORDINAL_DIRECTION` |
| Date elements and ages over 89 | `DATE`, `DATE_OF_BIRTH`, `TIME`, `AGE` |
| Telephone numbers | `PHONE` |
| Fax numbers | No dedicated canonical mapping |
| Email addresses | `EMAIL` |
| Social Security numbers | `SSN` |
| Medical record numbers | No dedicated canonical mapping |
| Health plan beneficiary numbers | No dedicated canonical mapping |
| Account numbers | Account and financial-identifier labels mapped by the canonical cross-map |
| Certificate and license numbers | No dedicated canonical mapping |
| Vehicle identifiers and serial numbers | `VIN`, `VEHICLE_REGISTRATION` |
| Device identifiers and serial numbers | `MAC_ADDRESS`, `IMEI` |
| Web URLs | `URL` |
| IP addresses | `IP_ADDRESS` |
| Biometric identifiers | No dedicated canonical mapping |
| Full-face photographs and comparable images | No dedicated canonical mapping |
| Other unique identifying numbers or codes | Remaining labels mapped to `UNIQUE_IDENTIFIER` by the canonical cross-map |

Categories without a canonical mapping are always flagged as residual-risk
items requiring qualified expert review. Specialized span producers can supply
a valid `hipaa_safe_harbor_class` / `safe_harbor_class` tag or a Safe Harbor
class in `regulatory_tags` to attribute a detection count more precisely. Such
a tag does not turn an otherwise uncovered category into canonical detector
coverage, so its residual-risk flag remains visible.

The report also flags a covered category when any observed detection used the
`keep` action.
