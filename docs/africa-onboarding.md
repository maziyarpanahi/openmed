# African Developer Onboarding

This guide is for teams building OpenMed into African clinical and public-health
systems where connectivity, power, hardware, and data-protection requirements
vary sharply between sites. It covers a small-model path that can be prepared
once and run offline, practical privacy-profile choices, and integration with
OpenMRS and DHIS2.

The commands use synthetic clinical text. Never paste real patient data into a
public issue, hosted notebook, translation service, or shared terminal log.

## Low-bandwidth quickstart

### Choose the 33M anatomy model

The examples use
[`OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-33M`](https://huggingface.co/OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-33M),
an English anatomy token-classification model with 33 million parameters. Its
required model and tokenizer artifacts are about 67 MB in total, making it a
better first download than the 109M or 434M registry entries.

Confirm the local registry estimate before downloading. The command is offline
by default; `OPENMED_OFFLINE=1` also makes that intent explicit:

```bash
OPENMED_OFFLINE=1 openmed models size \
  OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-33M
```

The table should report about 67.9 MB to download and 256 MB estimated peak
RAM. Use `openmed models size --budget-mb 100` to compare every registry model
that fits a 100 MB download budget. Add `--remote` only when you intentionally
want current Hub metadata and have a reliable connection.

This model recognizes anatomy in English text. The Swahili README is a product
and developer translation; it does not imply that this English checkpoint is a
Swahili clinical NER model.

### Install once on a connected machine

Use Python 3.11 and keep the package environment so it can be reproduced on a
machine with the same operating system and CPU architecture:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "openmed[hf]"
```

For multiple identical clinic machines, download the Python packages once into
a wheelhouse and transfer it with the model:

```bash
python -m pip download --dest wheelhouse "openmed[hf]"
```

On the offline machine, install without contacting a package index:

```bash
python -m pip install --no-index --find-links wheelhouse "openmed[hf]"
```

The wheelhouse can be much larger than the model because PyTorch and its
platform dependencies are included. Build it for each target OS, architecture,
and Python version rather than copying one wheelhouse across incompatible
machines.

### Prefetch the model with resumable downloads

Download only inference files. `snapshot_download()` resumes cached files and
checks their content hashes, while `max_workers=2` avoids saturating a fragile
connection:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-33M",
    local_dir="models/anatomy-33m",
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ],
    max_workers=2,
)
```

If the connection drops, run the same script again. For transfer over an
intermittent local link, preserve partial progress:

```bash
rsync --archive --partial --append-verify \
  models/anatomy-33m/ clinic-host:/opt/openmed/models/anatomy-33m/
```

USB or another approved encrypted removable medium is also suitable. Verify
the transfer before use:

```bash
find models/anatomy-33m -type f -print0 \
  | sort -z \
  | xargs -0 shasum -a 256 > anatomy-33m.sha256
```

Keep the checksum file with the deployment record. If you generated it on the
connected machine, verify it on the destination with `shasum -a 256 -c`.

### Run with outbound access disabled

Set local-only mode and point OpenMed at the transferred directory:

```bash
export OPENMED_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

```python
from openmed import OpenMedConfig, analyze_text

config = OpenMedConfig(device="cpu", local_only=True)
result = analyze_text(
    "Tenderness extends from the left kidney to the lower abdomen.",
    model_id="./models/anatomy-33m",
    config=config,
)

for entity in result.entities:
    print(entity.label, entity.text, round(entity.confidence, 3))
```

Because `model_id` is a local path and `local_only=True`, missing files fail
locally instead of triggering a Hub download. Run this smoke test while the
machine is disconnected before introducing any clinical workflow.

## POPIA and Nigeria NDPA pointers

South Africa's **Protection of Personal Information Act (POPIA)** treats health
information as special personal information. Start with the
[official Act](https://www.gov.za/sites/default/files/gcis_document/201409/3706726-11act4of2013protectionofpersonalinforcorrect.pdf)
and the Information Regulator's
[guidance on processing special personal information](https://inforegulator.org.za/wp-content/uploads/2020/07/InfoRegSA-GuidanceNote-Processing-SpecialPersonalInformation-20210628.pdf).

Nigeria's **Data Protection Act 2023 (NDPA)** defines health status, genetic
data, and biometric data used for unique identification as sensitive personal
data. Use the
[official Act](https://ndpc.gov.ng/wp-content/uploads/2024/03/Nigeria_Data_Protection_Act_2023.pdf)
and the [Nigeria Data Protection Commission](https://ndpc.gov.ng/) for current
guidance.

OpenMed policy profiles are technical controls, not legal certifications. No
bundled profile is named `popia` or `ndpa`, and selecting a profile does not by
itself establish a lawful basis, satisfy transparency duties, complete a data
protection impact assessment, authorize cross-border transfer, or set a lawful
retention period. Have the responsible information officer or data protection
officer approve the final configuration and data flow.

### Map deployment intent to existing profiles

| Deployment intent | Existing profile | What the checked-in config does | Review before production |
|---|---|---|---|
| Maximum-conservatism export or unknown recipient | `strict_no_leak` | Uses the strict threshold, high-recall union, mandatory safety sweep, and masks direct identifiers, quasi-identifiers, and clinical concepts | Utility is intentionally low; verify that masked clinical concepts still meet the purpose |
| Authorized care workflow inside the provider boundary | `clinical_minimal_redaction` | Masks direct identifiers while retaining many dates, locations, ages, and clinical concepts | Safety sweep is not mandatory; document why retained fields are necessary and authorized |
| Approved research or public-health dataset | `research_limited_dataset` | Masks direct identifiers, keeps many quasi-identifiers and clinical concepts, and requires the safety sweep | Assess re-identification risk, small cells, linkage risk, retention, access control, and publication rules |
| Health-data pseudonymization starting point | `gdpr_art9_health` | Uses the strict threshold, replaces direct identifiers, masks quasi-identifiers and clinical concepts, and keeps a reversible mapping | Protect the mapping separately; this is not a POPIA or NDPA opinion |

The source of truth is the JSON under `openmed/core/policies/`. Review the
actual actions rather than relying only on a profile name:

```bash
python -m json.tool openmed/core/policies/strict_no_leak.json
python -m json.tool openmed/core/policies/clinical_minimal_redaction.json
python -m json.tool openmed/core/policies/research_limited_dataset.json
python -m json.tool openmed/core/policies/gdpr_art9_health.json
```

Apply a selected profile explicitly. Request the PHI-safe audit report as a
separate result when you need review evidence:

```python
from openmed import deidentify

result = deidentify(
    "Synthetic patient Amina Example called +254 700 000 000.",
    policy="strict_no_leak",
    method="mask",
)

print(result.deidentified_text)

audit_report = deidentify(
    "Synthetic patient Amina Example called +254 700 000 000.",
    policy="strict_no_leak",
    method="mask",
    audit=True,
)
print(audit_report.policy)
print(audit_report.export_review_bundle())
```

Before rollout, record at least:

1. Purpose and lawful basis for each data flow.
2. Data fields retained, masked, replaced, or removed and why each is needed.
3. Whether pseudonym mappings exist, who can access them, and where they live.
4. Data residency, processor/subprocessor, cross-border transfer, and backup
   locations.
5. Retention and deletion schedules, access reviews, incident response, and
   data-subject request handling.
6. A synthetic leakage test set representative of local names, phone formats,
   addresses, facility identifiers, national IDs, and languages.

## OpenMRS integration with the FHIR exporter

OpenMRS exposes an R4 FHIR API at `/openmrs/ws/fhir2/R4`. OpenMed's
`to_bundle()` helper builds deterministic transaction bundles but does not
authenticate, send data, synthesize missing resources, or validate a site's
OpenMRS profiles.

Create a bundle from approved, already de-identified resources. This example
uses synthetic identifiers and an Observation with no real patient data:

```python
import json
from pathlib import Path

from openmed.clinical.exporters.fhir import to_bundle

resources = [
    {
        "resourceType": "Patient",
        "id": "synthetic-patient-001",
        "identifier": [
            {
                "system": "https://example.org/openmed/synthetic-id",
                "value": "SYN-001",
            }
        ],
    },
    {
        "resourceType": "Observation",
        "id": "synthetic-observation-001",
        "status": "final",
        "code": {"text": "Synthetic pulse rate"},
        "subject": {"reference": "Patient/synthetic-patient-001"},
        "valueQuantity": {"value": 72, "unit": "beats/min"},
    },
]

bundle = to_bundle(resources, doc_id="synthetic-openmrs-demo")
Path("openmed-fhir-bundle.json").write_text(
    json.dumps(bundle, indent=2) + "\n",
    encoding="utf-8",
)
```

Validate the bundle against the implementation guide and resource support of
the target OpenMRS deployment. Then set credentials in the environment and
POST the transaction bundle to the FHIR base endpoint:

```bash
export OPENMRS_BASE="https://openmrs.example.org"
export OPENMRS_USER="integration-user"
read -s OPENMRS_PASSWORD
export OPENMRS_PASSWORD

curl --fail-with-body \
  --user "$OPENMRS_USER:$OPENMRS_PASSWORD" \
  --header "Content-Type: application/fhir+json" \
  --data-binary @openmed-fhir-bundle.json \
  "$OPENMRS_BASE/openmrs/ws/fhir2/R4"
```

Use TLS, a least-privilege service account, server-side audit logs that do not
copy raw PHI, and a non-production OpenMRS instance for the first validation.
Do not assume every OpenMRS distribution enables the same FHIR resources or
profiles.

## DHIS2 integration through a reviewed mapping

DHIS2 is commonly used as an aggregate or Tracker data warehouse. It does not
accept an arbitrary FHIR transaction bundle as a drop-in replacement for its
own metadata model. Keep the OpenMed FHIR bundle as the standards-based
exchange artifact, then apply a site-owned, versioned mapping from FHIR codes
to DHIS2 program, program stage, organisation unit, and data element UIDs.

The following minimal adapter reads the synthetic Observation created above
and prepares a DHIS2 Tracker payload. Replace every UID with reviewed metadata
from the target DHIS2 instance:

```python
import json
import os
from pathlib import Path

bundle = json.loads(Path("openmed-fhir-bundle.json").read_text(encoding="utf-8"))
observation = next(
    entry["resource"]
    for entry in bundle["entry"]
    if entry["resource"]["resourceType"] == "Observation"
)

tracker_payload = {
    "events": [
        {
            "program": os.environ["DHIS2_PROGRAM_UID"],
            "programStage": os.environ["DHIS2_PROGRAM_STAGE_UID"],
            "orgUnit": os.environ["DHIS2_ORG_UNIT_UID"],
            "status": "ACTIVE",
            "occurredAt": "2026-07-18T00:00:00.000",
            "dataValues": [
                {
                    "dataElement": os.environ["DHIS2_PULSE_DATA_ELEMENT_UID"],
                    "value": observation["valueQuantity"]["value"],
                }
            ],
        }
    ]
}

Path("dhis2-tracker.json").write_text(
    json.dumps(tracker_payload, indent=2) + "\n",
    encoding="utf-8",
)
```

Validate before committing. Current DHIS2 Tracker deployments use
`POST /api/tracker`; `importMode=VALIDATE` checks the mapping without writing
the event:

```bash
export DHIS2_BASE="https://dhis2.example.org"
export DHIS2_USER="integration-user"
read -s DHIS2_PASSWORD
export DHIS2_PASSWORD

curl --fail-with-body \
  --user "$DHIS2_USER:$DHIS2_PASSWORD" \
  --header "Content-Type: application/json" \
  --data-binary @dhis2-tracker.json \
  "$DHIS2_BASE/api/tracker?async=false&importMode=VALIDATE&reportMode=FULL"
```

Only after reviewing a successful validation response should an authorized
operator repeat the request with `importMode=COMMIT`. Prefer aggregate or
de-identified values when the reporting purpose does not require person-level
data. Keep the FHIR-to-DHIS2 mapping in version control, test it with synthetic
fixtures, and treat changes to DHIS2 metadata UIDs as deployment changes.

## African developer communities

- [Masakhane](https://www.masakhane.io/) — a pan-African community for African
  language NLP research and open collaboration.
- [Data Science Nigeria](https://datasciencenigeria.org/) — data science and AI
  training, research, and community programs in Nigeria.
- [Zindi](https://www.zindi.africa/) — African data science challenges,
  learning resources, and community networks.
- [Deep Learning Indaba](https://deeplearningindaba.com/) — an African machine
  learning and AI research community with the annual Indaba and local IndabaX
  events.
- [Python Ghana](https://www.meetup.com/python-ghana/) — Python, PyData, PyLadies,
  mentorship, and community events in Ghana.
- [Python Nairobi](https://www.meetup.com/python-nairobi/) — a Python Software
  Foundation community group for developers and learners in Nairobi.

When asking for help, share a minimal synthetic reproducer, OpenMed version,
model identifier, hardware/OS, and whether local-only mode is enabled. Do not
share real clinical records, secrets, tokens, or production URLs.

## Production checklist

- [ ] Package and model artifacts are pinned, transferred, and checksum-verified.
- [ ] The target host runs successfully with networking disabled.
- [ ] Synthetic local-language and local-identifier leakage fixtures pass.
- [ ] The selected OpenMed policy actions were reviewed field by field.
- [ ] POPIA, NDPA, or other applicable legal review is recorded by the
      responsible organization.
- [ ] OpenMRS or DHIS2 mappings were validated in a non-production environment.
- [ ] Credentials are least privilege, rotated, and absent from source control.
- [ ] Logs, traces, errors, and audit exports contain no raw PHI.
- [ ] Backup, retention, deletion, incident, and recovery procedures are tested.
