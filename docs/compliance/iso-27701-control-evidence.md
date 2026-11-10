# ISO 27701/27001 control-evidence pack

OpenMed can generate a deterministic technical-evidence pack that maps its
privacy and release-safety capabilities to selected ISO/IEC 27701:2025 and
ISO/IEC 27001:2022 controls. The pack covers no-PHI logging, local-only offline
mode, tamper-evident audit records, surrogate-vault key lifecycle operations,
and release-gate evidence.

The output is an implementation crosswalk, not a certification audit,
Statement of Applicability, or organization-specific policy. An adopting
organization still owns its PIMS/ISMS scope, risk treatment, legal basis,
retention rules, key custody, access controls, operating evidence, and
independent assessment.

## Generate the pack

Generation uses the Python standard library, bundled mappings, and local file
writes only. It does not inspect patient data or make external calls.

```python
from pathlib import Path

from openmed.compliance import generate_control_evidence_pack

result = generate_control_evidence_pack(Path("audit-evidence/iso"))
print(result.manifest_path)
print(result.markdown_path)
```

The destination contains:

- `iso-27701-control-evidence.json`: structured manifest validated by the
  committed `control_evidence.schema.json` schema.
- `iso-27701-control-evidence.md`: human-readable rendering of the same
  controls, statuses, rationales, capabilities, and evidence pointers.

Repeated runs of the same OpenMed version produce identical file content. No
wall-clock timestamp or host-specific path is embedded.

## Status model

| Status | Meaning |
|---|---|
| `covered` | OpenMed directly implements and tests its product responsibility for the mapped control. Organization-level deployment obligations can still apply. |
| `partial` | OpenMed supplies relevant technical behavior or evidence, but an adopter must add organizational or deployment controls. |
| `out-of-scope` | The control is outside a software library's responsibility; the manifest includes an explicit rationale and does not claim evidence. |

Evidence pointers are machine-readable and typed as configuration flags, guard
tests, implementations, report artifacts, or documentation. They point to
concrete OpenMed sources such as `OPENMED_OFFLINE`, the no-raw-PHI logging guard,
the hash-chain audit log, vault rotation tests, and release-gate manifests.

## Validate the JSON manifest

The schema is bundled with the package and can be loaded without a network
schema resolver:

```python
from jsonschema import Draft202012Validator

from openmed.compliance import load_control_evidence_schema

schema = load_control_evidence_schema()
Draft202012Validator(schema).validate(result.manifest)
```

`jsonschema` is a development dependency rather than a runtime dependency. The
generator itself remains stdlib-only.

## Auditor handoff

Give both files to the control owner together with the referenced test results
and actual deployment evidence. Treat `partial` entries as a collection queue,
not as passed controls. Review out-of-scope rationales against the
organization's own Statement of Applicability before relying on them.

The mapping targets the current ISO/IEC 27701:2025 PIMS structure and
ISO/IEC 27001:2022 Annex A identifiers. It intentionally paraphrases control
topics and does not reproduce either standard.
