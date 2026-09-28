# Reproducible quality-measure evidence

`openmed.agent.workflows` can package a quality-measure calculation as a
canonical, locally signed evidence packet. The packet binds the measure
definition, value-set expansions, projected inputs, measurement period,
exclusions, implementation, ordered calculation trace, and aggregate result.
It performs no network, filesystem, or telemetry operation.

The contract accepts public developer-authored identifiers, versions,
aggregate counts, and SHA-256 digests. Patient identifiers, row values,
clinical text, credentials, and signing keys are deliberately outside the
packet. Digests and small aggregate counts can still be sensitive metadata;
apply the same access controls, disclosure rules, and retention limits used
for other clinical audit artifacts.

## Build and verify a packet

Use digests to bind inputs and intermediate outputs held inside the trusted
clinical-data boundary. The caller owns the HMAC key and its rotation policy;
OpenMed never persists it.

```python
from openmed.agent.workflows import (
    CalculationTraceStep,
    InputProjectionEvidence,
    MeasureDefinitionEvidence,
    MeasureImplementationEvidence,
    MeasureResultEvidence,
    MeasureTimeBoundaries,
    ValueSetEvidence,
    build_quality_measure_evidence_packet,
)

packet = build_quality_measure_evidence_packet(
    MeasureDefinitionEvidence(
        "CMS-example",
        "2026.1",
        "sha256:" + "a" * 64,
    ),
    (
        ValueSetEvidence(
            "valueset.denominator",
            "2026.1",
            "sha256:" + "b" * 64,
        ),
    ),
    InputProjectionEvidence(
        "sha256:" + "c" * 64,
        "sha256:" + "d" * 64,
        240,
    ),
    MeasureTimeBoundaries("2026-01-01T00:00:00Z", "2027-01-01T00:00:00Z"),
    (),
    MeasureImplementationEvidence(
        "openmed.quality.example",
        "1.0.0",
        "sha256:" + "e" * 64,
    ),
    (
        CalculationTraceStep(
            "denominator",
            "filter",
            "sha256:" + "f" * 64,
            "sha256:" + "d" * 64,
            "sha256:" + "1" * 64,
            120,
        ),
    ),
    MeasureResultEvidence(120, 84, 0, "sha256:" + "2" * 64),
    key_id="quality-evidence-2026",
    signing_key=load_local_signing_key(),
)

serialized = packet.to_json()
restored = type(packet).from_json(serialized)
restored.verify(load_local_signing_key())
```

Signatures use HMAC-SHA-256 over every canonical packet field except the
signature itself. Parsing rejects unknown or omitted fields, and verification
fails closed when any signed field changes. A key must contain at least 16
bytes. For deployments that require asymmetric signatures, wrap the canonical
JSON at an external evidence boundary instead of placing private-key handling
inside a measure calculation.

## Compare calculations

`compare_quality_measure_evidence()` separates semantic sources of drift from
the aggregate result change. It reports closed categories for:

- measure-definition changes;
- added, removed, or changed value sets;
- projected-input changes;
- measurement-period changes;
- exclusion-definition and exclusion-evidence changes;
- implementation changes; and
- calculation-logic, order, and output changes.

```python
from openmed.agent.workflows import compare_quality_measure_evidence

report = compare_quality_measure_evidence(previous_packet, current_packet)
if report.result_changed:
    send_to_local_review(report.to_dict())
```

The comparison report contains packet digests, closed change metadata, public
component identifiers, and a report digest. It never copies time boundaries,
input or trace digests, aggregate counts, signatures, or signing-key material.
Comparing packets does not verify their signatures because comparison callers
may use different key providers; verify each packet with its own trusted local
key before treating the report as evidence.

This packet is reproducibility and review evidence. It is not a compliance
certification, a correctness proof, or authorization for an autonomous
clinical decision or downstream action.
