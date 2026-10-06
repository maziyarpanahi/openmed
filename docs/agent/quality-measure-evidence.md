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

## Adapt a native measure run

`build_native_measure_evidence_packet()` bridges a completed Python
`MeasureRunResult` from `evaluate_native_measure()` to the existing signed
packet. It reads the definition, value-set bindings and engine identity from
the run's validated custody; callers do not supply independent replacements.
Measure evaluation and the CQL bridge are unchanged. This adapter is scoped to
the existing Python native-run contract; OpenMedKit has no corresponding native
run contract.

```python
from openmed.agent.workflows import build_native_measure_evidence_packet

# native_run is the successful StoreResult.value from evaluate_native_measure().
packet = build_native_measure_evidence_packet(
    native_run,
    input_schema_digest=local_input_schema_digest,
    key_id="quality-evidence-2026",
    key_provider=lambda key_id: local_keys[key_id],
)
restored = type(packet).from_json(packet.to_json())
assert restored.verify(local_keys[restored.key_id])
```

The schema digest must describe the caller's actual projected input schema;
the native run does not retain it and the adapter does not invent it. The
provider is trusted caller code and must resolve keys locally. It is invoked
once, after validating public metadata. Provider errors are replaced by the
value-free `key_provider_failed` diagnostic. Invalid keys fail closed under the
existing minimum-length contract. No key is persisted, and the adapter performs
no network, filesystem, telemetry or clinical action.

### Mapping and representability

- The packet binds the exact definition digest and each value-set version and
  digest. Its implementation digest covers the complete engine identity,
  including engine id, version, artifact digest and execution mode.
- The projection digest commits to both the native input projection digest and
  source snapshot digest. Record count means evaluated subjects, not source
  facts. Run, subject, result, snapshot, fact and evidence identifiers are never
  copied into the packet. Evaluation timestamps are also omitted.
- Native periods have inclusive ends. Whole-second boundaries are normalized to
  UTC; one second is added to the end to preserve the interval under the packet's
  exclusive-end contract. Fractional timestamps and overflowing ends fail
  closed without rounding. An instantaneous whole-second native interval becomes
  a one-second half-open interval.
- Exactly one denominator and one numerator population are required. Their raw
  `met` counts are preserved. The adapter neither subtracts exclusions nor
  intersects populations, and applies no initial-population or exception logic.
  A numerator count greater than the denominator is unrepresentable and rejected;
  multiple or missing denominator/numerator roles are rejected as well. These
  checks concern packet representability, not clinical measure correctness.
- Exclusion count is the number of subjects meeting at least one denominator
  exclusion, counted once. Each denominator exclusion and exception also has its
  own aggregate evidence entry; exception counts do not enter exclusion count.
- Each population becomes one trace step, ordered by population id. Its logic
  digest covers the population definition. Its input digest commits to the
  sorted multiset of native trace input digests. Its output digest commits to
  all four state counts and sorted multisets of native trace and population
  result digests, preserving multiplicity without exposing individual records.
  The native run retains rule commitments inside input digests, but not the
  original rule configuration; the adapter cannot reconstruct rule logic.
- Result digest covers the complete population/state count matrix, including
  initial populations, observations, exclusions, exceptions, `not_met`, `unknown`
  and `error`. Result drift tracks aggregate counts and their matrix layout,
  independent of run id,
  evaluation time or patient evidence changes. Empty runs retain a trace step
  for every defined population and zero counts.

These aggregate counts and commitments can still be sensitive, especially for
small populations. Keep packets inside the governed evidence boundary; this
adapter does not apply suppression or differential privacy.

### Native/packet comparator parity and intentional differences

The synthetic unit pairs and offline integration test execute both existing
comparators unchanged. Their category mapping is:

| Native reason | Packet category or flag |
| --- | --- |
| `definition_changed` | `measure_definition` |
| `engine_changed` | `implementation` |
| `measurement_period_changed` | `time_boundaries` for different UTC intervals |
| `source_snapshot_changed` | `input_projection` |
| `input_projection_changed` | `input_projection` |
| `population_counts_changed` | `result_changed`, with `calculation_output` for affected populations |

The packet comparator provides additional detail. Value-set digest, version,
addition or removal changes also yield `value_set`; native comparison includes
these in `definition_changed`. Population definition changes also yield
`calculation_logic`; exclusion/exception definition changes additionally yield
`exclusion_definition`. Native comparison has no separate categories for these.

Trace, evidence or reason changes with unchanged state counts yield
`calculation_output`, and `exclusion_evidence` for exclusion/exception changes.
Native comparison ignores this evidence-only drift; both result flags stay
false. A changed native rule is captured through input commitments and trace
outputs, rather than a reconstructed `calculation_logic` category. Changing the
caller-supplied input-schema digest yields packet `input_projection` drift;
there is no corresponding native-run field.

Two aggregate distinctions can also change the packet's result flag while the
native flag stays false: adding/removing an all-zero population changes the
count-matrix digest, and changing overlap among denominator exclusions changes
the unique excluded-subject count even when each exclusion's counts stay equal.
Native comparison checks per-population count deltas only. Synthetic controls
cover both differences; the adapter does not alter either comparator to hide
them.

Equivalent timestamp spellings (for example UTC and a matching offset) normalize
to the same packet boundaries. Native comparison compares the original strings
and can report period drift for the same interval. Both comparators ignore
run/snapshot identifier changes and evaluation-time changes when other custody
and counts remain equal. Packet key rotation changes signed packet identity but
neither comparator treats key metadata as measure drift. Verify each packet with
its trusted key before comparison.
