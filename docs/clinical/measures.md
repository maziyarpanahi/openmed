# Deterministic clinical measures

OpenMed measure contracts make population evaluation reproducible without
reimplementing a clinical quality language. Every result binds the exact
measure definition, population expressions, value-set digests, measurement
period, source Journey snapshot, input-projection digest, evaluator artifact,
and a value-free calculation trace.

Measure output is deterministic decision-support evidence for governed review.
It does not diagnose, treat, order, contact, submit payment, or trigger another
clinical action.

## Result states

Each population expression returns one explicit state:

| State | Meaning |
| --- | --- |
| `met` | The pinned expression evaluated true |
| `not_met` | The pinned expression evaluated false |
| `unknown` | Required source semantics are incomplete or uncertain |
| `error` | The evaluator returned a typed expression error |

The same state contract applies to initial population, denominator, numerator,
denominator exclusion, denominator exception, and measure-observation roles.
Unknown and error are never converted to `not_met`.

Every `PopulationResult` carries opaque Journey fact and evidence identifiers
plus derivation digests. Clinical values are absent. The calculation trace
stores only expression references, input digests, evidence digests, states, and
controlled reason codes.

## Small native measures

`evaluate_native_measure` supports intentionally small, validated measures
whose expressions are fact-type, status, and cardinality rules. It is useful
for local deterministic checks and synthetic validation; it is not a general
clinical quality language implementation.

```python
from openmed.clinical.measures import (
    MeasureDefinition,
    MeasureLanguage,
    MeasurePopulationDefinition,
    MeasureTimeWindow,
    NativePopulationRule,
    PopulationKind,
    evaluate_native_measure,
)

definition = MeasureDefinition(
    measure_id="measure_syntheticmeasure01",
    version="1.0.0",
    language=MeasureLanguage.NATIVE,
    populations=(
        MeasurePopulationDefinition(
            population_id="denominator",
            kind=PopulationKind.DENOMINATOR,
            expression_ref="native:denominator",
        ),
    ),
)
rules = (
    NativePopulationRule(
        population_id="denominator",
        fact_types=("condition",),
        allowed_statuses=("active",),
    ),
)

result = evaluate_native_measure(
    definition,
    rules,
    facts,
    subject_ids=("patient_syntheticpatient01",),
    source_snapshot_id="snapshot_syntheticsnapshot1",
    source_snapshot_digest="sha256:" + "1" * 64,
    measurement_period=MeasureTimeWindow(
        start="2025-01-01T00:00:00Z",
        end="2025-12-31T23:59:59Z",
    ),
    evaluated_at="2026-01-02T03:04:05Z",
)
```

Native input order cannot change the result. The input-projection digest covers
the exact fact versions and rule digests. Raw fact values are used only inside
their existing Journey hashes and are never copied into measure artifacts.

## CQL and ELM JSON

OpenMed does not parse or execute the language. `SubprocessCqlElmAdapter` and
`ServiceCqlElmAdapter` bridge an evaluator selected and pinned by the caller.
The caller supplies:

- an ELM JSON library whose digest must match the measure definition;
- an evaluator identifier, semantic version, artifact digest, and execution
  mode;
- declared required and supported feature sets;
- an explicit source snapshot, measurement period, and input projection; and
- bounded request, response, and timeout limits.

The subprocess adapter passes an argv sequence directly without a shell,
discards stderr, uses a minimal environment, and reports typed failure states.
The service adapter contains no network client: the caller must inject an
explicit transport and an HTTPS or loopback endpoint. Sending a protected input
projection to any remote service is therefore a deliberate caller decision,
not a library default.

Unsupported declared features fail before the evaluator runs. Responses must
echo the exact request and engine identities, cover every declared population,
and reference only projected subjects. Missing populations, engine drift,
unknown subjects, malformed evidence, oversized payloads, and version drift
fail closed.

## Aggregate evidence and drift

`MeasureRunResult.safe_summary()` is the only representation intended for logs
or aggregate evidence. It contains counts by population and state, subject
count, evaluator and definition digests, and the semantic fingerprint. It
contains no subject identifiers, patient-level facts, evidence identifiers, or
clinical values.

`compare_measure_runs` separates execution-semantic drift from population-count
drift. The report identifies changes to the definition, evaluator, measurement
period, source snapshot, or input projection and returns counts-only deltas.
It never lists changed subjects.

## Compatibility

Measure artifacts declare schema version `1.0.0` and `same_major`
compatibility. The bundled `clinical_measure.schema.json` validates definitions
and subject results. Deserialization verifies artifact type, advisory, exact
fields, content digests, evidence cardinality, population coverage, and trace
coverage.
