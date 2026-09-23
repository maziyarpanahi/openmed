# OMOP quality bridge and synthetic reconciliation

OpenMed normalizes aggregate OMOP 5.4 checks from an explicitly selected
adapter into a signed, PHI-safe report. The core package does not bundle or
reimplement an external quality engine, open a network connection, acquire a
vocabulary, or retain patient-level findings.

The public contract provides:

- digest-bound input and adapter-output custody;
- conformance, completeness, and plausibility categories;
- controlled failure and remediation codes instead of raw row values;
- aggregate row-count and mapping-coverage reconciliation;
- explicit `success`, `partial`, `unknown`, `conflict`, `unsupported`,
  `denied`, and `failure` outcomes; and
- caller-owned HMAC-SHA256 signatures over deterministic reports.

## Build reconciliation evidence

Convert an OpenMed fact projection into a count-only snapshot, then compare it
with a count-only snapshot produced by an independent ETL over the same frozen
synthetic cohort:

```python
from openmed.interop.omop import (
    OmopProjectionAggregate,
    reconcile_omop_aggregates,
)

openmed_snapshot = OmopProjectionAggregate.from_projection(
    projection,
    cohort_digest="sha256:" + "a" * 64,
)

reconciliation = reconcile_omop_aggregates(
    openmed_snapshot,
    reference_snapshot,
    expected_table_deltas={
        "person": 0,
        "visit_occurrence": 0,
        "note": 1,
        "condition_occurrence": 0,
        "drug_exposure": 0,
        "procedure_occurrence": 0,
        "measurement": 0,
        "observation": 0,
        "source_to_concept_map": 5,
    },
    expected_mapped_coverage_delta_ppm=0,
    semantic_difference_codes=(
        "openmed_emits_provenance_note",
        "openmed_emits_source_to_concept_rows",
    ),
)
```

Reconciliation refuses different cohort digests, different dataset splits,
different CDM versions, and reference snapshots without an allowed
redistributable license. Known semantic differences remain explicit; they are
not silently coerced into equality. An observed delta that differs from a
pinned expected delta forces a failed, reviewable quality verdict.

## Run a local adapter

`run_omop_quality_subprocess()` accepts an argv sequence and uses
`shell=False`. It sends one value-free request on standard input and accepts at
most 1 MB of aggregate JSON on standard output. Standard error is discarded so
an adapter cannot copy database values into OpenMed exceptions. The child gets
only `PATH`, `LANG`, `LC_ALL`, and environment entries explicitly supplied by
the caller; ambient credentials are not inherited.

```python
from openmed.interop.omop import run_omop_quality_subprocess

result = run_omop_quality_subprocess(
    quality_input,
    command=("/opt/local/bin/omop-quality-adapter",),
    reconciliation=reconciliation,
    signing_key=signing_key,
    key_id="local-quality-release",
)
```

The adapter request contains the versioned input manifest and its digest, but
no database connection string, row, note, code value, credential, or signing
key. The caller-selected adapter owns its database access and emits only:

```json
{
  "artifact_type": "openmed.omop_quality_tool_output",
  "schema_version": "1.0.0",
  "compatibility_policy": "same_major",
  "execution_mode": "local",
  "input_digest": "sha256:...",
  "tool_name": "local.quality_adapter",
  "tool_version": "1.0.0",
  "checks": [
    {
      "check_id": "cdm.visit.person_reference",
      "category": "conformance",
      "status": "fail",
      "severity": "error",
      "code": "person_reference_missing",
      "remediation_code": "repair_person_references",
      "affected_rows": 2,
      "table": "visit_occurrence",
      "requires_review": true
    }
  ],
  "output_digest": "sha256:..."
}
```

Use `build_omop_quality_tool_output()` in an adapter to calculate the exact
canonical output digest. Any mismatched input or output digest returns a typed
`conflict` with no accepted report.

## Run an explicitly configured remote job

OpenMed includes no default HTTP client for this bridge. A remote execution is
possible only when the application injects a runner callable:

```python
from openmed.interop.omop import run_omop_quality_remote

result = run_omop_quality_remote(
    quality_input,
    runner=application_owned_remote_runner,
    reconciliation=reconciliation,
    signing_key=signing_key,
)
```

The application remains responsible for transport security, authentication,
residency, consent, and ensuring the configured service receives only data it
is authorized to process. Core and offline tests never require that runner.

## Interpret the result

The `StoreResult` state describes whether the bridge produced acceptable
evidence. The report's `quality_verdict` describes the evaluated projection.
A failed check or reconciliation drift returns `StoreState.FAILURE` with the
signed report attached for review. Missing or unknown categories return
`PARTIAL` or `UNKNOWN`; they never become an apparent pass. Digest or split
mismatches return `CONFLICT`, policy rejection returns `DENIED`, and an
unsupported version or mode returns `UNSUPPORTED`.

Verify persisted evidence with `verify_omop_quality_report(report, key)`. The
key is never serialized. The bundled `omop_quality_report.schema.json` validates
the public JSON shape, while HMAC verification proves possession of the
caller-owned key and detects mutation.

## Frozen synthetic reference

The committed fixture uses only fabricated facts from
`tests/fixtures/interop/omop/fact_projection.json`. Its independent aggregate
reference is pinned in
`tests/fixtures/interop/omop/synthea_reference_omop_54_summary.json`.
The cohort file, reference summary, OpenMed projection, normalized tool output,
reconciliation, and final report each have separate SHA-256 custody.

The reference lane is based on the Apache-2.0
[Synthea generator](https://github.com/synthetichealth/synthea) and the
Apache-2.0 [ETL-Synthea package](https://github.com/OHDSI/ETL-Synthea), pinned
to release `2.1.1` for this fixture. The reference ETL supports OMOP CDM 5.4,
but requires caller-supplied vocabulary CSV files. OpenMed stores neither those
files nor terminology content. These aggregate checks are engineering evidence,
not clinical validation or regulatory certification.
