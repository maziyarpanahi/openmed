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

## Normalize a caller-supplied DataQualityDashboard file

The Python-only reference adapter reads the JSON produced by OHDSI
[DataQualityDashboard 2.9.0](https://github.com/OHDSI/DataQualityDashboard/tree/v2.9.0).
Its format is pinned to the upstream
[result writer](https://github.com/OHDSI/DataQualityDashboard/blob/v2.9.0/R/writeResultsTo.R),
[metadata and check envelope](https://github.com/OHDSI/DataQualityDashboard/blob/v2.9.0/R/executeDqChecks.R),
and [status flags](https://github.com/OHDSI/DataQualityDashboard/blob/v2.9.0/R/evaluateThresholds.R).
It does not run DQD, R, SQL or any database operation. Supply an existing local
results file and a custody manifest for that run:

```python
from openmed.interop.omop import normalize_dqd_results_file

output = normalize_dqd_results_file(
    "results.json",
    quality_input=quality_input,
)
# Pass output to normalize_omop_quality_output with reconciliation and signing_key.
```

The file must have a `Metadata` array containing one record with
`dqdVersion="2.9.0"`, and a `CheckResults` array. Missing or unknown versions,
unknown categories and tables outside the Journey projection's nine
`OMOP_FACT_TABLES` raise `OmopQualityUnsupportedError`. This deliberately does
not claim compatibility with other DQD releases or the entire OMOP table set.
Malformed JSON, duplicate keys, missing check fields, invalid flags/counts and
unreadable files raise `OmopQualityProtocolError` with controlled diagnostics.

Each record maps `category` case-insensitively to conformance, completeness or
plausibility, `cdmTableName` to an allowlisted lowercase table, and
`numViolatedRows` to an exact nonnegative integer `affected_rows` (at most
2^53 - 1). Check IDs are generated from zero-based result offsets. Array order
is part of the output digest. Source check IDs, descriptions, query text,
notes, error messages, field/concept values, metadata and all other fields are
discarded; changing them leaves the normalized digest unchanged.

| DQD flags/counts | Bridge status |
| --- | --- |
| `isError=1` or `notApplicable=1`, including with `passed=1` | `unknown` |
| Any null status flag, or both/neither `failed` and `passed` set | `unknown` |
| `failed=1`, `passed=0`, error/not-applicable flags zero | `fail` |
| `passed=1`, other flags zero, zero violated rows | `pass` |
| Otherwise passing, but count is null or positive | `unknown` |

Null counts become zero only as an unavailable-count placeholder; they cannot
establish a pass. DQD threshold passes may tolerate violated rows, while the
bridge requires a pass to have zero affected rows. Those checks retain their
counts with `dqd_threshold_pass_requires_review`. Errors and not-applicable
checks always require review. Empty results produce an unknown verdict through
the existing bridge; no check is silently omitted.

Use the module command directly with the existing subprocess runner:

```python
import sys
from openmed.interop.omop import run_omop_quality_subprocess

result = run_omop_quality_subprocess(
    quality_input,
    command=(sys.executable, "-m", "openmed.interop.omop.dqd", "results.json"),
    reconciliation=reconciliation,
    signing_key=signing_key,
)
```

The command reads the bridge request from stdin, validates its version and
input digest, and emits only canonical tool-output JSON on stdout. On rejection
it exits with status 2 and writes a controlled `state`/`code` diagnostic to
stderr, without payloads, paths or tracebacks. The existing runner discards
stderr and maps a nonzero adapter exit to `quality_adapter_failed`; the direct
Python API retains typed unsupported/protocol errors. Neither mode changes
signing, reconciliation or verdict logic.

Input files are bounded to 64 MiB, command requests to 64 KiB and normalized
outputs to the bridge's 1 MB limit. Oversized output is rejected in full;
select a suitably scoped DQD run upstream instead of truncating checks.
The entirely synthetic `tests/fixtures/interop/omop/dqd_results_2_9_0.json`
includes pass, fail, error, not-applicable and threshold-pass controls with
free-text leakage sentinels. No DQD code, runtime or vocabulary is bundled.

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
