# Cross-Runtime Contract Gate

OpenMed ships one reference behaviour in Python and several runtimes that must
reproduce it: OpenMedKit for Swift, OpenMedKit for Kotlin and Android, and the
web package. The shared parity fixtures under `fixtures/parity/` and
`tests/fixtures/parity/` pin that behaviour as synthetic vectors: spans,
offsets, actions, policy decisions, rejections and error reasons.

The contract gate says which runtime must pass which fixture, and records
per-case results from every runtime in one value-free format, so a missing or
failing cell is visible instead of hidden inside a runtime's own test log.

| File | Role |
| --- | --- |
| `tests/fixtures/parity/contract_gate.json` | The manifest: every parity file, its digest, version, case groups and runtime cells |
| `tests/fixtures/parity/contract_gate_result.schema.json` | JSON Schema (Draft 2020-12) for one runtime's result file |
| `scripts/release/contract_gate.py` | `verify`, `run-python`, `validate-result`, `aggregate` and `refresh` |
| `.github/workflows/contract-gate.yml` | The blocking `Contract gate` check: runtime result jobs plus the aggregation |
| `tests/unit/release/test_contract_gate.py` | Gate tests with synthetic fixtures and negative controls |

## Manifest

The manifest is a JSON object with schema `openmed.contract_gate.manifest`,
version `1` and algorithm `sha256`.

| Field | Meaning |
| --- | --- |
| `roots` | Repository-relative directories the gate covers. Every file below them must be listed. |
| `result_schema` | `path` and `sha256` of the result JSON Schema. |
| `runtimes` | Every runtime the gate knows, with its reporter state (below). |
| `fixtures` | One entry per fixture, in a stable order. |

Each fixture entry has:

| Field | Meaning |
| --- | --- |
| `id` | Stable snake_case fixture id, for example `offset_contract`. |
| `path`, `sha256` | The fixture document and the SHA-256 of its bytes. |
| `version` | The fixture contract version. When the document has a top-level integer `version`, the two must match. |
| `case_groups` | Where the fixture keeps its cases (see [Case ids](#case-ids)). |
| `companions` | Supporting files pinned with the fixture, such as a source text, a tokenizer or a fixture-specific manifest, each with `path` and `sha256`. |
| `runtimes` | One cell per declared runtime. |

A runtime cell is either:

- `{"status": "required", "consumers": [...]}`: the runtime must report a
  passing result for every case. `consumers` lists the repository-relative test
  files that read the fixture in that runtime.
- `{"status": "not_applicable", "reason": "..."}`: the fixture does not apply to
  the runtime. The reason is a short sentence and is mandatory; a
  `not_applicable` cell cannot list consumers.

Every fixture must declare a cell for every runtime, so no cell can be silently
absent.

### Reporter state

`runtimes.<id>.reporter` is `active` once the runtime writes result files, with
the `command` that writes them, or `pending` with a `tracking` issue while its
reporter is being built. A `required` cell of an `active` runtime must name at
least one consumer. A `required` cell of a `pending` runtime may have an empty
`consumers` list: the cell is still part of the contract and the runtime's
reporter has to cover it when it becomes active. Python and Kotlin are active;
the Swift and web reporters are tracked in #5327 and #5329.

## Case ids

Case ids come from the fixture content, so every runtime derives the same ids
without a hand-maintained list. For each entry of `case_groups`:

| Group | Case ids |
| --- | --- |
| `{"key": "cases", "id_field": "id"}` on a list | `cases/<item id>`, for example `cases/cjk-zh-hans-no-space-01` |
| `{"key": "invalid_tokens"}` on a list | `invalid_tokens/<index>`, starting at `0` |
| `{"key": "segment"}` on an object | `segment` |

Item ids must match `[A-Za-z0-9][A-Za-z0-9._-]*` and be unique within the
fixture. Case ids identify synthetic vectors only; they are never text taken
from a fixture's inputs or outputs.

## Result files

Each runtime writes one file, `<runtime>.json`, that validates against
`contract_gate_result.schema.json`:

```json
{
  "schema": "openmed.contract_gate.result",
  "schema_version": 1,
  "runtime": "python",
  "manifest_sha256": "<sha256 of the contract_gate.json bytes read>",
  "fixtures": [
    {
      "fixture_id": "offset_contract",
      "fixture_sha256": "<sha256 of the fixture bytes read>",
      "passed": ["cases/cjk-zh-hans-no-space-01"],
      "failed": [{"case_id": "cases/indic-deva-01", "reason": "consumer_failed"}],
      "skipped": [],
      "tests": {"passed": 104, "failed": 1, "skipped": 0}
    }
  ]
}
```

`fixture_sha256` is the digest of the bytes the runtime actually read, so a
runtime that tests a stale copy shows up as a digest mismatch. Every object is
closed (`additionalProperties: false`), reasons must be snake_case codes, and
case ids must match the case-id pattern, so a result file cannot carry fixture
text, inputs, outputs, assertion messages, paths or timestamps. `errors` holds
fixture-level reason codes when no case ids could be derived, and `tests` is an
optional count of the test executions that covered the fixture.

The Python runner writes these reason codes:

| Code | Meaning |
| --- | --- |
| `consumer_failed` | A test covering the case failed. |
| `consumer_error` | A test covering the case errored, including a module that failed to collect. |
| `consumer_skipped` | Every test covering the case was skipped. |
| `case_not_exercised` | No test covered the case. |
| `consumer_not_collected` | The consumer run produced no report, or none of its tests belonged to the fixture. |
| `consumer_missing` | A declared consumer file does not exist. |
| `fixture_unreadable` | The fixture could not be read or its case ids derived (fixture-level `errors`). |

Other reporters may add codes of their own, such as an offset or reason
mismatch, as long as they are snake_case and value-free.

## Commands

```bash
# Check listings, digests, versions, case ids and runtime cells.
python scripts/release/contract_gate.py verify

# Run the Python consumers and write results/python.json.
python scripts/release/contract_gate.py run-python --results-dir results

# Validate any runtime's result files against the schema.
python scripts/release/contract_gate.py validate-result results/*.json

# Combine every runtime's result file into one decision.
python scripts/release/contract_gate.py aggregate --results-dir results \
    --summary summary.md --report summary.json

# Rewrite the digests of files the manifest already lists.
python scripts/release/contract_gate.py refresh
```

`verify` fails with a code, fixture id and repository-relative path, for
example `digest_mismatch fixture=offset_contract
path=tests/fixtures/parity/offset_contract.json` or `unlisted_file
path=tests/fixtures/parity/new_vectors.json`. Hidden files such as `.DS_Store`
are ignored.

`run-python` runs every declared Python consumer once in a pytest subprocess
and maps the outcomes onto case ids. A test whose parameter id equals a case's
item id (or its full `<key>/<id>`) covers that case only; every other test in a
consumer file covers all of the fixture's cases. A case passes when at least
one covering test passed and none failed or errored. `--fixture <id>` limits
the run, `--output <file>` writes an exact path, and the results directory
defaults to `$OPENMED_CONTRACT_GATE_RESULTS_DIR`. The command exits 0 once a
schema-valid result is written, whatever the case outcomes; `--check` makes it
exit 1 when any case did not pass. Pytest's console output is discarded so the
gate never echoes fixture values; run the consumer file with pytest directly to
see assertion details.

Exit codes: `0` success, `1` verification or aggregation failure or a failing
`--check` run, `2` unreadable manifest or result, or a usage error.

## Aggregation and the CI check

`aggregate` reads every `*.json` result file below `--results-dir` (or the
files named on the command line) and makes one decision:

1. It runs `verify`; any finding blocks the gate.
2. Each result file must validate against the result schema, name a runtime the
   manifest declares, and be the only file for that runtime. Otherwise the gate
   fails with `invalid_result`, `result_unreadable`, `unknown_runtime` or
   `duplicate_result`; a result for a fixture the manifest does not list fails
   with `unknown_fixture`.
3. Every fixture-by-runtime cell gets a state:

| State | Meaning |
| --- | --- |
| `passed` | Every expected case id was reported as passed, with the manifest's fixture digest and the current manifest digest. |
| `failed` | A case failed (`case_failed`) or was skipped (`case_skipped`), an expected case is absent (`case_missing`), an unknown case was reported (`unknown_case`), the result came from another manifest (`manifest_mismatch`) or carries fixture-level errors (`fixture_error`). |
| `stale_digest` | The runtime read fixture bytes whose digest differs from the manifest. |
| `missing` | An active runtime wrote no result file (`missing_result`) or left the fixture out (`missing_cell`). |
| `pending` | The runtime's reporter is still `pending` and wrote no result. |
| `not_applicable` | The cell does not apply; the summary lists its reason. |

Only required cells of a runtime whose reporter is `active` block. A `pending`
runtime that does upload a result is evaluated and shown, but does not block,
so a reporter can be brought up before the manifest marks it active.

The decision is written as a Markdown table (`--summary`: fixture, runtime,
state, blocking, passed, failed, skipped and missing counts, codes; then the
`not_applicable` reasons and the pending reporters) and as JSON (`--report`,
schema `openmed.contract_gate.summary`). Both carry ids, counts, codes and the
manifest's own reasons only.

`.github/workflows/contract-gate.yml` runs on every push and pull request to
`master`, `main` and `release/v*` with no path filter, so the check always
reports. The `python-results` job runs `run-python` and uploads
`contract-gate-result-python`. The `kotlin-results` job runs every Kotlin
consumer the manifest declares and uploads `contract-gate-result-kotlin` (see
[Kotlin reporter](#kotlin-reporter)). The `contract-gate` job runs even when a runtime
job failed, downloads every `contract-gate-result-*` artifact, runs
`aggregate`, appends the summary to the job summary and uploads it as the
`contract-gate-summary` artifact. A runtime job that fails to produce its file
therefore fails the gate rather than skipping it. Make the `Contract gate`
check required in the branch ruleset to block merges on it.

## Adding or changing a fixture

1. Add or edit the fixture under one of the roots. Synthetic data only.
2. For a new file, add a fixture entry (or a companion of an existing one) with
   its case groups and a cell for every runtime. Give each `not_applicable`
   cell a reason, and name the consumer tests of each `required` cell.
3. Run `python scripts/release/contract_gate.py refresh`, then `verify`.
4. Run `python -m pytest tests/unit/release/test_contract_gate.py -q`.

`refresh` only rewrites digests. It never adds a file or decides which runtimes
must pass a fixture; those are reviewed manifest edits.

## Writing a reporter for another runtime

A reporter reads `contract_gate.json`, selects the fixtures whose cell for its
runtime is `required`, derives the case ids with the rules above, hashes the
fixture bytes it loads, records each case as passed, failed or skipped with a
reason code, and writes `<runtime>.json` into the configured results directory.
It should record known failures as failed cases rather than skip them, and it
must not decide what blocks a release: the gate does. Validate its output with
`python scripts/release/contract_gate.py validate-result`. When the reporter
lands, add a job to `.github/workflows/contract-gate.yml` that uploads
`contract-gate-result-<runtime>` and add it to the `contract-gate` job's
`needs`; in the same change set the runtime's `reporter` to `active` with its
`command` and fill in the consumers of every required cell.

## Kotlin reporter

The Android OpenMedKit unit tests write `kotlin.json`. Each Kotlin consumer
declares a JUnit rule for the fixture it reads, from
`android/openmedkit/src/test/kotlin/com/openmed/openmedkit/parity/ContractGate.kt`:

```kotlin
@get:Rule
val contractGate = ContractGateRule("offset_contract")
```

The rule fails the test with a value-free code when the class is not a declared
Kotlin consumer of a `required` fixture (`consumer_undeclared`,
`runtime_not_required`, `unknown_fixture`). The test loads the fixture through
`contractGate.fixtureBytes()` or `fixtureText()`, so `fixture_sha256` is the
digest of the bytes it read from the test classpath. As in the Python runner,
every test method of a consumer class covers all of the fixture's cases. A
method that records per-case outcomes with
`contractGate.cases(items, { "cases/${it.id}" }) { ... }` covers only the cases
it records: every item runs even after a failure, and the first failure is
rethrown at the end so the test still fails with its own assertion. A failure
outside those records fails every case. Case outcomes use the Python reason
codes: `consumer_failed` for an assertion failure, `consumer_error` for any
other exception, `consumer_skipped` for a failed assumption,
`case_not_exercised` for a case no covering test recorded, and
`consumer_not_collected` when no consumer of the fixture ran.

Reporting is off unless a results directory is set, so ordinary test runs write
nothing:

```bash
cd android
./gradlew :openmedkit:testDebugUnitTest -PopenmedContractGateResultsDir=/tmp/results
```

`OPENMED_CONTRACT_GATE_RESULTS_DIR` works as well, and a relative directory
resolves against the directory Gradle was started from. Only
`testDebugUnitTest` reports. Gradle deletes a `kotlin.json` left by an earlier
run, then each test method merges its outcome into the file under a lock and
replaces it atomically, so Robolectric and plain JVM test classes in the same
run share one file. Every write is a complete, schema-valid document; fixtures
whose consumers did not run are reported as failed with
`consumer_not_collected`. The `kotlin-results` CI job passes once the file is
written, whatever the case outcomes, and fails when no file was produced;
`ContractGateReporterTest` covers the reporter with synthetic manifests.
