# Offline agent governance self-check

Run the bundled metadata-only agreement checks after installing OpenMed:

```bash
openmed agent self-check
openmed agent self-check --json
```

The command needs no development extra, model, credential, network connection,
clock reading or sleep. It does not execute tools or connect to an EHR. The base
installation includes the MIT-licensed `jsonschema` validator, previously a
development dependency, for full Draft 2020-12 validation of the actual catalog.

## Library use

```python
from openmed.agent import run_agent_self_check

report = run_agent_self_check()
assert report.passed
print(report.to_json())
```

The runner accepts no input path or clinical payload. It reads the bounded,
bundled run-summary fixture resource and uses fixed opaque correlation and
relative timing vectors. The seven golden scenarios are empty, success,
abstention, denial, review, failure and mixed outcomes.

## Independent results

| Check | Agreement tested |
| --- | --- |
| `schema` | Complete versioned catalog fingerprints, canonical rendering, Draft 2020-12 validity, all outcome/reason pairs, serialized model examples and negative controls. |
| `golden` | Exact canonical JSON and Markdown digests against the public fixture and run-summary contracts; committed expectations cannot be silently regenerated. |
| `unsafe_fields` | Closed root, case, event, outcome and summary fields, opaque workflow IDs, bounded JSON, duplicate keys and non-finite values. |
| `outcomes` | All declared classes and reason codes, fixture outcomes, JSON round trips and cross-class rejection controls. |
| `correlations` | Opaque identifiers, null/root and parent/child records, byte-stable round trips and invalid identifier, self-parent and extra-field rejection. |
| `timings` | Relative integer intervals, computed durations, parent containment and graph validation; controls cover duplicates, cycles, missing parents, limits and forbidden overlap. |
| `commitments` | Every golden commitment against an independent domain-separated hash, public verification, malformed candidates and changed metadata. |

Every check runs even if another fails. Schema, golden expectation, unsafe-field,
correlation, timing and commitment mutations can fail their own check while the
other checks continue. Invalid shared fixture metadata can legitimately fail
several checks that consume that source. A missing or unreadable fixture fails
its dependent checks; it never becomes an empty passing fixture set.

Reports contain the normative versions checked, declared check names, closed
statuses and reasons, bounded case/control counts and SHA-256 digests over
validated metadata. The case count means positive vectors attempted, or catalog
entries for the schema check. Counts are not elapsed time or clinical cases.
Failed checks have no digest. Reports never include raw fixtures, identifiers,
paths, environment values, exception messages, prompts or clinical text.
JSON and text are deterministic and contain no timestamps.

`--json` uses the standard CLI `ok`, `command`, `data` envelope. A completed
check has `ok: true`; inspect `data.passed` and each result for contract failures.
The process exits **0** when all checks pass and **1** otherwise, while retaining
every independent result in the output. Unexpected report-construction failures
use a controlled CLI error with no exception detail.

## Maintaining the vectors

Schema and positive-result digests are versioned reference pins in the runner.
An intentional contract change needs a reviewed update to its version, actual
source catalog or golden resource, pins and mutation tests. Changing expected
bytes merely to obtain a passing check defeats the agreement guarantee.

The implementation consumes the schema catalog from issue #3043 and the golden
resource from #3044. Both must be present in the installed release. Development
validation of an unmerged stack must use their actual sources and verify the
assembled installed wheel; it must not replace missing contracts with copies or
fallback definitions in this runner.

This command is an offline contract check. It is not certification, clinical
validation, a sealed evaluation or a replacement for the full test suite.
