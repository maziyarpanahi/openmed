# Agent boundary threat model

The [effect-path inventory](#effect-path-inventory) is an independent source
guard for legacy writers and effect classification. The adversarial harness
below tests a caller-composed policy boundary rather than installing one.

OpenMed's adversarial agent-boundary suite is a deterministic, offline
conformance harness for the final policy boundary before a clinical agent can
dispatch a tool or touch a host resource. It supplies synthetic hostile inputs
to an application-owned adapter and verifies that attacks are denied before
the suite's dispatch callback runs. A benign control must pass through the same
capability and minimum-data policy context and dispatch exactly once.

The suite is release evidence, not an enforcement boundary. Applications must
compose the relevant capability grant, access ticket, delegation, argument
classification, tool-catalog, approval, credential-broker, and host-resource
controls in their adapter. Every consequential effect must remain behind the
callback supplied by the suite.

## Effect-path inventory

`scripts/security/check_effect_paths.py` parses package sources without
importing them and checks the explicit `scripts/security/effect_paths.json`
manifest. Run it without arguments to print only controlled classifications,
Python module names and function names. `--check` emits a fixed success or
failure code. Source literals, SQL, endpoints, payloads, source paths and
free-form exception messages are never printed.

The inventory conservatively identifies direct HTTP mutation methods, dynamic
request/SQL methods, database execution/commit calls and known writer wrappers.
Some candidates are reads, local callbacks or messaging operations sharing
these method names. Those candidates remain explicitly classified so a
future write cannot hide behind a broad module exemption. Inbound route
decorators and literal GET/SELECT calls are excluded. Each function candidate
needs an explicit entry; a new function in an existing module also fails.

| Classification | Meaning |
| --- | --- |
| `governed` | Existing approval-bound OMOP/lineage delegation contracts; the application-owned committer remains responsible for bounded execution |
| `legacy_explicit_opt_in` | Direct clinical writers or their transports, selected by a caller without the complete v3.1 agent authority/approval contract |
| `operational` | Local persistence, cache, transport or other conservative operational candidates; classification does not assert v3.1 governance |

The legacy FHIR client writes only with explicit `write=True` in
`FHIRServerClient.put_resource` or `fetch_and_deidentify`. Calling
`OpenMRSAdapter.write_back` delegates to the legacy REST/FHIR2 writers;
`dry_run` defaults to `False`, so invoking it can send a write without an
additional opt-in flag. These paths are outside v3.1 governed execution.
Neither path is changed or removed by this guard. Direct local OMOP writers
and PostgreSQL redaction are also classified as legacy; the reachability ban
below specifically targets the named FHIR/OpenMRS client paths.

The source guard checks built-in MCP, REST, CLI and Journey module dependencies
for legacy imports and writer references, including imports inside functions,
aliases, relative imports, transitive package imports, constant dynamic imports,
constant adapter requests and direct/getattr writer references. Unit integration
also constructs the real built-in tool registry/handlers, REST routes, CLI
parser handlers and Journey workflow definitions and checks their source roots.
No tool, route, clinical writer or Journey operation is dispatched by that check.

`make lint`, the local pre-commit `effect-paths` hook and the CI lint job run
the same `--check` guard. Adding/removing an effect requires a reviewed manifest
change; it receives no default classification. Updating the manifest cannot
waive the independent FHIR/OpenMRS reachability check. Legacy entries cannot
be relabeled as governed to pass that check.

This is a conservative repository source check, not a runtime sandbox or proof
against arbitrary dynamic Python, native code, monkeypatching or external
plugins. It audits the built-in surfaces with plugin discovery disabled in
the integration fixture; applications must review additional registered tools
and compose authority controls for them. Operational classification does not
authorize an effect. Local OMOP export/temporary database paths remain reachable
where explicitly exposed, and this guard does not add grant enforcement to them.
The separate governed execution adapters remain their own release work.

All fixture writes and transports are synthetic. Production effects still
require the application's explicit policy, approval and bounded executor;
no autonomous clinical action is introduced.

## Covered adversaries

The default corpus uses synthetic values and covers these boundary attacks:

| Attack class | Boundary property under test | Stable denial reason |
| --- | --- | --- |
| Instruction injection | Untrusted prompt text cannot override policy | `untrusted_instruction` |
| Hostile tool result | Tool output remains untrusted input | `hostile_tool_result` |
| Confused-deputy delegation | Delegated authority cannot be amplified | `delegation_scope_amplified` |
| Tool-catalog substitution | Runtime metadata must match the reviewed catalog | `catalog_substitution` |
| Credential leakage | Agents cannot receive or disclose brokered credentials | `credential_exposure` |
| Endpoint leakage | Content-free metadata cannot reveal configured endpoints | `endpoint_exposure` |
| Path traversal | Relative traversal cannot leave an allowed root | `path_escape` |
| URL abuse | Unsupported schemes and origins fail closed | `url_scheme_denied` |
| Filesystem escape | Unauthorized host file operations do not run | `filesystem_access_denied` |
| Network escape | Unauthorized connections do not run | `network_access_denied` |

The benign control uses the same
`capability:org.openmed/adversarial-boundary-probe@1.0.0` capability and
`policy:org.openmed/adversarial-minimum-data@1.0.0` policy profile as every
attack. The harness rejects a mixed policy context so an adapter cannot test
attacks under a stricter policy than its control.

## Adapter contract

An adapter receives one immutable `AdversarialAttempt` and a zero-argument
dispatch callback. Expected decisions and reason codes are deliberately absent
from the attempt. The adapter returns only a closed `BoundaryVerdict`. Attack
cases pass when the verdict is `deny`, the reason matches the suite's expected
stable code, and dispatch count remains zero. The control passes only when the
verdict is `allow` and dispatch count is exactly one.

```python
from openmed.agent.security import (
    AdversarialReasonCode,
    AttackClass,
    BoundaryVerdict,
    assert_adversarial_suite,
)


def boundary(attempt, dispatch):
    # Run the application's capability and minimum-data checks for every case.
    verify_active_authority(attempt.capability, attempt.policy_profile)

    if attempt.attack_class is AttackClass.NETWORK_ESCAPE:
        return BoundaryVerdict.deny(
            AdversarialReasonCode.NETWORK_ACCESS_DENIED
        )
    if attempt.attack_class is AttackClass.BENIGN_CONTROL:
        dispatch()
        return BoundaryVerdict.allow()
    return evaluate_other_boundary_controls(attempt)


report = assert_adversarial_suite(boundary)
store_content_free_evidence(report.to_dict())
```

Do not pass a callback that performs a real clinical write. The suite's
callback is a dispatch probe, so adapters should execute a synthetic or fake
host implementation after authorization. Application integration tests can
also assert that fake filesystem, network, ledger, token, and approval adapters
remain untouched for every denied case.

## Content-free evidence

`AdversarialSuiteReport` contains only:

- a fixed schema version;
- validated case and attack-class identifiers;
- `allow`, `deny`, or `error`;
- a closed reason code;
- dispatch counts and pass/fail booleans; and
- aggregate case and pass counts.

Fixture payloads, credentials, endpoints, paths, URLs, clinical text, adapter
return values, and exception messages never enter the report. Fixture and
report representations omit payloads. If an adapter raises, the harness
discards the exception and records only `boundary_error`. An untyped verdict is
reduced to `invalid_verdict`. `AdversarialSuiteFailure` has a fixed message and
retains only the safe report.

The default payloads are synthetic canaries. Do not replace them with real PHI,
credentials, tenant configuration, production paths, or live endpoints. Store
reports only under the workflow's audit retention and access controls.

## Fail-closed fixture handling

Fixture IDs, governance identifiers, payload shapes, nesting depth, string
length, corpus size, and JSON scalar types are bounded and validated. Payload
mappings and sequences are recursively snapshotted into immutable containers
before an adapter sees them. A suite must include at least one attack and one
benign control, use unique case IDs, and share one capability-policy context.

The harness catches ordinary adapter failures so a free-form exception cannot
become evidence. Process interrupts still propagate. No network client,
filesystem writer, telemetry exporter, model, or hosted scanner is imported or
enabled by the suite.

## Limits and review responsibilities

Passing this corpus does not establish protection from novel attacks, prove
that a tool implementation honors its declared contract, or authorize a
clinical action. It also cannot detect side effects performed outside the
provided callback. Reviewers must confirm that all dispatch, filesystem,
network, credential, and write paths remain behind the tested boundary and
must extend the corpus when a new attack surface or policy reason is added.

Run the focused gate with:

```text
.venv/bin/python -m pytest tests/unit/agent/security tests/integration/agent/test_hostile_tool_boundaries.py -q
```
