# Agent boundary threat model

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
.venv/bin/python -m pytest tests/unit/agent/security tests/integration/agent/test_hostile_tool_boundaries.py tests/integration/agent/test_fhir_input_boundary.py -q
```

## FHIR input screening

`InjectionGuard.guard_fhir_input(resource)` screens untrusted FHIR reads and
Subscription payloads before they become agent context. The default strict mode
raises `PromptInjectionDetected` before dispatch on an injection finding or an
attachment quarantine. `guard_input` also detects nested objects containing
`resourceType`, including resources inside Bundles. Applications must pass their
FHIR input through this boundary; fetching a resource alone does not screen it.
This Python agent/FHIR boundary does not change the Swift document intake API.

```python
from openmed.agent.security import InjectionGuard

# The transport and resource remain application-owned.
resource = fhir_client.get_resource("Observation", "synthetic-observation")
prepared = InjectionGuard().guard_fhir_input(resource)
agent_context = prepared.value
# Record prepared.finding_dicts(), never prepared.value or the resource.
```

The shared narrative parser in `openmed.interop.fhir_server` removes markup,
decodes character and numeric entities, and maps each visible character back to
its original XHTML span. The guard scans both source and visible text. Returned
narratives are rebuilt as escaped visible text in an XHTML `div`: hidden script,
style and template bodies, comments, attributes and remote references are not
forwarded. In allow mode, matching text is replaced with inert quarantine
markers; source-markup findings quarantine the entire narrative.

FHIR findings add `element_path` to the existing controlled pattern ID, severity,
and half-open `start`/`end` character offsets. Narrative offsets address the
original XHTML, ordinary strings address the original element value, and decoded
attachment findings address decoded UTF-8 text (HTML attachment findings address
its decoded source markup). Attachment quarantine offsets cover the encoded
value; unavailable/empty values use the sentinel span `[0, 1)`. Unknown property
names become opaque `element_<digest>` path components. No source text, attachment
bytes, arbitrary key names or external URLs enter findings, exceptions or logs.
`GuardedInput` and `InjectionScan` representations omit their payloads. Use
`finding_dicts()` for evidence, not general dataclass serialization.

Attachments are screened entirely in memory. Only UTF-8 `text/plain`,
`text/markdown`, `text/html` and `application/xhtml+xml` are accepted, with absent,
UTF-8 or US-ASCII charset declarations. Data must be strict base64 and at most
65,536 decoded bytes. The encoded length and predicted decoded size are checked
before decoding. Non-text, oversized, invalid-base64, invalid-UTF-8,
unsupported-charset, missing-data and remote-only attachments become empty
objects with controlled findings. In allow mode these objects may be forwarded,
but their original data and URL never are. No URL is fetched. Accepted embedded
attachments are scanned, then encoded from the screened text; external URLs and
stale size/hash metadata are removed from that copy.

FHIR input has no arbitrary key-name exemptions. Explicit resource-relative
coded paths exempt coded primitives (such as `Patient.gender`,
`Observation.status`, and `Bundle.type`) and the `system`, `version`, `code`,
`display`, `userSelected` leaves in declared `coding[]` paths:

| Resource | CodeableConcept paths |
| --- | --- |
| Observation | `code`, `category[]`, `interpretation[]`, `component[].code` |
| Condition | `code`, `category[]`, `clinicalStatus`, `verificationStatus` |
| DiagnosticReport | `code`, `category[]`, `conclusionCode[]` |
| DocumentReference | `type`, `category[]` |
| Procedure | `code`, `category[]` |
| MedicationRequest | `medicationCodeableConcept` |
| AllergyIntolerance | `code`, `clinicalStatus`, `verificationStatus` |

`DocumentReference.content[].format` is a direct Coding path with the same
exempt leaves. Concept `text`, unexpected children, strings under `category`,
and unknown paths are still scanned. The implementation is conservative for other resource types;
this path table is not a complete FHIR schema validator or an authorization
policy. Coded values still require the application's FHIR validation and context
projection. Resource traversal is limited to 10,000 nodes and depth 64; ordinary
text values to 131,072 characters. Exceeding those bounds quarantines the input.

`FHIR_ADVERSARIAL_FIXTURES` is part of the default offline corpus. It contains all
five reported evasions and eight synthetic benign clinical narratives. The fixed
false-positive budget is **0 flagged benign cases out of those 8**, with all five
attacks denied in strict mode. This is a deterministic regression budget for that
small corpus, not a measured clinical population rate, model benchmark or
release qualification. Non-English cue packs, OCR regions and result-scope
authorization remain separate boundaries.

```text
.venv/bin/python -m pytest tests/unit/interop/test_fhir_screening.py tests/unit/agent/test_injection_guard.py tests/unit/agent/security tests/unit/interop/test_fhir_server.py tests/integration/agent/test_hostile_tool_boundaries.py tests/integration/agent/test_fhir_input_boundary.py -q
```
