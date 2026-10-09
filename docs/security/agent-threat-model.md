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
.venv/bin/python -m pytest tests/unit/agent/security tests/integration/agent/test_hostile_tool_boundaries.py -q
```

## Multilingual lexical screening

The existing `InjectionGuard` applies deterministic lexical cues before MCP
dispatch. Localized packs add instruction override, named tool spoofing and
data exfiltration patterns after its existing NFKC/casefold normalization and
format-character removal. All packs scan every string without a language
detector or language-hint gate, so mixed-language content does not select a
weaker pack. English pattern IDs remain unchanged. Localized findings use
`instruction_override.<language>`, `tool_name_spoofing.<language>` and
`data_exfiltration.<language>`, original codepoint offsets and existing severity
values. They contain no matched text, language-detection guesses or source
values. Use `to_dict()` for audit evidence; the scan object's original and
quarantined text remain protected application data.

In strict mode a cue refuses dispatch. In allow mode matched spans become inert
quarantine markers before the copied input reaches the handler. Neither mode
grants authority. Existing structural-field handling is unchanged; capability,
catalog, consent, approval and host controls must protect every actual effect.
FHIR structure handling and model-based detection are outside this lexical
slice.

### Coverage matrix

This matrix covers the current PII language catalog, including distinct
caller-supplied and national-ID-only routes. A PII route does not imply lexical
injection coverage. Fluent contributor review is pending for all six new
packs; their engineering results are draft evidence, not reviewed translations.
No reviewer, date or approval is inferred from passing tests.

| Language codes | Lexical coverage | Review or routing boundary |
| --- | --- | --- |
| `en` | Existing English override, tool, delimiter and exfiltration cues | Existing baseline; unchanged |
| `es` | Spanish override, named tool and exfiltration cues | Draft; fluent review pending |
| `fr` | French override, named tool and exfiltration cues | Draft; fluent review pending |
| `de` | German override, named tool and exfiltration cues | Draft; fluent review pending |
| `pt` | Portuguese override, named tool and exfiltration cues | Draft; fluent review pending |
| `zh` | Selected Simplified and Traditional Chinese cues | Draft; fluent review pending; not all regional forms |
| `hi` | Devanagari, selected Roman Hindi and mixed-script cues | Draft; fluent review pending; not general Indic coverage |
| `am, ar, as, bn, cs, da, el, fa, gu, he, id, it, ja, kn, ko, ml, mr, nl, no, or, pa, pl, ro, ru, sv, sw, ta, te, th, tr, uk, ur, vi, xh, zu` | No native cue pack | Default PII routes; authority controls only for uncovered native cues |
| `ne` | No native cue pack | Caller-supplied PII model route; authority controls only |
| `af, bg, et, fi, ha, hr, hu, ig, lv, ms, rw, sk, sr, tl, yo` | No native cue pack | Deterministic national-ID-only routes; authority controls only |

English strings and shared machine delimiters can still match in an uncovered
language. Absence of a finding never proves that text is benign. New localized
packs do not translate the delimiter category; machine syntax stays covered by
the existing shared patterns.

### Synthetic budgets and fluent review

The synthetic fixture
`tests/fixtures/agent/multilingual-injection-cues.json` has six hostile and six
benign clinical sentences per new language. Hindi adds three Roman Hindi and
three mixed-script hostile cases plus four benign controls. There are 42
hostile and 40 benign cases in total. Each language must detect **100%** of its
expected hostile cue categories and flag **0%** of its benign cases. Tests also
cover uppercase, fullwidth, format-split and canonically decomposed accent
variants, original offsets and actual MCP strict/allow dispatch. These small
curated budgets do not estimate field recall or false-positive rates.

Before accepting translations, a fluent contributor for each language must
review the expressions and every native/variant fixture for meaning,
naturalness, word boundaries, plausible benign clinical usage and overbroad
matches. Hindi review includes Devanagari combining marks, danda punctuation,
Roman Hindi and mixed-script forms. Chinese review includes both script forms.
Record an attributable reviewer, date and review receipt in the fixture's
`linguistic_reviews` entry only after actual review. All entries currently say
`pending` with empty reviewer/date/receipt fields. Rerun the corpus and dispatch
gates after any correction; do not mark the issue complete or merge this draft
while linguistic review is pending.

The scanner cannot understand negation, quoted hostile instructions, synonyms,
novel paraphrases or arbitrary transliteration. Quoted or reported cues may
still be quarantined. Structural authority controls remain required for every
language, including covered ones. No cue grants a tool, network, credential or
clinical-write capability, and these tests establish no clinical validation.

```text
.venv/bin/python -m pytest tests/unit/agent/test_injection_guard.py tests/unit/agent/test_multilingual_injection_cues.py tests/integration/agent/test_multilingual_injection_boundary.py -q
```
