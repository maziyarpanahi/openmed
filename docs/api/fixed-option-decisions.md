# Fixed-option decision API

OpenMed provides one versioned contract for choosing among caller-supplied
options, ordering them, selecting multiple labels, or returning a bounded
scalar. The same request and result shapes are available through Python,
`POST /v1/decisions`, the typed REST clients, and the `openmed_decide` MCP tool.

This API is deliberately closed-world: options are data, never instructions,
and no result authorizes diagnosis, treatment, enrollment, outreach, ordering,
or another autonomous clinical action. Every result requires human review.

## Modes and abstention

| Mode | Input shape | Successful output |
| --- | --- | --- |
| `fixed_choice` | Two to 64 options | One `choice`, all `option_scores`, and a stable `ranking` |
| `boolean_choice` | Exactly two caller-defined options | One `choice`; labels are not forced to `true` and `false` |
| `ordered_preference` | Two to 64 options | One `choice` plus a full stable `ranking` |
| `multi_label` | Two to 64 options | Zero or more thresholded `choices` plus a ranking |
| `scalar_score` | No options | One number in the inclusive range 0 through 1 |

Abstention is a result state, not a fabricated option. A request returns
`state: abstained` when confidence or margin does not satisfy its calibration
profile. Unsupported modes, missing calibration profiles, policy denial,
backend conflicts, timeouts, and backend failures remain distinct typed states.

The result always includes the backend identity and revision, calibration ID
and version, warnings, access-policy decision, schema version, compatibility
policy, and `autonomous_action: false`. Failed, denied, unsupported, or
abstained results never carry an apparent successful choice.

## Python

```python
from openmed.structured import DecisionRequest, decide

request = DecisionRequest(
    mode="fixed_choice",
    input_text="Synthetic review priority is urgent.",
    options=("urgent", "routine"),
)
result = decide(request)

if result.state.value == "success":
    print(result.choice, result.confidence)
else:
    print(result.state.value, result.code)
```

The dependency-free deterministic backend is conservative and intended for
tests, explicit lexical decisions, and offline fallback. Applications can
inject an `EncoderDecisionBackend`, `CrossEncoderDecisionBackend`, or
`SpecialistDecisionBackend`. Learned adapters require a permissive license
identifier and preserve the configured model ID, immutable revision, runtime,
and calibration identity in every result.

## REST and typed clients

```bash
curl --request POST http://127.0.0.1:8080/v1/decisions \
  --header 'Content-Type: application/json' \
  --data '{
    "mode": "fixed_choice",
    "input_text": "Synthetic review priority is urgent.",
    "options": ["urgent", "routine"]
  }'
```

Python clients can pass either a `DecisionRequest` or a matching mapping:

```python
from openmed.service.client import OpenMedClient
from openmed.structured import DecisionRequest

with OpenMedClient("http://127.0.0.1:8080") as client:
    result = client.decision(
        DecisionRequest(
            mode="boolean_choice",
            input_text="Synthetic eligibility flag is yes.",
            options=("no", "yes"),
        )
    )
```

TypeScript uses `client.decision(request)`. Go uses
`client.Decision(ctx, openmed.FixedOptionDecisionRequest{...})`. Both clients
expose the typed result and all non-success states without remapping them to
success.

## MCP

`openmed_decide` exposes the same request fields and canonical result schema.
It is declared read-only, non-destructive, idempotent, and closed-world. The
tool registry is generated from the same schema used by Python and REST, and
cross-surface drift tests compare complete synthetic results.

## Limits and privacy

- input text is capped at 8,192 characters and 32,768 UTF-8 bytes;
- each request accepts at most 64 options, each bounded by characters and
  encoded bytes;
- timeouts range from 1 through 30,000 milliseconds;
- batches contain at most 32 requests and have an aggregate character cap;
- duplicate normalized options, control characters, invalid versions, and
  unknown fields fail closed;
- safe metadata contains lengths, counts, policy identifiers, and backend
  provenance, never input text or option values.

The default path makes no network calls. Model weights, calibration fixtures,
and restricted clinical datasets are not bundled. Calibration reports use
synthetic or operator-provided fixtures and emit aggregate metrics and digests
without retaining source text.

## Compatibility

The current request and result schema version is `1.0.0` with `same_major`
compatibility. Same-major migration preserves unknown result fields under
`extensions`; different major versions are rejected. The committed JSON
Schemas, OpenAPI artifact, MCP registry, and generated client surfaces are
checked for drift in CI.
