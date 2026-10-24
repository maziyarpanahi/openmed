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
| `scalar_score` | No options; optional ordered `levels` | One number in the inclusive range 0 through 1, plus an `expected_level` when levels are declared |
| `question_set` | Up to 32 named `questions` over one input | One result per question in `questions` |

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

## Question sets and score levels

Schema `1.1.0` adds optional fields only. A request can name one question with
`question_id`, word it with `instructions`, describe options with
`option_descriptions` (aligned with `options`, `null` for no description), and
declare ordered `scalar_score` `levels`, lowest first. `mode: question_set`
scores up to 32 named questions against one input, each with its own mode,
options, descriptions, instructions, and levels. `input_format: json` declares
the input as a JSON object or array, and `max_input_chars` raises the input
bound up to 131,072 characters.

```python
from openmed.structured import DecisionRequest, decide

request = DecisionRequest.from_dict(
    {
        "mode": "question_set",
        "input_text": "Synthetic review priority is urgent; severity is high.",
        "questions": [
            {
                "question_id": "priority",
                "mode": "fixed_choice",
                "options": ["urgent", "routine"],
                "option_descriptions": ["Same-day synthetic review", None],
                "instructions": "Which review priority applies?",
            },
            {
                "question_id": "severity",
                "mode": "scalar_score",
                "levels": ["Low", "Medium", "High"],
            },
        ],
    }
)
result = decide(request)
for question in result.questions:
    print(question.question_id, question.state.value, question.code)
```

All questions share one access decision and calibration profile. Each question
is then scored, calibrated, and abstains on its own, so one abstaining or
failing question never hides the others. The top-level `state` is `success`
when every question succeeds, the shared state when all questions agree, and
`partial` otherwise; read each entry of `questions` for its own state and
code. A backend that implements `score_questions` scores the whole set in one
call; any other backend receives one single-question request per question.

For levelled scores, a backend returns one score per level. OpenMed normalizes
them, reports `level_scores`, the `expected_level` index, a `scalar_score`
equal to the expected level divided by the highest index, and a confidence
equal to the most likely level's probability. The question abstains when that
confidence is below the calibration profile's minimum.

## System One and OpenAI-compatible Decisions bodies

`openmed.structured.decision_interop` converts requests and responses of the
decision-model wire shapes without network I/O:

| System One | OpenAI-compatible Decisions | OpenMed question |
| --- | --- | --- |
| `choice` | `choice` | `fixed_choice`, options in declared order |
| `noul` | `predicate` | `boolean_choice` over `("true", "false")` |
| `score` | `score` | `scalar_score` with ordered `levels` |

```python
from openmed.structured.decision_interop import (
    from_system_one_request,
    from_system_one_response,
    to_system_one_request,
    to_system_one_response,
)

request = from_system_one_request(system_one_body)
assert to_system_one_request(request) == system_one_body
outputs = from_system_one_response(request, server_response)
```

`from_system_one_request` and `from_openai_decisions_request` return a
`question_set` request; `to_system_one_request` and
`to_openai_decisions_request` render it back without loss for canonical
bodies. System One questions always carry `instructions` (`null` when unset);
OpenAI-compatible questions omit unset instructions and descriptions. Namespace,
purpose, calibration, and timeout are OpenMed policy fields that the wire
shapes do not carry, so callers pass them to the converters. OpenAI-compatible
inputs must be text; message and media inputs are rejected.

`from_system_one_response` and `from_openai_decisions_response` parse server
answers into one backend output per question, so OpenMed applies its own
calibration and abstention instead of trusting a server's choice or
confidence. Answers that do not match the declared questions, options, levels,
or probability bounds raise `DecisionConflictError`. Published servers shape
`score` answers differently, so the System One parser accepts a `legend` given
as `{"0": label}` or as a label list, `probabilities` keyed by level index or
level label or listed in level order, and an expected level reported as
`score` or `expected_level`; every variant must still match the declared
levels. `to_system_one_response`
and `to_openai_decisions_response` render OpenMed results; a question that did
not succeed carries a `null` answer plus its OpenMed `state` and `code` and is
never given a fabricated answer.

## Local decision servers

Many decision models need a CUDA serving stack, a custom head, or a pinned
runtime, and ship their own server for `POST /v1/systemone` or the
OpenAI-compatible `POST /v1/decisions`. `DecisionServerBackend` is a
`specialist` backend that sends text requests to such a server through the
converters above and parses its probabilities, so OpenMed's calibration and
abstention still decide every question.

```python
from openmed.structured import decide
from openmed.structured.decision_server import DecisionServerBackend

backend = DecisionServerBackend(
    endpoint="http://127.0.0.1:8000/v1/systemone",
    model_id="vllm-sr/d3-mini",
    revision="<commit of the served weights>",
    license_id="Apache-2.0",
)
result = decide(request, backend=backend)  # a question set is sent once
```

Pass `protocol="openai_decisions"` for a `/v1/decisions` route, and
`served_model` when the server knows the model by another name; it is sent as
`model` and accepted when the server reports it. Reported model names are
compared without their organization prefix and letter case, and a different
model yields `conflict` (`decision_server_model_mismatch`). The model ID,
revision, and declared license are recorded in the backend identity, whose
`runtime` names the protocol and scope, for example
`system_one_server:loopback`. Undeclared or non-permissive licenses are
rejected as for every backend. An optional `bearer_token` is sent as
`Authorization: Bearer` and never appears in representations, errors, or logs.

**Endpoint policy.** Endpoints are checked with the shared
[upstream endpoint policy](../security/mcp-upstream-policy.md):

- loopback is accepted by default: `127.0.0.0/8`, `::1`, or `localhost` when
  every resolved address is loopback, plus a Unix domain socket given as
  `unix_socket="/run/decider.sock"` (requests carry `Host: localhost`);
- a private-network host is reachable only when the operator lists its exact
  origin in `allowed_private_origins` **and** constructs the backend with
  `contains_phi=False`; every resolved address must be private (RFC 1918 or
  unique-local), and offline mode (`OPENMED_OFFLINE`) refuses it;
- `contains_phi` defaults to `True`, so requests marked as containing PHI never
  leave loopback; public hosts are always refused.

The endpoint is validated again on every call, and the client connects only to
the addresses approved for that call. Same-origin `307` and `308` redirects
are followed up to `max_redirects` (two by default); any other redirect, or a
redirect to another scheme, host, or port, is refused. Connection-level
retries (`max_retries`, one by default) reuse the same approved addresses, so
a request body is never resent to a different host. `timeout_ms` bounds the
whole exchange, including connection, redirects, and retries, and response
bodies are capped by `max_response_bytes` (1 MiB by default).

**States.** Server problems are typed states, never fabricated answers:

| Outcome | State | Code |
| --- | --- | --- |
| HTTP 400, 413, or 422 (for example over the server's input limit) | `unsupported` | `decision_server_input_rejected` |
| HTTP 404, 405, or 501 | `unsupported` | `decision_server_route_unsupported` |
| A question type without a wire shape | `unsupported` | `backend_unsupported` |
| Timeout, HTTP 408, or HTTP 504 | `failure` | `backend_timeout` |
| Refused or reset connection after retries | `failure` | `decision_server_unreachable` |
| HTTP 401 or 403 | `failure` | `decision_server_unauthorized` |
| HTTP 429 or 5xx | `failure` | `decision_server_unavailable` |
| Body that is not a JSON object, or answers that do not match the request | `failure` | `decision_server_response_invalid` |
| Body larger than `max_response_bytes` | `failure` | `decision_server_response_too_large` |
| Refused redirect | `failure` | `decision_server_redirect_refused` |
| Endpoint refused when the call is made | `failure` | `decision_server_endpoint_refused` |
| Reported model differs from the declaration | `conflict` | `decision_server_model_mismatch` |

Backend exceptions may carry a controlled `decision_code`, which `decide`
reports as the result code; the exception class still selects the state.
Errors and log records contain codes, counts, and timings only; error bodies
from the server are never read into results. Inputs are text or JSON text;
image and video inputs are not sent.

**Recipe: one model behind its own server.** The integration test
`tests/integration/test_decision_server_recipe.py` sends a synthetic
three-question request to a running server. It is marked `integration` and
`slow` and skips unless `OPENMED_DECISION_SERVER_ENDPOINT` is set, so it never
runs in CI. For example, with `vllm-sr/d3-mini` and the stdlib server packaged
with the model (follow its model card for hardware and dependencies; the
server binds `127.0.0.1` by default):

```bash
# In a separate environment for the model's own runtime:
hf download vllm-sr/d3-mini --local-dir d3-mini
pip install -r d3-mini/requirements.txt
python d3-mini/d3_server.py --model ./d3-mini --device cuda:0 --port 8000

# In the OpenMed development environment:
OPENMED_DECISION_SERVER_ENDPOINT=http://127.0.0.1:8000/v1/systemone \
OPENMED_DECISION_SERVER_MODEL=vllm-sr/d3-mini \
OPENMED_DECISION_SERVER_REVISION=<commit of the downloaded weights> \
pytest tests/integration/test_decision_server_recipe.py -m integration -q
```

Other servers documented by their model cards work the same way, for example
the h2o-lightning-4b shim on port 8741 (set
`OPENMED_DECISION_SERVER_PROTOCOL=openai_decisions` for `/v1/decisions`) or a
Mica server on port 8010. Optional variables set the license
(`OPENMED_DECISION_SERVER_LICENSE`, `Apache-2.0` by default), the served model
name, and a bearer key. Serve only weights whose license you have declared;
weights under non-commercial or other non-permissive licenses are rejected.

## MCP

`openmed_decide` exposes the same request fields and canonical result schema.
It is declared read-only, non-destructive, idempotent, and closed-world. The
tool registry is generated from the same schema used by Python and REST, and
cross-surface drift tests compare complete synthetic results.

## Limits and privacy

- input text is capped at 8,192 characters and 32,768 UTF-8 bytes unless
  `max_input_chars` raises the bound, up to 131,072 characters and four bytes
  per character;
- each request accepts at most 64 options, each bounded by characters and
  encoded bytes;
- question sets hold at most 32 questions; instructions are capped at 2,048
  characters, option descriptions at 1,024, and all question text at 64 KiB;
- timeouts range from 1 through 30,000 milliseconds;
- batches contain at most 32 requests and have an aggregate character cap;
- duplicate normalized options, control characters, invalid versions, and
  unknown fields fail closed;
- safe metadata contains lengths, counts, policy identifiers, and backend
  provenance, never input text or option values;
- for `1.1.0` requests, safe metadata and `decision_audit_record` add question
  IDs, modes, option and level counts, and a SHA-256 digest of each question
  definition, never the input, instructions, options, or option descriptions.

The default path makes no network calls. `DecisionServerBackend` is opt-in
and reaches only loopback, a Unix socket, or an operator-approved
private-network origin for requests not marked as containing PHI. Model
weights, calibration fixtures,
and restricted clinical datasets are not bundled. Calibration reports use
synthetic or operator-provided fixtures and emit aggregate metrics and digests
without retaining source text.

## Compatibility

The current request and result schema version is `1.1.0` with `same_major`
compatibility. Every `1.0.0` request remains valid and decides exactly as
before: a request that uses no `1.1.0` field and omits `schema_version`
resolves to `1.0.0`, and its result has exactly the `1.0.0` fields. A request
that declares `1.0.0` cannot use `1.1.0` fields. `1.1.0` results always carry
`questions`, `level_scores`, and `expected_level`.

Same-major migration preserves unknown result fields under `extensions`;
different major versions are rejected. Migrating a `1.1.0` result to `1.0.0`
moves non-empty `1.1.0` fields into `extensions`, and migrating back restores
them; question-set results cannot be expressed in `1.0.0`. The committed JSON
Schemas, OpenAPI artifact, MCP registry, and generated client surfaces are
checked for drift in CI.
