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

## GLiClass v4 backend

`openmed.structured.decision_gliclass` provides an encoder backend for
`knowledgator/gliclass-v4-edge` (Apache-2.0, 0.4B DeBERTa-v3 uni-encoder). It
scores every option of a request in one forward pass, with an optional
instruction prompt and few-shot examples, and reads the model's learned
calibrator.

| Runtime | Install | Artifact |
| --- | --- | --- |
| `torch` | `pip install "openmed[gliclass]"` | `model.safetensors` through `gliclass` 0.1.21 |
| `onnx` | `pip install "openmed[onnx-runtime]"` | `onnx/model_q8.onnx` on the CPU execution provider |

`gliclass` requires `transformers>=5` and scikit-learn, so the `gliclass`
extra is kept out of the base install. It cannot be combined with the
`scispacy` extra, which caps NumPy below 2. When an extra is missing,
`from_pretrained` raises `MissingDependencyError` with the install command.

```python
from openmed.structured import DecisionRequest
from openmed.structured.decision_gliclass import (
    GLiClassDecisionBackend,
    decide_with_gliclass,
)

backend = GLiClassDecisionBackend.from_pretrained(
    runtime="onnx",  # or "torch"
    prompt="How urgent is the follow-up?",
)
result = decide_with_gliclass(
    DecisionRequest(
        mode="fixed_choice",
        input_text="Synthetic referral note: the follow-up is a routine annual check.",
        options=("urgent", "routine"),
    ),
    backend=backend,
)
print(result.state.value, result.ranking[0])
print(result.extensions["gliclass"]["scores"])
```

Each mode maps to GLiClass labels and a `classification_type`:

| Mode | Labels | `classification_type` | Scores |
| --- | --- | --- | --- |
| `fixed_choice`, `boolean_choice`, `ordered_preference` | request options | `single-label` | Softmax over the options |
| `multi_label` | request options | `multi-label` | Independent sigmoid per option; the profile threshold selects `choices` |
| `scalar_score` | backend `score_levels`, lowest first | `single-label` | Expected level scaled to 0 through 1; confidence is the most likely level's probability |

Decision requests at schema 1.0.0 carry no per-question instructions, so the
instruction prompt, few-shot examples (`GLiClassExample` or
`{"text": ..., "labels": [...]}`) and score levels are backend settings.
`backend.with_instructions(prompt)` returns a backend for another question that
reuses the loaded model. These values are never logged or returned; results
carry a `prompt_digest` instead.

Calibrated scores drive the decision and its abstention. `decide_with_gliclass`
also adds `extensions["gliclass"]` to `success` and `abstained` results:
calibrated `scores`, uncalibrated `initial_scores`, `classification_type`,
`label_count`, `token_count`, `prompt_digest` and `example_count`. The torch
path reports `initial_scores`. The `onnx/model_q8.onnx` graph only emits
calibrated logits, so on that path `initial_scores` is `null` and
`initial_scores_available` is `false`. Plain `decide(request, backend=backend)` returns the same decision
without the extension.

Model files are downloaded once at a pinned revision, and only the files the
runtime needs: the fp32 ONNX graph is never fetched. Pass `local_dir=` to use a
pre-downloaded copy; the path is not recorded in the backend identity. Scoring
makes no network calls. Inputs that would be truncated at `max_length` tokens
(2,048 by default) return `unsupported` instead of a decision, as do requests
whose text or options contain GLiClass control tokens such as `<<SEP>>`.

The int8 graph keeps the torch model's top option on the synthetic fixtures in
`tests/integration/test_gliclass_decision_onnx.py`, but its probabilities are
flatter (for example 0.69 rather than 1.00 for the leading option), so it
abstains more often under the default profile. Fit a calibration profile for
each runtime on frozen, operator-owned slices before relying on its success
rate. The opt-in check downloads about 1.5 GB:

```bash
OPENMED_RUN_GLICLASS_V4_TEST=1 python -m pytest \
  tests/integration/test_gliclass_decision_onnx.py -m "integration and slow"
```

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
