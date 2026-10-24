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

## Qualify a backend offline

Set abstention thresholds per decision on your own labelled cases before a
backend is used in a workflow. `openmed decision-qualify` fits thresholds on a
development file, freezes them, and reports how they hold on a separate
evaluation file. It makes no network calls and loads nothing that is not
already installed.

```bash
openmed decision-qualify \
  --backend-factory my_site.decisions:build_backend \
  --development /private/decisions/development.jsonl \
  --evaluation /private/decisions/evaluation.jsonl \
  --calibration-id site.triage \
  --profile-output /private/decisions/profiles.json \
  --json
```

Select exactly one backend. `--backend deterministic` uses the
dependency-free lexical backend. `--backend-factory module:function` imports a
trusted, already installed function that takes no arguments and returns a
`DecisionBackend`, such as an `EncoderDecisionBackend` around a local model.
The factory is called once per run. The command never registers the backend
or its profiles as a default, and the bundled calibration profile is
unchanged.

Each line of both files is one labelled case (this one is synthetic):

```json
{"id": "case-001", "mode": "fixed_choice", "input_text": "Synthetic review priority is clearly urgent.", "options": ["urgent", "routine"], "expected_choices": ["urgent"]}
```

`mode` is `fixed_choice`, `boolean_choice`, `ordered_preference` or
`multi_label`. Single-choice modes take exactly one expected choice; for
`ordered_preference` it is the option expected first. `scalar_score` is not
supported because the 1.0.0 calibration evaluator scores option choices. `id`
is optional and opaque. Malformed lines, unknown fields, duplicate IDs and
requests outside the limits below fail with a controlled error code and no
input values.

Thresholds are chosen per mode from the development file only:

- each development case is scored once, and the candidates are the observed
  development confidences, margins and option scores;
- the command keeps the candidate that covers the most development cases while
  selective accuracy stays at or above `--accuracy-floor` (default 0.9); ties
  prefer higher accuracy, then stricter thresholds;
- `fixed_choice`, `boolean_choice` and `ordered_preference` fit
  `minimum_confidence` and `minimum_margin`; `multi_label` fits
  `multi_label_threshold`. Fields that do not apply to a mode are set to 1.0,
  so a profile reused for another mode abstains;
- evaluation cases are scored only after the thresholds are fixed and never
  change them.

For each mode the report gives development and evaluation accuracy on covered
cases, coverage, abstention rate, expected calibration error (from
`evaluate_decision_calibration`) and result-state counts. Per option it gives
development and evaluation support and evaluation recall, keyed by the
`sha256:` digest of the option after NFKC normalization, case folding and
whitespace collapsing. Inputs appear only as the SHA-256 digest of each file's
bytes and a record count. The report holds no input text, option values,
record IDs or file paths.

`qualified` is true only when both files contain the same modes, no case
(same mode, normalized text and options) or ID appears in both files, every
declared option has at least `--min-examples-per-option` evaluation examples
(default 20), the backend returned no errors, and evaluation accuracy and
coverage meet `--accuracy-floor` and `--coverage-floor` (default 0.5). Data
problems give `status: insufficient`; unmet floors or backend errors give
`not_qualified`. The exit status is 0 only for `qualified`.

Profiles appear in the report's `profiles` field, and in `--profile-output`
(a new file with mode 0600 that is never overwritten), only when the run
qualifies. One profile is emitted per mode and named
`<calibration-id>.<mode>`. Its `in_domain_digest` pins the development file
and its `out_of_domain_digest` pins the evaluation file. Calibration IDs in
the `openmed` namespace are reserved for bundled profiles. Pass the profiles
to `decide` explicitly:

```python
import json

from openmed.structured import DecisionCalibrationProfile, DecisionRequest, decide

with open("/private/decisions/profiles.json", encoding="utf-8") as handle:
    profiles = {
        key: DecisionCalibrationProfile(**value)
        for key, value in json.load(handle).items()
    }

result = decide(
    DecisionRequest(
        mode="fixed_choice",
        input_text="Synthetic review priority is clearly urgent.",
        options=("urgent", "routine"),
        calibration_id="site.triage.fixed_choice",
    ),
    backend=backend,
    calibration_profiles=profiles,
)
```

The same run is available from Python as `qualify_decision_backend` in
`openmed.eval.decision_qualification`, which returns the report and its
`profiles()`. A qualified report means only that the backend met the selected
floors on the supplied files. It is not clinical validation, the defaults are
tooling defaults rather than clinical requirements, and every decision still
requires human review.

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
