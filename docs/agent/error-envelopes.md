# Agent Error Envelopes

`ErrorEnvelope` gives an agent adapter a machine-readable failure that a
caller can act on, without copying anything out of the source exception. It
does not log, retry, or replace the existing exception classes.

```python
from openmed.agent.errors import ErrorStage, envelope_from_exception

try:
    run_tool()
except Exception as exception:
    envelope = envelope_from_exception(
        exception, stage=ErrorStage.TOOL_CALL, run_id=run_id
    )
    print(envelope.to_json())
```

## What an envelope carries

Exactly seven fields: `schema_version`, `error_class`, `error_code`, `stage`,
`retryable`, `run_id` and `action_id`. There is deliberately **no** free-text
detail, message, reason, or traceback field, so nothing an exception was
carrying can be copied through.

`retryable` is derived from the code rather than supplied, so an envelope
cannot claim a retryability its code does not have. `run_id` and `action_id`
are optional and validated as opaque
[correlation identifiers](event-correlation.md).

## Classes, codes and stages

| Class | Codes |
| --- | --- |
| `validation` | `invalid_configuration`, `invalid_input`, `schema_violation`, `unsupported_input` |
| `policy` | `consent_required`, `policy_denied`, `purpose_mismatch` |
| `authorization` | `audience_mismatch`, `capability_expired`, `capability_missing` |
| `resource` | `budget_exceeded`, `model_unavailable`, `optional_dependency_missing`, `resource_exhausted` |
| `provider` | `provider_failed`, `provider_protocol_error`, `provider_unavailable` |
| `timeout` | `deadline_exceeded`, `request_timeout` |
| `internal` | `internal_error` |

Each code belongs to exactly one class, and `ErrorEnvelope.for_code` derives
the class so the two cannot disagree. `RETRYABLE_ERROR_CODES` is
`deadline_exceeded`, `provider_unavailable`, `request_timeout` and
`resource_exhausted`; every other code is not retryable.

`ErrorStage` is `validation`, `authorization`, `planning`, `tool_call`,
`provider` or `post_processing`, and each class declares which stages it can
legitimately arise at. An `authorization` failure at the `provider` stage, for
example, fails closed with `invalid_stage_for_class` rather than being
recorded as a plausible-looking lie. `internal` is valid at every stage, and
`tool_call` is valid for every class.

## Mapping exceptions

`envelope_from_exception` matches the exception's **type** against the
documented [public failure contract](../api-reference.md) by method resolution
order, so a subclass maps to its most specific documented ancestor:

| Exception | Code |
| --- | --- |
| `InputError` | `invalid_input` |
| `ConfigurationError` | `invalid_configuration` |
| `PolicyError` | `policy_denied` |
| `BudgetExceededError` | `budget_exceeded` |
| `MissingExtraError` | `optional_dependency_missing` |
| `ModelLoadError` | `model_unavailable` |
| `CapabilityError` | `optional_dependency_missing` |
| `InferenceError` | `provider_failed` |
| `InternalError`, `OpenMedError` | `internal_error` |
| `TimeoutError` | `request_timeout` |
| `MemoryError` | `resource_exhausted` |

Nothing else about the exception is read. Its `str()` is never called, its
`args`, `__dict__`, class name, module and traceback are never inspected, and
no cause is chained, so none of them can reach the envelope. Every unmapped
exception becomes `internal_error` in the `internal` class.

The `authorization` codes are not produced by this mapping; an adapter raises
them explicitly, for example from a capability lifetime check.

## Verification

```bash
uv run --frozen --extra dev pytest tests/unit/agent/test_errors.py -q
```

Fixtures are synthetic. Tests pin every documented failure class to its code
and retryability, check subclass specificity, assert that each class has a
valid stage set, and prove that a sentinel containing a name, an MRN, a bearer
token, a path and a prompt never reaches the envelope — including for an
exception whose `__str__` raises if it is ever called.
