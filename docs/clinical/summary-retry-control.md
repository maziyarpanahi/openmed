# Bounded local-summary retries

`openmed.clinical.summary_retry_control` prevents failed local summary
generation from retrying indefinitely. State is isolated by the exact pair of
immutable input and model digests. The controller is deterministic, in-memory,
thread-safe, and performs no network calls.

```python
import hashlib

from openmed.clinical.summary_retry_control import (
    SummaryRetryController,
    SummaryRetryPolicy,
)


def digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


controller = SummaryRetryController(SummaryRetryPolicy(max_attempts=3))
input_digest = digest("already-deidentified-input")
model_digest = digest("local-model-artifact")

permit = controller.begin_attempt(input_digest, model_digest)
if permit.action == "generate":
    # Run exactly one local generation attempt, then finish its state.
    decision = controller.record_failure(
        input_digest,
        model_digest,
        "synthetic-safety-gate",
    )
```

Call `begin_attempt()` immediately before every generation. Finish each permit
exactly once with `record_failure()` or `record_success()`. A failure at the
configured attempt ceiling returns
`summary_retry_ceiling_exhausted`. Reaching the configured occurrence limit
for the same failure class returns `summary_repeated_failure_class`, even when
attempts remain. Refusals are sticky for that input/model pair.

Decisions and `audit_log()` contain only an opaque generation-key digest,
attempt counts, fixed actions, and refusal codes. Raw inputs, model names, and
failure classes are never emitted. Failure classes are fingerprinted for
internal comparison. Invalid-state exceptions are categorical and do not echo
caller values.

This controller bounds attempts; it does not generate text, choose a model,
judge clinical correctness, or provide a compliance guarantee.
