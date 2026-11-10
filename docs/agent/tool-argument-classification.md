# Tool argument classification

`openmed.agent.tools.argument_classifier` applies a deterministic privacy
policy to structured tool arguments before a local tool is invoked. It walks
nested dictionaries, lists, and tuples without flattening them, classifies
every scalar leaf through developer-authored path rules, and then allows,
redacts, or blocks each value.

The classifier performs no network request and no content inference. It is a
policy enforcement boundary, not a detector. A caller must construct rules
from the trusted tool schema or from a separate preflight detector. Unmatched
and multiply matched values fail closed.

## Define a complete policy

Paths are tuples of mapping fields and sequence indexes. A `"*"` segment
matches every list or tuple index. Mapping fields use a restricted
developer-authored grammar so a report path cannot embed a patient or record
identifier.

```python
from openmed.agent.tools import (
    ArgumentAction,
    ArgumentClassificationPolicy,
    ArgumentClassifier,
    ArgumentDataClassDecision,
    ArgumentPathRule,
)

clinical = "data:org.example/clinical-text@1.0.0"
non_sensitive = "data:org.example/non-sensitive@1.0.0"
policy_profile = "policy:org.example/minimum-necessary@1.0.0"

classifier = ArgumentClassifier(
    rules=(
        ArgumentPathRule(("documents", "*", "text"), clinical),
        ArgumentPathRule(("documents", "*", "format"), non_sensitive),
    ),
    policy=ArgumentClassificationPolicy(
        policy_profile=policy_profile,
        decisions=(
            ArgumentDataClassDecision(
                clinical,
                ArgumentAction.REDACT,
                replacement="synthetic-redaction",
            ),
            ArgumentDataClassDecision(non_sensitive, ArgumentAction.ALLOW),
        ),
    ),
    hash_key=b"replace-with-32-or-more-local-key-bytes",
)
```

A redaction replacement is a developer-authored JSON scalar. Choose a value
accepted by the tool's typed schema. Use `BLOCK` instead when no safe scalar
replacement exists. The classifier copies the argument tree and never mutates
the caller's object. Blocked values are replaced with `None` in the internal
copy and cannot reach the dispatch wrapper.

## Bind classification to a signed grant

The policy profile is meaningful only when it is authorized for the proposed
tool operation. `dispatch_with_argument_classification` requires an existing
capability-grant request and verifier. It first verifies the signed grant, then
requires the request's policy profile to match the classifier policy, walks the
arguments, and invokes the callback only when there are no blocked findings.

```python
from openmed.agent.tools import dispatch_with_argument_classification

result = dispatch_with_argument_classification(
    manifest,
    request,
    verifier,
    classifier,
    {
        "documents": [
            {"text": "synthetic-example", "format": "text/plain"}
        ]
    },
    lambda safe_arguments: local_tool(**safe_arguments),
)
```

The returned `ClassifiedDispatchResult` contains the local tool result and a
safe classification report. A `BLOCK` decision raises
`ArgumentDispatchBlockedError` before the callback runs; its `report` remains
safe to record.

## PHI-safe evidence

Each report groups decisions under `allowed`, `redacted`, and `blocked` and
records only:

- a schema path such as `/documents/0/text`;
- a developer-authored data-class identifier; and
- a path- and data-class-bound HMAC-SHA-256 hash of the canonical scalar value.

The hash key is application-owned, local, and at least 32 bytes. Keyed hashes
reduce offline guessing and cross-field correlation risk but remain linkable
within the same classified path; protect and rotate the key, and retain reports
only as long as policy requires. Reports,
exceptions, and object representations never copy argument values, redaction
replacements, signing keys, or tool results. Do not log the original or
sanitized argument objects.

Classification does not establish consent, discover PHI, validate a clinical
purpose, authorize operating-system access, or guarantee compliance. Combine
it with purpose-bound access tickets when record selectors and minimum-
necessary projections also need verification.

Run the focused offline tests with:

```text
.venv/bin/python -m pytest tests/unit/agent/tools/test_argument_classifier.py -q
```
