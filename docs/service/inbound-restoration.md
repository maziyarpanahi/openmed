# Local inbound placeholder restoration

`openmed.service.privacy_proxy.inbound` restores placeholders in a local
inference response after the response returns from a model. The mapping stays
inside the caller's process: this boundary makes no network request and does
not write mapping values to logs, reports, temporary files, or audit output.

## Request-scoped usage

Create immutable state for one request, use a restorer once, and close it when
the request ends:

```python
from openmed.service.privacy_proxy.inbound import (
    InboundPlaceholderRestorer,
    InboundRestorationState,
)

state = InboundRestorationState.from_mapping(
    {
        "<<OPENMED_PHI_NAME_DEADBEEF_000001>>": "Synthetic Patient",
    },
    request_id="request-123",
)

with InboundPlaceholderRestorer(state) as restorer:
    restored = restorer.restore(
        {
            "choices": [
                {
                    "message": {
                        "content": "Hello <<OPENMED_PHI_NAME_DEADBEEF_000001>>"
                    }
                }
            ]
        }
    )
```

The result preserves the response's JSON-like shape and replaces strings at
any nesting level. Lists, tuples, mappings, booleans, numbers, and `None` are
supported. Unsupported Python objects are rejected instead of being coerced.
Mapping keys are checked too, so a placeholder cannot remain hidden in a
structured response key.

For a service handling multiple concurrent requests, use
`InboundRestorationStore`. `put()` validates a request mapping, `restore()`
consumes it by default, and `remove()` can clean up an abandoned request. The
store has both active-request and aggregate mapping-byte limits. It refuses a
new request when full; it never evicts another active request's mapping.

## Validation policy

The default `InboundRestorationPolicy` is strict:

- valid tokens use the form
  `<<OPENMED_PHI_<CATEGORY>_<8-HEX-DIGITS>_<6+-DIGITS>>`;
- unknown, malformed, and duplicated response placeholders are rejected;
- duplicate mapping keys are rejected before a state is created;
- mapping entries, response bytes, restored bytes, response nodes, nesting
  depth, and placeholder occurrences are bounded;
- restoration errors contain reason codes and constant messages, never the
  response token or mapped value.

Some response formats intentionally repeat a reference. They can opt into
that behavior without disabling unknown or malformed-token checks:

```python
from openmed.service.privacy_proxy.inbound import InboundRestorationPolicy

policy = InboundRestorationPolicy(reject_duplicates=False)
```

Use `restore_text()` for a text-only response and
`restore_structured_response()` for a JSON-like response tree. Both functions
apply the same policy and accept either validated state or a one-call mapping.
The state form is preferred because it makes request ownership and cleanup
explicit.
