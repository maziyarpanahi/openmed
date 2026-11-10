# Structured public errors

OpenMed exposes one rooted error contract across the public Python PII API,
the REST service, and MCP tools. Catch `OpenMedError` when one recovery path is
enough, or catch a subclass when the application can correct the failure.

```python
import openmed

try:
    result = openmed.deidentify(payload, method="mask")
except openmed.InputError as error:
    # Correct the request. Branch on the code, not the prose message.
    print(error.code, error.details)
except openmed.CapabilityError as error:
    # Install/configure the requested local capability before retrying.
    print(error.code)
except openmed.OpenMedError as error:
    # All other expected OpenMed failures remain rooted here.
    print(error.code)
```

Every error has a stable, lowercase `code`, an actionable `message`, PHI-safe
`details`, and `to_dict()`. Human-readable messages may improve between
releases; application logic must use `code` or the exception class instead of
matching message text.

## Taxonomy and transport mapping

| Python exception | Stable code | Compatible built-in bases | REST status | MCP code |
|---|---|---|---:|---|
| `OpenMedError` | `openmed_error` | `Exception` | 500 | `openmed_error` |
| `InputError` | `input_error` | `ValueError`, `TypeError` | 400 | `input_error` |
| `ConfigurationError` | `configuration_error` | `ValueError`, `TypeError`, `KeyError` | 400 | `configuration_error` |
| `CapabilityError` | `capability_error` | `ImportError` | 503 | `capability_error` |
| `MissingExtraError` | `missing_extra` | `ImportError` | 503 | `missing_extra` |
| `ModelLoadError` | `model_load_error` | `ImportError`, `ValueError` | 503 | `model_load_error` |
| `PolicyError` | `policy_error` | `ValueError`, `TypeError` | 400 | `policy_error` |
| `BudgetExceededError` | `budget_exceeded` | `RuntimeError` | 503 | `budget_exceeded` |
| `InternalError` | `internal_error` | `RuntimeError` | 500 | `internal_error` |
| `InferenceError` | `inference_error` | `RuntimeError` | 500 | `inference_error` |

`MissingExtraError` and `ModelLoadError` descend from `CapabilityError`.
`InferenceError` descends from `InternalError`. The compatible built-in bases
keep existing `except ValueError`, `except ImportError`, and
`except RuntimeError` handlers working while applications migrate.

`openmed.ERROR_CODES` is the machine-readable class-name registry. Codes are
never reused for a different meaning. Removing or changing a published code is
a compatibility break; adding a new subclass or specialized validation code is
additive.

Shared input validation retains these more specific stable leaf codes:
`text_required`, `text_type`, `invalid_encoding`, `empty_text`, `min_chars`,
`max_chars`, `max_bytes`, `language_required`, `language_type`,
`unsupported_language`, and `suspicious_content`. These errors are still
`InputError` instances. Existing service-specific codes such as `bad_request`,
`validation_error`, and authentication or resilience codes remain supported.

## PHI-safe diagnostics

OpenMed error messages never include source clinical text, detected identifier
surfaces, reversible mappings, credentials, or secret key material. Structured
details are restricted to safe diagnostics such as counts and limits, offsets,
canonical labels, public option/package/model identifiers, type names,
checkpoints, and hashes.

Use `redact_detail()` when untrusted text must be correlated locally:

```python
from openmed import redact_detail

descriptor = redact_detail(untrusted_value)
# <redacted bytes=... sha256=...>
```

The descriptor contains only a UTF-8 byte count and a full SHA-256 digest. Do
not put the original value into a custom error message or `details` mapping.

## REST errors

Taxonomy failures use the standard service envelope:

```json
{
  "error": {
    "code": "input_error",
    "message": "The request input is malformed. Correct the documented field and retry.",
    "details": {"argument": "text"}
  }
}
```

Caller-correctable input, configuration, and policy failures use HTTP 400.
Unavailable capabilities and exceeded budgets use HTTP 503. Internal and
inference failures use HTTP 500. Server-side responses set `details` to `null`
so internal context is not exposed. FastAPI request-schema failures continue to
use HTTP 422 with `validation_error`.

## MCP errors

MCP tools return the same code and message in structured content and set both
the protocol error flag and `is_error`:

```json
{
  "error": {
    "code": "missing_extra",
    "message": "The optional runtime is unavailable. Install the documented extra and retry.",
    "details": {"extra": "mcp"}
  },
  "is_error": true
}
```

MCP security, authorization, consent, and tool-schema errors keep their
existing specialized codes. They do not include the rejected input or upstream
exception text.
