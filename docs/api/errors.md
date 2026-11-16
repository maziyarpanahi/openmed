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

The envelope is published as `service_error_envelope.schema.json` in the bundled
schema package (`openmed.core.schemas.json`) and registered in the schema
fingerprint snapshot, so this shape is covered by drift tests. `error` always
carries `code`, `message`, and `details`, and includes `request_id` only when a
request id is active. The job and webhook counterparts are documented in
[Async De-identification Jobs](../serving/async-jobs.md).

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

## Rerooting legacy exceptions

Many public exceptions predate this taxonomy and derive straight from a
builtin such as `ValueError` or `RuntimeError`. They join the contract through
`openmed.core.errors.LegacyRootedError`, a compatibility subclass of
`OpenMedError` that changes nothing callers already observe:

```python
from openmed.core.errors import LegacyRootedError


class AecgError(ValueError, LegacyRootedError):
    """Raised when a synthetic aECG document is malformed."""
```

A rerooted class:

- is caught by `except OpenMedError` and by every handler for its builtin
  bases;
- accepts any positional arguments, including none, and keeps `args`,
  `str()`, `repr()`, pickling and copying byte-identical to the builtin
  behavior;
- keeps its method resolution order: `LegacyRootedError` and `OpenMedError`
  are inserted directly in front of `Exception`, so `except` ordering,
  `isinstance` checks and MRO-based dispatch tables resolve as before;
- gains only a class-level `code` and `to_dict()`. A `code` or `to_dict()` the
  class or its instances already define wins. No other attribute and no
  instance attribute is added, so `hasattr(error, "message")`,
  `hasattr(error, "details")` and `vars(error)` answer as before.
  `to_dict()` reports the code, `str(error)` as the message and empty
  `details`: attributes of a legacy class were never reviewed against the
  PHI-safe details contract, so they are not promoted into it. Unlike other
  `OpenMedError` subclasses, read `str(error)` or `to_dict()` rather than
  `error.message` and `error.details`.

### Rules

1. Put the builtin base first and `LegacyRootedError` last:
   `class AecgError(ValueError, LegacyRootedError)`. Subclasses of a rerooted
   class are rooted automatically.
2. Replace a bare `Exception` base with the root itself:
   `class ToolError(LegacyRootedError)`. Python rejects
   `class ToolError(Exception, LegacyRootedError)`.
3. A class that combines a rerooted parent with another builtin base lists
   that builtin's ancestors before the root, for example
   `class AecgKeyError(AecgError, KeyError, LookupError, LegacyRootedError)`.
   Otherwise the roots would land in front of `LookupError`.
4. The stable `code` is the nearest explicit `code` class attribute or, when
   there is none, the snake_case class name (`AecgError` reports
   `aecg_error`). Existing explicit codes, and the codes subclasses inherit
   from them, never change. Set `code` explicitly to keep a code across a
   class rename. `openmed.core.errors.legacy_error_code()` returns the code a
   class reports, including before it is rerooted.
5. Messages stay value-free: rerooting does not make a message that echoes
   input acceptable, because `to_dict()` exposes `str(error)`.

`LegacyRootedError` is not part of `ERROR_CODES`, and rerooted classes do not
move onto the taxonomy envelopes: the REST service and MCP tools keep the
status codes and envelopes these errors had before they were rooted, and agent
error envelopes map them as before. Moving a class onto a taxonomy envelope is
a separate, explicit change.

### The unrooted-exception gate

`tests/unit/core/test_error_roots.py` runs
`scripts/release/check_error_roots.py`, which imports every public module (no
path component starting with `_`), collects the exception classes they expose
under public names (warning categories excluded), and reports each by its
qualified name only, `module:QualName`. Modules that cannot be imported because
an optional dependency is missing are listed as skipped. The package-wide scan
is a `slow` test (pushes to `master` and `full-ci` pull requests); run
`python scripts/release/check_error_roots.py` locally before opening a pull
request that adds or moves an exception class.

`tests/fixtures/error_roots/unrooted_allowlist.txt` names the exceptions that
are not rooted yet. It only shrinks:

- a public exception that is neither rooted nor allowlisted fails the gate, so
  new exceptions must be rooted from the start;
- an entry whose class is now rooted, or no longer public, fails the gate as
  stale until its line is deleted;
- every `LegacyRootedError` subclass must keep the roots directly in front of
  `Exception` and report a valid code.

To reroot a package:

```bash
# 1. Before editing, record each class's MRO, code and behavior on synthetic
#    arguments.
python scripts/release/check_error_roots.py --record openmed.multimodal \
  --record-output tests/fixtures/error_roots/rerooted/multimodal.json
# 2. Add LegacyRootedError to the classes, then delete the now-stale lines.
python scripts/release/check_error_roots.py --prune
```

Commit the snapshot with a parametrized test that replays it through
`replay_snapshot()` from the same script: the rerooted classes must report the
recorded code and keep the recorded MRO (ignoring the two roots), `args`,
`str()` and pickling results. Snapshots contain synthetic tokens, class names
and exception type names only.
