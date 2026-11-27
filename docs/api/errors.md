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

## Error-code registry

`openmed/core/error_codes.json` is the single source of truth for every stable
error code OpenMed reports. Each entry names one code and records, for that
code:

| Field | Meaning |
| --- | --- |
| `category` | One of the transport-neutral categories in [`protocol-map.json`](./protocol-map.json). |
| `description` | A one-line, value-free description. |
| `portable` | Whether a client kit (Swift, Kotlin, web) can raise the code. |
| `retryable` | Whether retrying the unchanged request may succeed, or `null` when current behaviour does not establish it. |
| `retryable_basis` | The evidence for `retryable` (see below). |
| `python` | The Python classes that carry the code. |
| `http` | REST status and `error.code`, whether `Retry-After` is sent, and in-band NDJSON `stream_event` codes. |
| `grpc` | gRPC status and `openmed-reason` trailer, or the status details when no trailer is sent. |
| `graphql` | `errors[].extensions.code` as emitted today, with the GraphQL HTTP status. |
| `mcp` | The MCP tool error code, or the code of a failed stage inside a successful result. |
| `cli` | The CLI exit code (0 success, 1 failure, 2 usage) and `error.code` in `--json` output. |
| `divergences` | Recorded differences between surfaces, listed below. |

The registry describes current behaviour. It changes no wire format, status,
trailer, extension code, MCP result or exit code. A surface field lists how
that surface reports the same failure, which may be under a different code:
`forbidden` maps to the gRPC reason `insufficient_scope` and the GraphQL code
`OPENMED_FORBIDDEN`. An empty list means that surface's own envelope never
carries the failure. When HTTP middleware refuses a `/graphql` request before
execution, the answer is the REST envelope, so it appears under `http`.

The document is validated by `error_code_registry.schema.json` in the bundled
schema package and is registered in the schema fingerprint snapshot. Load it
with the standard-library loader:

```python
from openmed.core.error_registry import load_error_registry, lookup_error_code

registry = load_error_registry()
entry = lookup_error_code("input_error")
print(entry.http_status, entry.cli_exit_code, entry.retryable)
print([item.code for item in registry.portable_codes()])
```

`retryable` is derived from existing behaviour, never guessed:

| Basis | Retryable | Evidence |
| --- | --- | --- |
| `retry_after_header` | yes | REST answers with a `Retry-After` header. |
| `retryable_signal` | yes | A surface sends an explicit retryable flag or trailer. |
| `transient_capacity` | yes | A capacity refusal whose message says to retry later. |
| `request_must_change` | no | The same request is refused until the caller changes it. |
| `configuration_must_change` | no | A setting, installed capability or deployment state must change first. |
| `server_fault` | no | An internal failure the contract says cannot safely continue. |
| `undetermined` | `null` | Current behaviour and documentation do not establish it. |

### Portable codes

These are the codes a client kit can raise. They are generated into one small
typed enum per kit:

| Portable code | Category | HTTP | CLI exit | Retryable |
| --- | --- | --- | --- | --- |
| `budget_exceeded` | `resource_exhausted` | 503 | 1 | no |
| `capability_error` | `unavailable` | 503 | 1 | no |
| `configuration_error` | `invalid_argument` | 400 | 1 | no |
| `inference_error` | `internal` | 500 | 1 | no |
| `input_error` | `invalid_argument` | 400 | 1 | no |
| `internal_error` | `internal` | 500 | 1 | no |
| `model_load_error` | `unavailable` | 503 | 1 | no |
| `policy_error` | `invalid_argument` | 400 | 1 | no |

`scripts/generate_error_code_enums.py` writes them to
`swift/OpenMedKit/Sources/OpenMedKit/Generated/OpenMedErrorCode.swift`
(`OpenMedErrorCode`, raw value plus `isRetryable`),
`android/openmedkit/src/main/kotlin/com/openmed/openmedkit/OpenMedErrorCode.kt`
(`OpenMedErrorCode`, `code` plus `retryable`) and
`js/openmedkit-web/src/error-codes.ts` (`OPENMED_ERROR_CODES`,
`OPENMED_ERROR_CODE_RETRYABLE` and the `OpenMedErrorCode` type). The enums are
public API but no kit behaviour uses them yet. Run the script with `--check`
to verify the committed files; the unit tests do this.

### Shared error-code vectors

`tests/fixtures/parity/error_codes_v1.json` (`openmed.error_code_vectors`
version 1) tells every runtime which failure must carry which code. It lists
each portable code with its category, HTTP status, CLI exit code, retryable
flag and the scenarios that expect it, plus offline scenarios with synthetic
inputs:

| Scenario | Code | Python trigger |
| --- | --- | --- |
| `empty_hash_secret` | `configuration_error` | `hmac_text_hash` or `Pipeline(hmac_secret="")` |
| `invalid_offsets` | `input_error` | `OpenMedSpan` with `end < start` |
| `unknown_policy_profile` | `policy_error` | `canonical_policy_name` |
| `model_missing_offline` | `model_load_error` | `prefetch_model` in offline mode with an empty cache |
| `invalid_sdk_config` | `configuration_error` | pending the portable SDK configuration loader (#5342) |

Each scenario marks every runtime (`python`, `swift`, `kotlin`, `web`) as
`required`, `pending` with the owning work, or `not_applicable` with a
reason. A kit cell may name its entry point and what the kit raises today,
which tells kit adoption what to map. Only the code is part of the contract;
messages are not. `tests/unit/core/test_error_code_vectors.py` checks the
fixture against the registry and runs the Python scenarios.

Three Python failures used to raise a bare builtin and now carry a code. Each
keeps its builtin base and its message, so existing handlers still catch it:

- An empty `hmac_secret` raises `ConfigurationError` (a `ValueError`).
- `OpenMedSpan` offsets that are negative or end before they start raise
  `InputError` (a `ValueError`).
- A model that is not cached while offline mode is on raises
  `openmed.core.offline.OfflineModelUnavailableError`, an `OfflineModeError`
  (a `RuntimeError`) that is also a `ModelLoadError`.

Like every taxonomy class, they also gain the taxonomy's extra builtin bases:
`TypeError` for both, plus `KeyError` for `ConfigurationError`, and
`ImportError` and `ValueError` for `OfflineModelUnavailableError`, so an
`except ImportError` around a model load now also catches an offline cache
miss. The adapters now map them as taxonomy errors: REST answers
`400 configuration_error` or `400 input_error` instead of
`400 bad_request`, and `503 model_load_error` instead of a generic 500 for
the offline miss; MCP reports `configuration_error`, `input_error` and
`model_load_error` instead of `invalid_arguments` and `execution_error`; and
agent error envelopes report `invalid_configuration`, `invalid_input` and
`model_unavailable` instead of `internal_error`.

### Recorded divergences

Where surfaces disagree today, the registry records the difference instead of
changing a contract. Entries that share one name the same id:

- `rest_grpc_code_names`: REST and gRPC name the same outcome differently,
  such as `forbidden` and `insufficient_scope`.
- `grpc_collapses_taxonomy_codes`: gRPC reports taxonomy and runtime failures
  as `invalid_request` or `internal_error`.
- `graphql_resolver_error_masks_category`: GraphQL reports resolver failures as
  `OPENMED_RESOLVER_ERROR`.
- `graphql_code_casing`: GraphQL mixes `OPENMED_*` codes with lower-case codes.
- `graphql_limits_and_deadlines_uncoded`: the service text limit and execution
  timeout inside a resolver are uncoded on GraphQL.
- `rest_stream_errors_collapse_to_internal`: in-band REST stream events
  collapse to `bad_request` or `internal_error`.
- `rest_limits_reported_as_validation`: REST reports text and chunk limits as
  `422 validation_error`.
- `rest_validation_wraps_input_leaf_codes`: input-validation leaf codes raised
  while REST parses a body become `422 validation_error`.
- `drain_refusal_codes`: a drain refusal is `not_ready`, `shutting_down`, a gRPC
  status-details code or `execution_error`, depending on the surface.
- `grpc_status_details_codes`: gRPC drain and peer-contract refusals carry their
  code in the status details, without an `openmed-reason` trailer.
- `budget_exceeded_status_split`: `budget_exceeded` is 503 from the service and
  413 from the redaction app.
- `mcp_payload_ignores_instance_code`: MCP reports a base taxonomy class by its
  class code even when the instance carries a more specific one.
- `mcp_collapses_service_failures`: MCP reports runtime refusals by builtin
  family.
- `rest_inventory_lists_unemitted_codes`: the REST compatibility inventory lists
  privacy-proxy codes that no route emits.
- `rest_inventory_omits_emitted_codes`: REST emits codes the inventory does not
  list.
- `admission_grpc_status_unused`: admission declares gRPC statuses the gRPC
  server never sends.
- `protocol_map_graphql_codes_incomplete`: the protocol map lists only part of
  the GraphQL codes.
- `cli_reports_python_errors_as_runtime_error`: the CLI reports an escaped
  Python error as `runtime_error`.
- `grounding_failed_meanings`: `grounding_failed` has one meaning on MCP and
  another in the CLI gates command.
- `cli_command_local_codes`: command-local CLI codes are not registered; the
  CLI exit code is only 0, 1 or 2.

### Adding or changing a code

Codes are never reused. To add a code, add its entry to
`openmed/core/error_codes.json` in sorted order, bump `contract_version`, and
run `tests/unit/core/test_error_registry.py` and
`tests/unit/service/test_error_registry_surfaces.py`: they read the code sets
from the real tables and source of every surface and fail until the registry
matches. A code that stops being emitted moves to `retired_codes` and can never
return. When a portable code changes, run
`python scripts/generate_error_code_enums.py` and commit the regenerated kit
files.
