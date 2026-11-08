# Offline media task commands

`openmed multimodal process-document`, `openmed multimodal transcribe` and
`openmed multimodal draft` adapt a **trusted injected local session service**.
They do not install a provider. There is no default model, download, microphone
capture, cloud fallback, automatic clinical finalization or record submission.
This Python CLI slice owns command adapters; it does not replace the Python or
OpenMedKit ambient session policy, preflight, ASR or note-evidence implementations.

## Explicit invocation

```sh
openmed multimodal process-document ./document.pdf \
  --model-artifact ./installed-model \
  --session-factory my_application.media:create_service \
  --confirm-review --output-directory ./private-results --json

openmed multimodal transcribe ./audio.wav \
  --model-artifact ./installed-asr \
  --session-factory my_application.media:create_service \
  --consent-evidence ./consent-receipt.json \
  --confirm-review --output-directory ./private-transcript --json

openmed multimodal draft ./finalized-transcript.json \
  --model-artifact ./installed-note-model \
  --session-factory my_application.media:create_service \
  --consent-evidence ./consent-receipt.json \
  --confirm-review --output-directory ./private-draft --json
```

Media, consent and model artifacts must already exist locally. Source/consent
files must be nonempty regular files, not symlinks or devices. A model artifact
may be a file or directory, not a symlink. Input reads are bounded to 64 MiB;
consent to 64 KiB; protected provider JSON to 8 MiB. The provider validates actual
media, installed model contents and the semantic validity/currentness of consent.
Presence of a receipt is **not** proof of valid consent. Transcribe and draft
refuse a missing receipt before opening a service; a document service can also
require consent. The service must refuse invalid, withdrawn or stale consent.

`--confirm-review` explicitly authorizes production of a protected draft. Both
outputs retain a non-diagnostic notice and `requires_clinical_review: true`.
Confirmation does not declare a clinical record finalized or submit it anywhere.

## Injection and library boundary

`openmed.cli.multimodal_inference` exposes the narrow, command-owned
`LocalSessionService`, `LocalSession`, `LocalSessionRequest` and
`LocalSessionResult` protocols/data carriers. A factory is an installed
`module:function` returning a service. Embedding applications can instead pass
`service=` to `handle_multimodal_inference`.

The service declares `local_only=True` and supported `LocalTask` values. It opens
one request with bounded media bytes, the selected artifact, opaque consent bytes
and reviewer confirmation. `run()` returns a complete protected JSON object and
an existing `ProviderResultEnvelope`, or a controlled `SessionRefusal`. The CLI
revalidates the envelope and checks its input and output SHA-256 bindings. It
never fills in citations, confidence, provider evidence or inferred statements.
Source offsets/citations in protected results remain exactly service-owned.

The application adapter is responsible for using its library's PHI policy,
media validation, finalized evidence, consent, provenance and retention rules.
It must not capture, download, contact a network, write a clinical record or log
source values. `cancel()` invalidates pending work; `close()` verifies cleanup
and destroys service-owned retained data. Both are idempotent. If opening fails,
the service must clean its own allocations because no session was returned.
A successful `close()` is required before output creation. Python buffers are
released normally; this CLI does not claim secure memory erasure, swap/crash-dump
protection or clinical/provider qualification.

Factories execute trusted installed code. Python stdout/stderr from their calls
are discarded defensively; native writes, preconfigured logging handlers and
hostile plugins are not sandboxed. Factories must meet the no-raw-PHI logging
contract themselves. Diagnostics never reflect exception messages or private
paths. No inference or capture contract is duplicated by this adapter.

## Output permissions and completion

On macOS and Linux/POSIX, the command exclusively creates a **new** user-selected
output directory with mode `0700`. Inside it, `result.json`, `evidence.json` and
`COMPLETE` are exclusively created with mode `0600` (subject to a stricter umask).
Use a trusted local parent directory; filesystem ACLs, backups and disk encryption
remain host responsibilities. Existing directories, destination symlinks and
existing files are never overwritten. There are no plaintext temporary files,
caches or implicit retained artifacts outside the selected destination.

`result.json` contains protected service results, bound notices and review flags.
`evidence.json` contains the value-free provider envelope, task/review flags and
the SHA-256 digest of the exact protected file. JSON stdout contains that evidence
only; plain stdout contains the non-diagnostic notice. It never prints source
media, transcripts, drafts, filenames, consent values or credential text.

`COMPLETE` is written only after both output files have been flushed/fsynced and
service cleanup has succeeded. Consumers must require successful exit **and**
the marker, and verify `result_file_digest`; files alone do not prove completion.
Writes to two files are not an atomic transaction. On errors, Ctrl-C/SIGINT or
SIGTERM, owned outputs are removed and no pair is finalized. A cleanup failure
is a refusal, never success. SIGKILL, power loss, filesystem removal failures and
hostile in-process code cannot guarantee cleanup: an incomplete directory may
remain in the explicitly selected protected destination and needs manual removal.
The command never resumes such a directory. Embedded worker threads use service
cancellation; signal handling is installed/restored only in the main thread.

Windows refuses with `unsupported_permissions` before any service is opened or
output created: POSIX modes do not enforce Windows ACL privacy. Windows output
support requires a separate reviewed ACL implementation; it is not simulated here.
Tests exercise native POSIX permissions and a controlled Windows refusal.

## Error codes

Failures use the standard CLI error envelope with controlled messages.

| Code | Exit | Meaning |
| --- | --- | --- |
| `invalid_arguments` | 2 | Invalid command syntax, without echoing argument values |
| `provider_unavailable` | 1 | No factory, unsupported task or unavailable provider |
| `offline_required` | 1 | Service does not declare local-only execution |
| `consent_required` | 2 | Missing or rejected consent evidence |
| `review_required` | 2 | Explicit reviewer confirmation missing |
| `malformed_media` | 2 | Empty input or provider-declared malformed media |
| `resource_limit` | 2 | Bounded read or provider limit exceeded |
| `invalid_artifact` | 2 | Missing/unreadable/symlink/special-file local input |
| `invalid_result` | 1 | Incomplete, malformed or unbound provider result |
| `session_refused` | 1 | Other provider abstention/validation failure |
| `session_failed` | 1 | Unexpected service failure or unrecognized refusal |
| `output_failed` | 1 | Destination exists or output cannot be created/written |
| `cleanup_failed` | 1 | Service destruction or owned-output removal failed |
| `unsupported_permissions` | 1 | Protected output permissions unavailable |
| `cancelled` | 130 | SIGINT/SIGTERM or service cancellation |

## Synthetic offline example

From a repository checkout, create explicit synthetic inputs (not model weights):

```sh
mkdir -m 700 synthetic-cli
printf 'synthetic media\n' > synthetic-cli/media.txt
printf 'synthetic consent\n' > synthetic-cli/consent.txt
printf 'synthetic placeholder\n' > synthetic-cli/artifact.txt
.venv/bin/python -c 'from openmed.cli import main; raise SystemExit(main())' \
  multimodal draft synthetic-cli/media.txt \
  --model-artifact synthetic-cli/artifact.txt \
  --consent-evidence synthetic-cli/consent.txt \
  --session-factory examples.multimodal.offline_cli:create_service \
  --confirm-review --output-directory synthetic-cli/results --json
```

The fake factory accepts only the exact synthetic bytes above and returns
`Synthetic draft only` with synthetic offsets in the protected file. It runs no
model or media decoder and makes no clinical-validation or benchmark claim.
Unit and subprocess integration tests cover all three commands, success,
unavailability, missing/rejected consent, malformed media, refusals, digest
mismatch, permission modes, cancellation and cleanup failures.
