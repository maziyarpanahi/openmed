# Tauri sidecar de-identification

This example keeps a desktop note on the local machine. The Tauri host sends
the note to a persistent OpenMed sidecar over newline-delimited JSON, receives
only the redacted text and canonical `OpenMedSpan` records, and shuts the
process down when the application exits.

The sidecar always enables OpenMed's local-only mode. The model is a local
directory shipped with the application, and the trusted Rust host refuses to
start the sidecar unless the executable and every model file match a pinned
manifest. Renderer requests cannot select a model path. A missing, changed or
incompatible artifact fails with a typed error before any text is sent; it
never falls back to a network request or a download.

## Build the self-contained executable

Install the model runtime and build tool in the development environment:

```bash
uv sync --extra hf
uv run --extra hf --with pyinstaller pyinstaller \
  --clean \
  --onefile \
  --name openmed-sidecar \
  --collect-all openmed \
  --collect-all transformers \
  scripts/openmed_sidecar_entry.py
```

The executable in `dist/` contains the Python runtime, so the desktop user does
not need Python installed. Model weights remain separate application resources
and are loaded from that local directory.

Tauri requires a target-triple suffix for bundled external binaries. For an
Apple Silicon development build, for example:

```bash
mkdir -p js/openmedkit-tauri/src-tauri/binaries
cp dist/openmed-sidecar \
  js/openmedkit-tauri/src-tauri/binaries/openmed-sidecar-aarch64-apple-darwin
```

Use `rustc --print host-tuple` to obtain the current target triple. Build one
binary per supported desktop target. Code signing, installers, and application
auto-update are intentionally outside this example.

## Pin the sidecar and model in a manifest

Place the model directory in the application resources (for example
`src-tauri/models/openmed-pii`), then write the manifest for each target:

```bash
cd js/openmedkit-tauri/src-tauri
cargo run --locked --bin openmed-sidecar-manifest -- \
  --sidecar binaries/openmed-sidecar-aarch64-apple-darwin \
  --model-root models/openmed-pii \
  --model-id OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 \
  --out openmed-sidecar.json
```

The command prints only the manifest's SHA-256. Pass `--platform` and `--arch`
when you write a manifest for another target. The manifest
(`openmed.tauri.sidecar`, schema version 1) records:

- the protocol version and the commands the sidecar supports
  (`deidentify`, `ping`, `shutdown`);
- the sidecar's platform, architecture, size and SHA-256;
- the model identifier and every file in the model directory, sorted by path,
  each with a role, size and SHA-256.

It is stored as sorted-key compact JSON, so the bytes have one canonical form
and the digest is the manifest's identity. The file inventory uses the same
`path`, `role`, `size_bytes` and `sha256` fields as the portable model-bundle
manifest; the two formats are expected to converge.

The model directory must contain exactly the listed files. An added file,
including an operating-system file such as `.DS_Store`, refuses with
`model_file_unlisted`. Regenerate the manifest whenever the sidecar or model
changes.

## Configure Tauri v2

Add the binary stem and the model directory to `src-tauri/tauri.conf.json`:

```json
{
  "bundle": {
    "externalBin": ["binaries/openmed-sidecar"],
    "resources": ["models/openmed-pii/**/*"]
  }
}
```

Copy `js/openmedkit-tauri/src-tauri/sidecar_command.rs` and
`js/openmedkit-tauri/src-tauri/sidecar_manifest.rs` into the application crate
and add these dependencies:

```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sha2 = "0.10"
tauri = "2"
tokio = { version = "1", features = ["io-util", "process", "rt", "sync", "time"] }
```

The host starts the sidecar itself, so the shell plugin is not needed. Embed
the manifest in the application binary, register the state and commands, and
terminate the sidecar when the application exits:

```rust
mod sidecar_command;
mod sidecar_manifest;

use sidecar_command::{
    openmed_sidecar_deidentify, openmed_sidecar_ping,
    openmed_sidecar_shutdown, OpenMedSidecarState,
};
use sidecar_manifest::SidecarBundle;
use tauri::{path::BaseDirectory, Manager, RunEvent};

const SIDECAR_MANIFEST: &[u8] = include_bytes!("../openmed-sidecar.json");

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let model_root = app
                .path()
                .resolve("models/openmed-pii", BaseDirectory::Resource)?;
            let bundle = SidecarBundle::new(
                SIDECAR_MANIFEST,
                SidecarBundle::bundled_sidecar_path()?,
                model_root,
            );
            app.manage(OpenMedSidecarState::from_bundle(bundle));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            openmed_sidecar_ping,
            openmed_sidecar_deidentify,
            openmed_sidecar_shutdown,
        ])
        .build(tauri::generate_context!())
        .expect("Tauri application failed to build")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                app.state::<OpenMedSidecarState>().terminate_blocking();
            }
        });
}
```

Embedding the manifest with `include_bytes!` lets the signed application
binary decide which sidecar and model it accepts. If you ship the manifest as
a separate file instead, load it with `SidecarBundle::from_manifest_file` and
pin its digest with `with_pinned_manifest_sha256`. `OpenMedSidecarState` no
longer has `default()` or `with_model()`: every sidecar start is qualified
against a bundle. To check the bundle at launch instead of on the first
request, call `state.qualify().await`; it returns the generation id, manifest
digest, protocol version, capabilities and file count, and nothing else.

The webview calls only the three bounded application commands above. Do not
add the shell plugin or grant `shell:allow-spawn`, `shell:allow-stdin-write`,
or another direct shell permission: the trusted Rust wrapper owns sidecar
creation and stdio so a renderer cannot bypass the manifest, model pinning,
limits, validation, or safe errors.

## Lifecycle

Before each process start the host verifies, in this order: the manifest size,
pinned digest, schema, canonical form and structure; the protocol version,
required commands and platform; the executable and model files (presence,
sizes, unlisted files, then SHA-256). Only then does it start the executable,
with `OPENMED_OFFLINE`, `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` set, the
verified model directory in `OPENMED_SIDECAR_MODEL`, and Python start-up
variables such as `PYTHONPATH` removed.

Each process gets a fresh, random generation id. The host sends a `ping`
handshake within a one-minute start-up deadline and requires an offline
sidecar on protocol version 1. Every request id carries the generation, and a
response from another generation or for another request fails with
`SIDECAR_PROTOCOL` and discards the process. If the renderer abandons a
request before its response arrives, the next command discards that process
and starts a new generation, so a late response is never delivered to another
caller.

The wrapper serializes requests through one long-lived child process. This
lets the Python runtime reuse the same offline loader and cached model. A
concurrent renderer call fails with `SIDECAR_BUSY`. A request that exceeds the
two-minute host deadline fails with `SIDECAR_TIMEOUT`. If the process exits
while a request is running, the command rejects with `SIDECAR_TERMINATED`
instead of returning partial JSON. Each of these transport failures discards
the child before a later request can start a new one. Stderr is drained and
discarded; the host recognises only the sidecar's own `SIDECAR_START_FAILED`
and `SIDECAR_RUNTIME_FAILED` terminal records and reports them as reasons.
`terminate()` (or `terminate_blocking()` from the exit handler) interrupts any
running request with `SIDECAR_NOT_RUNNING` and kills the process; the child is
also killed if the state is dropped.

Errors carry a fixed `code`, a fixed `message`, and, where useful, a stable
snake_case `reason`. None of them contains source text, stderr output, paths,
file names or digests:

| Code | Reasons |
| --- | --- |
| `SIDECAR_MANIFEST_INVALID` | `manifest_unreadable`, `manifest_too_large`, `manifest_digest_mismatch`, `manifest_malformed`, `manifest_schema_unsupported`, `manifest_not_canonical`, `path_unsafe`, `duplicate_path` |
| `SIDECAR_ARTIFACT_INCOMPATIBLE` | `protocol_unsupported`, `capability_missing`, `platform_mismatch` |
| `SIDECAR_ARTIFACT_MISSING` | `sidecar_missing`, `model_root_missing`, `model_file_missing`, `artifact_unreadable` |
| `SIDECAR_ARTIFACT_MISMATCH` | `sidecar_size_mismatch`, `sidecar_digest_mismatch`, `model_file_size_mismatch`, `model_file_digest_mismatch`, `model_file_unlisted`, `model_tree_too_large` |
| `SIDECAR_PROTOCOL` | `handshake_invalid`, `stale_generation`, `response_id_mismatch`, `response_too_large`, `response_malformed`, `result_invalid` |
| `SIDECAR_TERMINATED` | `process_exited`, `sidecar_start_failed`, `sidecar_runtime_failed` |
| `SIDECAR_TIMEOUT` | none, or `startup_timeout` |
| `SIDECAR_NOT_RUNNING` | `host_terminated` |

The TypeScript client exposes `error.code` and, when it is a stable code,
`error.reason`.

## Redact a synthetic note

Install the typed client and Tauri API in the front-end workspace, or import
`js/openmedkit-tauri/src/client.ts` directly while developing this repository:

```ts
import {
  deidentify,
  shutdownSidecar,
} from "@openmed/openmedkit-tauri";

const note =
  "Callback 425-555-0100 or email rowan@example.test.";

const result = await deidentify(note, {
  policy: "hipaa_safe_harbor",
  docId: "synthetic-tauri-example",
});

document.querySelector("#redacted")!.textContent = result.deidentifiedText;
console.info(`Redacted ${result.spans.length} spans`);

window.addEventListener("beforeunload", () => {
  void shutdownSidecar();
});
```

Do not log `note`, the protocol request, or the returned redacted text. The
sidecar writes structured operational records to stderr containing only event
names, counts, durations, and a keyed request-ID hash. It never logs request
text, document IDs, model paths, detected surfaces, or exception details.

Requests are limited to 1,000,000 Unicode code points and 4,000,000 UTF-8
bytes. Returned span offsets use OpenMed's cross-platform Unicode code-point
coordinate space, not JavaScript UTF-16 code units. The clients reject invalid,
overlapping, out-of-range, or excessive span records before returning them to
application code.

For a rules-only smoke test that does not load a learned model, set
`deterministicOnly: true`. Production PHI workflows should use a locally
bundled PII model plus the mandatory policy safety sweep.

## Qualification

`cargo test` in `js/openmedkit-tauri/src-tauri` runs the fake-process
qualification. It copies a synthetic `openmed-fake-sidecar` executable into a
temporary bundle, pins it in a manifest, and drives the real host through
missing, changed, added and incompatible artifacts, crashes mid-request,
stderr failures, stale and mismatched responses, abandoned requests,
deadlines, teardown, and oversized or malformed responses. It also checks that
the host returns every shared offset-contract vector in
`tests/fixtures/parity/offset_contract.json` unchanged. That fake engine
replays the reference spans; it shows the host carries code-point offsets and
Unicode text intact, not that a model finds them.

The packaged lane checks a real sidecar. It is defined as the
`packaged-sidecar` job in `.github/workflows/tauri-sidecar.yml` for Linux,
macOS and Windows, and runs only when the workflow is dispatched manually with
`packaged_lane` enabled. It builds the PyInstaller sidecar, provisions the
model into a local directory in a separate setup step, writes the manifest,
and runs `tests/packaged_sidecar_lane.rs`. That test compares the sidecar with
the shared golden vector in
`tests/unit/service/fixtures/sidecar_stdio_golden.json`, runs one model-backed
request as a leakage check, confirms that a tampered or missing model refuses
before start, and uploads value-free evidence (platform, architecture,
manifest digest, counts). To run it locally:

```bash
cd js/openmedkit-tauri/src-tauri
OPENMED_TAURI_LANE_SIDECAR=/path/to/openmed-sidecar \
OPENMED_TAURI_LANE_MANIFEST=/path/to/openmed-sidecar.json \
OPENMED_TAURI_LANE_MODEL_ROOT=/path/to/model \
OPENMED_SIDECAR_HASH_SECRET=sidecar-golden-secret \
cargo test --locked --test packaged_sidecar_lane -- --ignored
```

## Wire protocol

The Tauri wrapper normally owns the protocol. For diagnostics, one request is
one JSON line on stdin. The host uses `<generation>:<sequence>` request ids:

```json
{"id":"0f3c9a52d4e1b7a08c6e2f1d9b4a7c35:1","operation":"deidentify","text":"Callback 425-555-0100.","options":{"deterministic_only":true,"policy":"hipaa_safe_harbor"}}
```

Stdout returns exactly one correlated line. This display abbreviates each span;
the actual response includes every canonical `OpenMedSpan` field:

```json
{"id":"0f3c9a52d4e1b7a08c6e2f1d9b4a7c35:1","ok":true,"result":{"deidentified_text":"Callback [phone_number].","spans":[{"canonical_label":"PHONE","start":9,"end":21,"action":"mask","replacement":"[phone_number]"}]}}
```

Supported operations are `ping`, `deidentify`, and `shutdown`. Closing stdin,
sending `SIGINT`/`SIGTERM`, or sending `shutdown` releases cached model objects
and exits cleanly. Direct protocol hosts can pin the pre-provisioned model with
`OPENMED_SIDECAR_MODEL`; `options` deliberately does not accept a model path.
Duplicate JSON keys, non-finite numbers, empty text, and response lines above
64,000,000 bytes fail closed.
