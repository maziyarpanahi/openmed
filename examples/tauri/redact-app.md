# Tauri sidecar de-identification

This example keeps a desktop note on the local machine. The Tauri host sends
the note to a persistent OpenMed sidecar over newline-delimited JSON, receives
only the redacted text and canonical `OpenMedSpan` records, and shuts the
process down when the application exits.

The sidecar always enables OpenMed's local-only mode. Download the model during
application installation or development, then pass its local directory as
`modelName`. A missing local model fails with a structured error; it never
falls back to a network request.

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
and are loaded from disk or an already-populated OpenMed cache.

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

## Configure Tauri v2

Add the binary stem to `src-tauri/tauri.conf.json`:

```json
{
  "bundle": {
    "externalBin": ["binaries/openmed-sidecar"]
  }
}
```

Copy `js/openmedkit-tauri/src-tauri/sidecar_command.rs` into the application
crate and add these dependencies:

```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tauri = "2"
tauri-plugin-shell = "2"
tokio = { version = "1", features = ["sync"] }
```

Register the persistent state, shell plugin, and commands in the Tauri builder:

```rust
mod sidecar_command;

use sidecar_command::{
    openmed_sidecar_deidentify, openmed_sidecar_ping,
    openmed_sidecar_shutdown, OpenMedSidecarState,
};

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(OpenMedSidecarState::default())
        .invoke_handler(tauri::generate_handler![
            openmed_sidecar_ping,
            openmed_sidecar_deidentify,
            openmed_sidecar_shutdown,
        ])
        .run(tauri::generate_context!())
        .expect("Tauri application failed");
}
```

The command wrapper serializes requests through one long-lived child process.
This lets the Python runtime reuse the same offline loader and cached model. If
the process is killed while a request is running, the command rejects with
`SIDECAR_TERMINATED` instead of returning partial JSON or an unstructured
process error.

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
  modelName: "/absolute/app-resource/models/openmed-pii",
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

For a rules-only smoke test that does not load a learned model, set
`deterministicOnly: true`. Production PHI workflows should use a locally
bundled PII model plus the mandatory policy safety sweep.

## Wire protocol

The Tauri wrapper normally owns the protocol. For diagnostics, one request is
one JSON line on stdin:

```json
{"id":"1","operation":"deidentify","text":"Callback 425-555-0100.","options":{"deterministic_only":true,"policy":"hipaa_safe_harbor"}}
```

Stdout returns exactly one correlated line. This display abbreviates each span;
the actual response includes every canonical `OpenMedSpan` field:

```json
{"id":"1","ok":true,"result":{"deidentified_text":"Callback [phone_number].","spans":[{"canonical_label":"PHONE","start":9,"end":21,"action":"mask","replacement":"[phone_number]"}]}}
```

Supported operations are `ping`, `deidentify`, and `shutdown`. Closing stdin,
sending `SIGINT`/`SIGTERM`, or sending `shutdown` releases cached model objects
and exits cleanly.
