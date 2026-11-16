# OpenMedKit for Tauri

Typed Tauri v2 commands and a front-end client for the local OpenMed
de-identification sidecar. The trusted Rust host pins the packaged sidecar
and the offline model in a versioned manifest (`sidecar_manifest.rs`) and
verifies both before every process start; the renderer can submit bounded text
and policy options but cannot choose local filesystem paths. Each sidecar
process is bound to a generation id, and crashes, stale responses, deadlines
and teardown return typed, value-free errors (`sidecar_command.rs`).

See [`examples/tauri/redact-app.md`](../../examples/tauri/redact-app.md) for the
self-contained binary build, manifest generation, Tauri registration, offline
model setup, lifecycle and error codes, graceful shutdown, and the
qualification lanes.

```bash
cd src-tauri && cargo test   # fake-process qualification
npm test                     # TypeScript client
```
