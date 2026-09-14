# OpenMedKit for Tauri

Typed Tauri v2 commands and a front-end client for the local OpenMed
de-identification sidecar. The trusted Rust host pins the offline model; the
renderer can submit bounded text and policy options but cannot choose local
filesystem paths.

See [`examples/tauri/redact-app.md`](../../examples/tauri/redact-app.md) for the
self-contained binary build, Tauri registration, offline model setup, and
graceful shutdown flow.
