# Self-hosted Compose deployment

`deploy/openmed-compose.yaml` is a single-service, offline-first bundle for
running the OpenMed REST service on a host you control. It uses the hardened
non-root image, publishes only to loopback by default, keeps the container root
filesystem read-only, and persists model artifacts in a named Docker volume.

This bundle does not start a database, model registry, telemetry collector,
OpenHIM, MCP server, or remote model provider. A running service does not need
any outbound network connection after its image and model artifacts have been
prepared.

## Prepare the image and model artifacts

Build the image while the build host has access to the package indexes and base
image registry required by the checked-in Dockerfile:

```bash
docker compose -f deploy/openmed-compose.yaml build
```

Before an offline hand-off, populate the `openmed-cache` volume with the model
artifacts that the service will use, or place a local model directory under
`models/` in the repository. Local model directories are mounted at `/models`
read-only. Set `OPENMED_SERVICE_PRELOAD_MODELS` to a path such as
`/models/synthetic-ner` when the service should load one of those artifacts
during startup. Do not put patient data, credentials, or restricted datasets in
the image, cache, or model directory.

## Start offline

Once the image and model artifacts are present, start without building or
pulling anything:

```bash
docker compose -f deploy/openmed-compose.yaml up -d --no-build
```

The bundle sets `OPENMED_OFFLINE=1`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`. Missing model files
therefore fail locally instead of silently triggering a download. Check the
container and its readiness endpoint:

```bash
docker compose -f deploy/openmed-compose.yaml ps
curl --fail http://127.0.0.1:8080/readyz
```

The default port mapping is `127.0.0.1:8080:8080`. To use another local port,
set `OPENMED_PORT`; to intentionally expose the service on another interface,
set `OPENMED_BIND_ADDRESS` and configure authentication, TLS, and a trusted-host
allowlist at the same time.

## Volumes and permissions

The service runs as UID/GID `65532:65532` with a read-only root filesystem:

- `openmed-cache:/cache` is the persistent writable volume. It contains the
  Hugging Face cache under `/cache/huggingface` and OpenMed data under
  `/cache/openmed`.
- `${OPENMED_MODEL_DIR:-../models}:/models:ro` is an optional read-only bind
  mount for pre-staged local model artifacts. The default resolves to the
  repository-level `models/` directory because the Compose file is under
  `deploy/`.
- `/tmp` is the only other writable location, supplied as a bounded `tmpfs`.

Docker normally creates the named cache volume with the image's permissions.
If it is replaced with a host bind mount, create that directory and make it
writable by `65532:65532`; keep the model bind mount readable by that UID and
read-only. Do not make the whole repository writable by the container.

## Optional integrations

The defaults explicitly disable the privacy gateway endpoint, OpenHIM
mediator, Prometheus metrics, and OTLP tracing. `HF_TOKEN` is not included in
the bundle. Enable an integration only through a reviewed Compose override or
secret store, and document its network boundary separately. In particular,
keep credentials out of YAML, images, logs, and support artifacts. Enabling a
remote integration is an intentional opt-in and is outside the offline
startup guarantee.

The REST service keeps operational logs free of request bodies. Use synthetic
inputs for smoke checks and keep any reversible mappings or raw clinical text
outside container logs and the persistent cache.

This deployment mechanism is an operational packaging aid, not a compliance
certification or a clinical decision guarantee.
