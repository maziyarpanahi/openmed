# Hardened Distroless Service Image

OpenMed ships a distroless production image for the REST service at
`deploy/docker/Dockerfile.distroless`. It keeps build tools, shells, package
managers, and source checkout metadata out of the runtime layer, then runs the
service as the fixed non-root UID `65532`.

## Build

Build the current service image and the hardened image from the repository
root:

```bash
docker build -t openmed:service .
docker build -f deploy/docker/Dockerfile.distroless -t openmed:distroless .
```

Inspect their compressed runtime sizes:

```bash
docker images 'openmed' \
  --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}'
```

## Run

Run the distroless image with a read-only root filesystem, no Linux
capabilities, no privilege escalation, a writable model-cache volume, and a
tmpfs for temporary files:

```bash
docker volume create openmed-cache

docker run --rm -p 8080:8080 \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=128m \
  --mount type=volume,source=openmed-cache,target=/cache \
  -e OPENMED_PROFILE=prod \
  -e OPENMED_SERVICE_KEEP_ALIVE=10m \
  -e OPENMED_SERVICE_MAX_RESIDENT_MODELS=2 \
  openmed:distroless
```

The image sets `OPENMED_CACHE_DIR=/cache/openmed` and
`HF_HOME=/cache/huggingface`; mount `/cache` as a volume so model downloads are
reused across restarts. If you bind-mount a host directory instead of using a
Docker volume, make it writable by UID/GID `65532`.

## Probes

The Docker `HEALTHCHECK` calls `/readyz`, matching the service's readiness
contract. Orchestrators should keep liveness and readiness split:

```bash
curl http://127.0.0.1:8080/livez
curl http://127.0.0.1:8080/readyz
```

Use `/livez` for process liveness and `/readyz` for startup readiness. The
legacy `/health` endpoint remains available for clients that need version and
profile metadata, but it is not used by the hardened image healthcheck.

## Smoke Test

The default test suite keeps container and Hugging Face downloads opt-in. Run
the full distroless smoke test explicitly when Docker is available:

```bash
OPENMED_RUN_DISTROLESS_IMAGE_TEST=1 \
  .venv/bin/python -m pytest tests/integration/test_distroless_image.py -q
```

The smoke test builds the distroless image, starts it with `--read-only`,
`--cap-drop=ALL`, `no-new-privileges`, a `/tmp` tmpfs, and a writable `/cache`
volume, verifies `/livez` and `/readyz`, and runs a synthetic
`/pii/deidentify` request.

## Runtime Comparison

| Area | Current service image (`Dockerfile`) | Distroless image |
|---|---|---|
| Runtime base | `python:3.11-slim` | `gcr.io/distroless/python3-debian12:nonroot` |
| Runtime user | Root by default | Fixed non-root UID/GID `65532` |
| Shell and package manager | Debian shell, `apt` database, and `pip` remain available | No shell or package manager in the final stage |
| Python dependencies | Installed into the image's system environment | Copied from the builder into `/opt/openmed/python` |
| Model cache | `/root/.cache/huggingface` | Writable `/cache` volume with `OPENMED_CACHE_DIR=/cache/openmed` |
| Root filesystem | Writable unless runtime flags override it | Designed for `--read-only` with a `/tmp` tmpfs |
| Linux capabilities | Docker default capability set unless overridden | Run command drops all capabilities |
| Healthcheck | `/health` | `/readyz`, with `/livez` kept separate for liveness |

The distroless image removes interactive debugging tools from production by
design. Use the standard image for local diagnosis if you need a shell, then
promote the distroless image for production deployments.
