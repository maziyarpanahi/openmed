# Multi-Arch Container Images

OpenMed publishes a service container image for both `linux/amd64` and
`linux/arm64`. The multi-platform tags are manifest lists, so Docker, containerd,
Kubernetes, and compatible runtimes select the native image for the node that
pulls the tag.

The supported production platforms are:

| Platform | Typical hosts |
|---|---|
| `linux/amd64` | x86_64 servers, most CI runners, Intel or AMD developer machines |
| `linux/arm64` | AWS Graviton, Ampere servers, Apple-silicon Docker Desktop |

The image is published to GitHub Container Registry:

```bash
docker pull ghcr.io/maziyarpanahi/openmed:latest
```

To force a specific architecture, pass `--platform`:

```bash
docker pull --platform linux/arm64 ghcr.io/maziyarpanahi/openmed:latest
docker pull --platform linux/amd64 ghcr.io/maziyarpanahi/openmed:latest
```

Run the service with the same environment variables used by the local Docker
and Compose workflows:

```bash
docker run --rm -p 8080:8080 \
  -e OPENMED_PROFILE=prod \
  -e OPENMED_SERVICE_KEEP_ALIVE=10m \
  ghcr.io/maziyarpanahi/openmed:latest
```

Inspect the manifest list before promoting a tag:

```bash
docker buildx imagetools inspect ghcr.io/maziyarpanahi/openmed:latest
```

The output should include both `linux/amd64` and `linux/arm64` platform entries.

## Local Builds

The production Dockerfile lives at `deploy/docker/Dockerfile` and pins its
Python base image by digest. Build a native local image with:

```bash
docker build -f deploy/docker/Dockerfile -t openmed:local .
```

For a registry-backed multi-platform build, use Buildx:

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f deploy/docker/Dockerfile \
  -t ghcr.io/maziyarpanahi/openmed:dev \
  --push .
```

CI validates each architecture by importing `openmed` inside the built image and
running a synthetic de-identification path. Pull request runs build and smoke-test
both architectures without publishing. Pushes to `master`, version tags, and
manual dispatches publish manifest-list tags after the per-architecture smoke
checks pass. Published image digests are signed and attested by the follow-up
[Container Image Signing](../supply-chain/image-signing.md) workflow.
