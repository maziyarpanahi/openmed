# gRPC Service

OpenMed ships a schema-first gRPC service for polyglot integrations that need
typed contracts and server-streaming de-identification. The gRPC service uses
the same `ServiceRuntime` and warm-pool model loader as the REST service, so
preloaded models, resident-model limits, and keep-alive behavior stay
consistent across transports.

## Install

Install the service extra to get the gRPC runtime dependencies:

```bash
uv pip install -e ".[service]"
```

Developers regenerating protobuf stubs need dev dependencies:

```bash
uv sync --extra dev
make grpc-proto
```

## Contract

The OpenMed service contract is committed at
`openmed/service/proto/openmed.proto`. The client-only KServe V2 wire subset
used by the [remote inference backend](kserve-triton.md) is committed at
`openmed/service/proto/kserve_v2.proto`. Generated Python modules for both are
committed under `openmed/service/proto/generated/`.

The `OpenMedService` exposes:

- `Analyze`: unary equivalent of `POST /analyze`
- `Extract`: unary equivalent of `POST /pii/extract`
- `Deidentify`: unary equivalent of `POST /pii/deidentify`
- `StreamDeidentify`: unary request with server-streamed redacted fragments

Response messages include typed entity records and the canonical `OpenMedSpan`
message shape for callers that consume schema-versioned span contracts.

## Start A Server

```python
from openmed.service.grpc_server import serve

server = serve("127.0.0.1:50051")
server.wait_for_termination()
```

## Client Example

```python
import grpc

from openmed.service.proto.generated import openmed_pb2, openmed_pb2_grpc

channel = grpc.insecure_channel("127.0.0.1:50051")
stub = openmed_pb2_grpc.OpenMedServiceStub(channel)

response = stub.Deidentify(
    openmed_pb2.DeidentifyRequest(
        text="Patient Maria Garcia can be reached at maria@example.test.",
        method="mask",
        lang="en",
    )
)

print(response.deidentified_text)
for entity in response.pii_entities:
    print(entity.label, entity.start, entity.end)
```

## Streaming De-Identification

`StreamDeidentify` accepts either explicit `chunks` or a single `request.text`
with an optional `chunk_size`. The stream yields redacted text fragments as they
become safe to emit and ends with a final event that carries aggregate stream
metadata and final spans.

```python
request = openmed_pb2.DeidentifyStreamRequest(
    request=openmed_pb2.DeidentifyRequest(
        text="Long synthetic clinical note...",
        method="mask",
    ),
    chunk_size=4096,
    max_buffer=8192,
)

for event in stub.StreamDeidentify(request):
    if event.redacted_text:
        print(event.redacted_text, end="")
```

## Regeneration Gate

Run this before committing proto changes:

```bash
make grpc-proto
make grpc-proto-check
```

The unit test suite also runs the same drift check, so CI fails when
`openmed.proto` and the committed generated stubs disagree.

## Standard health probes

The service extra pins `grpcio-health-checking==1.81.1`, the Apache-2.0
upstream implementation of `grpc.health.v1.Health`. This matches the committed
OpenMed generated stubs' grpcio version; the package requires grpcio 1.81.1 or
newer and protobuf 6.33.5 through the 6.x series. No reflection service is enabled.

`Check` and `Watch` expose only the standard status enum for the overall server
(empty service name) and `openmed.service.v1.OpenMedService`. Both start
`NOT_SERVING`. `server.preload_runtime()` invokes the configured runtime's actual
preload and changes them to `SERVING` only after success. `serve()` performs this
preload after starting the listener. Failed preload exposes only
`grpc_preload_failed` and `serve()` stops its listener.

For a custom listener, use `create_grpc_server()`, configure the listener,
call `start()`, then `preload_runtime()`. The wrapper forwards `add_secure_port`
credentials unchanged, so health and OpenMed RPCs share the same TLS listener.
Call `begin_drain()` before the host stops accepting work or drains existing
requests; `stop(grace)` also publishes terminal `NOT_SERVING` before stopping.
A preload finishing after drain cannot restore readiness. This notification seam
does not itself implement coordinated admission or runtime lease draining (#3576).
Listener binding and credential policy remain owned by #3546.

For Compose, an image that explicitly includes the separately maintained
`grpc_health_probe` executable can use an exec healthcheck:

```yaml
healthcheck:
  test: ["CMD", "/bin/grpc_health_probe", "-addr=127.0.0.1:50051", "-service=openmed.service.v1.OpenMedService"]
  interval: 10s
  timeout: 3s
```

The same container-local exec probe supports a loopback-bound Helm deployment:

```yaml
readinessProbe:
  exec:
    command: ["/bin/grpc_health_probe", "-addr=127.0.0.1:50051", "-service=openmed.service.v1.OpenMedService"]
  periodSeconds: 10
  timeoutSeconds: 3
```

These examples require a plaintext local listener and the executable in the image;
OpenMed does not install it. A TLS listener requires that probe's TLS flags and
trusted certificate configuration, rather than an additional insecure health
port. Use the same health protocol over your configured TLS channel. The examples
are configuration guidance; the focused tests exercise actual ephemeral loopback
and TLS servers, not a deployed Compose or Helm cluster.
