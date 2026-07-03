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

The source protobuf contract is committed at
`openmed/service/proto/openmed.proto`. Generated Python modules are committed
under `openmed/service/proto/generated/`.

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
