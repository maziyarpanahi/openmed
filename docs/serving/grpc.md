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

server = serve()  # plaintext, loopback only: 127.0.0.1:50051
server.wait_for_termination()
```

`serve()` validates the listener before it creates the server, binds it, and
only then starts serving. It raises `GrpcStartupError` instead of returning a
server when the configuration is refused or the address cannot be bound, so a
returned server is always listening.

## Transport Security

The default listener is `127.0.0.1:50051`. Plaintext is accepted only on
loopback addresses (`127.0.0.0/8`, `[::1]`, `localhost`) and on local
`unix:` / `unix-abstract:` sockets. Wildcard (`0.0.0.0`, `[::]`), routable and
named hosts require server TLS credentials:

```python
from openmed.service.grpc_security import GrpcTLSConfig
from openmed.service.grpc_server import serve

server = serve(
    "0.0.0.0:50051",
    tls=GrpcTLSConfig(
        certificate_chain_file="/etc/openmed/grpc/server-chain.pem",
        private_key_file="/etc/openmed/grpc/server-key.pem",
    ),
)
```

Add `client_ca_file` to require mutual TLS. Clients must then present a
certificate issued by one of the CAs in that bundle; clients without a
certificate, or with one from another CA, fail the handshake:

```python
tls = GrpcTLSConfig(
    certificate_chain_file="/etc/openmed/grpc/server-chain.pem",
    private_key_file="/etc/openmed/grpc/server-key.pem",
    client_ca_file="/etc/openmed/grpc/client-ca.pem",
)
```

A mutual-TLS client trusts the server CA and presents its own certificate:

```python
from pathlib import Path

import grpc

credentials = grpc.ssl_channel_credentials(
    root_certificates=Path("/etc/openmed/grpc/server-ca.pem").read_bytes(),
    private_key=Path("/etc/openmed/grpc/client-key.pem").read_bytes(),
    certificate_chain=Path("/etc/openmed/grpc/client-chain.pem").read_bytes(),
)
channel = grpc.secure_channel("openmed-grpc.internal:50051", credentials)
```

`require_client_cert` defaults to whether a client CA is configured. Setting it
to `True` without a client CA, or to `False` with one, is refused rather than
silently ignored.

Credential material is loaded and checked before anything is bound. Startup is
refused when a file is missing or unreadable, is not valid PEM, is empty or
larger than 1 MiB, when the private key is encrypted or does not match the leaf
certificate, or when the leaf certificate is expired or not yet valid. The
certificate chain file must list the leaf first, followed by any
intermediates. The private key must be unencrypted PEM, so protect it with file
permissions or a secret mount.

### Development override

For a disposable lab that needs plaintext on a non-loopback address, pass
`allow_insecure_remote_dev=True`. The override is accepted only when the
`OPENMED_PROFILE` (or the supplied runtime's profile) is `dev` or `test`, and
it logs `grpc_insecure_dev_override_active`. Under any other profile,
including `prod` and an unset `OPENMED_PROFILE`, setting the override fails
startup with `grpc_insecure_dev_override_rejected`, even for a loopback
address.

### Startup reason codes

`GrpcStartupError` exposes `reason`, an optional `category`
(`certificate_chain`, `private_key`, `client_ca`, `client_auth`, `listener`, or
`override`) and, when a certificate was parsed, `fingerprint_sha256`. Messages
never include file paths, addresses, PEM text or key bytes.

| Reason | Meaning |
| --- | --- |
| `grpc_address_invalid` | The bind address is not `host:port`, `[ipv6]:port`, `unix:` or `unix-abstract:` |
| `grpc_insecure_remote_bind` | Plaintext was requested on a non-loopback address |
| `grpc_insecure_dev_override_rejected` | The development override was set outside `dev`/`test` |
| `grpc_tls_incomplete` | A certificate, key, or required client CA is not configured |
| `grpc_tls_client_auth_ambiguous` | A client CA was configured with `require_client_cert=False` |
| `grpc_tls_unreadable` | A credential file could not be read |
| `grpc_tls_malformed` | A credential file is empty, oversized, encrypted or not valid PEM |
| `grpc_tls_key_mismatch` | The private key does not match the leaf certificate |
| `grpc_tls_certificate_expired` | The leaf certificate is past its `notAfter` |
| `grpc_tls_certificate_not_yet_valid` | The leaf certificate is before its `notBefore` |
| `grpc_bind_failed` | grpcio raised or reported that nothing was bound |

### Custom listeners

`create_grpc_server()` returns an unbound server and accepts `interceptors`.
Use `resolve_grpc_listener()` and `bind_grpc_listener()` to apply the same
policy to a server you construct yourself. `GrpcListener.report()` returns a
value-free summary (loopback, transport, override use, certificate and client
CA fingerprints, and certificate expiry) for deployment self-checks.

### Certificate rotation

Certificates and keys are read once at startup. To rotate, write the new
chain and key, then restart the process; the restart re-runs every validation
above and a bad replacement fails startup instead of serving. In-place reload
without a restart is not supported yet.

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
