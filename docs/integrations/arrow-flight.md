# Arrow Flight De-identification

OpenMed's Arrow Flight integration accepts a stream of Arrow record batches,
de-identifies one configured string column, and returns each redacted batch as
soon as it is processed. The service preserves the input schema, row count,
nulls, and every non-target column. It never materializes the complete stream.

Install the optional columnar dependency:

```bash
uv pip install -e ".[columnar]"
```

## Start a Server

```python
from openmed.integrations.arrow_flight import (
    ArrowFlightDeidentificationServer,
)

server = ArrowFlightDeidentificationServer(
    "grpc://127.0.0.1:8815",
    batch_size=512,
)
server.serve()
```

`ArrowFlightDeidentificationServer` uses OpenMed's PII model and
`process_batch(operation="deidentify")` by default. You can set server-wide
defaults with `text_column=` and `policy=`, or select them per exchange in the
Flight command descriptor.

## Exchange Record Batches

```python
import pyarrow as pa
import pyarrow.flight as flight

from openmed.integrations.arrow_flight import make_deidentify_descriptor

client = flight.connect("grpc://127.0.0.1:8815")
descriptor = make_deidentify_descriptor(
    "clinical_note",
    policy="hipaa_safe_harbor",
)
writer, reader = client.do_exchange(descriptor)

schema = pa.schema(
    [
        ("record_id", pa.int64()),
        ("clinical_note", pa.string()),
        ("status", pa.string()),
    ]
)
writer.begin(schema)

for input_batch in source_batches:
    writer.write_batch(input_batch)
    output_batch = reader.read_chunk().data
    consume(output_batch)

writer.done_writing()
```

The helper creates a versioned JSON `FlightDescriptor` command. For
`DoExchange`, the descriptor is the request-metadata channel; its policy takes
precedence over a server default. Keeping configuration in the descriptor lets
one server apply different OpenMed policy profiles without mixing raw clinical
cells into RPC metadata.

## Privacy and Streaming Contract

- Each incoming `RecordBatch` is passed to `process_batch` and returned before
  the server reads the entire stream.
- Only the target string column is converted to Python values for redaction.
  All other Arrow arrays are passed through unchanged.
- Response batches retain the input schema and number of rows, including null
  placement.
- The service does not log raw or redacted cell values. Errors identify only
  the column and row position.
- The command descriptor must not contain patient data. It is only for the
  text-column name, policy name, and descriptor version.

## Authentication and TLS Hooks

The server constructor forwards Arrow Flight's `auth_handler`,
`tls_certificates`, `verify_client`, `root_certificates`, and `middleware`
options. These are deployment hooks, not a complete security policy. Production
operators remain responsible for certificate management, authentication design,
network isolation, and authorization.
