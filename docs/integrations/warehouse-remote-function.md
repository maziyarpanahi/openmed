# Warehouse Remote-Function Handler

`openmed.integrations.remote_function` exposes a FastAPI application for
BigQuery's HTTP remote-function protocol. It accepts the batched `calls`
envelope, performs vectorized OpenMed de-identification, and returns a matching
`replies` array. Install the optional service and model dependencies:

```bash
uv pip install -e ".[hf,service]"
```

Run the packaged application with access logging disabled:

```bash
uvicorn openmed.integrations.remote_function:app \
  --host 127.0.0.1 \
  --port 8080 \
  --no-access-log
```

The public Python surface is:

- `create_app(process_batch_fn=None)`: build an embeddable FastAPI app and
  optionally inject a batch implementation for offline tests.
- `redact_remote_function_batch(payload, request_policy=None,
  process_batch_fn=None)`: validate and process one envelope without HTTP.
- `app`: the module-level ASGI entrypoint used by Uvicorn.

Each call row contains `text` and may contain `policy`. A request-wide policy
can instead come from `X-OpenMed-Policy`, `?policy=`, or
`userDefinedContext.policy`. Conflicting selectors fail closed. The handler
canonicalizes policy aliases, groups mixed-policy rows into vector batches,
and restores original order. Nulls and empty strings pass through unchanged.

Malformed or empty batches return a bounded `{"errorMessage": ...}` response
with status `400`; processing failures return a generic `503` envelope. The
adapter does not log bodies, request metadata, policy values, output, or
exception messages. Errors identify only protocol fields and row offsets.

For the pinned container entrypoint, BigQuery DDL, synthetic request, IAM
boundary, model pre-staging, and `OPENMED_OFFLINE=1` guidance, use the
[`examples/warehouse-remote-function/`](https://github.com/maziyarpanahi/openmed/tree/master/examples/warehouse-remote-function)
deployment example.
