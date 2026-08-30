# Warehouse Remote-Function Handler

This example deploys OpenMed as a BigQuery-compatible HTTP remote function.
BigQuery sends a batch of scalar calls in one request; the handler groups the
non-empty rows by policy, invokes `process_batch(operation="deidentify")`, and
returns one ordered value per input row.

The application accepts the documented BigQuery envelope:

```json
{
  "requestId": "synthetic-request-001",
  "caller": "//bigquery.googleapis.com/projects/example/jobs/synthetic-job",
  "sessionUser": "warehouse.user@example.test",
  "calls": [
    ["Patient Jane Roe called 555-0101.", "hipaa_safe_harbor"],
    [null, "hipaa_safe_harbor"]
  ]
}
```

A successful response has the same number of replies:

```json
{"replies":["Patient [PERSON] called [PHONE].",null]}
```

Malformed requests use BigQuery's failure envelope and a non-retryable `400`
status. Processing failures use a generic `503` response. Neither response
contains input text or an exception message.

## Policy selection

The SQL function's optional second call value selects a policy for that row:

```sql
SELECT `PROJECT_ID.DATASET_ID.openmed_deidentify`(
  note,
  'hipaa_safe_harbor'
)
FROM `PROJECT_ID.DATASET_ID.synthetic_notes`;
```

The HTTP service also accepts a request-wide policy from
`X-OpenMed-Policy`, the `policy` query parameter, or
`userDefinedContext.policy`. Request-wide selectors must agree. A row policy
must also agree with a request-wide policy, so a SQL argument cannot weaken a
profile enforced by a trusted gateway. Rows without any selection use
`hipaa_safe_harbor`.

Different row policies are processed in separate vector batches, then restored
to their original reply positions. SQL `NULL` and empty strings are preserved
without loading the model.

## Run locally with synthetic data

Install the model and service extras, then start the module-level application:

```bash
uv pip install -e ".[hf,service]"
uvicorn openmed.integrations.remote_function:app \
  --host 127.0.0.1 \
  --port 8080 \
  --no-access-log
```

Post only synthetic text during setup:

```bash
curl --fail-with-body \
  --header 'Content-Type: application/json' \
  --header 'X-OpenMed-Policy: hipaa_safe_harbor' \
  --data '{"calls":[["Patient Jane Roe called 555-0101."]]}' \
  http://127.0.0.1:8080/
```

## Build and deploy the container

Build from the repository root so the descriptor can install the current
source tree:

```bash
docker build \
  --file examples/warehouse-remote-function/Dockerfile \
  --tag REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY/openmed-warehouse:VERSION \
  .
docker push REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY/openmed-warehouse:VERSION
```

Deploy the image as an authenticated Cloud Run service. Do not make an endpoint
that receives clinical text public:

```bash
gcloud run deploy openmed-warehouse-handler \
  --image REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY/openmed-warehouse:VERSION \
  --region REGION \
  --port 8080 \
  --no-allow-unauthenticated
```

The BigQuery connection's service account needs permission to invoke that
service. Project creation, connection creation, IAM grants, and VPC controls
are intentionally operator-owned and are not automated here. See Google's
[remote-function guide](https://cloud.google.com/bigquery/docs/remote-functions)
for those steps.

Copy `create_function.sql`, replace its project, dataset, region, connection,
and endpoint placeholders, and run it in BigQuery. `max_batching_rows` bounds
one HTTP request; tune it against the selected model's memory and latency.
BigQuery can retry a batch, so the handler is stateless and idempotent.

## Privacy and offline operation

- The application does not log request bodies, BigQuery request metadata,
  policy values, replies, or exception messages. The container entrypoint also
  disables Uvicorn access logs. Keep raw text out of URL paths, query strings,
  headers, tracing attributes, crash reporters, and platform log labels.
- The endpoint still receives raw clinical text. Deploy it only inside an
  approved cloud/VPC and regional processing boundary, and review BigQuery,
  Cloud Run, load-balancer, support, backup, and query-history retention.
- Pre-stage the configured OpenMed model in an encrypted image layer or mounted
  cache. After that explicit download succeeds, run with `OPENMED_OFFLINE=1`
  so request-time inference cannot make an outbound model request.
- The checked-in request and SQL examples are synthetic. Do not commit real
  warehouse rows, HTTP captures, model traces, or platform logs.
